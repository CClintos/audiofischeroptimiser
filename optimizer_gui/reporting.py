from __future__ import annotations

import base64
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QBuffer, QIODevice, QMarginsF, QPointF, QRectF, QSizeF, Qt
from PySide6.QtGui import (
    QColor, QFont, QImage, QPageLayout, QPageSize, QPainter, QPainterPath,
    QPdfWriter, QPen, QTextDocument,
)
from PySide6.QtWidgets import QApplication

from .warning_text import warning_info
from . import theme

GROUP_LABELS = {
    "front_voicing": "Whole front stage (matched voicing)",
    "sub": "Subwoofer",
    "high_sym": "Both tweeters (symmetric)",
    "high_crossover_sym": "Both tweeters near the crossover",
    "fl_high": "Front L tweeter",
    "fr_high": "Front R tweeter",
    "mid_sym": "Both midrange drivers (symmetric)",
    "mid_crossover_sym": "Both midrange drivers near the crossover",
    "fl_mid": "Front L midrange",
    "fr_mid": "Front R midrange",
    "low_sym": "Both midbass drivers (symmetric)",
    "low_crossover_sym": "Both midbass drivers near the crossover",
    "fl_low": "Front L midbass",
    "fr_low": "Front R midbass",
}

RESULT_METRICS = (
    ("tonal_error_db", "Tonal accuracy"),
    ("presence_error_db", "Vocal region"),
    ("narrow_peak_penalty_db", "Narrow peaks"),
    ("balance_penalty_db", "L/R balance"),
)
MEANINGFUL_IMPROVEMENT_PERCENT = 1.0

_REPORT_QT_APP: QApplication | None = None


def _ensure_qt_app() -> None:
    global _REPORT_QT_APP
    if QApplication.instance() is None:
        _REPORT_QT_APP = QApplication([])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "Not available"
    return f"{float(value):.{digits}f}{suffix}"


def _delta(base: Any, best: Any) -> str:
    if not isinstance(base, (int, float)) or not isinstance(best, (int, float)):
        return "-"
    value = float(best) - float(base)
    return f"{value:+.3f}"


def _objective_improvement(base: Any, best: Any) -> str:
    if not isinstance(base, (int, float)) or not isinstance(best, (int, float)) or not base:
        return "Not available"
    value = (float(base) - float(best)) / abs(float(base)) * 100.0
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:.1f}%"


def _best_filters(summary: dict[str, Any], report_path: Path) -> dict[str, list[list[float]]]:
    stored = (summary.get("best") or {}).get("added_filters")
    if isinstance(stored, dict):
        return {str(key): list(value) for key, value in stored.items() if value}
    if not report_path.exists():
        return {}
    text = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^### Rank 1:.*?$(.*?)(?=^### |\Z)", text, re.MULTILINE | re.DOTALL)
    if not match:
        return {}
    parsed: dict[str, list[list[float]]] = {}
    band_pattern = re.compile(r"([0-9.]+) Hz Q([0-9.]+) ([+-]?[0-9.]+) dB")
    for line in match.group(1).splitlines():
        if not line.startswith("- ") or ":" not in line:
            continue
        group, values = line[2:].split(":", 1)
        if group not in GROUP_LABELS or "no added filters" in values:
            continue
        bands = [[float(f), float(q), float(g)] for f, q, g in band_pattern.findall(values)]
        if bands:
            parsed[group] = bands
    return parsed


def _table(rows: list[tuple[str, str]], header: tuple[str, str] | None = None) -> str:
    body = []
    if header:
        body.append(f"<tr><th>{html.escape(header[0])}</th><th>{html.escape(header[1])}</th></tr>")
    body.extend(
        f"<tr><td>{html.escape(str(left))}</td><td><b>{html.escape(str(right))}</b></td></tr>"
        for left, right in rows
    )
    return "<table>" + "".join(body) + "</table>"


def _component_table(baseline: dict[str, Any], best: dict[str, Any]) -> str:
    labels = (
        ("objective", "Overall objective"),
        ("tonal_error_db", "Tonal error"),
        ("presence_error_db", "Vocal / presence error"),
        ("target_shape_error_db", "Target contour error (anchor-independent)"),
        ("peak_penalty_db", "Broad peak penalty"),
        ("narrow_peak_penalty_db", "Narrow / raw peak penalty"),
        ("balance_penalty_db", "L/R balance penalty"),
        ("positive_gain_penalty_db", "Positive-gain / headroom penalty"),
        ("protective_output_trim_db", "Protective front output trim"),
        ("spatial_worst_db", "Worst-position error"),
        ("guardrail_penalty", "Guardrail penalty"),
        ("target_rms_null_excluded_db", "Target RMS, nulls excluded (decision metric)"),
        ("target_rms_null_included_db", "Target RMS, nulls included"),
    )
    rows = []
    for key, label in labels:
        if key not in baseline and key not in best:
            continue
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                html.escape(label), _fmt(baseline.get(key)), _fmt(best.get(key)),
                html.escape(_delta(baseline.get(key), best.get(key))),
            )
        )
    return (
        "<table><tr><th>Score component</th><th>Baseline</th><th>Candidate</th>"
        "<th>Delta</th></tr>" + "".join(rows) + "</table>"
    )


def _warning_items(summary: dict[str, Any], mode: str) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for item in summary.get("warnings") or []:
        warnings.append(warning_info(item))
    best = summary.get("best") or {}
    if best.get("left_alone"):
        warnings.append(warning_info(best["left_alone"]))
    if mode == "phase" and summary.get("phase_actions"):
        warnings.append({
            "severity": "warning",
            "colour": "#9a6500",
            "text": "Phase changes are predictions. Fix: re-measure every affected crossover in-car.",
        })
    if not warnings:
        warnings.append({
            "severity": "info",
            "colour": "#2b6684",
            "text": "No blocking measurement or candidate warnings were reported.",
        })
    return warnings[:8]


def _png_data_uri(image: QImage) -> str:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    encoded = base64.b64encode(bytes(buffer.data())).decode("ascii")
    return "data:image/png;base64," + encoded


def _line_chart(
    series: list[dict[str, Any]],
    width: int = 1080,
    height: int = 350,
    zero_line: bool = True,
) -> str:
    _ensure_qt_app()
    clean = []
    all_x: list[float] = []
    all_y: list[float] = []
    for item in series:
        points = []
        for x_value, y_value in zip(item.get("x") or [], item.get("y") or []):
            try:
                x = float(x_value)
                y = float(y_value)
            except (TypeError, ValueError):
                continue
            if x > 0 and math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if len(points) >= 2:
            clean.append((item, points))
            all_x.extend(point[0] for point in points)
            all_y.extend(point[1] for point in points)
    if not clean:
        return ""

    x_min = min(all_x)
    x_max = max(all_x)
    ordered = sorted(all_y)
    low = ordered[max(0, int(len(ordered) * 0.02) - 1)]
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.98))]
    if zero_line:
        low = min(low, 0.0)
        high = max(high, 0.0)
    padding = max(1.0, 0.10 * max(high - low, 4.0))
    y_min = math.floor((low - padding) / 2.0) * 2.0
    y_max = math.ceil((high + padding) / 2.0) * 2.0
    if y_max - y_min < 6.0:
        middle = 0.5 * (y_min + y_max)
        y_min = middle - 3.0
        y_max = middle + 3.0

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#ffffff"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setFont(QFont("Segoe UI", 9))
    left, right, top, bottom = 74.0, 24.0, 46.0, 42.0
    plot = QRectF(left, top, width - left - right, height - top - bottom)
    log_min = math.log10(x_min)
    log_span = max(math.log10(x_max) - log_min, 1e-9)

    def px(value: float) -> float:
        return plot.left() + (math.log10(value) - log_min) / log_span * plot.width()

    def py(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return plot.top() + (y_max - clipped) / (y_max - y_min) * plot.height()

    painter.setPen(QPen(QColor("#d8dde2"), 1))
    painter.drawRect(plot)
    tick_count = 5
    for index in range(tick_count + 1):
        value = y_min + (y_max - y_min) * index / tick_count
        y = py(value)
        painter.setPen(QPen(QColor("#edf0f2"), 1))
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(QColor("#68717a"), 1))
        painter.drawText(QRectF(2, y - 9, left - 9, 18), Qt.AlignmentFlag.AlignRight, f"{value:+.0f} dB")
    frequency_ticks = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    for frequency in frequency_ticks:
        if not x_min <= frequency <= x_max:
            continue
        x = px(float(frequency))
        painter.setPen(QPen(QColor("#edf0f2"), 1))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        label = f"{frequency / 1000:g}k" if frequency >= 1000 else str(frequency)
        painter.setPen(QPen(QColor("#68717a"), 1))
        painter.drawText(QRectF(x - 28, plot.bottom() + 8, 56, 18), Qt.AlignmentFlag.AlignCenter, label)
    if zero_line and y_min <= 0 <= y_max:
        painter.setPen(QPen(QColor("#727b84"), 2))
        painter.drawLine(QPointF(plot.left(), py(0.0)), QPointF(plot.right(), py(0.0)))

    legend_x = plot.left()
    for item, _points in clean:
        color = QColor(str(item.get("color") or "#1f7a5a"))
        pen = QPen(color, 3)
        if item.get("dashed"):
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(legend_x, 22), QPointF(legend_x + 24, 22))
        painter.setPen(QPen(QColor("#30363b"), 1))
        painter.drawText(QPointF(legend_x + 31, 26), str(item.get("label") or "Series"))
        legend_x += 46 + painter.fontMetrics().horizontalAdvance(str(item.get("label") or "Series"))

    painter.save()
    painter.setClipRect(plot)
    for item, points in clean:
        path = QPainterPath()
        path.moveTo(px(points[0][0]), py(points[0][1]))
        for x, y in points[1:]:
            path.lineTo(px(x), py(y))
        pen = QPen(QColor(str(item.get("color") or "#1f7a5a")), 3)
        if item.get("dashed"):
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawPath(path)
    painter.restore()
    painter.end()
    return _png_data_uri(image)


def line_chart_data_uri(
    series: list[dict[str, Any]],
    width: int = 1080,
    height: int = 350,
    zero_line: bool = True,
) -> str:
    """Render the same local chart used by PDF reports for native GUI views."""
    return _line_chart(series, width=width, height=height, zero_line=zero_line)

def _paired_bar_chart(rows: list[tuple[str, float, float]], width: int = 1080) -> str:
    _ensure_qt_app()
    rows = [(label, float(before), float(after)) for label, before, after in rows
            if isinstance(before, (int, float)) and isinstance(after, (int, float))]
    if not rows:
        return ""
    height = 70 + 64 * len(rows)
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#ffffff"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setFont(QFont("Segoe UI", 9))
    painter.setPen(QPen(QColor("#525b63"), 1))
    painter.drawText(QPointF(300, 24), "Baseline")
    painter.drawText(QPointF(410, 24), "Candidate")
    maximum = max(max(before, after) for _label, before, after in rows)
    scale = 460.0 / max(maximum, 0.001)
    for index, (label, before, after) in enumerate(rows):
        y = 48 + index * 64
        painter.setPen(QPen(QColor("#30363b"), 1))
        painter.drawText(QRectF(4, y - 5, 260, 42), Qt.AlignmentFlag.AlignVCenter, label)
        painter.fillRect(QRectF(300, y, before * scale, 13), QColor("#9aa3aa"))
        painter.fillRect(QRectF(300, y + 22, after * scale, 13), QColor("#16805d"))
        painter.drawText(QPointF(310 + before * scale, y + 11), f"{before:.2f}")
        painter.drawText(QPointF(310 + after * scale, y + 33), f"{after:.2f}")
    painter.end()
    return _png_data_uri(image)


def _response_plot(summary: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    plot = full.get("response_plot") or {}
    if plot.get("frequency_hz"):
        return plot
    rows = (((summary.get("best") or {}).get("fixed_anchor_response") or {}).get("checkpoints") or [])
    if not rows:
        return {}
    return {
        "frequency_hz": [row.get("frequency_hz") for row in rows],
        "baseline_error_db": [row.get("baseline_error_db") for row in rows],
        "candidate_error_db": [row.get("candidate_error_db") for row in rows],
        "raw_system_delta_db": [row.get("raw_system_delta_db") for row in rows],
        "pairs": {},
        "drivers": {},
    }


def load_response_plot(summary_path: Path) -> dict[str, Any]:
    """Load the full fixed-anchor response plot, with older-run fallback."""
    summary_path = Path(summary_path)
    summary = _read_json(summary_path)
    details = summary.get("details") or {}
    full_path = summary_path.parent / str(details.get("optimizer_summary", "optimizer_summary.json"))
    return _response_plot(summary, _read_json(full_path))



def result_comparison_plots(
    summary: dict[str, Any], full: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return fixed-anchor response plots for each auditable result stage."""
    comparisons = full.get("response_comparisons") or {}
    if comparisons:
        labels = {
            "supplied": "Supplied",
            "rehabilitated": "Existing tune improved",
            "final": "Final",
        }
        result = {}
        for key, value in comparisons.items():
            if not isinstance(value, dict) or not value.get("frequency_hz"):
                continue
            plot = dict(value)
            plot.setdefault("candidate_label", labels.get(str(key), str(key).title()))
            result[str(key)] = plot
        return result
    final = _response_plot(summary, full)
    if not final:
        return {}
    supplied = dict(final)
    supplied["candidate_error_db"] = list(
        final.get("baseline_error_db") or final.get("candidate_error_db") or []
    )
    supplied["candidate_label"] = "Supplied"
    final = dict(final)
    final["candidate_label"] = "Final"
    return {"supplied": supplied, "final": final}


def load_response_comparisons(summary_path: Path) -> dict[str, dict[str, Any]]:
    """Load every fixed-anchor result stage for selectable GUI comparison."""
    summary_path = Path(summary_path)
    summary = _read_json(summary_path)
    details = summary.get("details") or {}
    full_path = summary_path.parent / str(
        details.get("optimizer_summary", "optimizer_summary.json")
    )
    return result_comparison_plots(summary, _read_json(full_path))

def response_chart_series(plot: dict[str, Any]) -> list[dict[str, Any]]:
    """Build native-GUI series, including opt-in per-driver predicted changes."""
    frequencies = plot.get("frequency_hz") or []
    if not frequencies:
        return []
    series = [
        {
            "label": "Before",
            "x": frequencies,
            "y": plot.get("baseline_error_db"),
            "color": theme.SERIES_BASELINE,
            "visible": True,
        },
        {
            "label": str(plot.get("candidate_label") or "Candidate"),
            "x": frequencies,
            "y": plot.get("candidate_error_db"),
            "color": theme.SERIES_CANDIDATE,
            "visible": True,
        },
        {
            "label": "Target",
            "x": frequencies,
            "y": [0.0] * len(frequencies),
            "color": theme.SERIES_TARGET,
            "dashed": True,
            "visible": True,
        },
    ]
    driver_colours = theme.DRIVER_SERIES
    for index, (role, data) in enumerate((plot.get("drivers") or {}).items()):
        driver_x = data.get("frequency_hz") or frequencies
        driver_y = data.get("change_db") or []
        if len(driver_x) < 2 or len(driver_y) < 2:
            continue
        series.append({
            "label": f"{role} change",
            "x": driver_x,
            "y": driver_y,
            "color": driver_colours[index % len(driver_colours)],
            "visible": False,
            "driver": True,
        })
    return series


def _improvement(base: Any, best: Any) -> tuple[float | None, float | None]:
    if not isinstance(base, (int, float)) or not isinstance(best, (int, float)):
        return None, None
    delta = float(base) - float(best)
    percent = 100.0 * delta / abs(float(base)) if float(base) else 0.0
    return delta, percent


def metric_card_data(
    baseline: dict[str, Any], best: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the same named before/after metrics used by the PDF and GUI."""
    cards = []
    for key, label in RESULT_METRICS:
        delta, percent = _improvement(baseline.get(key), best.get(key))
        cards.append({
            "key": key,
            "label": label,
            "delta_db": delta,
            "percent": percent,
            "state": (
                "neutral" if delta is None or abs(delta) <= 0.02
                else "good" if delta > 0.0 else "warn"
            ),
        })
    return cards


def improvement_verdict(
    baseline: dict[str, Any], best: dict[str, Any],
) -> dict[str, Any]:
    """Describe overall improvement without making the raw objective the headline."""
    delta, percent = _improvement(baseline.get("objective"), best.get("objective"))
    meaningful = bool(
        delta is not None and percent is not None
        and delta > 0.0 and percent >= MEANINGFUL_IMPROVEMENT_PERCENT
    )
    if delta is None or percent is None:
        return {
            "meaningful": False,
            "percent": None,
            "delta": None,
            "heading": "Improvement versus the current tune is unavailable",
            "detail": "This older result does not contain comparable baseline metrics.",
        }
    if meaningful:
        return {
            "meaningful": True,
            "percent": percent,
            "delta": delta,
            "heading": f"{percent:.1f}% improvement versus the current tune",
            "detail": "The candidate cleared the minimum 1% modelled-improvement threshold.",
        }
    return {
        "meaningful": False,
        "percent": percent,
        "delta": delta,
        "heading": "Not meaningfully better than your current tune",
        "detail": (
            "The modelled change is within the repeatability threshold. "
            "Treat it as a tie and keep the baseline unless listening and "
            "re-measurement prove a worthwhile advantage."
        ),
    }


def _metric_cards(
    baseline: dict[str, Any], best: dict[str, Any], *, suppress_percent: bool = False,
) -> str:
    cells = []
    for card in metric_card_data(baseline, best):
        label = card["label"]
        delta = card["delta_db"]
        percent = card["percent"]
        if delta is None:
            value = "Not available"
            detail = ""
            css = "neutral"
        elif suppress_percent:
            value = "No meaningful change"
            detail = f"{abs(delta):.2f} dB modelled difference; within repeatability"
            css = "neutral"
        else:
            value = f"{abs(percent):.0f}% {'better' if delta >= 0 else 'worse'}"
            detail = f"{abs(delta):.2f} dB {'less' if delta >= 0 else 'more'} error"
            css = card["state"]
        cells.append(
            f'<td class="metric {css}"><span>{html.escape(label)}</span><br>'
            f'<b>{html.escape(value)}</b><br><small>{html.escape(detail)}</small></td>'
        )
    return '<table class="metric-grid"><tr>' + ''.join(cells[:2]) + '</tr><tr>' + ''.join(cells[2:]) + '</tr></table>'


def _frequency_region(frequency: float) -> str:
    if frequency < 80:
        return "deep bass weight"
    if frequency < 250:
        return "upper-bass warmth and thickness"
    if frequency < 500:
        return "lower-mid body"
    if frequency < 1200:
        return "vocal body and boxiness"
    if frequency < 4000:
        return "vocal clarity and presence"
    if frequency < 8000:
        return "attack and brightness"
    return "top-end air"


def _plain_findings(summary: dict[str, Any], phase_mode: bool) -> list[str]:
    best = summary.get("best") or {}
    if phase_mode:
        actions = summary.get("phase_actions") or []
        if not actions:
            return ["No polarity, delay or all-pass change cleared the evidence gates. Keeping the baseline is the result."]
        findings = []
        for action in actions[:3]:
            source = str(action.get("source") or "Crossover")
            if action.get("delay_samples") is not None:
                findings.append(f"{source}: delay changes by {int(action['delay_samples']):+d} samples to improve crossover summation.")
            elif action.get("apf_f"):
                findings.append(f"{source}: a residual all-pass correction is proposed near {float(action['apf_f']):.0f} Hz.")
            elif action.get("polarity_channels"):
                findings.append(f"{source}: a polarity change produced the strongest supported summation result.")
        return findings

    findings = []
    audit = best.get("fixed_anchor_response") or {}
    checkpoints = audit.get("checkpoints") or []
    numeric = [row for row in checkpoints if isinstance(row.get("raw_system_delta_db"), (int, float))]
    if numeric:
        largest = max(numeric, key=lambda row: abs(float(row["raw_system_delta_db"])))
        frequency = float(largest["frequency_hz"])
        delta = float(largest["raw_system_delta_db"])
        direction = "reduces" if delta < 0 else "adds"
        findings.append(
            f"The largest predicted system change is {delta:+.1f} dB near {frequency:.0f} Hz. "
            f"That mainly {direction} {_frequency_region(frequency)}."
        )
    baseline = summary.get("baseline") or {}
    components = best.get("components") or {}
    for key, label in (
        ("tonal_error_db", "overall target tracking"),
        ("target_shape_error_db", "the requested target contour"),
        ("presence_error_db", "the vocal region"),
        ("balance_penalty_db", "left/right consistency"),
    ):
        _delta_value, percent = _improvement(baseline.get(key), components.get(key))
        if percent is not None and percent >= 5.0:
            findings.append(f"Modelled {label} improves by about {percent:.0f}%.")
    if not findings:
        findings.append("The candidate is deliberately subtle; its advantage comes from several small, supported corrections rather than one large tonal change.")
    return findings[:4]


def _format_rehabilitation_band(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    return (
        f"{float(value.get('frequency_hz', 0.0)):.1f} Hz / "
        f"Q {float(value.get('q', 0.0)):.2f} / "
        f"{float(value.get('gain_db', 0.0)):+.2f} dB"
    )


def _rehabilitation_report_html(rehabilitation: dict[str, Any]) -> str:
    if not rehabilitation:
        return ""
    verdict = str(rehabilitation.get("verdict") or "unknown")
    verdict_text = (
        "Meaningful improvement beyond the repeatability threshold."
        if verdict == "meaningful_improvement" else
        "No meaningful improvement: the numerical difference is within measurement repeatability."
        if verdict == "no_meaningful_improvement" else
        verdict.replace("_", " ").title()
    )
    accepted_rows = []
    for row in rehabilitation.get("accepted_operations") or []:
        role = str(row.get("channel_role") or GROUP_LABELS.get(
            str(row.get("group") or ""), str(row.get("group") or "DSP")
        ))
        channel = row.get("channel")
        if channel is not None:
            role += f" (channel {int(channel)})"
        slot = "-" if row.get("slot") is None else str(int(row["slot"]))
        accepted_rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                html.escape(str(row.get("operation") or "change").title()),
                html.escape(role), html.escape(slot),
                html.escape(_format_rehabilitation_band(row.get("old"))),
                html.escape(_format_rehabilitation_band(row.get("new"))),
                html.escape(str(row.get("reason") or "")),
            )
        )
    accepted_html = (
        "<table><tr><th>Action</th><th>Driver / channel</th><th>Slot</th>"
        "<th>Previous F / Q / G</th><th>New F / Q / G</th><th>Why</th></tr>"
        + "".join(accepted_rows) + "</table>"
        if accepted_rows else
        '<div class="empty">No existing-filter edit cleared the repeatability and safety gates.</div>'
    )
    rejected_rows = [
        (
            "%s, slot %s" % (
                str(row.get("channel_role") or f"Channel {row.get('channel', '-') }"),
                row.get("slot", "-"),
            ),
            str(row.get("reason") or "Rejected by a hard gate"),
        )
        for row in rehabilitation.get("rejected_operations") or []
    ]
    rejected_html = _table(rejected_rows) if rejected_rows else "<p>No operation was rejected after entering the retained decision set.</p>"
    delta_rows = []
    for stage, values in (rehabilitation.get("component_deltas") or {}).items():
        for component, value in (values or {}).items():
            if isinstance(value, (int, float)):
                delta_rows.append((
                    f"{str(stage).replace('_', ' ')} / {str(component).replace('_', ' ')}",
                    f"{float(value):+.4f}",
                ))
    delta_html = _table(delta_rows) if delta_rows else "<p>No comparable component deltas were available.</p>"
    headroom_rows = []
    for stage, values in (rehabilitation.get("headroom") or {}).items():
        numeric = [float(value) for value in (values or {}).values() if isinstance(value, (int, float))]
        headroom_rows.append((str(stage).title(), f"{min(numeric):.2f} dB minimum" if numeric else "Not available"))
    headroom_html = _table(headroom_rows) if headroom_rows else "<p>Headroom details were not available.</p>"
    consolidation = rehabilitation.get("consolidation") or {}
    mismatch = consolidation.get("max_cascade_error_db")
    audit_rows = [
        ("Repeatability verdict", verdict_text),
        ("Candidate evaluations", str(rehabilitation.get("evaluation_count", 0))),
        ("Cache source", str(rehabilitation.get("cache_source") or "Built in this run")),
        ("Rejected operations", str(len(rehabilitation.get("rejected_operations") or []))),
        ("Consolidation", "Accepted" if consolidation.get("accepted") else "Not used"),
        ("Consolidation mismatch", f"{float(mismatch):.3f} dB" if isinstance(mismatch, (int, float)) else "Not available"),
    ]
    return (
        '<h2>Existing Tune Rehabilitation</h2>'
        f'<div class="callout"><b>{html.escape(verdict_text)}</b></div>'
        '<h3>Accepted DSP operations</h3>' + accepted_html
        + '<h3>Rejected by evidence or safety gates</h3>' + rejected_html
        + '<h3>Objective component deltas</h3>' + delta_html
        + '<h3>Available headroom</h3>' + headroom_html
        + '<h3>Decision audit</h3>' + _table(audit_rows)
    )

def build_report_html(summary: dict[str, Any], full: dict[str, Any], summary_path: Path) -> str:
    mode = str((summary.get("search") or {}).get("mode") or (full.get("run_config") or {}).get("mode") or "peq")
    phase_mode = mode == "phase"
    title = "Phase / Timing Diagnostic" if phase_mode else "PEQ Tuning Report"
    baseline = summary.get("baseline") or {}
    best = summary.get("best") or {}
    best_components = best.get("components") or {}
    inputs = summary.get("inputs") or {}
    source = ((inputs.get("baseline") or {}).get("file") or Path(str(full.get("baseline", "Baseline tune"))).name)
    target = ((inputs.get("target") or {}).get("file") or Path(str(full.get("target", "Target curve"))).name)
    candidate = best.get("file") or "No candidate"
    sessions = (summary.get("gates") or {}).get("measurement_session") or {}
    positions = ["centre", *[str(item) for item in sessions.get("spatial_positions") or []]]
    validation = (summary.get("gates") or {}).get("pair_validation") or []
    noise_guard = (summary.get("gates") or {}).get("measurement_noise_guard") or {}
    active_weights = summary.get("active_weights") or {}
    search_audit = summary.get("search") or {}
    warnings = _warning_items(summary, mode)
    filters = _best_filters(summary, summary_path.parent / "optimizer_report.md")
    actions = summary.get("phase_actions") or []
    families = summary.get("families") or {}
    rehabilitation = summary.get("rehabilitation") or {}
    rehabilitation_tie = (
        rehabilitation.get("verdict") == "no_meaningful_improvement"
    )
    rehabilitation_html = _rehabilitation_report_html(rehabilitation)
    verdict = (
        "A phase candidate was produced. Re-measure every changed crossover before calling it final."
        if phase_mode and actions else
        "No phase write cleared the evidence gates; the baseline remains the recommended result."
        if phase_mode else
        "No PEQ change cleared the measurement-repeatability threshold; keep the supplied tune."
        if rehabilitation_tie else
        "A PEQ candidate was produced from the supplied magnitude measurements and conservative guardrails."
    )
    overview = _table([
        ("Analysis", "Phase / delay / APF only; PEQ preserved" if phase_mode else "PEQ from magnitude / RTA data"),
        ("Source tune", source),
        ("Output candidate", candidate),
        ("Target", target),
        ("Measurements", f"{len(positions)} position(s): {', '.join(positions)}"),
        (
            "Worst-position weight",
            f"{active_weights.get('worst', 0.0):.2f} (active - {len(positions)} positions)"
            if active_weights.get("worst") else
            f"0.00 (single position - would double-count the tonal term, see CHANGELOG.md)",
        ),
        ("Candidates retained", str(summary.get("candidate_count", 0))),
        ("Baseline objective", _fmt(baseline.get("objective"), 4)),
        ("Best objective", _fmt(best.get("objective"), 4)),
        ("Objective improvement", (
            "No meaningful improvement (within repeatability)"
            if rehabilitation_tie else
            _objective_improvement(baseline.get("objective"), best.get("objective"))
        )),
        ("Phase-valid session", "Yes" if sessions.get("phase_valid") else "No / not required"),
    ])
    warning_html = "".join(
        '<li><b style="color:%s">%s</b> %s</li>'
        % (
            item.get("colour", "#9a6500"),
            html.escape(item.get("severity", "warning").upper()),
            html.escape(item.get("text", "")),
        )
        for item in warnings
    )

    validation_rows = []
    for item in validation:
        if item.get("pass") is None:
            detail = "NOT AVAILABLE: measured Together trace was not supplied; PEQ used the solo/system fallback"
        else:
            status = "PASS" if item.get("pass") else "FAIL"
            detail = f"{status}: {item.get('rms_db', '-')} dB RMS against {item.get('threshold_db', '-')} dB limit"
        validation_rows.append((str(item.get("pair", "Pair")).title(), detail))
    validation_html = _table(validation_rows) if validation_rows else "<p>No solo/together validation rows were available.</p>"
    noise_rows = []
    for item in noise_guard.get("midbass") or []:
        noise_rows.append((
            f"Midbass {item.get('range_hz', '')} Hz",
            f"{item.get('floor_db', '-')} dB floor",
        ))
    for item in noise_guard.get("tweeter") or []:
        noise_rows.append((
            f"Tweeter {item.get('range_hz', '')} Hz",
            f"{item.get('floor_db', '-')} dB floor",
        ))
    for branch, points in (noise_guard.get("branches") or {}).items():
        for item in points:
            noise_rows.append((
                f"{str(branch).title()} {float(item.get('frequency_hz', 0.0)):.0f} Hz",
                f"{float(item.get('floor_db', 0.0)):.2f} dB empirical floor",
            ))
    if noise_guard.get("known_eq_delta_status"):
        noise_rows.append((
            "Known EQ delta",
            str(noise_guard.get("known_eq_delta_status")),
        ))
    noise_rows.append((
        "Filter threshold",
        f"{noise_guard.get('required_multiplier', 2.5)}x the local floor",
    ))
    noise_html = _table(noise_rows)
    census = search_audit.get("problem_census") or {}
    worth_rows = [
        (
            f"{item.get('group', 'Group')} @ {float(item.get('frequency_hz', 0.0)):.0f} Hz",
            f"{item.get('source', 'tonal')} - priority {float(item.get('recoverable_error', 0.0)):.2f}",
        )
        for item in (census.get("worth_fixing") or [])[:10]
    ]
    skipped_rows = [
        (
            str(item.get("reason", "suppressed")).replace("_", " "),
            (
                f"{float(item.get('frequency_hz')):.0f} Hz"
                if item.get("frequency_hz") is not None else "affected driver band"
            ),
        )
        for item in (census.get("deliberately_skipped") or [])[:10]
    ]
    census_html = (
        "<h3>Worth fixing</h3>" + (_table(worth_rows) if worth_rows else "<p>No eligible correction centres were found.</p>")
        + "<h3>Deliberately skipped</h3>"
        + (_table(skipped_rows) if skipped_rows else "<p>No proposal suppressions were recorded.</p>")
    )

    filter_rows = []
    for group, bands in filters.items():
        for frequency, q_value, gain in bands:
            filter_rows.append(
                "<tr><td>%s</td><td>%.1f Hz</td><td>%.2f</td><td>%+.2f dB</td></tr>"
                % (html.escape(GROUP_LABELS.get(group, group)), frequency, q_value, gain)
            )
    filters_html = (
        "<table><tr><th>Driver / group</th><th>Frequency</th><th>Q</th><th>Gain</th></tr>"
        + "".join(filter_rows) + "</table>"
        if filter_rows else "<p>No added PEQ filters were reported for the selected best candidate.</p>"
    )

    action_rows = []
    for action in actions:
        details = []
        if action.get("polarity_channels"):
            details.append("polarity on ch " + ", ".join(str(v) for v in action["polarity_channels"]))
        if action.get("delay_samples") is not None:
            details.append(f"delay {int(action['delay_samples']):+d} samples")
        if action.get("apf_f"):
            details.append(f"APF {float(action['apf_f']):.1f} Hz, Q {float(action.get('apf_q', 0)):.2f}")
        action_rows.append((
            str(action.get("source", "Crossover")),
            "; ".join(details) + f"; confidence: {action.get('confidence', 'unknown')}",
        ))
    actions_html = _table(action_rows) if action_rows else "<p>No polarity, delay or APF write cleared the confidence gates.</p>"

    crossover_rows = []
    for item in full.get("crossover_phase_confidence") or []:
        crossover_rows.append((
            str(item.get("label", item.get("name", "Crossover"))),
            "%s; phase %s; summation %s; predicted-sum match %s"
            % (
                item.get("band", "band unavailable"), item.get("phase_stability", "unknown"),
                item.get("summation_quality", "unknown"), item.get("predicted_sum_match", "unknown"),
            ),
        ))
    crossover_html = _table(crossover_rows) if crossover_rows else "<p>No crossover diagnostic rows were available.</p>"

    family_rows = [(role.title(), f"{data.get('file', '-')} | objective {_fmt(data.get('objective'), 4)}")
                   for role, data in families.items()]
    family_html = _table(family_rows) if family_rows else "<p>No family variants were generated.</p>"
    remeasure = summary.get("remeasure") or []
    if phase_mode:
        checks = [
            "Load the candidate without changing any other DSP setting.",
            *[str(item) for item in remeasure],
            "Compare the changed crossover pair before and after at the same microphone positions.",
            "Play a mono vocal and confirm that the centre image remains locked and focused.",
            "Reject the change if it creates a new hole, ringing, image pull, or weaker summed output.",
        ]
    else:
        checks = [
            "Load the chosen PEQ candidate and keep a copy of the baseline for A/B listening.",
            "Confirm tonal balance at normal listening level with familiar material.",
            "A confirmation MMM is optional; use it only if listening exposes a concern.",
            "Take fresh phase-valid sweeps later only if delay, polarity or APF work is needed.",
        ]
    check_html = "".join(f"<li>{html.escape(item)}</li>" for item in checks)
    verification_title = "Load And Re-measure Checklist" if phase_mode else "Load And Check"
    status_text = (
        "Candidate generated from the supplied measurements. In-car re-measurement and listening remain required."
        if phase_mode else
        "Candidate generated from the supplied MMM measurements. A/B listening is recommended; a confirmation MMM is optional."
    )
    confidence_text = (
        "Destructive interference, crossover skirts, driver roll-off and unsupported one-position corrections are not rewarded as ordinary EQ targets. The loaded-and-re-measured result is the final proof."
        if phase_mode else
        "Destructive interference, crossover skirts, driver roll-off and unsupported one-position corrections are not rewarded as ordinary EQ targets. Use A/B listening as the decision step; a confirmation MMM is optional."
    )

    method = (
        "Only the crossover bands are judged. The tool first checks that the solo traces predict the measured "
        "pair, then tests polarity, bounded delay and finally a residual all-pass filter. A change is written only "
        "when the evidence and predicted improvement clear the gates."
        if phase_mode else
        "The target is anchored once from the baseline measurement. Every candidate uses that same anchor. "
        "The score combines tonal accuracy, anchor-independent target-contour accuracy, extra vocal-band importance, "
        "peak control, L/R consistency, spatial robustness, headroom and filter restraint. Lower error is better."
    )

    plot = _response_plot(summary, full)
    frequencies = plot.get("frequency_hz") or []
    system_chart = _line_chart([
        {"label": "Before", "x": frequencies, "y": plot.get("baseline_error_db"), "color": "#a34b43"},
        {"label": "Candidate", "x": frequencies, "y": plot.get("candidate_error_db"), "color": "#16805d"},
    ]) if frequencies and not phase_mode else ""
    delta_chart = _line_chart([
        {"label": "Predicted system change", "x": frequencies, "y": plot.get("raw_system_delta_db"), "color": "#2368a2"},
    ]) if frequencies and not phase_mode else ""
    component_rows = []
    for key, label in (
        ("tonal_error_db", "Tonal accuracy error"),
        ("presence_error_db", "Vocal / presence error"),
        ("target_shape_error_db", "Target contour error"),
        ("peak_penalty_db", "Broad peak penalty"),
        ("narrow_peak_penalty_db", "Narrow / raw peak penalty"),
        ("balance_penalty_db", "L/R balance error"),
        ("spatial_worst_db", "Worst-position error"),
    ):
        if isinstance(baseline.get(key), (int, float)) and isinstance(best_components.get(key), (int, float)):
            component_rows.append((label, float(baseline[key]), float(best_components[key])))
    component_chart = _paired_bar_chart(component_rows) if not phase_mode else ""

    lr_blocks = []
    pair_labels = {"low": "Midbass L/R", "mid": "Midrange L/R", "high": "Tweeter L/R"}
    for name, data in (plot.get("pairs") or {}).items():
        chart = _line_chart([
            {"label": "Before L-R", "x": data.get("frequency_hz"), "y": data.get("baseline_lr_db"), "color": "#a34b43"},
            {"label": "Candidate L-R", "x": data.get("frequency_hz"), "y": data.get("candidate_lr_db"), "color": "#16805d"},
        ])
        if chart:
            lr_blocks.append(
                f'<h2>{html.escape(pair_labels.get(name, str(name).title()))}</h2>'
                '<p class="chart-note">Closer to the 0 dB line means the two sides are more evenly matched.</p>'
                f'<img class="chart" src="{chart}">'
            )

    crossover_items = full.get("crossover_phase_confidence") or []
    phase_rows = []
    confidence_table_rows = []
    for item in crossover_items:
        ladder = item.get("crossover_ladder") if isinstance(item.get("crossover_ladder"), dict) else {}
        before = ladder.get("score_before")
        after = ladder.get("score_after_apf", ladder.get("score_after_polarity_delay"))
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            phase_rows.append((str(item.get("label", item.get("name", "Crossover"))), float(before), float(after)))
        decision = "WRITE" if ladder.get("write_eligible") else "LEAVE ALONE"
        confidence_table_rows.append(
            "<tr><td>%s<br><small>%s</small></td><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b><br><small>%s</small></td></tr>"
            % (
                html.escape(str(item.get("label", item.get("name", "Crossover")))),
                html.escape(str(item.get("band", ""))),
                html.escape(str(item.get("phase_stability", "unknown"))),
                html.escape(str(item.get("summation_quality", "unknown"))),
                html.escape(str(item.get("predicted_sum_match", "unknown"))),
                decision,
                html.escape(str(ladder.get("reason", ""))),
            )
        )
    phase_chart = _paired_bar_chart(phase_rows)
    confidence_matrix = (
        "<table><tr><th>Crossover</th><th>Phase stability</th><th>Summation</th>"
        "<th>Solo prediction</th><th>Decision</th></tr>" + "".join(confidence_table_rows) + "</table>"
        if confidence_table_rows else "<p>No crossover confidence rows were available.</p>"
    )

    findings_html = "".join(f"<li>{html.escape(item)}</li>" for item in _plain_findings(summary, phase_mode))
    chart_or_message = (
        f'<img class="chart hero-chart" src="{phase_chart}"><p class="chart-note">Lower bars are better. The candidate bar includes the final supported polarity, delay and APF step.</p>'
        if phase_mode and phase_chart else
        f'<img class="chart hero-chart" src="{system_chart}"><p class="chart-note">This graph is dB relative to the target. The 0 dB line is the target; closer is better. Before and candidate share one fixed anchor.</p>'
        if system_chart else
        '<div class="empty">A response graph was not available for this older run. New runs include fixed-anchor plot data automatically.</div>'
    )
    evidence_visual = (
        (f'<img class="chart" src="{phase_chart}">' if phase_chart else "") + confidence_matrix
        if phase_mode else
        (f'<img class="chart" src="{component_chart}">' if component_chart else "") + "".join(lr_blocks)
    )
    changes_visual = (
        actions_html if phase_mode else
        rehabilitation_html + filters_html + (f'<h2>Predicted system change</h2><img class="chart" src="{delta_chart}">' if delta_chart else "")
    )

    return f"""
<!doctype html><html><head><meta charset="utf-8"><style>
body {{ font-family:'Segoe UI',Arial,sans-serif; color:#20262b; font-size:10pt; line-height:1.38; }}
h1 {{ font-size:25pt; margin:3px 0 4px; color:#1d252b; }} h2 {{ font-size:15pt; margin:15px 0 6px; color:#1d252b; }}
h3 {{ color:#176b4d; font-size:9pt; text-transform:uppercase; letter-spacing:1px; margin:14px 0 5px; }}
p {{ margin:4px 0 9px; }} .eyebrow {{ color:#a05f00; font-size:8pt; letter-spacing:1.4px; text-transform:uppercase; }}
.muted {{ color:#68717a; }} .callout {{ background:#eaf6f1; border:1px solid #8dcab5; padding:11px; margin:12px 0; color:#145a43; }}
.callout.warn {{ background:#fff4d9; border-color:#d9a441; color:#6a4700; }}
.empty {{ background:#f2f4f5; border:1px solid #d8dde1; padding:14px; color:#68717a; margin:10px 0; }}
table {{ width:100%; border-collapse:collapse; margin:7px 0 12px; }} th {{ background:#252d32; color:white; text-align:left; padding:6px; }}
td {{ border-bottom:1px solid #d8dde1; padding:6px; vertical-align:top; }} tr:nth-child(even) td {{ background:#f5f7f7; }}
.metric-grid td {{ width:50%; border:6px solid white; padding:10px; background:#f1f4f3; }}
.metric span {{ display:block; color:#5e6870; font-size:8pt; text-transform:uppercase; }} .metric b {{ display:block; font-size:17pt; color:#176b4d; }}
.metric small {{ color:#68717a; }} .metric.warn b {{ color:#a34b43; }} .metric.neutral b {{ color:#4f5960; }}
ul,ol {{ margin:4px 0 10px 20px; }} li {{ margin-bottom:5px; }} .page {{ page-break-before:always; }}
img.chart {{ width:100%; margin:7px 0 4px; }} .hero-chart {{ margin-top:10px; }}
.chart-note {{ color:#66717a; font-size:8.5pt; margin-top:2px; }} small {{ color:#6c757d; }}
.footer {{ color:#7a828a; font-size:8pt; border-top:1px solid #d8dde1; padding-top:6px; margin-top:16px; }}
</style></head><body>
<div class="eyebrow">AudioFischer Optimizer - Executive Tuning Report</div>
<h1>{html.escape(title)}</h1>
<div class="muted">Generated {datetime.now().strftime('%d %B %Y, %I:%M %p')} | Source: {html.escape(source)} | Candidate: {html.escape(candidate)}</div>
<div class="callout {'warn' if phase_mode else ''}"><b>{html.escape(verdict)}</b></div>
{confidence_matrix if phase_mode else _metric_cards(baseline, best_components, suppress_percent=rehabilitation_tie)}
{chart_or_message}
<h2>What You Should Notice</h2><ul>{findings_html}</ul>
<h3>Worth checking</h3><ul>{warning_html}</ul>

<div class="page"><div class="eyebrow">1 - Evidence you can read</div><h1>{'Crossover Evidence' if phase_mode else 'Why This Candidate Won'}</h1>
<p>{html.escape(method)}</p>
{evidence_visual}
<h2>Detailed score values</h2>{_component_table(baseline, best_components)}
<h2>Measurement validation</h2>{validation_html}
<h2>Measurement repeatability guard</h2>{noise_html}
<p>{html.escape(str(noise_guard.get('note') or 'Filters below the assumed local measurement floor are rejected.'))}</p>
<h2>Pre-search problem census</h2>{census_html}
<div class="callout"><b>How to read this:</b> {html.escape(confidence_text)}</div></div>

<div class="page"><div class="eyebrow">2 - {'Phase / timing changes' if phase_mode else 'PEQ changes'}</div><h1>{'What Was Written' if phase_mode else 'What Changed In The DSP'}</h1>
{changes_visual}
<h2>{'Crossover detail' if phase_mode else 'Candidate variants'}</h2>{crossover_html if phase_mode else family_html}
<h2>Deliberately left alone</h2><p>{html.escape(str(best.get('left_alone') or 'No additional left-alone note was reported.'))}</p></div>

<div class="page"><div class="eyebrow">3 - Verification</div><h1>{html.escape(verification_title)}</h1>
<ol>{check_html}</ol>
<h2>Run details</h2>{overview}
<h2>Integrity statement</h2><p>The baseline file is never overwritten. PEQ mode does not alter delay, polarity, APF or crossovers. Phase mode preserves PEQ and writes only evidence-gated phase actions. Crossovers remain unchanged unless a future explicit crossover workflow says otherwise.</p>
<div class="callout"><b>Status:</b> {html.escape(status_text)}</div>
<div class="footer">Source: {html.escape(source)} | Candidate: {html.escape(candidate)} | Report generated locally</div></div>
</body></html>"""


def generate_tuning_report(summary_path: Path, output_path: Path | None = None) -> Path:
    summary_path = Path(summary_path)
    summary = _read_json(summary_path)
    details = summary.get("details") or {}
    full_path = summary_path.parent / str(details.get("optimizer_summary", "optimizer_summary.json"))
    full = _read_json(full_path)
    output = output_path or (summary_path.parent / "SQ_Tuning_Report.pdf")
    writer = QPdfWriter(str(output))
    writer.setTitle("AudioFischer Optimizer SQ Tuning Report")
    writer.setCreator("AudioFischer Optimizer")
    writer.setResolution(120)
    writer.setPageLayout(QPageLayout(
        QPageSize(QPageSize.A4), QPageLayout.Portrait,
        QMarginsF(14, 14, 14, 14), QPageLayout.Millimeter,
    ))
    document = QTextDocument()
    document.setHtml(build_report_html(summary, full, summary_path))
    document.setPageSize(QSizeF(writer.width(), writer.height()))
    document.print_(writer)
    return output
