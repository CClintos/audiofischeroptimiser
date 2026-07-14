from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QMarginsF, QSizeF
from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument


GROUP_LABELS = {
    "sub": "Subwoofer",
    "high_sym": "Both tweeters (symmetric)",
    "fl_high": "Front L tweeter",
    "fr_high": "Front R tweeter",
    "mid_sym": "Both midrange drivers (symmetric)",
    "fl_mid": "Front L midrange",
    "fr_mid": "Front R midrange",
    "low_sym": "Both midbass drivers (symmetric)",
    "fl_low": "Front L midbass",
    "fr_low": "Front R midbass",
}


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
        ("peak_penalty_db", "Peak penalty"),
        ("balance_penalty_db", "L/R balance penalty"),
        ("positive_gain_penalty_db", "Positive-gain / headroom penalty"),
        ("spatial_worst_db", "Worst-position error"),
        ("guardrail_penalty", "Guardrail penalty"),
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


def _warning_items(summary: dict[str, Any], mode: str) -> list[str]:
    warnings = [str(item) for item in summary.get("warnings") or []]
    best = summary.get("best") or {}
    if best.get("left_alone"):
        warnings.append(str(best["left_alone"]))
    if mode == "phase" and summary.get("phase_actions"):
        warnings.append("Phase changes are model predictions until the affected crossover is re-measured in-car.")
    if not warnings:
        warnings.append("No blocking measurement or candidate warnings were reported.")
    return warnings[:8]


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
    warnings = _warning_items(summary, mode)
    filters = _best_filters(summary, summary_path.parent / "optimizer_report.md")
    actions = summary.get("phase_actions") or []
    families = summary.get("families") or {}

    verdict = (
        "A phase candidate was produced. Re-measure every changed crossover before calling it final."
        if phase_mode and actions else
        "No phase write cleared the evidence gates; the baseline remains the recommended result."
        if phase_mode else
        "A PEQ candidate was produced from the supplied magnitude measurements and conservative guardrails."
    )
    overview = _table([
        ("Analysis", "Phase / delay / APF only; PEQ preserved" if phase_mode else "PEQ from magnitude / RTA data"),
        ("Source tune", source),
        ("Output candidate", candidate),
        ("Target", target),
        ("Measurements", f"{len(positions)} position(s): {', '.join(positions)}"),
        ("Candidates retained", str(summary.get("candidate_count", 0))),
        ("Baseline objective", _fmt(baseline.get("objective"), 4)),
        ("Best objective", _fmt(best.get("objective"), 4)),
        ("Objective improvement", _objective_improvement(baseline.get("objective"), best.get("objective"))),
        ("Phase-valid session", "Yes" if sessions.get("phase_valid") else "No / not required"),
    ])
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)

    validation_rows = []
    for item in validation:
        status = "PASS" if item.get("pass") else "FAIL"
        validation_rows.append((
            str(item.get("pair", "Pair")).title(),
            f"{status}: {item.get('rms_db', '-')} dB RMS against {item.get('threshold_db', '-')} dB limit",
        ))
    validation_html = _table(validation_rows) if validation_rows else "<p>No solo/together validation rows were available.</p>"

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
        "The phase stage checks only crossover bands. It validates whether solo traces predict the measured "
        "together trace, then tests polarity first, bounded relative delay second, and residual all-pass correction last."
        if phase_mode else
        "The PEQ stage minimizes one named scalar objective. It combines target-shape accuracy, extra vocal-band "
        "weight, peak cost, L/R mismatch, spatial robustness, positive-gain/headroom cost, and filter restraint."
    )
    return f"""
<!doctype html><html><head><meta charset="utf-8"><style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; color:#202327; font-size:10pt; line-height:1.35; }}
h1 {{ font-size:25pt; margin:3px 0 3px 0; }} h2 {{ font-size:15pt; margin:12px 0 7px; }}
h3 {{ color:#176b4d; font-size:10pt; text-transform:uppercase; margin:13px 0 5px; }}
p {{ margin:4px 0 8px; }} .eyebrow {{ color:#9b5c00; font-size:8pt; letter-spacing:1px; }}
.muted {{ color:#68717a; }} .callout {{ background:#fff4d9; border:1px solid #d9a441; padding:9px; margin:12px 0; color:#6a4700; }}
table {{ width:100%; border-collapse:collapse; margin:7px 0 11px; }} th {{ background:#202327; color:white; text-align:left; padding:6px; }}
td {{ border-bottom:1px solid #d8dde1; padding:6px; vertical-align:top; }} tr:nth-child(even) td {{ background:#f4f6f7; }}
ul,ol {{ margin:4px 0 9px 20px; }} li {{ margin-bottom:4px; }} .page {{ page-break-before:always; }}
.footer {{ color:#7a828a; font-size:8pt; border-top:1px solid #d8dde1; padding-top:6px; margin-top:16px; }}
</style></head><body>
<div class="eyebrow">AUDIOFISCHER OPTIMIZER - SQ TUNING SESSION</div>
<h1>{html.escape(title)}</h1>
<div class="muted">Generated {datetime.now().strftime('%d %B %Y, %I:%M %p')}</div>
<div class="callout"><b>{html.escape(verdict)}</b></div>
{overview}
<h3>Worth checking</h3><ul>{warning_html}</ul>
<h3>Contents</h3><ol><li>Measurement and score evidence</li><li>{'Phase changes and crossover confidence' if phase_mode else 'Added PEQ filters and candidate families'}</li><li>What was deliberately left alone</li><li>Verification checklist</li></ol>

<div class="page"><div class="eyebrow">1 - EVIDENCE AND OBJECTIVE</div><h1>How This Candidate Was Judged</h1>
<p>{html.escape(method)}</p>
<p><b>Lower is better.</b> Component values are decision metrics, not claims that the cabin response has been acoustically verified after loading.</p>
{_component_table(baseline, best_components)}
<h2>Measurement validation</h2>{validation_html}
<h2>Confidence boundaries</h2><p>{html.escape(confidence_text)}</p></div>

<div class="page"><div class="eyebrow">2 - {'PHASE / TIMING' if phase_mode else 'PEQ CHANGES'}</div><h1>{'Written Phase Changes' if phase_mode else 'Added Parametric EQ'}</h1>
{actions_html if phase_mode else filters_html}
<h2>{'Crossover confidence' if phase_mode else 'Candidate families'}</h2>{crossover_html if phase_mode else family_html}
<h2>Deliberately left alone</h2><p>{html.escape(str(best.get('left_alone') or 'No additional left-alone note was reported.'))}</p></div>

<div class="page"><div class="eyebrow">3 - VERIFICATION</div><h1>{html.escape(verification_title)}</h1>
<ol>{check_html}</ol>
<h2>Integrity statement</h2><p>The optimizer never overwrites the baseline. PEQ mode does not write delay, polarity, APF or crossover changes. Phase mode preserves PEQ and writes only evidence-gated phase actions. Crossovers remain unchanged.</p>
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
