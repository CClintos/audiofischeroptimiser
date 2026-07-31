"""Dark, pro-audio design tokens: palette, QSS, and a hand-drawn icon set.

Every color used by the GUI (optimizer_gui/window.py) traces back to a name in
this module. PDF report rendering (optimizer_gui/reporting.py's _line_chart,
_paired_bar_chart, build_report_html) intentionally stays on its own light
palette for print and is not touched here.

Contrast ratios below are WCAG relative-luminance ratios against the stated
background, computed once during design (see PR notes); re-check before
changing any of the base tokens.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

# ---- base surface ----------------------------------------------------------
BG_BASE = "#14171a"        # window background
BG_PANEL = "#1b1f23"       # cards, tab pane, table/tree backgrounds
BG_RAISED = "#242a2f"      # input fields, buttons at rest
BORDER = "#343b41"
BORDER_STRONG = "#454d54"
CARD_HIGHLIGHT = "#3d444b"  # subtle top-edge highlight standing in for a drop shadow
                            # (QGraphicsDropShadowEffect on cards with several child
                            # labels causes visible per-child compositing seams - avoid it)

# ---- text -------------------------------------------------------------------
TEXT_PRIMARY = "#e8eaec"   # 14.9:1 on BG_BASE, 13.7:1 on BG_PANEL
TEXT_MUTED = "#98a1a8"     # 6.3:1 on BG_PANEL, 6.9:1 on BG_BASE
TEXT_ON_ACCENT = "#ffffff"

# ---- accent (kept close to the existing brand green so the app and the
# printed PDF report still read as the same product) -------------------------
ACCENT_FILL = "#176b4d"        # solid button backgrounds; white text 6.5:1
ACCENT_FILL_HOVER = "#1c7a58"  # hover state; white text 5.3:1
ACCENT_LINE = "#33c48f"        # text/icons/progress/chart "candidate" series; 6.5:1 on BG_RAISED
ACCENT_SOFT_BG = "#12271f"     # success-callout background (e.g. "meaningful improvement" banner)
ACCENT_SOFT_TEXT = "#cdf3e3"

# ---- status -------------------------------------------------------------------
WARN = "#e6a23c"
WARN_SOFT_BG = "#3a2c12"
WARN_TEXT = "#f4cf8e"          # 9.1:1 on WARN_SOFT_BG
DANGER = "#eb5457"             # 4.7:1 on BG_PANEL, 5.1:1 on BG_BASE
DANGER_SOFT_BG = "#3a1618"
DANGER_TEXT = "#f4a7a9"        # 8.4:1 on DANGER_SOFT_BG
INFO = "#5b9bd5"               # 5.6:1 on BG_PANEL
INFO_SOFT_BG = "#12283a"
INFO_TEXT = "#a9d2f0"          # 9.5:1 on INFO_SOFT_BG

# ---- chart tokens (live GUI charts only; PDF stays on its own palette) -----
CHART_GRID = "#2c3236"
CHART_AXIS_TEXT = TEXT_MUTED
CHART_ZERO_LINE = "#5c656d"
CHART_MARKER = WARN
SERIES_BASELINE = "#e5726b"    # "before" / warm reference line; 5.5:1 on BG_PANEL
SERIES_CANDIDATE = ACCENT_LINE  # "after" / candidate; matches the accent
SERIES_PREDICTED = INFO         # third series (e.g. predicted response)
SERIES_TARGET = TEXT_MUTED      # dashed reference/target line
DRIVER_SERIES = (
    "#a688d6", "#e08fc9", "#5aa9e6", "#8bc463", "#e6a559", "#d1b370", "#5fb8c4",
)  # opt-in per-driver toggle lines; all >=5.6:1 on BG_PANEL

FONT_FAMILY = "Segoe UI"
FONT_MONO = '"Consolas", "Cascadia Mono", "Segoe UI"'


def severity_colour(severity: str) -> str:
    """GUI (dark-bg) text color for a warning severity.

    Distinct from warning_text.SEVERITY_COLOURS, which is tuned for the white
    PDF report page and must not change with the app theme.
    """
    return {"error": DANGER, "warning": WARN, "info": INFO}.get(severity, TEXT_MUTED)


# Ordered (first match wins) keyword -> state mapping for the run_badge label.
# Checked case-insensitively against the badge's own text so every status
# string window.py already sets (READY, RUNNING, VALIDATED, FAILED, ...)
# gets a state without a second source of truth to keep in sync.
_BADGE_RULES = (
    ("REPORT FAILED", "warn"),
    ("FAILED", "danger"),
    ("BLOCKED", "danger"),
    ("RUNNING", "info"),
    ("VALIDATING", "info"),
    ("WRITING", "info"),
    ("STOPPING", "info"),
    ("COMPLETE", "good"),
    ("VALIDATED", "good"),
    ("CANCELLED", "warn"),
    ("STOPPED", "warn"),
    ("NEEDS", "warn"),
    ("MAPPING", "warn"),
)
_BADGE_LOOK = {
    "good": (ACCENT_SOFT_BG, ACCENT_LINE, ACCENT_SOFT_TEXT),
    "warn": (WARN_SOFT_BG, WARN, WARN_TEXT),
    "danger": (DANGER_SOFT_BG, DANGER, DANGER_TEXT),
    "info": (INFO_SOFT_BG, INFO, INFO_TEXT),
}


def badge_style(text: str) -> str:
    """Inline stylesheet for the top-right status badge, colour-coded by state.

    Returns "" for the neutral/default state so the static QLabel#badge QSS
    rule in build_stylesheet() applies instead of an inline override.
    """
    upper = text.upper()
    for keyword, state in _BADGE_RULES:
        if keyword in upper:
            bg, border, fg = _BADGE_LOOK[state]
            return (
                f"background:{bg}; color:{fg}; border:1px solid {border}; "
                "padding:6px 11px; border-radius:4px; font-weight:700;"
            )
    return ""


def apply_palette(app) -> None:
    """Dark QPalette so Qt-drawn native chrome (QMessageBox, QInputDialog,
    QComboBox popups, disabled-state text) matches the theme instead of
    falling back to a light OS palette. Pair with QApplication.setStyle
    ("Fusion") - the native Windows style mostly ignores QPalette/QSS for
    sliders, checkboxes, and spin/combo arrows.
    """
    from PySide6.QtGui import QPalette

    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG_BASE,
        QPalette.ColorRole.WindowText: TEXT_PRIMARY,
        QPalette.ColorRole.Base: BG_RAISED,
        QPalette.ColorRole.AlternateBase: BG_PANEL,
        QPalette.ColorRole.Text: TEXT_PRIMARY,
        QPalette.ColorRole.Button: BG_RAISED,
        QPalette.ColorRole.ButtonText: TEXT_PRIMARY,
        QPalette.ColorRole.BrightText: DANGER,
        QPalette.ColorRole.Highlight: ACCENT_FILL,
        QPalette.ColorRole.HighlightedText: TEXT_ON_ACCENT,
        QPalette.ColorRole.ToolTipBase: BG_RAISED,
        QPalette.ColorRole.ToolTipText: TEXT_PRIMARY,
        QPalette.ColorRole.PlaceholderText: TEXT_MUTED,
        QPalette.ColorRole.Link: ACCENT_LINE,
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(BORDER_STRONG))
    app.setPalette(palette)




def make_app_icon() -> QIcon:
    """A simple EQ-bars monogram for the window/taskbar icon."""
    icon = QIcon()
    for px in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT_FILL))
        radius = px * 0.22
        painter.drawRoundedRect(QRectF(0, 0, px, px), radius, radius)
        painter.setBrush(QColor(TEXT_ON_ACCENT))
        bars = (0.30, 0.62, 0.46, 0.74, 0.62, 0.50)
        count = len(bars)
        gap = px * 0.08
        bar_width = (px - gap * (count + 1)) / count
        base_y = px * 0.78
        for index, height_ratio in enumerate(bars):
            height = px * 0.56 * height_ratio + px * 0.08
            x = gap + index * (bar_width + gap)
            rect = QRectF(x, base_y - height, bar_width, height)
            painter.drawRoundedRect(rect, bar_width * 0.3, bar_width * 0.3)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def build_stylesheet() -> str:
    return f"""
        QMainWindow, QWidget {{ background: {BG_BASE}; color: {TEXT_PRIMARY}; font-size: 13px; }}
        QLabel#title {{ font-size: 26px; font-weight: 700; color: {TEXT_PRIMARY}; }}
        QLabel#subtitle {{ color: {TEXT_MUTED}; font-size: 12px; }}
        QLabel#badge {{ background: {BG_RAISED}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER};
            padding: 6px 11px; border-radius: 4px; font-weight: 700; }}
        QLabel#warning {{ background: {WARN_SOFT_BG}; border-left: 4px solid {WARN}; padding: 9px;
            color: {WARN_TEXT}; }}
        QLabel#resultBanner {{ background: {ACCENT_SOFT_BG}; border-left: 4px solid {ACCENT_LINE}; padding: 10px;
            color: {ACCENT_SOFT_TEXT}; font-weight: 650; }}
        QFrame#card {{ background: {BG_PANEL}; border: 1px solid {BORDER};
            border-top: 1px solid {CARD_HIGHLIGHT}; border-radius: 8px; }}
        QFrame#cardAccent {{ background: {BG_PANEL}; border: 1px solid {BORDER};
            border-top: 1px solid {CARD_HIGHLIGHT}; border-left: 3px solid {ACCENT_LINE};
            border-radius: 8px; }}
        QFrame#metricCard {{ background: {BG_PANEL}; border: 1px solid {BORDER};
            border-top: 1px solid {CARD_HIGHLIGHT}; border-radius: 6px; }}
        QLabel#metricName {{ color: {TEXT_MUTED}; font-size: 11px; text-transform: uppercase; }}
        QLabel#metricValue {{ color: {TEXT_PRIMARY}; font-size: 19px; font-weight: 700;
            font-family: {FONT_MONO}; }}
        QLabel#sectionTitle {{ font-size: 19px; font-weight: 650; color: {TEXT_PRIMARY}; }}
        QLabel#workflowTitle {{ font-size: 15px; font-weight: 650; color: {TEXT_PRIMARY}; margin-top: 8px; }}
        QLabel#workflowSummary {{ font-size: 15px; font-weight: 650; color: {TEXT_PRIMARY}; }}
        QLabel#chart {{ background: {BG_PANEL}; border: 1px solid {BORDER}; color: {TEXT_MUTED}; padding: 4px; }}
        QLabel#chartNote {{ color: {TEXT_MUTED}; font-size: 11px; }}
        QScrollArea#runScroll, QWidget#runContent {{ background: {BG_BASE}; }}
        QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG_PANEL}; border-radius: 6px; }}
        QTabBar::tab {{ background: {BG_RAISED}; color: {TEXT_MUTED}; border: 1px solid {BORDER};
            padding: 9px 14px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
        QTabBar::tab:selected {{ background: {BG_PANEL}; color: {TEXT_PRIMARY}; border-bottom-color: {BG_PANEL};
            font-weight: 650; }}
        QTabBar::tab:disabled {{ color: {BORDER_STRONG}; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {{
            background: {BG_RAISED}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER};
            border-radius: 3px; padding: 6px; selection-background-color: {ACCENT_FILL};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
            border: 1px solid {ACCENT_LINE};
        }}
        QTableWidget {{ gridline-color: {BORDER}; alternate-background-color: {BG_PANEL};
            selection-background-color: {ACCENT_FILL}; selection-color: {TEXT_ON_ACCENT}; }}
        QComboBox QAbstractItemView {{ background: {BG_RAISED}; color: {TEXT_PRIMARY};
            selection-background-color: {ACCENT_FILL}; border: 1px solid {BORDER}; }}
        QComboBox::drop-down {{ border: none; width: 24px; }}
        QComboBox::down-arrow {{ image: none; width: 0; height: 0;
            border-left: 4px solid transparent; border-right: 4px solid transparent;
            border-top: 5px solid {TEXT_MUTED}; margin-right: 8px; }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: top right; width: 18px;
            border-left: 1px solid {BORDER}; background: {BG_RAISED}; border-top-right-radius: 3px; }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: bottom right; width: 18px;
            border-left: 1px solid {BORDER}; background: {BG_RAISED}; border-bottom-right-radius: 3px; }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: none; width: 0; height: 0;
            border-left: 3px solid transparent; border-right: 3px solid transparent;
            border-bottom: 4px solid {TEXT_MUTED}; }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: none; width: 0; height: 0;
            border-left: 3px solid transparent; border-right: 3px solid transparent;
            border-top: 4px solid {TEXT_MUTED}; }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: #2b323a; }}
        QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
        QSlider::sub-page:horizontal {{ background: {ACCENT_LINE}; border-radius: 2px; }}
        QSlider::add-page:horizontal {{ background: {BORDER}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: {TEXT_PRIMARY}; border: 2px solid {ACCENT_LINE};
            width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; }}
        QSlider::handle:horizontal:hover {{ background: {ACCENT_LINE}; }}
        QMenu {{ background: {BG_RAISED}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER}; }}
        QMenu::item {{ padding: 5px 22px; }}
        QMenu::item:selected {{ background: {ACCENT_FILL}; color: {TEXT_ON_ACCENT}; }}
        QPushButton, QToolButton {{ background: {BG_RAISED}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER};
            border-radius: 4px; padding: 7px 12px; }}
        QPushButton:hover, QToolButton:hover {{ background: #2b323a; border-color: {BORDER_STRONG}; }}
        QPushButton#primary {{ background: {ACCENT_FILL}; color: {TEXT_ON_ACCENT}; border-color: {ACCENT_FILL};
            font-weight: 650; }}
        QPushButton#primary:hover {{ background: {ACCENT_FILL_HOVER}; }}
        QPushButton:disabled, QToolButton:disabled {{ color: {BORDER_STRONG}; background: {BG_PANEL};
            border-color: {BORDER}; }}
        QProgressBar {{ border: 1px solid {BORDER}; background: {BG_RAISED}; height: 16px;
            text-align: center; color: {TEXT_PRIMARY}; border-radius: 3px; }}
        QProgressBar::chunk {{ background: {ACCENT_LINE}; border-radius: 3px; }}
        QHeaderView::section {{ background: {BG_RAISED}; color: {TEXT_MUTED}; border: 0;
            border-bottom: 1px solid {BORDER}; padding: 7px; font-weight: 650; }}
        QCheckBox {{ color: {TEXT_PRIMARY}; spacing: 8px; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px;
            border: 1px solid {BORDER_STRONG}; background: {BG_RAISED}; }}
        QCheckBox::indicator:hover {{ border-color: {ACCENT_LINE}; }}
        QCheckBox::indicator:checked {{ background: {ACCENT_FILL}; border-color: {ACCENT_FILL}; }}
        QCheckBox::indicator:disabled {{ border-color: {BORDER}; background: {BG_PANEL}; }}
        QToolTip {{ background: {BG_RAISED}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER}; padding: 4px; }}
        QScrollBar:vertical {{ background: {BG_BASE}; width: 12px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 5px; min-height: 24px; }}
        QScrollBar::handle:vertical:hover {{ background: {TEXT_MUTED}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{ background: {BG_BASE}; height: 12px; margin: 0; }}
        QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 5px; min-width: 24px; }}
        QScrollBar::handle:horizontal:hover {{ background: {TEXT_MUTED}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """


# ---- icon set ---------------------------------------------------------------
# Small flat/line-style glyphs drawn on a 0..1 unit grid (scaled by `size`),
# replacing QStyle.SP_* system dialog icons everywhere in window.py.

def _pt(x: float, y: float, size: float) -> QPointF:
    return QPointF(x * size, y * size)


def _stroke_path(painter: QPainter, points: list[tuple[float, float]], size: float,
                  color: QColor, close: bool = False) -> None:
    path = QPainterPath()
    path.moveTo(_pt(*points[0], size))
    for x, y in points[1:]:
        path.lineTo(_pt(x, y, size))
    if close:
        path.closeSubpath()
    pen = QPen(color)
    pen.setWidthF(max(1.3, size * 0.075))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(path)


def _draw_arrow_right(painter: QPainter, size: float, color: QColor) -> None:
    _stroke_path(painter, [(0.16, 0.5), (0.72, 0.5)], size, color)
    _stroke_path(painter, [(0.58, 0.30), (0.84, 0.5), (0.58, 0.70)], size, color)


def _draw_check(painter: QPainter, size: float, color: QColor) -> None:
    _stroke_path(painter, [(0.18, 0.52), (0.42, 0.76), (0.84, 0.24)], size, color)


def _draw_play(painter: QPainter, size: float, color: QColor) -> None:
    path = QPainterPath()
    path.moveTo(_pt(0.30, 0.20, size))
    path.lineTo(_pt(0.30, 0.80, size))
    path.lineTo(_pt(0.82, 0.50, size))
    path.closeSubpath()
    painter.setPen(Qt.NoPen)
    painter.setBrush(color)
    painter.drawPath(path)


def _draw_stop(painter: QPainter, size: float, color: QColor) -> None:
    rect = QRectF(_pt(0.26, 0.26, size), _pt(0.74, 0.74, size))
    painter.setPen(Qt.NoPen)
    painter.setBrush(color)
    painter.drawRoundedRect(rect, size * 0.05, size * 0.05)


def _folder_body(painter: QPainter, size: float, color: QColor) -> None:
    _stroke_path(
        painter,
        [(0.13, 0.30), (0.38, 0.30), (0.46, 0.38), (0.87, 0.38), (0.87, 0.80), (0.13, 0.80)],
        size, color, close=True,
    )


def _draw_folder(painter: QPainter, size: float, color: QColor) -> None:
    _folder_body(painter, size, color)


def _draw_folder_open(painter: QPainter, size: float, color: QColor) -> None:
    _stroke_path(
        painter,
        [(0.13, 0.28), (0.38, 0.28), (0.46, 0.36), (0.87, 0.36), (0.87, 0.46), (0.13, 0.46)],
        size, color, close=True,
    )
    _stroke_path(
        painter,
        [(0.08, 0.46), (0.92, 0.46), (0.80, 0.84), (0.18, 0.84)],
        size, color, close=True,
    )


def _draw_folder_plus(painter: QPainter, size: float, color: QColor) -> None:
    _folder_body(painter, size, color)
    _stroke_path(painter, [(0.64, 0.50), (0.64, 0.70)], size, color)
    _stroke_path(painter, [(0.54, 0.60), (0.74, 0.60)], size, color)


def _draw_export(painter: QPainter, size: float, color: QColor) -> None:
    _stroke_path(painter, [(0.5, 0.16), (0.5, 0.60)], size, color)
    _stroke_path(painter, [(0.32, 0.44), (0.5, 0.64), (0.68, 0.44)], size, color)
    _stroke_path(painter, [(0.20, 0.68), (0.20, 0.84), (0.80, 0.84), (0.80, 0.68)], size, color)


def _document(painter: QPainter, size: float, color: QColor, lines: int) -> None:
    _stroke_path(
        painter,
        [(0.26, 0.12), (0.60, 0.12), (0.76, 0.28), (0.76, 0.88), (0.26, 0.88)],
        size, color, close=True,
    )
    _stroke_path(painter, [(0.60, 0.12), (0.60, 0.28), (0.76, 0.28)], size, color)
    ys = [0.44, 0.58, 0.72][:lines]
    for y in ys:
        width = 0.62 if y != ys[-1] else 0.52
        _stroke_path(painter, [(0.36, y), (0.36 + width - 0.36, y)], size, color)


def _draw_file(painter: QPainter, size: float, color: QColor) -> None:
    _document(painter, size, color, lines=1)


def _draw_report(painter: QPainter, size: float, color: QColor) -> None:
    _document(painter, size, color, lines=3)


_ICON_DRAWERS = {
    "arrow-right": _draw_arrow_right,
    "check": _draw_check,
    "play": _draw_play,
    "stop": _draw_stop,
    "folder": _draw_folder,
    "folder-open": _draw_folder_open,
    "folder-plus": _draw_folder_plus,
    "export": _draw_export,
    "file": _draw_file,
    "report": _draw_report,
}


def make_icon(name: str, color: str = TEXT_PRIMARY) -> QIcon:
    drawer = _ICON_DRAWERS[name]
    qcolor = QColor(color)
    icon = QIcon()
    for px in (16, 20, 24, 32):
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        drawer(painter, float(px), qcolor)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def step_badge_icon(number: int, *, active: bool, size: int = 20) -> QIcon:
    """A small circular step-number badge for the numbered workflow tabs."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    fill = QColor(ACCENT_FILL if active else BG_RAISED)
    border = QColor(ACCENT_LINE if active else BORDER_STRONG)
    text_color = QColor(TEXT_ON_ACCENT if active else TEXT_MUTED)
    pen = QPen(border)
    pen.setWidthF(max(1.2, size * 0.07))
    painter.setPen(pen)
    painter.setBrush(fill)
    margin = size * 0.09
    painter.drawEllipse(QRectF(margin, margin, size - 2 * margin, size - 2 * margin))
    painter.setPen(text_color)
    font = painter.font()
    font.setPixelSize(int(size * 0.52))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, str(number))
    painter.end()
    return QIcon(pixmap)
