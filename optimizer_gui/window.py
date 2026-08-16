from __future__ import annotations

import base64
import html
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    QObject, QPointF, QRunnable, QRectF, QSettings, QThreadPool,
    QTimer, Qt, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from .backend import (
    APP_NAME, RunConfig, RunRootBusyError, active_run_pid, candidate_files,
    claim_run_root, collect_progress, create_measurement_template, default_export_name,
    default_target, discover_baseline, export_candidate, load_summary,
    load_target_curve, locate_summary, measurement_checklist,
    memory_guard_status, merge_progress_fraction, powershell_command, process_is_running,
    process_tree_memory, record_run_decision, release_run_claim, save_role_map, stop_process_tree,
    start_detached_process, suggest_measurement_role, timestamped_run_root,
    runner_completed_successfully, runner_failure_reason, update_run_claim,
    validate_config,
)
from . import __version__
from .reporting import (
    GROUP_LABELS, generate_tuning_report, improvement_verdict, load_response_comparisons, load_response_plot,
    metric_card_data, response_chart_series,
)
from .warning_text import warning_info
from . import theme
from scripts.make_measurement_manifest import ALL_MEASUREMENT_ROLES
from scripts.verify_achieved_response import verify_run


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class BackgroundTask(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.cancel_event = threading.Event()
        self.signals = TaskSignals()

    def cancel(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        try:
            self.signals.result.emit(self.function(self.cancel_event))
        except Exception as exc:
            self.signals.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.signals.finished.emit()


class RoleMappingDialog(QDialog):
    def __init__(
        self, files: list[str], resolved_roles: dict[str, str],
        remembered: dict[str, str], parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Map Measurement Files")
        self.resize(720, 520)
        self.mapping: dict[str, str] = {}
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Choose the speaker role represented by each TXT file. Files that are not REW "
            "measurements, such as target curves, should remain set to Ignore."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(len(files), 2)
        self.table.setHorizontalHeaderLabels(["TXT file", "Measurement role"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        resolved_by_file = {
            Path(path).name.lower(): role for role, path in resolved_roles.items()
            if ":" not in role
        }
        self.combos: list[tuple[str, QComboBox]] = []
        for row, filename in enumerate(files):
            self.table.setItem(row, 0, QTableWidgetItem(filename))
            combo = QComboBox()
            combo.addItem("Ignore", "")
            for role in ALL_MEASUREMENT_ROLES:
                combo.addItem(role, role)
            guess = resolved_by_file.get(filename.lower()) or suggest_measurement_role(
                filename, remembered,
            )
            if guess:
                index = combo.findData(guess)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.table.setCellWidget(row, 1, combo)
            self.combos.append((filename, combo))
        layout.addWidget(self.table, 1)
        self.remember_check = QCheckBox("Remember this naming for next time")
        self.remember_check.setChecked(True)
        layout.addWidget(self.remember_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        mapping: dict[str, str] = {}
        for filename, combo in self.combos:
            role = str(combo.currentData() or "")
            if not role:
                continue
            if role in mapping:
                QMessageBox.warning(
                    self, "Duplicate role",
                    f"{role} is assigned to both {mapping[role]} and {filename}.",
                )
                return
            mapping[role] = filename
        self.mapping = mapping
        super().accept()


class DropLineEdit(QLineEdit):
    pathDropped = Signal(str)

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        for path in paths:
            if (self.mode == "folder" and path.is_dir()) or (self.mode == "file" and path.is_file()):
                self.setText(str(path))
                self.pathDropped.emit(str(path))
                event.acceptProposedAction()
                return


class _ChartCanvas(QWidget):
    enlargeRequested = Signal()

    def __init__(self, placeholder: str, allow_enlarge: bool = True, parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.allow_enlarge = allow_enlarge
        self.series: list[dict] = []
        self.markers: list[float] = []
        self.static_pixmap = QPixmap()
        self.setMouseTracking(True)
        self.setMinimumHeight(210)

    def set_plot(self, series: list[dict], markers: list[float] | None = None):
        self.static_pixmap = QPixmap()
        self.series = [dict(item) for item in series]
        self.markers = [
            float(value) for value in (markers or [])
            if isinstance(value, (int, float)) and float(value) > 0
        ]
        self.update()

    def set_static(self, pixmap: QPixmap):
        self.series = []
        self.markers = []
        self.static_pixmap = pixmap
        self.update()

    def clear_plot(self, message: str):
        self.placeholder = message
        self.series = []
        self.markers = []
        self.static_pixmap = QPixmap()
        self.update()

    def _clean_series(self) -> list[tuple[dict, list[tuple[float, float]]]]:
        clean = []
        for item in self.series:
            if not item.get("visible", True):
                continue
            points = []
            for x_value, y_value in zip(item.get("x") or [], item.get("y") or []):
                try:
                    x, y = float(x_value), float(y_value)
                except (TypeError, ValueError):
                    continue
                if x > 0 and math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))
            if len(points) >= 2:
                clean.append((item, points))
        return clean

    def _geometry(self, clean):
        rect = QRectF(58, 18, max(20, self.width() - 78), max(20, self.height() - 52))
        all_x = [x for _item, points in clean for x, _y in points]
        all_y = [y for _item, points in clean for _x, y in points]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(min(all_y), 0.0), max(max(all_y), 0.0)
        span = max(2.0, y_max - y_min)
        y_min -= span * 0.1
        y_max += span * 0.1
        log_min, log_max = math.log10(x_min), math.log10(x_max)

        def point(x, y):
            px = rect.left() + (math.log10(x) - log_min) / max(log_max - log_min, 1e-9) * rect.width()
            py = rect.bottom() - (y - y_min) / max(y_max - y_min, 1e-9) * rect.height()
            return QPointF(px, py)

        return rect, x_min, x_max, y_min, y_max, point

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.BG_PANEL))
        if not self.static_pixmap.isNull():
            scaled = self.static_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            painter.drawPixmap(
                (self.width() - scaled.width()) // 2,
                (self.height() - scaled.height()) // 2,
                scaled,
            )
            return
        clean = self._clean_series()
        if not clean:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.placeholder)
            return
        rect, x_min, x_max, y_min, y_max, point = self._geometry(clean)
        painter.setPen(QPen(QColor(theme.CHART_GRID), 1))
        for index in range(5):
            value = y_min + (y_max - y_min) * index / 4
            y = point(x_min, value).y()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor(theme.CHART_AXIS_TEXT))
            painter.drawText(QRectF(2, y - 9, 52, 18), Qt.AlignRight | Qt.AlignVCenter, f"{value:+.1f}")
            painter.setPen(QPen(QColor(theme.CHART_GRID), 1))
        for frequency in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            if x_min <= frequency <= x_max:
                x = point(frequency, 0).x()
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                label = f"{frequency // 1000}k" if frequency >= 1000 else str(frequency)
                painter.setPen(QColor(theme.CHART_AXIS_TEXT))
                painter.drawText(QRectF(x - 22, rect.bottom() + 4, 44, 20), Qt.AlignHCenter, label)
                painter.setPen(QPen(QColor(theme.CHART_GRID), 1))
        if y_min <= 0 <= y_max:
            painter.setPen(QPen(QColor(theme.CHART_ZERO_LINE), 1, Qt.DashLine))
            y = point(x_min, 0).y()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for frequency in self.markers:
            if x_min <= frequency <= x_max:
                x = point(frequency, 0).x()
                painter.setPen(QPen(QColor(theme.CHART_MARKER), 1, Qt.DotLine))
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                painter.drawEllipse(QPointF(x, rect.top() + 6), 3, 3)
        for item, points in clean:
            path = QPainterPath()
            first = point(*points[0])
            path.moveTo(first)
            for x_value, y_value in points[1:]:
                path.lineTo(point(x_value, y_value))
            pen = QPen(QColor(str(item.get("color") or theme.SERIES_CANDIDATE)), 2.2)
            if item.get("dashed"):
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.setPen(QColor(theme.CHART_AXIS_TEXT))
        painter.drawText(QRectF(2, 0, 54, 18), Qt.AlignRight, "dB")
        painter.drawText(
            QRectF(self.width() - 96, 0, 94, 18),
            Qt.AlignRight, "Frequency (Hz)",
        )

    def mouseMoveEvent(self, event):
        clean = self._clean_series()
        if not clean:
            return
        rect, x_min, x_max, _y_min, _y_max, _point = self._geometry(clean)
        if not rect.contains(event.position()):
            QToolTip.hideText()
            return
        ratio = (event.position().x() - rect.left()) / max(rect.width(), 1.0)
        frequency = 10 ** (math.log10(x_min) + ratio * (math.log10(x_max) - math.log10(x_min)))
        lines = [f"{frequency:.0f} Hz"]
        for item, points in clean:
            nearest = min(points, key=lambda value: abs(math.log(value[0] / frequency)))
            lines.append(f"{item.get('label', 'Series')}: {nearest[1]:+.2f} dB")
        QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)

    def mousePressEvent(self, event):
        if self.allow_enlarge and event.button() == Qt.LeftButton and (
            self.series or not self.static_pixmap.isNull()
        ):
            self.enlargeRequested.emit()
        super().mousePressEvent(event)


class ChartLabel(QWidget):
    """Interactive chart with series toggles, hover values and click-to-enlarge."""

    def __init__(self, placeholder: str, parent=None, allow_enlarge: bool = True):
        super().__init__(parent)
        self._placeholder = placeholder
        self._series: list[dict] = []
        self._markers: list[float] = []
        self.setObjectName("chart")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        self.toggle_row = QHBoxLayout()
        self.toggle_row.addStretch()
        layout.addLayout(self.toggle_row)
        self.canvas = _ChartCanvas(placeholder, allow_enlarge=allow_enlarge)
        self.canvas.enlargeRequested.connect(self._open_large)
        layout.addWidget(self.canvas, 1)
        self.setMinimumHeight(235)

    def _clear_toggles(self):
        while self.toggle_row.count():
            item = self.toggle_row.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def set_series(
        self, series: list[dict], markers: list[float] | None = None,
        fallback: str | None = None,
    ):
        if not series:
            self.clear_chart(fallback or self._placeholder)
            return
        self._series = [dict(item) for item in series]
        self._markers = list(markers or [])
        self._clear_toggles()
        for index, item in enumerate(self._series):
            checkbox = QCheckBox(str(item.get("label") or f"Series {index + 1}"))
            checkbox.setChecked(bool(item.get("visible", True)))
            checkbox.setStyleSheet(f"color:{item.get('color', theme.TEXT_PRIMARY)};")
            checkbox.toggled.connect(
                lambda checked, row=index: self._toggle_series(row, checked)
            )
            self.toggle_row.addWidget(checkbox)
        self.toggle_row.addStretch()
        self.canvas.set_plot(self._series, self._markers)

    def _toggle_series(self, index: int, checked: bool):
        self._series[index]["visible"] = checked
        self.canvas.set_plot(self._series, self._markers)

    def set_data_uri(self, data_uri: str, fallback: str | None = None):
        encoded = data_uri.partition(",")[2] if data_uri else ""
        pixmap = QPixmap()
        if encoded and pixmap.loadFromData(base64.b64decode(encoded)):
            self._clear_toggles()
            self._series = []
            self._markers = []
            self.canvas.set_static(pixmap)
            return
        self.clear_chart(fallback or self._placeholder)

    def clear_chart(self, message: str | None = None):
        self._clear_toggles()
        self._series = []
        self._markers = []
        self.canvas.clear_plot(message or self._placeholder)

    def _open_large(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Response chart")
        dialog.resize(1100, 680)
        layout = QVBoxLayout(dialog)
        chart = ChartLabel(self._placeholder, dialog, allow_enlarge=False)
        chart.setMinimumHeight(580)
        if self._series:
            chart.set_series(self._series, self._markers)
        elif not self.canvas.static_pixmap.isNull():
            chart.canvas.set_static(self.canvas.static_pixmap)
        layout.addWidget(chart)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.exec()

class OptimizerWindow(QMainWindow):
    TAB_HOME = 0
    TAB_PEQ = 1
    TAB_PHASE = 2
    TAB_RUN = 3
    TAB_RESULTS = 4
    TAB_VERIFY = 5
    TAB_ABOUT = 6
    TAB_RETARGET = 7

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(1120, 760)
        self.setMinimumSize(920, 650)
        self.settings = QSettings("AudioFischer Optimizer", "AudioFischer Optimizer")
        self.process_pid = 0
        self.process_finished_handled = False
        self.runner_log_offset = 0
        self.config: RunConfig | None = None
        self.summary_path: Path | None = None
        self.report_path: Path | None = None
        self.summary: dict = {}
        self.started_monotonic = 0.0
        self.memory_limit_hits = 0
        self.memory_guard_available, self.memory_guard_reason = memory_guard_status()
        self.memory_guard_error_logged = False
        self.stop_requested_reason = ""
        self.active_mode = "peq"
        self.validated_signatures: dict[str, tuple] = {}
        self.validated_configs: dict[str, RunConfig] = {}
        self.validation_task: BackgroundTask | None = None
        self.validation_context: tuple | None = None
        self.validation_diagnostics: dict[str, dict] = {}
        self.report_task: BackgroundTask | None = None
        self.shutdown_task: BackgroundTask | None = None
        self.verify_task: BackgroundTask | None = None
        self.close_after_stop = False
        self.current_run_phase = ""
        self.last_progress_value = 0
        self.pending_role_dialog: tuple[str, QTextEdit, RunConfig, dict] | None = None
        self.role_mapping_attempted: set[str] = set()
        self.thread_pool = QThreadPool.globalInstance()
        self._build_ui()
        self._apply_style()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_run)
        self.data_edit.textChanged.connect(lambda: self._workflow_input_changed("peq"))
        self.baseline_edit.textChanged.connect(lambda: self._workflow_input_changed("peq"))
        self.target_edit.textChanged.connect(lambda: self._workflow_input_changed("peq"))
        self.retarget_data_edit.textChanged.connect(lambda: self._workflow_input_changed("retarget"))
        self.retarget_baseline_edit.textChanged.connect(lambda: self._workflow_input_changed("retarget"))
        self.retarget_target_edit.textChanged.connect(lambda: self._workflow_input_changed("retarget"))
        self.retarget_target_edit.textChanged.connect(self._update_retarget_target_chart)
        self.phase_data_edit.textChanged.connect(lambda: self._workflow_input_changed("phase"))
        self.phase_baseline_edit.textChanged.connect(lambda: self._workflow_input_changed("phase"))
        self.phase_target_edit.textChanged.connect(lambda: self._workflow_input_changed("phase"))
        self._set_defaults()
        self._refresh_recent_runs()
        if not self.memory_guard_available:
            self._set_memory_guard_unavailable(self.memory_guard_reason)

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 20)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("AudioFischer Optimizer")
        title.setObjectName("title")
        title.setMinimumHeight(38)
        subtitle = QLabel("Local AFPX tuning, measurement validation and candidate export")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.run_badge = QLabel("READY")
        self.run_badge.setObjectName("badge")
        header.addWidget(self.run_badge)
        outer.addLayout(header)

        busy_line = QHBoxLayout()
        self.busy_label = QLabel("")
        self.busy_label.setObjectName("subtitle")
        self.busy_progress = QProgressBar()
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setMaximumWidth(240)
        self.busy_label.hide()
        self.busy_progress.hide()
        busy_line.addWidget(self.busy_label)
        busy_line.addWidget(self.busy_progress)
        busy_line.addStretch()
        outer.addLayout(busy_line)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_home_tab(), "Home")
        self.tabs.addTab(self._build_inputs_tab(), "PEQ / RTA")
        self.tabs.addTab(self._build_phase_tab(), "Sweeps / Phase")
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.tabs.addTab(self._build_verify_tab(), "Verify")
        self.tabs.addTab(self._build_about_tab(), "About")
        self.tabs.addTab(self._build_retarget_tab(), "Retarget")
        self._tab_steps = {
            self.TAB_PEQ: 1, self.TAB_PHASE: 2, self.TAB_RUN: 3, self.TAB_RESULTS: 4,
        }
        self._set_tab_enabled(self.TAB_RUN, False)
        self.tabs.setTabToolTip(self.TAB_RUN, "Validate one workflow before opening Run.")
        self._set_tab_enabled(self.TAB_RESULTS, False)
        self.tabs.setTabToolTip(self.TAB_RESULTS, "Complete or open a run before viewing Results.")
        for index in self._tab_steps:
            self._update_step_badge(index, self.tabs.isTabEnabled(index))
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _set_badge(self, text: str):
        self.run_badge.setText(text)
        self.run_badge.setStyleSheet(theme.badge_style(text))

    def _set_tab_enabled(self, index: int, enabled: bool):
        self.tabs.setTabEnabled(index, enabled)
        self._update_step_badge(index, enabled)

    def _update_step_badge(self, index: int, enabled: bool):
        step = self._tab_steps.get(index)
        if step is not None:
            self.tabs.setTabIcon(index, theme.step_badge_icon(step, active=enabled))

    def _card(self, *, accent: bool = False) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("cardAccent" if accent else "card")
        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(8)
        return frame, card_layout

    def _build_home_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(18)

        heading = QLabel("Choose the right workflow")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        intro = QLabel(
            "For a normal tune, complete PEQ first, load the result into the DSP, then take fresh "
            "sweeps for phase alignment. Retarget is only for changing the tonal curve later."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        folder_card, folder_layout = self._card()
        folder_title = QLabel("Measurement folder readiness")
        folder_title.setObjectName("workflowTitle")
        folder_title.setStyleSheet("margin-top:0;")
        folder_layout.addWidget(folder_title)
        home_row, self.home_measurement_edit = self._path_row(
            "folder", self._browse_home_measurements,
        )
        self.home_measurement_edit.pathDropped.connect(self._home_folder_selected)
        self.home_measurement_edit.textChanged.connect(self._update_home_checklist)
        folder_layout.addWidget(home_row)
        template_line = QHBoxLayout()
        self.template_layout_combo = QComboBox()
        self.template_layout_combo.addItem("2-way front + sub", "front_2way_plus_sub")
        self.template_layout_combo.addItem("3-way front + sub", "front_3way_plus_sub")
        template_button = QPushButton("Create Measurement Folder Template")
        template_button.setIcon(theme.make_icon("folder-plus"))
        template_button.clicked.connect(self._create_measurement_template)
        template_line.addWidget(self.template_layout_combo)
        template_line.addWidget(template_button)
        template_line.addStretch()
        folder_layout.addLayout(template_line)
        self.home_checklist = QLabel("Choose a measurement folder to see the required files.")
        self.home_checklist.setWordWrap(True)
        self.home_checklist.setTextFormat(Qt.RichText)
        folder_layout.addWidget(self.home_checklist)
        layout.addWidget(folder_card)

        peq_card, peq_layout = self._card(accent=True)
        peq_title = QLabel("1. PEQ / RTA - start here")
        peq_title.setObjectName("workflowTitle")
        peq_title.setStyleSheet("margin-top:0;")
        peq_layout.addWidget(peq_title)
        peq_text = QLabel(
            "Use fresh moving-mic or magnitude measurements to tune tonal balance and L/R response. "
            "This stage writes PEQ only and preserves delay, polarity, crossovers and APFs."
        )
        peq_text.setWordWrap(True)
        peq_layout.addWidget(peq_text)
        peq_button = QPushButton("Open PEQ / RTA")
        peq_button.setObjectName("primary")
        peq_button.setIcon(theme.make_icon("arrow-right", theme.TEXT_ON_ACCENT))
        peq_button.clicked.connect(
            lambda _checked=False: self.tabs.setCurrentIndex(self.TAB_PEQ)
        )
        peq_layout.addWidget(peq_button, 0, Qt.AlignLeft)
        layout.addWidget(peq_card)

        phase_card, phase_layout = self._card()
        phase_title = QLabel("2. Sweeps / Phase - after PEQ")
        phase_title.setObjectName("workflowTitle")
        phase_title.setStyleSheet("margin-top:0;")
        phase_layout.addWidget(phase_title)
        phase_text = QLabel(
            "Load the selected PEQ result into the DSP, take fresh phase-valid sweeps, then use this "
            "stage for supported polarity, delay and residual APF changes. Existing PEQ is preserved."
        )
        phase_text.setWordWrap(True)
        phase_layout.addWidget(phase_text)
        phase_button = QPushButton("Open Sweeps / Phase")
        phase_button.setIcon(theme.make_icon("arrow-right"))
        phase_button.clicked.connect(
            lambda _checked=False: self.tabs.setCurrentIndex(self.TAB_PHASE)
        )
        phase_layout.addWidget(phase_button, 0, Qt.AlignLeft)
        layout.addWidget(phase_card)

        retarget_card, retarget_layout = self._card()
        retarget_title = QLabel("Retarget - use later when changing the tonal curve")
        retarget_title.setObjectName("workflowTitle")
        retarget_title.setStyleSheet("margin-top:0;")
        retarget_layout.addWidget(retarget_title)
        retarget_text = QLabel(
            "Use fresh MMM/RTA measurements of the current tune plus a different target curve. "
            "It creates a new PEQ candidate without changing phase controls or the baseline."
        )
        retarget_text.setWordWrap(True)
        retarget_layout.addWidget(retarget_text)
        retarget_button = QPushButton("Open Retarget")
        retarget_button.setIcon(theme.make_icon("arrow-right"))
        retarget_button.clicked.connect(
            lambda _checked=False: self.tabs.setCurrentIndex(self.TAB_RETARGET)
        )
        retarget_layout.addWidget(retarget_button, 0, Qt.AlignLeft)
        layout.addWidget(retarget_card)

        recent_card, recent_layout = self._card()
        recent_title = QLabel("Recent runs")
        recent_title.setObjectName("workflowTitle")
        recent_title.setStyleSheet("margin-top:0;")
        recent_layout.addWidget(recent_title)
        self.recent_runs_table = QTableWidget(0, 5)
        self.recent_runs_table.setAlternatingRowColors(True)
        self.recent_runs_table.setHorizontalHeaderLabels(
            ["Date", "Workflow", "Status", "Best objective", "Path"]
        )
        self.recent_runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.recent_runs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.recent_runs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.recent_runs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.recent_runs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.recent_runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.recent_runs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_runs_table.setMaximumHeight(175)
        self.recent_runs_table.cellDoubleClicked.connect(self._open_recent_run)
        recent_layout.addWidget(self.recent_runs_table)
        layout.addWidget(recent_card)

        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _path_row(self, mode: str, browse_slot):
        edit = DropLineEdit(mode)
        edit.setPlaceholderText("Drop a folder here" if mode == "folder" else "Drop a file here")
        button = QToolButton()
        button.setIcon(theme.make_icon("folder-open" if mode == "folder" else "file"))
        button.setToolTip("Browse")
        button.clicked.connect(browse_slot)
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return box, edit

    def _build_inputs_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(14)

        intro = QLabel(
            "First stage: use fresh magnitude or RTA measurements captured at one consistent level "
            "to optimize PEQ. Phase, delay and APF writes are disabled in this stage."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        data_row, self.data_edit = self._path_row("folder", self._browse_data)
        base_row, self.baseline_edit = self._path_row("file", self._browse_baseline)
        target_row, self.target_edit = self._path_row("file", self._browse_target)
        self.data_edit.pathDropped.connect(self._data_dropped)
        self.baseline_edit.pathDropped.connect(self._baseline_dropped)
        form.addRow("Measurements", data_row)
        form.addRow("Baseline AFPX", base_row)
        form.addRow("Target curve", target_row)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.validate_button = QPushButton("Validate RTA / Prepare PEQ")
        self.validate_button.setIcon(theme.make_icon("check"))
        self.validate_button.clicked.connect(self.validate_inputs)
        self.resume_button = QPushButton("Open Existing Run")
        self.resume_button.setIcon(theme.make_icon("folder-open"))
        self.resume_button.clicked.connect(self._open_existing_run)
        self.copy_peq_diagnostics = QPushButton("Copy Diagnostics")
        self.copy_peq_diagnostics.clicked.connect(
            lambda: self._copy_validation_diagnostics("peq")
        )
        self.copy_peq_diagnostics.setEnabled(False)
        actions.addWidget(self.validate_button)
        actions.addWidget(self.resume_button)
        actions.addWidget(self.copy_peq_diagnostics)
        actions.addStretch()
        layout.addLayout(actions)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setPlaceholderText("Validation results appear here.")
        layout.addWidget(self.validation_text, 1)
        return page

    def _build_retarget_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(14)

        intro = QLabel(
            "Retune an existing tune to a different tonal target using fresh MMM or RTA "
            "measurements taken with that tune loaded. This runs the same conservative Beam "
            "optimizer as PEQ / RTA; delay, polarity, crossovers and APFs remain untouched."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        data_row, self.retarget_data_edit = self._path_row("folder", self._browse_retarget_data)
        base_row, self.retarget_baseline_edit = self._path_row("file", self._browse_retarget_baseline)
        target_row, self.retarget_target_edit = self._path_row("file", self._browse_retarget_target)
        self.retarget_data_edit.pathDropped.connect(self._retarget_data_dropped)
        self.retarget_baseline_edit.pathDropped.connect(self._retarget_baseline_dropped)
        self.retarget_target_edit.setPlaceholderText("Drop the new target curve here")
        form.addRow("Fresh measurements", data_row)
        form.addRow("Current tune AFPX", base_row)
        form.addRow("New target curve", target_row)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.validate_retarget_button = QPushButton("Validate / Prepare Retarget")
        self.validate_retarget_button.setIcon(theme.make_icon("check"))
        self.validate_retarget_button.clicked.connect(self.validate_retarget_inputs)
        self.copy_retarget_diagnostics = QPushButton("Copy Diagnostics")
        self.copy_retarget_diagnostics.clicked.connect(
            lambda: self._copy_validation_diagnostics("retarget")
        )
        self.copy_retarget_diagnostics.setEnabled(False)
        actions.addWidget(self.validate_retarget_button)
        actions.addWidget(self.copy_retarget_diagnostics)
        actions.addStretch()
        layout.addLayout(actions)

        note = QLabel(
            "The measurements must describe the current AFPX baseline. Retargeting changes only "
            "supported PEQ bands and writes a new candidate file; the baseline is never overwritten."
        )
        note.setObjectName("warning")
        note.setWordWrap(True)
        layout.addWidget(note)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(14)
        chart_box = QVBoxLayout()
        chart_title = QLabel("Target shape preview")
        chart_title.setObjectName("workflowTitle")
        chart_box.addWidget(chart_title)
        self.retarget_target_chart = ChartLabel("Choose a new target curve to preview it.")
        chart_box.addWidget(self.retarget_target_chart)
        chart_note = QLabel(
            "Curves are normalized to 0 dB at 1 kHz so their tonal shape can be compared."
        )
        chart_note.setObjectName("chartNote")
        chart_note.setWordWrap(True)
        chart_box.addWidget(chart_note)
        preview_row.addLayout(chart_box, 2)

        validation_box = QVBoxLayout()
        validation_title = QLabel("Validation")
        validation_title.setObjectName("workflowTitle")
        validation_box.addWidget(validation_title)
        self.retarget_validation_text = QTextEdit()
        self.retarget_validation_text.setReadOnly(True)
        self.retarget_validation_text.setPlaceholderText("Retarget validation results appear here.")
        validation_box.addWidget(self.retarget_validation_text, 1)
        preview_row.addLayout(validation_box, 1)
        layout.addLayout(preview_row, 1)
        return page

    def _build_phase_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(14)

        intro = QLabel(
            "Use this stage directly if the current tune's PEQ is already dialled in. Otherwise, "
            "load the PEQ result into the DSP first, take fresh phase-valid sweeps, and use that "
            "tune as the baseline here. Existing PEQ is preserved."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        data_row, self.phase_data_edit = self._path_row("folder", self._browse_phase_data)
        base_row, self.phase_baseline_edit = self._path_row("file", self._browse_phase_baseline)
        target_row, self.phase_target_edit = self._path_row("file", self._browse_phase_target)
        self.phase_data_edit.pathDropped.connect(self._phase_data_dropped)
        self.phase_baseline_edit.pathDropped.connect(self._phase_baseline_dropped)
        form.addRow("Fresh sweep folder", data_row)
        form.addRow("Current / PEQ tune AFPX", base_row)
        form.addRow("Target curve", target_row)
        layout.addLayout(form)

        action_line = QHBoxLayout()
        self.validate_phase_button = QPushButton("Validate Sweeps / Prepare Phase")
        self.validate_phase_button.setIcon(theme.make_icon("check"))
        self.validate_phase_button.clicked.connect(self.validate_phase_inputs)
        self.phase_use_peq_button = QPushButton("Use Latest PEQ Result")
        self.phase_use_peq_button.clicked.connect(self._use_latest_peq_result)
        self.copy_phase_diagnostics = QPushButton("Copy Diagnostics")
        self.copy_phase_diagnostics.clicked.connect(
            lambda: self._copy_validation_diagnostics("phase")
        )
        self.copy_phase_diagnostics.setEnabled(False)
        action_line.addWidget(self.validate_phase_button)
        action_line.addWidget(self.phase_use_peq_button)
        action_line.addWidget(self.copy_phase_diagnostics)
        action_line.addStretch()
        layout.addLayout(action_line)

        note = QLabel(
            "Only gated polarity, relative delay and residual APF changes can be written. "
            "No new PEQ filters are searched in this stage."
        )
        note.setObjectName("warning")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.phase_validation_text = QTextEdit()
        self.phase_validation_text.setReadOnly(True)
        self.phase_validation_text.setPlaceholderText("Sweep and phase validation results appear here.")
        layout.addWidget(self.phase_validation_text, 1)
        return page

    def _build_run_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        self.run_scroll = QScrollArea()
        self.run_scroll.setWidgetResizable(True)
        self.run_scroll.setFrameShape(QFrame.NoFrame)
        self.run_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.run_scroll.setObjectName("runScroll")

        self.run_content = QWidget()
        self.run_content.setObjectName("runContent")
        layout = QVBoxLayout(self.run_content)
        layout.setContentsMargins(22, 20, 22, 22)
        layout.setSpacing(12)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        setup_title = QLabel("Optimizer run settings")
        setup_title.setObjectName("sectionTitle")
        layout.addWidget(setup_title)
        setup_help = QLabel(
            "Choose the search time and resource limits. CPU controls how many workers "
            "run; the RAM setting is a safety stop, not a memory-use target."
        )
        setup_help.setObjectName("chartNote")
        setup_help.setWordWrap(True)
        layout.addWidget(setup_help)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setColumnMinimumWidth(0, 165)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, 210)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Quick check - 2 minutes", 120)
        self.preset_combo.addItem("Short run - 5 minutes", 300)
        self.preset_combo.addItem("Normal - 20 minutes", 1200)
        self.preset_combo.addItem("Thorough - 40 minutes", 2400)
        self.preset_combo.addItem("Custom", 0)
        self.preset_combo.setCurrentIndex(2)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(30, 14400)
        self.seconds_spin.setValue(1200)
        self.seconds_spin.setSuffix(" s")

        self.cpu_slider = QSlider(Qt.Horizontal)
        self.cpu_slider.setRange(20, 80)
        self.cpu_slider.setValue(60)
        self.cpu_label = QLabel("60%")
        self.cpu_slider.valueChanged.connect(lambda value: self.cpu_label.setText(f"{value}%"))
        self.cpu_control = QWidget()
        cpu_box = QHBoxLayout(self.cpu_control)
        cpu_box.setContentsMargins(0, 0, 0, 0)
        cpu_box.addWidget(self.cpu_slider, 1)
        cpu_box.addWidget(self.cpu_label)

        self.phase_run_value = QLabel("Automatic diagnostic - usually under 1 minute")
        self.phase_run_value.setObjectName("metricValue")
        self.phase_run_value.hide()
        self.phase_cpu_value = QLabel("1 bounded worker")
        self.phase_cpu_value.setObjectName("metricValue")
        self.phase_cpu_value.hide()

        self.ram_slider = QSlider(Qt.Horizontal)
        self.ram_slider.setRange(20, 70)
        self.ram_slider.setValue(50)
        self.ram_label = QLabel("50% of RAM")
        self.ram_slider.valueChanged.connect(lambda value: self.ram_label.setText(f"{value}% of RAM"))
        ram_help = (
            "Safety ceiling only. The optimizer does not try to fill this amount of RAM; "
            "it stops safely if the complete process tree reaches the selected percentage."
        )
        self.ram_slider.setToolTip(ram_help)
        self.ram_label.setToolTip(ram_help)
        ram_title = QLabel("RAM safety stop limit")
        ram_title.setToolTip(ram_help)
        ram_box = QHBoxLayout()
        ram_box.addWidget(self.ram_slider, 1)
        ram_box.addWidget(self.ram_label)

        self.workflow_value = QLabel("PEQ / RTA - Beam search + guided continuation")
        self.workflow_value.setObjectName("workflowSummary")
        self.workflow_value.setWordWrap(True)
        self.workflow_value.setMinimumHeight(54)

        grid.addWidget(QLabel("Run length"), 0, 0)
        grid.addWidget(self.preset_combo, 0, 1)
        grid.addWidget(self.seconds_spin, 0, 2)
        grid.addWidget(self.phase_run_value, 0, 1, 1, 2)
        grid.addWidget(QLabel("CPU target"), 1, 0)
        grid.addWidget(self.cpu_control, 1, 1, 1, 2)
        grid.addWidget(self.phase_cpu_value, 1, 1, 1, 2)
        grid.addWidget(ram_title, 2, 0)
        grid.addLayout(ram_box, 2, 1, 1, 2)
        grid.addWidget(QLabel("Workflow"), 3, 0)
        grid.addWidget(self.workflow_value, 3, 1, 1, 2)
        layout.addLayout(grid)

        options_title = QLabel("Optional run outputs")
        options_title.setObjectName("sectionTitle")
        layout.addWidget(options_title)
        options_help = QLabel(
            "These do not change the main optimization method. Enable only the extra "
            "files or recommendation you want."
        )
        options_help.setObjectName("chartNote")
        options_help.setWordWrap(True)
        layout.addWidget(options_help)

        option_line = QVBoxLayout()
        option_line.setSpacing(10)
        self.voicing_check = QCheckBox("Create voicing audition files")
        self.sub_blend_check = QCheckBox("Report sub level recommendation")
        self.voicing_check.setToolTip(
            "Creates neutral tonal audition alternatives for listening comparisons. "
            "It does not choose a preferred voicing; matched positive front voicing may "
            "use uniform protective attenuation when required."
        )
        self.sub_blend_check.setToolTip(
            "Calculates a recommendation only when measurement levels are calibrated and "
            "you declare available subwoofer headroom. It never writes an output-level change."
        )
        self.sub_blend_check.toggled.connect(lambda checked: self.headroom_spin.setEnabled(checked))
        self.headroom_spin = QDoubleSpinBox()
        self.headroom_spin.setRange(0.0, 12.0)
        self.headroom_spin.setValue(3.0)
        self.headroom_spin.setSuffix(" dB headroom")
        self.headroom_spin.setEnabled(False)
        voicing_box = QVBoxLayout()
        voicing_box.addWidget(self.voicing_check)
        voicing_help = QLabel(
            "Creates optional neutral listening files; no preferred tonal balance is selected."
        )
        voicing_help.setObjectName("metricName")
        voicing_help.setWordWrap(True)
        voicing_box.addWidget(voicing_help)
        option_line.addLayout(voicing_box)
        sub_box = QVBoxLayout()
        sub_head = QHBoxLayout()
        sub_head.addWidget(self.sub_blend_check)
        sub_head.addWidget(self.headroom_spin)
        sub_head.addStretch()
        sub_box.addLayout(sub_head)
        sub_help = QLabel(
            "Recommendation only; requires calibrated measurement level and declared headroom."
        )
        sub_help.setObjectName("metricName")
        sub_help.setWordWrap(True)
        sub_box.addWidget(sub_help)
        option_line.addLayout(sub_box)
        layout.addLayout(option_line)

        self.phase_warning = QLabel("PEQ stage uses Beam and cannot write phase changes. Phase stage preserves PEQ and writes only changes that pass the evidence gates. The baseline is never overwritten.")
        self.phase_warning.setObjectName("warning")
        self.phase_warning.setWordWrap(True)
        layout.addWidget(self.phase_warning)

        self.memory_guard_warning = QLabel()
        self.memory_guard_warning.setObjectName("warning")
        self.memory_guard_warning.setWordWrap(True)
        self.memory_guard_warning.hide()
        layout.addWidget(self.memory_guard_warning)

        progress_title = QLabel("Start and monitor the run")
        progress_title.setObjectName("sectionTitle")
        layout.addWidget(progress_title)

        action_line = QHBoxLayout()
        self.start_button = QPushButton("Start Optimizer")
        self.start_button.setObjectName("primary")
        self.start_button.setIcon(theme.make_icon("play", theme.TEXT_ON_ACCENT))
        self.start_button.clicked.connect(self._start_clicked)
        self.start_button.setEnabled(False)
        self.cancel_button = QPushButton("Stop Safely")
        self.cancel_button.setIcon(theme.make_icon("stop", theme.DANGER))
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setEnabled(False)
        self.open_run_button = QPushButton("Open Run Folder")
        self.open_run_button.setIcon(theme.make_icon("folder-open"))
        self.open_run_button.clicked.connect(self._open_run_folder)
        self.open_run_button.setEnabled(False)
        action_line.addWidget(self.start_button)
        action_line.addWidget(self.cancel_button)
        action_line.addWidget(self.open_run_button)
        action_line.addStretch()
        layout.addLayout(action_line)

        progress_line = QHBoxLayout()
        self.phase_label = QLabel("Ready")
        self.phase_label.setFixedWidth(220)
        self.phase_label.setObjectName("workflowTitle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        progress_line.addWidget(self.phase_label)
        progress_line.addWidget(self.progress, 1)
        layout.addLayout(progress_line)

        status_grid = QGridLayout()
        self.elapsed_value = QLabel("00:00")
        self.worker_value = QLabel("0")
        self.trial_value = QLabel("0")
        self.best_value = QLabel("-")
        self.memory_value = QLabel("-")
        for index, (label, widget) in enumerate((
            ("Elapsed", self.elapsed_value), ("Workers", self.worker_value),
            ("Candidates checked", self.trial_value), ("Best objective", self.best_value),
            ("Optimizer memory", self.memory_value),
        )):
            row, col = divmod(index, 3)
            cell = QVBoxLayout()
            name = QLabel(label)
            name.setObjectName("metricName")
            widget.setObjectName("metricValue")
            cell.addWidget(name)
            cell.addWidget(widget)
            status_grid.addLayout(cell, row, col)
            status_grid.setColumnStretch(col, 1)
        layout.addLayout(status_grid)
        self.convergence_label = QLabel(
            "Convergence: waiting for the first worker checkpoint."
        )
        self.convergence_label.setObjectName("chartNote")
        self.convergence_label.setWordWrap(True)
        layout.addWidget(self.convergence_label)
        self.convergence_table = QTableWidget(0, 2)
        self.convergence_table.setAlternatingRowColors(True)
        self.convergence_table.setHorizontalHeaderLabels(
            ["Elapsed", "Best objective"]
        )
        self.convergence_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.convergence_table.verticalHeader().setVisible(False)
        self.convergence_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.convergence_table.setSelectionMode(QTableWidget.NoSelection)
        self.convergence_table.setMinimumHeight(112)
        self.convergence_table.setMaximumHeight(145)
        self.convergence_table.setToolTip(
            "Recent objective improvements from the live worker checkpoints."
        )
        layout.addWidget(self.convergence_table)

        self.run_log = QTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setMinimumHeight(150)
        self.run_log.setPlaceholderText("Run status appears here. Full worker logs remain in the run folder.")
        layout.addWidget(self.run_log)

        self.run_scroll.setWidget(self.run_content)
        outer.addWidget(self.run_scroll)
        return page

    def _build_results_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)
        self.result_heading = QLabel("No completed run loaded")
        self.result_heading.setObjectName("sectionTitle")
        self.result_heading.setWordWrap(True)
        layout.addWidget(self.result_heading)

        self.improvement_banner = QLabel(
            "Load a completed run to compare it with the current tune."
        )
        self.improvement_banner.setObjectName("resultBanner")
        self.improvement_banner.setWordWrap(True)
        layout.addWidget(self.improvement_banner)

        metric_grid = QGridLayout()
        self.result_metric_values: list[QLabel] = []
        self.result_metric_details: list[QLabel] = []
        for column, label in enumerate((
            "Tonal accuracy", "Vocal region", "Narrow peaks", "L/R balance",
        )):
            frame = QFrame()
            frame.setObjectName("metricCard")
            box = QVBoxLayout(frame)
            box.setContentsMargins(10, 8, 10, 8)
            name = QLabel(label)
            name.setObjectName("metricName")
            value = QLabel("-")
            value.setObjectName("metricValue")
            detail = QLabel("Waiting for results")
            detail.setObjectName("metricName")
            detail.setWordWrap(True)
            box.addWidget(name)
            box.addWidget(value)
            box.addWidget(detail)
            metric_grid.addWidget(frame, 0, column)
            self.result_metric_values.append(value)
            self.result_metric_details.append(detail)
        layout.addLayout(metric_grid)

        self.results_chart = ChartLabel(
            "Complete or open a PEQ / Retarget run to see before and predicted-after response."
        )
        self.results_chart.setMinimumHeight(285)
        self.results_chart.setMaximumHeight(360)
        layout.addWidget(self.results_chart)
        self.results_chart_note = QLabel(
            "Toggle series, hover for frequency and dB, or click the chart to enlarge it."
        )
        self.results_chart_note.setObjectName("chartNote")
        self.results_chart_note.setWordWrap(True)
        layout.addWidget(self.results_chart_note)

        candidate_title = QLabel("Candidate comparison")
        candidate_title.setObjectName("workflowTitle")
        layout.addWidget(candidate_title)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setHorizontalHeaderLabels(["Choice", "Decision score", "File"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_table.itemSelectionChanged.connect(self._result_selected)
        self.result_table.setMaximumHeight(190)
        layout.addWidget(self.result_table)

        result_actions = QHBoxLayout()
        self.export_button = QPushButton("Export Selected AFPX")
        self.export_button.setIcon(theme.make_icon("export"))
        self.export_button.clicked.connect(self._export_selected)
        self.export_button.setEnabled(False)
        self.open_results_button = QPushButton("Open Results Folder")
        self.open_results_button.setIcon(theme.make_icon("folder-open"))
        self.open_results_button.clicked.connect(self._open_results_folder)
        self.open_results_button.setEnabled(False)
        self.open_report_button = QPushButton("Open Tuning Report")
        self.open_report_button.setIcon(theme.make_icon("report"))
        self.open_report_button.clicked.connect(self._open_report)
        self.open_report_button.setEnabled(False)
        self.record_decision_button = QPushButton("Record Listening Decision")
        self.record_decision_button.clicked.connect(self._record_listening_decision)
        self.record_decision_button.setEnabled(False)
        self.repeat_run_button = QPushButton("Repeat This Run")
        self.repeat_run_button.clicked.connect(self._repeat_loaded_run)
        self.repeat_run_button.setEnabled(False)
        result_actions.addWidget(self.export_button)
        result_actions.addWidget(self.open_report_button)
        result_actions.addWidget(self.open_results_button)
        result_actions.addWidget(self.record_decision_button)
        result_actions.addWidget(self.repeat_run_button)
        result_actions.addStretch()
        layout.addLayout(result_actions)

        filter_heading = QHBoxLayout()
        filter_title = QLabel("DSP changes")
        filter_title.setObjectName("workflowTitle")
        self.copy_filters_button = QPushButton("Copy Filters as Text")
        self.copy_filters_button.clicked.connect(self._copy_filters)
        self.copy_filters_button.setEnabled(False)
        filter_heading.addWidget(filter_title)
        filter_heading.addStretch()
        filter_heading.addWidget(self.copy_filters_button)
        layout.addLayout(filter_heading)
        self.filter_table = QTableWidget(0, 6)
        self.filter_table.setAlternatingRowColors(True)
        self.filter_table.setWordWrap(True)
        self.filter_table.setHorizontalHeaderLabels(
            ["Action", "Driver / group", "Slot", "Previous", "New", "Why"],
        )
        for column in (0, 1, 2, 3, 4):
            self.filter_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents,
            )
        self.filter_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.Stretch,
        )
        self.filter_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.filter_table.setMaximumHeight(280)
        layout.addWidget(self.filter_table)

        lower = QHBoxLayout()
        warning_box = QVBoxLayout()
        warning_title = QLabel("Warnings and things deliberately left alone")
        warning_title.setObjectName("workflowTitle")
        self.results_warnings = QTextEdit()
        self.results_warnings.setReadOnly(True)
        self.results_warnings.setMaximumHeight(145)
        warning_box.addWidget(warning_title)
        warning_box.addWidget(self.results_warnings)
        lower.addLayout(warning_box, 1)
        check_box = QVBoxLayout()
        check_title = QLabel("What to check in the car")
        check_title.setObjectName("workflowTitle")
        self.results_remeasure = QTextEdit()
        self.results_remeasure.setReadOnly(True)
        self.results_remeasure.setMaximumHeight(145)
        check_box.addWidget(check_title)
        check_box.addWidget(self.results_remeasure)
        lower.addLayout(check_box, 1)
        layout.addLayout(lower)

        self.results_notice = QLabel()
        self.results_notice.setObjectName("chartNote")
        self.results_notice.setWordWrap(True)
        layout.addWidget(self.results_notice)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return page

    def _build_verify_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(14)
        heading = QLabel("Verify predicted response against a post-load measurement")
        heading.setObjectName("sectionTitle")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        note = QLabel(
            "Load the recommended AFPX in PC-Tool, capture the same REW roles again, "
            "then select that folder here. Capture level is aligned, but response-shape "
            "differences remain visible."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.verify_run_edit = DropLineEdit("folder")
        self.verify_post_edit = DropLineEdit("folder")
        run_row = QHBoxLayout()
        run_row.addWidget(self.verify_run_edit, 1)
        run_browse = QPushButton("Browse")
        run_browse.clicked.connect(lambda: self._browse_verify_folder(self.verify_run_edit))
        run_row.addWidget(run_browse)
        post_row = QHBoxLayout()
        post_row.addWidget(self.verify_post_edit, 1)
        post_browse = QPushButton("Browse")
        post_browse.clicked.connect(lambda: self._browse_verify_folder(self.verify_post_edit))
        post_row.addWidget(post_browse)
        run_widget = QWidget()
        run_widget.setLayout(run_row)
        post_widget = QWidget()
        post_widget.setLayout(post_row)
        form.addRow("Completed run", run_widget)
        form.addRow("Post-load REW folder", post_widget)
        layout.addLayout(form)
        action_row = QHBoxLayout()
        self.verify_button = QPushButton("Compare Predicted vs Achieved")
        self.verify_button.clicked.connect(self._start_achieved_verification)
        action_row.addWidget(self.verify_button)
        action_row.addStretch()
        layout.addLayout(action_row)
        self.verify_status = QLabel("No post-load verification has been run.")
        self.verify_status.setObjectName("resultBanner")
        self.verify_status.setWordWrap(True)
        layout.addWidget(self.verify_status)
        self.verify_chart = ChartLabel(
            "Choose a completed run and post-load measurements to see predicted vs achieved."
        )
        self.verify_chart.setMinimumHeight(300)
        layout.addWidget(self.verify_chart, 1)
        self.verify_details = QTextEdit()
        self.verify_details.setReadOnly(True)
        self.verify_details.setMaximumHeight(150)
        layout.addWidget(self.verify_details)
        return page

    def _browse_verify_folder(self, field: QLineEdit):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose folder", field.text().strip() or str(Path.home())
        )
        if folder:
            field.setText(folder)

    def _start_achieved_verification(self):
        run_folder = Path(self.verify_run_edit.text().strip())
        post_folder = Path(self.verify_post_edit.text().strip())
        if not run_folder.is_dir() or not post_folder.is_dir():
            QMessageBox.warning(
                self, "Verification inputs missing",
                "Choose a completed run folder and the post-load REW measurement folder.",
            )
            return
        self.verify_button.setEnabled(False)
        self.verify_status.setText("Comparing predicted and achieved response...")
        self._show_busy("Verifying achieved response")
        task = BackgroundTask(
            lambda _cancel: verify_run(run_folder, post_folder)
        )
        self.verify_task = task
        task.signals.result.connect(self._achieved_verification_ready)
        task.signals.error.connect(self._achieved_verification_failed)
        task.signals.finished.connect(self._achieved_verification_finished)
        self.thread_pool.start(task)

    def _achieved_verification_ready(self, payload):
        system = payload.get("system") or {}
        frequencies = system.get("frequency_hz") or []
        self.verify_chart.set_series([
            {
                "label": "Predicted", "x": frequencies,
                "y": system.get("predicted_db") or [], "color": theme.SERIES_PREDICTED,
            },
            {
                "label": "Achieved", "x": frequencies,
                "y": system.get("achieved_db") or [], "color": theme.SERIES_CANDIDATE,
            },
        ])
        verdict = str(payload.get("verdict", "")).replace("_", " ").title()
        self.verify_status.setText(
            f"{verdict}. System difference {float(system.get('difference_rms_db', 0.0)):.2f} dB RMS; "
            f"vocal region {float(system.get('vocal_difference_rms_db', 0.0)):.2f} dB RMS."
        )
        rows = [
            f"{role}: {float(item.get('difference_rms_db', 0.0)):.2f} dB RMS"
            for role, item in (payload.get("drivers") or {}).items()
        ]
        rows.append(f"Saved: {payload.get('file', '')}")
        self.verify_details.setPlainText("\n".join(rows))

    def _achieved_verification_failed(self, error: str):
        self.verify_status.setText(f"Verification failed: {error}")

    def _achieved_verification_finished(self):
        self.verify_button.setEnabled(True)
        self._hide_busy()
        self.verify_task = None

    def _build_about_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        about = QTextEdit()
        about.setReadOnly(True)
        about.setHtml(f"""
        <h1>AudioFischer Optimizer</h1>
        <p><b>Version {html.escape(__version__)}</b></p>
        <p>A local, conservative tuning tool for Helix and Audiotec Fischer AFPX files. It reads REW measurements, predicts supported changes, writes new candidate tunes, and never overwrites the baseline.</p>
        <h2>Workflows</h2>
        <p><b>1. PEQ / RTA:</b> Start here with fresh magnitude or moving-mic RTA measurements to improve tonal balance and L/R consistency. Delay, polarity, APF and crossovers remain untouched.</p>
        <p><b>2. Sweeps / Phase:</b> After the PEQ result is loaded, fresh phase-valid sweeps are used to test crossover polarity, bounded relative delay and residual all-pass correction. Existing PEQ remains unchanged.</p>
        <p><b>Retarget:</b> Use later when changing tonal balance. Fresh MMM or RTA measurements of the current tune are optimized against a different supplied target curve while phase controls remain untouched.</p>
        <p>The Retarget tab previews target shapes at a common 1 kHz anchor. Results graphs the measured baseline and predicted candidate against the target using one fixed anchor.</p>
        <h2>How PEQ is judged</h2>
        <ul>
          <li>ERB-smoothed tonal error against the supplied target.</li>
          <li>Anchor-independent target-contour accuracy through 1.3-5 kHz.</li>
          <li>Extra weighting through the vocal and presence region.</li>
          <li>Peaks cost more than comparable dips.</li>
          <li>L/R signed bias plus absolute and RMS mismatch from solo traces.</li>
          <li>Centre, left-ear and right-ear robustness when those positions are supplied.</li>
          <li>Penalties for positive gain, excessive filter count, narrow/deep filters, wasted bands and unsupported asymmetry.</li>
        </ul>
        <h2>How phase is judged</h2>
        <p>The tool validates that solo traces reproduce the measured together trace, then checks only the crossover band. It tests polarity first, relative delay second and an APF only for a supported residual. Weak or inconsistent evidence is rejected or clearly warned.</p>
        <h2>What it deliberately avoids</h2>
        <ul>
          <li>Boosting destructive acoustic nulls or crossover cancellations with PEQ.</li>
          <li>EQ outside a driver's useful passband or at physical roll-off edges.</li>
          <li>Changing crossover frequency, slope, shelves or arbitrary output levels automatically.</li>
          <li>A matched whole-front voicing boost may add uniform protective attenuation only; it never raises an output level.</li>
          <li>Claiming that a predicted tune is verified before it is loaded and re-measured.</li>
        </ul>
        <h2>Objective</h2>
        <p>Lower is better. The displayed objective is a weighted decision score made from the named components above, not a single raw flatness number. Candidate reports show why a result won, what changed, what was left alone and what must be checked in-car.</p>
        """)
        layout.addWidget(about)
        return page

    def _apply_style(self):
        self.setStyleSheet(theme.build_stylesheet())

    def _set_defaults(self):
        target = default_target()
        fallback_target = str(target) if target.exists() else ""
        values = {
            "peq": (self.data_edit, self.baseline_edit, self.target_edit),
            "phase": (self.phase_data_edit, self.phase_baseline_edit, self.phase_target_edit),
            "retarget": (
                self.retarget_data_edit, self.retarget_baseline_edit,
                self.retarget_target_edit,
            ),
        }
        for workflow, fields in values.items():
            data_value = str(self.settings.value(f"paths/{workflow}/data", "") or "")
            baseline_value = str(self.settings.value(f"paths/{workflow}/baseline", "") or "")
            target_value = str(
                self.settings.value(f"paths/{workflow}/target", fallback_target) or ""
            )
            fields[0].setText(data_value)
            fields[1].setText(baseline_value)
            fields[2].setText(target_value)
        self.home_measurement_edit.setText(self.data_edit.text())
        self._update_home_checklist()

    def _show_busy(self, text: str):
        self.busy_label.setText(text)
        self.busy_label.show()
        self.busy_progress.show()

    def _hide_busy(self):
        self.busy_label.hide()
        self.busy_progress.hide()

    def _set_memory_guard_unavailable(self, reason: str):
        self.memory_guard_available = False
        self.memory_guard_reason = reason or "unknown error"
        self.memory_value.setText("Unavailable")
        self.memory_guard_warning.setText(
            "Memory guard unavailable. Automatic RAM-limit stopping is disabled for this "
            f"session. Reason: {self.memory_guard_reason}"
        )
        self.memory_guard_warning.show()
        if not self.memory_guard_error_logged:
            self.run_log.append(f"Memory guard unavailable: {self.memory_guard_reason}")
            self.memory_guard_error_logged = True

    def _workflow_fields(self, workflow: str) -> tuple[QLineEdit, QLineEdit, QLineEdit]:
        if workflow == "phase":
            return self.phase_data_edit, self.phase_baseline_edit, self.phase_target_edit
        if workflow == "retarget":
            return self.retarget_data_edit, self.retarget_baseline_edit, self.retarget_target_edit
        return self.data_edit, self.baseline_edit, self.target_edit

    def _workflow_input_changed(self, workflow: str):
        data, baseline, target = self._workflow_fields(workflow)
        self.settings.setValue(f"paths/{workflow}/data", data.text().strip())
        self.settings.setValue(f"paths/{workflow}/baseline", baseline.text().strip())
        self.settings.setValue(f"paths/{workflow}/target", target.text().strip())
        self.validated_signatures.pop(workflow, None)
        self.validated_configs.pop(workflow, None)
        if workflow == "peq" and self.home_measurement_edit.text() != data.text():
            self.home_measurement_edit.setText(data.text())
        if workflow == self.active_mode:
            self.start_button.setEnabled(False)
            self._set_tab_enabled(self.TAB_RUN, False)
            self.tabs.setTabToolTip(self.TAB_RUN, "Validate this workflow before opening Run.")
            self._set_badge("NEEDS VALIDATION")

    def _input_changed(self):
        self._workflow_input_changed(self.active_mode)

    def _browse_home_measurements(self):
        path = QFileDialog.getExistingDirectory(self, "Select measurement folder")
        if path:
            self._home_folder_selected(path)

    def _home_folder_selected(self, value: str):
        self.home_measurement_edit.setText(value)
        self.data_edit.setText(value)
        baseline = discover_baseline(Path(value))
        if baseline:
            self.baseline_edit.setText(str(baseline))

    def _update_home_checklist(self, _value: str = ""):
        folder = Path(self.home_measurement_edit.text().strip())
        if not folder.is_dir():
            self.home_checklist.setText(
                "Choose a measurement folder to see the required files."
            )
            return
        role_map = ""
        config = self.validated_configs.get("peq")
        if config:
            role_map = config.role_map
        checklist = measurement_checklist(folder, role_map or None)
        layout_name = (
            "3-way front + sub"
            if checklist["layout"] == "front_3way_plus_sub"
            else "2-way front + sub"
        )
        lines = [f"<b>Detected layout: {html.escape(layout_name)}</b>"]
        for row in checklist["rows"]:
            if row["ready"]:
                marker, colour, detail = "&#10003;", theme.ACCENT_LINE, Path(row["path"]).name
            elif not row.get("required", True):
                marker, colour, detail = "&#9675;", theme.WARN, f"{row['expected']} (optional)"
            elif row["empty"]:
                marker, colour, detail = "&#10007;", theme.DANGER, f"{row['expected']} (empty placeholder)"
            else:
                marker, colour, detail = "&#10007;", theme.DANGER, row["expected"]
            lines.append(
                f'<span style="color:{colour};font-weight:600">{marker}</span> '
                f"{html.escape(row['role'])}: {html.escape(detail)}"
            )
        self.home_checklist.setText("<br>".join(lines))

    def _create_measurement_template(self):
        parent = QFileDialog.getExistingDirectory(
            self, "Choose where to create the measurement folder",
        )
        if not parent:
            return
        name, accepted = QInputDialog.getText(
            self, "Measurement folder name", "Folder name:",
            text="AudioFischer Measurements",
        )
        if not accepted or not name.strip():
            return
        folder_name = name.strip()
        if Path(folder_name).name != folder_name or folder_name in (".", ".."):
            QMessageBox.warning(
                self, "Invalid folder name",
                "Enter a single folder name without slashes or parent-directory references.",
            )
            return
        destination = Path(parent) / folder_name
        try:
            created = create_measurement_template(
                destination, str(self.template_layout_combo.currentData()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Could not create template", str(exc))
            return
        self._home_folder_selected(str(destination))
        QMessageBox.information(
            self, "Template created",
            f"Created {len(created) - 1} empty REW placeholders and an instruction file in:\n"
            f"{destination}",
        )

    def _recent_run_paths(self) -> list[str]:
        try:
            values = json.loads(str(self.settings.value("recent_runs", "[]") or "[]"))
        except ValueError:
            values = []
        return [str(path) for path in values if path]

    def _remember_run(self, config: RunConfig):
        paths = [config.run_root, *self._recent_run_paths()]
        unique = list(dict.fromkeys(paths))[:12]
        self.settings.setValue("recent_runs", json.dumps(unique))
        self._refresh_recent_runs()

    def _refresh_recent_runs(self):
        if not hasattr(self, "recent_runs_table"):
            return
        rows = []
        for value in self._recent_run_paths():
            root = Path(value)
            try:
                config = RunConfig.load(root)
            except Exception:
                continue
            summary_path = locate_summary(root)
            best = "-"
            if summary_path:
                try:
                    objective = (load_summary(summary_path).get("best") or {}).get("objective")
                    best = "-" if objective is None else f"{float(objective):.5f}"
                except (OSError, ValueError, TypeError):
                    pass
            date = config.started_at.replace("T", " ") if config.started_at else "-"
            rows.append((date, config.ui_workflow, config.status, best, str(root)))
        self.recent_runs_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                if column == 4:
                    item.setData(Qt.UserRole, str(value))
                self.recent_runs_table.setItem(row_index, column, item)

    def _open_recent_run(self, row: int, _column: int):
        item = self.recent_runs_table.item(row, 4)
        if item:
            self._open_run_root(Path(str(item.data(Qt.UserRole) or item.text())))

    def _update_retarget_target_chart(self, _value: str = ""):
        selected_path = Path(self.retarget_target_edit.text().strip())
        selected = load_target_curve(selected_path)
        if not selected:
            self.retarget_target_chart.clear_chart("Choose a valid new target curve to preview it.")
            return
        reference = load_target_curve(default_target())
        series = [{
            "label": selected.get("file", "New target"),
            "x": selected.get("frequency_hz"),
            "y": selected.get("relative_db"),
            "color": theme.SERIES_CANDIDATE,
        }]
        try:
            same_target = selected_path.resolve() == default_target().resolve()
        except OSError:
            same_target = False
        if reference and not same_target:
            series.append({
                "label": "Built-in ResoNix 2026",
                "x": reference.get("frequency_hz"),
                "y": reference.get("relative_db"),
                "color": theme.SERIES_BASELINE,
                "dashed": True,
            })
        self.retarget_target_chart.set_series(
            series, fallback="The selected target could not be plotted.",
        )

    def _browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "Select measurement folder")
        if path:
            self.data_edit.setText(path)
            baseline = discover_baseline(Path(path))
            if baseline:
                self.baseline_edit.setText(str(baseline))

    def _data_dropped(self, value: str):
        baseline = discover_baseline(Path(value))
        if baseline:
            self.baseline_edit.setText(str(baseline))

    def _baseline_dropped(self, value: str):
        if not self.data_edit.text():
            self.data_edit.setText(str(Path(value).parent))

    def _browse_baseline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select baseline tune", self.data_edit.text(), "AFPX tune (*.afpx)")
        if path:
            self.baseline_edit.setText(path)
            if not self.data_edit.text():
                self.data_edit.setText(str(Path(path).parent))

    def _browse_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select target curve", "", "Text files (*.txt);;All files (*)")
        if path:
            self.target_edit.setText(path)

    def _browse_retarget_data(self):
        path = QFileDialog.getExistingDirectory(self, "Select fresh retarget measurement folder")
        if path:
            self.retarget_data_edit.setText(path)
            baseline = discover_baseline(Path(path))
            if baseline:
                self.retarget_baseline_edit.setText(str(baseline))

    def _retarget_data_dropped(self, value: str):
        baseline = discover_baseline(Path(value))
        if baseline:
            self.retarget_baseline_edit.setText(str(baseline))

    def _retarget_baseline_dropped(self, value: str):
        if not self.retarget_data_edit.text():
            self.retarget_data_edit.setText(str(Path(value).parent))

    def _browse_retarget_baseline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select current tune", self.retarget_data_edit.text(), "AFPX tune (*.afpx)"
        )
        if path:
            self.retarget_baseline_edit.setText(path)
            if not self.retarget_data_edit.text():
                self.retarget_data_edit.setText(str(Path(path).parent))

    def _browse_retarget_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select new target curve", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            self.retarget_target_edit.setText(path)

    def _browse_phase_data(self):
        path = QFileDialog.getExistingDirectory(self, "Select fresh sweep folder")
        if path:
            self.phase_data_edit.setText(path)

    def _browse_phase_baseline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PEQ result tune", "", "AFPX tune (*.afpx)")
        if path:
            self.phase_baseline_edit.setText(path)

    def _browse_phase_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select target curve", "", "Text files (*.txt);;All files (*)")
        if path:
            self.phase_target_edit.setText(path)

    def _phase_data_dropped(self, _value: str):
        return

    def _phase_baseline_dropped(self, _value: str):
        return

    def _use_latest_peq_result(self):
        if not self.summary_path or not self.summary:
            QMessageBox.information(self, "No PEQ result loaded", "Complete or open a PEQ run first.")
            return
        best = self.summary.get("best") or {}
        candidate = self.summary_path.parent / str(best.get("file", ""))
        if not candidate.exists():
            QMessageBox.warning(self, "PEQ result missing", "The best PEQ candidate file could not be found.")
            return
        self.phase_baseline_edit.setText(str(candidate))

    def _current_config(self, run_root: Path | None = None, mode: str | None = None) -> RunConfig:
        workflow = mode or self.active_mode
        if workflow == "phase":
            data_root = self.phase_data_edit.text().strip()
            baseline = self.phase_baseline_edit.text().strip()
            target = self.phase_target_edit.text().strip()
        elif workflow == "retarget":
            data_root = self.retarget_data_edit.text().strip()
            baseline = self.retarget_baseline_edit.text().strip()
            target = self.retarget_target_edit.text().strip()
        else:
            data_root = self.data_edit.text().strip()
            baseline = self.baseline_edit.text().strip()
            target = self.target_edit.text().strip()
        backend_mode = "phase" if workflow == "phase" else "peq"
        return RunConfig(
            data_root=data_root, baseline=baseline, target=target,
            run_root=str(run_root or timestamped_run_root()), mode=backend_mode,
            workflow=workflow,
            seconds=30 if workflow == "phase" else self.seconds_spin.value(),
            cpu_percent=20 if workflow == "phase" else self.cpu_slider.value(),
            ram_percent=self.ram_slider.value(), proposal="beam",
            phase_writes="auto" if workflow == "phase" else "off",
            voicing_variants=("audition" if self.voicing_check.isChecked() else "off") if workflow == "peq" else "off",
            sub_blend=("recommend" if self.sub_blend_check.isChecked() else "off") if workflow == "peq" else "off",
            headroom_db=(self.headroom_spin.value() if self.sub_blend_check.isChecked() else None) if workflow == "peq" else None,
            level_calibration="",
        )

    def validate_inputs(self):
        return self._validate_workflow("peq", self.validation_text)

    def validate_phase_inputs(self):
        return self._validate_workflow("phase", self.phase_validation_text)

    def validate_retarget_inputs(self):
        return self._validate_workflow("retarget", self.retarget_validation_text)

    def _validate_workflow(self, mode: str, output: QTextEdit):
        if self.validation_task:
            return
        config = self._current_config(mode=mode)
        self._begin_validation(mode, output, config)

    def _begin_validation(self, mode: str, output: QTextEdit, config: RunConfig):
        buttons = {
            "peq": (self.validate_button, "Validate RTA / Prepare PEQ", self.validate_inputs),
            "phase": (
                self.validate_phase_button, "Validate Sweeps / Prepare Phase",
                self.validate_phase_inputs,
            ),
            "retarget": (
                self.validate_retarget_button, "Validate / Prepare Retarget",
                self.validate_retarget_inputs,
            ),
        }
        button, original_text, callback = buttons[mode]
        for candidate in (self.validate_button, self.validate_phase_button, self.validate_retarget_button):
            candidate.setEnabled(candidate is button)
        button.clicked.disconnect()
        button.setText("Cancel Validation")
        button.clicked.connect(self.cancel_validation)
        output.setPlainText("Validating measurements and tune data...")
        self._set_badge("VALIDATING")
        self._show_busy("Validating inputs")
        self.validation_context = (mode, output, button, original_text, callback, config)
        task = BackgroundTask(lambda cancelled: validate_config(config, cancelled))
        self.validation_task = task
        task.signals.result.connect(self._validation_result)
        task.signals.error.connect(self._validation_error)
        task.signals.finished.connect(self._validation_finished)
        self.thread_pool.start(task)

    def cancel_validation(self):
        if not self.validation_task or not self.validation_context:
            return
        _mode, output, button, _text, _callback, _config = self.validation_context
        self.validation_task.cancel()
        button.setEnabled(False)
        button.setText("Cancelling...")
        output.append("\nCancelling validation...")

    def _validation_result(self, result: dict):
        if not self.validation_context:
            return
        mode, output, _button, _text, _callback, config = self.validation_context
        diagnostics = dict(result.get("diagnostics") or {})
        if diagnostics:
            self.validation_diagnostics[mode] = diagnostics
            self._diagnostics_button(mode).setEnabled(True)
        if result.get("cancelled"):
            output.setPlainText("Validation cancelled. No optimizer run was started.")
            self.start_button.setEnabled(False)
            self._set_badge("VALIDATION CANCELLED")
            return
        compact = result.get("compact") or {}
        missing_roles = list(compact.get("missing_roles") or [])
        available_txt = list(compact.get("available_txt") or [])
        if (
            missing_roles and available_txt
            and config.run_root not in self.role_mapping_attempted
        ):
            self.role_mapping_attempted.add(config.run_root)
            self.pending_role_dialog = (mode, output, config, compact)
            output.setPlainText(
                "Some required roles could not be identified from the filenames. "
                "The role-mapping dialog will open next."
            )
            self.start_button.setEnabled(False)
            self._set_badge("ROLE MAPPING NEEDED")
            return
        if result["compact"]:
            layout_label = {
                "front_2way_plus_sub": "2-way front + sub",
                "front_3way_plus_sub": "3-way front + sub",
            }.get(str(compact["detected_layout"]), str(compact["detected_layout"]))
            mode_label = {
                "magnitude_only_peq": "PEQ from magnitude data",
                "crossover_ladder_available": "PEQ plus evidence-gated phase analysis",
            }.get(str(compact["safe_mode"]), str(compact["safe_mode"]))
            lines = [
                f"Layout: {layout_label}",
                f"Mode: {mode_label}",
                f"Measurements found: {compact['measurement_count']}",
                f"Phase files: {compact['phase_file_count']}  |  Coherence: {compact['coherence_file_count']}  |  Impulses: {compact['impulse_file_count']}",
                f"Spatial positions: {', '.join(compact['spatial_positions']) or 'centre only'}",
            ]
            if compact["missing"]:
                lines.append("\nMissing required:\n- " + "\n- ".join(compact["missing"]))
            if compact.get("optional_missing"):
                lines.append(
                    "\nOptional pair traces not supplied (PEQ can continue):\n- "
                    + "\n- ".join(compact["optional_missing"])
                )
            preflight = result.get("preflight") or {}
            for row in preflight.get("pair_validation", []):
                if row.get("pass") is None:
                    lines.append(
                        f"Pair gate {row.get('pair')}: NOT AVAILABLE - "
                        "using individual drivers plus System Sum for PEQ"
                    )
                else:
                    verdict = "PASS" if row.get("pass") else "FAIL"
                    lines.append(
                        f"Pair gate {row.get('pair')}: {row.get('rms_db')} dB / "
                        f"{row.get('threshold_db')} dB - {verdict}"
                    )
            audit = preflight.get("measurement_session") or {}
            if audit:
                lines.append(
                    f"Tonal session: {'PASS' if audit.get('tonal_valid') else 'FAIL'}  |  "
                    f"Phase session: {'PASS' if audit.get('phase_valid') else 'DISABLED'}"
                )
            census = preflight.get("problem_census") or {}
            worth = census.get("worth_fixing") or []
            skipped = census.get("deliberately_skipped") or []
            if worth:
                lines.append("\nHighest-value problems the run can address:")
                lines.extend(
                    f"- {item.get('group')}: {float(item.get('frequency_hz', 0.0)):.0f} Hz "
                    f"({item.get('source')}, priority {float(item.get('recoverable_error', 0.0)):.2f})"
                    for item in worth[:5]
                )
            if skipped:
                lines.append("\nDeliberately skipped before search:")
                lines.extend(
                    f"- {item.get('reason')} at "
                    + (
                        f"{float(item.get('frequency_hz')):.0f} Hz"
                        if item.get("frequency_hz") is not None else "the affected pair band"
                    )
                    for item in skipped[:5]
                )
            if result["errors"]:
                lines.append("\nBlocked:\n- " + "\n- ".join(result["errors"]))
            if result["valid"]:
                lines.append("\nPASS: the optimizer can start with this input set.")
            body = (
                '<div style="white-space:pre-wrap">'
                + html.escape("\n".join(lines))
                + "</div>"
            )
            warning_tokens = list(compact["warnings"])
            warning_tokens.extend(
                token for token in audit.get("warnings", [])
                if token not in warning_tokens
            )
            if warning_tokens:
                warning_rows = []
                for token in warning_tokens:
                    info = warning_info(token)
                    warning_rows.append(
                        '<div style="margin-top:8px;color:%s"><b>%s</b><br>%s</div>'
                        % (
                            theme.severity_colour(info["severity"]),
                            html.escape(info["severity"].upper()),
                            html.escape(info["text"]),
                        )
                    )
                body += "<h3>Warnings and fixes</h3>" + "".join(warning_rows)
            output.setHtml(body)
        else:
            output.setPlainText("\n".join(result["errors"]))
        self.start_button.setEnabled(bool(result["valid"]))
        self._set_badge("VALIDATED" if result["valid"] else "INPUT BLOCKED")
        if result["valid"]:
            self.validated_signatures[mode] = self._config_signature(config)
            self.validated_configs[mode] = config
            self.active_mode = mode
            phase_mode = mode == "phase"
            tone_options = mode == "peq"
            labels = {
                "phase": "Sweeps / Phase - preserve PEQ, gated phase writes only",
                "retarget": "Retarget - Beam search + guided continuation, no phase writes",
                "peq": "PEQ / RTA - Beam search + guided continuation, no phase writes",
            }
            self.workflow_value.setText(labels[mode])
            self.preset_combo.setEnabled(not phase_mode)
            self.seconds_spin.setEnabled(not phase_mode)
            self.cpu_slider.setEnabled(not phase_mode)
            self.preset_combo.setVisible(not phase_mode)
            self.seconds_spin.setVisible(not phase_mode)
            self.cpu_control.setVisible(not phase_mode)
            self.phase_run_value.setVisible(phase_mode)
            self.phase_cpu_value.setVisible(phase_mode)
            self.voicing_check.setEnabled(tone_options)
            self.sub_blend_check.setEnabled(tone_options)
            self.headroom_spin.setEnabled(tone_options and self.sub_blend_check.isChecked())
            self._set_tab_enabled(self.TAB_RUN, True)
            self.tabs.setTabToolTip(self.TAB_RUN, "Run the validated workflow.")
            self.tabs.setCurrentIndex(self.TAB_RUN)
            self._update_home_checklist()

    def _validation_error(self, error: str):
        if not self.validation_context:
            return
        mode, output, _button, _text, _callback, config = self.validation_context
        self.validation_diagnostics[mode] = {
            "job_config": config.__dict__,
            "manifest": None,
            "stderr": error,
            "exception_before_result": True,
        }
        self._diagnostics_button(mode).setEnabled(True)
        output.setPlainText(f"Validation failed:\n{error}")
        self.start_button.setEnabled(False)
        self._set_badge("VALIDATION FAILED")

    def _validation_finished(self):
        if not self.validation_context:
            return
        _mode, _output, button, original_text, callback, _config = self.validation_context
        button.clicked.disconnect()
        button.setText(original_text)
        button.clicked.connect(callback)
        for candidate in (self.validate_button, self.validate_phase_button, self.validate_retarget_button):
            candidate.setEnabled(True)
        self.validation_task = None
        self.validation_context = None
        self._hide_busy()
        pending = self.pending_role_dialog
        self.pending_role_dialog = None
        if pending:
            QTimer.singleShot(0, lambda: self._show_role_mapping_dialog(pending))

    def _diagnostics_button(self, mode: str) -> QPushButton:
        return {
            "peq": self.copy_peq_diagnostics,
            "phase": self.copy_phase_diagnostics,
            "retarget": self.copy_retarget_diagnostics,
        }[mode]

    def _copy_validation_diagnostics(self, mode: str):
        diagnostics = self.validation_diagnostics.get(mode)
        if not diagnostics:
            return
        QApplication.clipboard().setText(json.dumps(
            diagnostics, indent=2, default=str,
        ))
        self._diagnostics_button(mode).setText("Diagnostics Copied")
        QTimer.singleShot(
            1600,
            lambda: self._diagnostics_button(mode).setText("Copy Diagnostics"),
        )

    def _show_role_mapping_dialog(
        self, context: tuple[str, QTextEdit, RunConfig, dict],
    ):
        mode, output, config, compact = context
        remembered = self._remembered_role_names()
        dialog = RoleMappingDialog(
            list(compact.get("available_txt") or []),
            dict(compact.get("resolved_roles") or {}),
            remembered, self,
        )
        if dialog.exec() != QDialog.Accepted or not dialog.mapping:
            output.setPlainText(
                "Role mapping was not saved. Validation remains blocked until the "
                "required measurements are mapped."
            )
            self._set_badge("INPUT BLOCKED")
            return
        mapped = dialog.mapping
        three_way_roles = {
            "FL Mid", "FR Mid", "Mids Together",
            "FL Low", "FR Low", "Mid Bass Together",
        }
        layout = (
            "front_3way_plus_sub"
            if three_way_roles.issubset(mapped)
            else "front_2way_plus_sub"
        )
        role_map_path = save_role_map(
            Path(config.run_root) / "role_map.json", mapped, layout,
        )
        config.role_map = str(role_map_path)
        config.save()
        if dialog.remember_check.isChecked():
            remembered.update({
                filename.lower(): role for role, filename in mapped.items()
            })
            self.settings.setValue(
                "remembered_role_names", json.dumps(remembered),
            )
        output.setPlainText("Role mapping saved. Revalidating the mapped files...")
        self._begin_validation(mode, output, config)

    def _remembered_role_names(self) -> dict[str, str]:
        try:
            values = json.loads(
                str(self.settings.value("remembered_role_names", "{}") or "{}")
            )
        except ValueError:
            values = {}
        return {
            str(name).lower(): str(role)
            for name, role in dict(values).items()
            if role in ALL_MEASUREMENT_ROLES
        }

    @staticmethod
    def _config_signature(config: RunConfig) -> tuple:
        return (
            config.ui_workflow, config.data_root, config.baseline, config.target,
            config.level_calibration, config.role_map,
        )

    def _preset_changed(self):
        seconds = self.preset_combo.currentData()
        if seconds:
            self.seconds_spin.setValue(seconds)

    def _set_run_progress(self, value: int) -> None:
        self.last_progress_value = max(0, min(1000, int(value)))
        self.progress.setRange(0, 1000)
        self.progress.setValue(self.last_progress_value)

    def _freeze_run_progress(self) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(self.last_progress_value)

    def _start_clicked(self, _checked: bool = False):
        self.start_run()

    def start_run(self, resume_root: Path | None = None):
        if isinstance(resume_root, bool):
            resume_root = None
        if self._run_is_active():
            return
        if resume_root is None:
            validated = self.validated_configs.get(self.active_mode)
            if not validated:
                QMessageBox.information(
                    self, "Validation required",
                    "Validate the current inputs before starting the optimizer.",
                )
                return
            current = self._current_config(
                run_root=Path(validated.run_root), mode=self.active_mode,
            )
            current.role_map = validated.role_map
            if self.validated_signatures.get(self.active_mode) != self._config_signature(current):
                QMessageBox.information(
                    self, "Validation required",
                    "The active workflow inputs changed. Validate them again before starting.",
                )
                return
        if resume_root is not None:
            config = RunConfig.load(resume_root)
            config.status = "resuming"
            config.error = ""
        else:
            config = current
        try:
            claim_run_root(Path(config.run_root))
        except RunRootBusyError as exc:
            QMessageBox.warning(
                self, "Run already active",
                f"{exc}\n\nOpen that run from Recent Runs instead of starting it twice.",
            )
            return
        config.started_at = datetime.now().isoformat(timespec="seconds")
        config.status = "running"
        config.save()
        self._remember_run(config)
        self.config = config
        self.started_monotonic = time.monotonic()
        self.memory_limit_hits = 0
        self.memory_guard_error_logged = False
        self.stop_requested_reason = ""
        self.current_run_phase = ""
        self.process_finished_handled = False
        self.runner_log_offset = 0
        program, args = powershell_command(config)
        try:
            pid = start_detached_process(
                program, args, Path(config.run_root).parent,
                Path(config.run_root) / "gui_runner.log",
            )
        except OSError as exc:
            release_run_claim(Path(config.run_root))
            config.status = "failed"
            config.error = f"The detached optimizer process could not be started: {exc}"
            config.save()
            QMessageBox.critical(self, "Could not start", config.error)
            return
        self.process_pid = int(pid)
        update_run_claim(Path(config.run_root), self.process_pid)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.open_run_button.setEnabled(True)
        self._set_badge("RUNNING")
        memory_line = (
            f"Memory stop limit: {config.ram_percent}% of physical RAM"
            if self.memory_guard_available else
            f"Memory guard unavailable: {self.memory_guard_reason}"
        )
        self.run_log.setPlainText(
            f"Run folder: {config.run_root}\nWorkers: {config.workers}\n"
            f"{memory_line}\n"
        )
        self.memory_guard_error_logged = not self.memory_guard_available
        self.phase_label.setText(
            "Preparing phase diagnostics"
            if config.mode == "phase" else "Preparing existing tune"
        )
        self.last_progress_value = 0
        self._set_run_progress(0)
        self.poll_timer.start(1000)
        self.tabs.setCurrentIndex(self.TAB_RUN)

    def _run_is_active(self) -> bool:
        return bool(self.process_pid and process_is_running(self.process_pid))

    def _read_runner_log(self):
        if not self.config:
            return
        path = Path(self.config.run_root) / "gui_runner.log"
        if not path.exists():
            return
        try:
            with path.open("rb") as stream:
                stream.seek(self.runner_log_offset)
                payload = stream.read()
                self.runner_log_offset = stream.tell()
        except OSError:
            return
        text = payload.decode("utf-8", errors="replace").strip()
        if text:
            self.run_log.append(text)

    def _poll_run(self):
        if not self.config or not self.process_pid:
            return
        self._read_runner_log()
        if not self._run_is_active():
            if not self.process_finished_handled:
                exit_code = (
                    0
                    if runner_completed_successfully(Path(self.config.run_root))
                    else 1
                )
                self._process_finished(exit_code, None)
            return
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        self.elapsed_value.setText(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        if self.stop_requested_reason:
            self._freeze_run_progress()
            return
        progress = collect_progress(Path(self.config.run_root))
        phase = progress["phase"]
        phase_names = {
            "preparing": (
                "Preparing phase diagnostics"
                if self.config.mode == "phase" else "Preparing existing tune"
            ),
            "searching": "Searching",
            "merging": "Merging",
            "verifying": "Verifying",
            "reporting": "Writing report",
            "complete": "Finalizing",
        }
        if phase != self.current_run_phase:
            self.current_run_phase = phase
            self.run_log.append(f"Phase: {phase_names.get(phase, phase.title())}")
        if phase == "preparing":
            self.phase_label.setText(phase_names["preparing"])
            self._set_run_progress(0)
        elif phase == "searching":
            self.phase_label.setText("Searching")
            worker_elapsed = float(progress.get("elapsed_worker_seconds", 0.0))
            self._set_run_progress(
                min(800, round(800 * worker_elapsed / max(self.config.seconds, 1)))
            )
        elif phase == "merging":
            merge = progress.get("merge_progress") or {}
            fraction = merge_progress_fraction(merge)
            stage = str(merge.get("stage") or "").replace("_", " ").strip()
            self.phase_label.setText("Merging" + (f" - {stage}" if stage else ""))
            self._set_run_progress(round(800 + 100 * fraction))
        elif phase == "verifying":
            verified = progress["verified_candidates"]
            candidates = progress["verification_candidates"]
            suffix = f" {verified}/{candidates}" if candidates else ""
            self.phase_label.setText(f"Verifying{suffix}")
            fraction = verified / candidates if candidates else 0.0
            self._set_run_progress(round(900 + 80 * min(1.0, fraction)))
        elif phase == "reporting":
            self.phase_label.setText("Writing report")
            self._set_run_progress(990)
        else:
            self.phase_label.setText("Finalizing")
            self._set_run_progress(995)
        self.worker_value.setText(str(progress["workers_reporting"]))
        self.trial_value.setText(f"{progress['trials']:,}")
        objective = progress["best_objective"]
        self.best_value.setText("-" if objective is None else f"{objective:.5f}")
        convergence = progress.get("convergence") or {}
        verdict = str(convergence.get("verdict", "deterministic_plateau"))
        stalled = float(convergence.get("stalled_seconds", 0.0))
        convergence_text = {
            "still_improving": "Still improving; a longer run may find more.",
            "stalled": f"No improvement for {stalled / 60.0:.1f} minutes; the search is stalled.",
            "deterministic_plateau": "Deterministic pass complete; waiting for a guided improvement.",
        }.get(verdict, verdict.replace("_", " ").title())
        self.convergence_label.setText("Convergence: " + convergence_text)
        events = sorted(
            (
                event for event in convergence.get("events", [])
                if "elapsed_seconds" in event and "objective" in event
            ),
            key=lambda event: float(event["elapsed_seconds"]),
        )[-8:]
        self.convergence_table.setRowCount(len(events))
        for row, event in enumerate(events):
            elapsed_event = max(0, round(float(event["elapsed_seconds"])))
            elapsed_text = (
                f"{elapsed_event // 60:02d}:{elapsed_event % 60:02d}"
            )
            self.convergence_table.setItem(
                row, 0, QTableWidgetItem(elapsed_text)
            )
            self.convergence_table.setItem(
                row, 1, QTableWidgetItem(f"{float(event['objective']):.5f}")
            )
        pid = self.process_pid
        rss, total, memory_error = process_tree_memory(pid)
        if total:
            percent = 100.0 * rss / total
            self.memory_value.setText(f"{rss / 2**30:.2f} GB ({percent:.1f}%)")
            self.memory_limit_hits = self.memory_limit_hits + 1 if percent >= self.config.ram_percent else 0
            if self.memory_limit_hits >= 3:
                self.run_log.append(
                    f"Memory safety stop: optimizer reached {percent:.1f}% of physical RAM "
                    f"(limit {self.config.ram_percent}%). State is preserved for resume."
                )
                self.cancel_run(memory_stop=True)
        elif memory_error and not self.memory_guard_error_logged:
            self._set_memory_guard_unavailable(memory_error)

    def cancel_run(self, memory_stop: bool = False):
        if not self._run_is_active():
            return
        pid = self.process_pid
        self.stop_requested_reason = "memory" if memory_stop else "user"
        stop_file = Path(self.config.run_root) / "stop_requested" if self.config else None
        if stop_file:
            stop_file.write_text(self.stop_requested_reason, encoding="ascii")
        self._freeze_run_progress()
        self._set_badge("STOPPING SAFELY")
        self.phase_label.setText("Stopping safely")
        self.cancel_button.setEnabled(False)
        self._show_busy("Stopping safely and preserving checkpoints")
        if self.config:
            self.config.status = "memory_stopped" if memory_stop else "stopped"
            self.config.error = "Memory safety limit reached" if memory_stop else "Stopped by user"
            self.config.save()
            self._remember_run(self.config)
        task = BackgroundTask(lambda _cancelled: stop_process_tree(pid))
        self.shutdown_task = task
        task.signals.result.connect(self._shutdown_result)
        task.signals.error.connect(self._shutdown_error)
        task.signals.finished.connect(self._shutdown_finished)
        self.thread_pool.start(task)

    def _shutdown_result(self, result: dict):
        if result.get("forced"):
            self.run_log.append("The optimizer did not stop cooperatively; its process tree was ended.")
        if result.get("error"):
            self.run_log.append(f"Shutdown warning: {result['error']}")
        if result.get("forced"):
            self.process_finished_handled = False

    def _shutdown_error(self, error: str):
        self.run_log.append(f"Shutdown warning: {error}")

    def _shutdown_finished(self):
        self.shutdown_task = None
        self._hide_busy()
        if not self._run_is_active():
            self._set_badge("STOPPED - RESUMABLE")
            self.phase_label.setText("Stopped")
        if self.close_after_stop:
            self.close_after_stop = False
            QTimer.singleShot(0, self.close)

    def _process_finished(self, exit_code: int, _status):
        if self.process_finished_handled:
            return
        self.process_finished_handled = True
        self._read_runner_log()
        self.poll_timer.stop()
        self.cancel_button.setEnabled(False)
        if not self.config:
            return
        finished_pid = self.process_pid
        self.process_pid = 0
        release_run_claim(Path(self.config.run_root), finished_pid)
        summary = locate_summary(Path(self.config.run_root))
        if exit_code == 0 and summary:
            stopped = bool(self.stop_requested_reason)
            self.config.status = "stopped_complete" if stopped else "complete"
            self.config.summary_path = str(summary)
            self.config.completed_at = datetime.now().isoformat(timespec="seconds")
            self.config.error = ""
            self.config.save()
            self._remember_run(self.config)
            self._set_badge("WRITING REPORT")
            self.load_results(summary)
            self.tabs.setCurrentIndex(self.TAB_RESULTS)
        elif self.config.status not in ("stopped", "memory_stopped"):
            self.config.status = "failed"
            self.config.error = runner_failure_reason(Path(self.config.run_root))
            self.config.save()
            self._remember_run(self.config)
            self._set_badge("FAILED")
            self._freeze_run_progress()
            self.phase_label.setText("Failed")
            self.run_log.append(self.config.error)
        self.start_button.setEnabled(True)

    def _open_existing_run(self):
        folder = QFileDialog.getExistingDirectory(self, "Open optimizer run folder")
        if not folder:
            return
        self._open_run_root(Path(folder))

    def _open_run_root(self, root: Path):
        try:
            config = RunConfig.load(root)
        except Exception as exc:
            QMessageBox.warning(self, "Not a GUI run", str(exc))
            return
        self.config = config
        self.active_mode = config.ui_workflow
        if self.active_mode == "phase":
            self.phase_data_edit.setText(config.data_root)
            self.phase_baseline_edit.setText(config.baseline)
            self.phase_target_edit.setText(config.target)
        elif self.active_mode == "retarget":
            self.retarget_data_edit.setText(config.data_root)
            self.retarget_baseline_edit.setText(config.baseline)
            self.retarget_target_edit.setText(config.target)
        else:
            self.data_edit.setText(config.data_root)
            self.baseline_edit.setText(config.baseline)
            self.target_edit.setText(config.target)
        active_pid = active_run_pid(root)
        if active_pid:
            self.process_pid = active_pid
            self.process_finished_handled = False
            self.runner_log_offset = 0
            self.started_monotonic = time.monotonic()
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.open_run_button.setEnabled(True)
            self._set_badge("RUNNING IN BACKGROUND")
            self.run_log.setPlainText(
                f"Reattached to background run:\n{root}\nProcess: {active_pid}"
            )
            self.poll_timer.start(1000)
            self._set_tab_enabled(self.TAB_RUN, True)
            self.tabs.setCurrentIndex(self.TAB_RUN)
            return
        summary = locate_summary(root)
        legacy_complete = (
            summary is not None
            and config.status in ("complete", "stopped_complete")
            and not (root / ".runner_failed").exists()
        )
        if summary and (runner_completed_successfully(root) or legacy_complete):
            if config.status in ("running", "running_detached", "resuming"):
                config.status = "complete"
                config.summary_path = str(summary)
                config.completed_at = config.completed_at or datetime.now().isoformat(
                    timespec="seconds",
                )
                config.save()
                self._remember_run(config)
            release_run_claim(root)
            self.load_results(summary)
            self._set_tab_enabled(self.TAB_RESULTS, True)
            self.tabs.setTabToolTip(self.TAB_RESULTS, "Review and export completed candidates.")
            self.tabs.setCurrentIndex(self.TAB_RESULTS)
        else:
            failure = runner_failure_reason(root) if (root / ".runner_failed").exists() else ""
            if failure:
                config.status = "failed"
                config.error = failure
                config.save()
                self._remember_run(config)
                self._set_badge("FAILED - RESUMABLE")
                self.run_log.setPlainText(failure)
            prompt = (
                f"The previous run failed before producing a verified result.\n\n{failure}\n\n"
                "Resume it from the intact checkpoints?"
                if failure else
                "No merged result exists. Resume this run from its checkpoints?"
            )
            reply = QMessageBox.question(self, "Resume run", prompt)
            if reply == QMessageBox.Yes:
                self.start_run(root)

    def load_results(self, summary_path: Path):
        self._set_tab_enabled(self.TAB_RESULTS, True)
        self.tabs.setTabToolTip(self.TAB_RESULTS, "Review and export completed candidates.")
        self.summary_path = summary_path
        self.summary = load_summary(summary_path)
        self.verify_run_edit.setText(str(summary_path.parent))
        self.repeat_run_button.setEnabled(True)
        self.record_decision_button.setEnabled(True)
        rows = candidate_files(self.summary, summary_path)
        self.result_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            role = QTableWidgetItem(str(row["role"]))
            role.setData(Qt.UserRole, row.get("path", ""))
            role.setData(Qt.UserRole + 1, bool(row.get("exportable")))
            role.setData(Qt.UserRole + 2, bool(row.get("is_baseline")))
            role.setData(Qt.UserRole + 3, str(row.get("comparison_stage") or "final"))
            objective = "-" if row["objective"] is None else f"{float(row['objective']):.6f}"
            self.result_table.setItem(index, 0, role)
            self.result_table.setItem(index, 1, QTableWidgetItem(objective))
            self.result_table.setItem(index, 2, QTableWidgetItem(str(row["file"])))
        best = self.summary.get("best") or {}
        baseline = self.summary.get("baseline") or {}
        best_components = best.get("components") or {}
        best_components = dict(best_components)
        if isinstance(best.get("objective"), (int, float)):
            best_components.setdefault("objective", best["objective"])
        verdict = improvement_verdict(baseline, best_components)
        self.result_heading.setText(verdict["heading"])
        self.improvement_banner.setText(verdict["detail"])
        self.improvement_banner.setStyleSheet(
            f"background:{theme.ACCENT_SOFT_BG};border-left:4px solid {theme.ACCENT_LINE};padding:10px;"
            f"color:{theme.ACCENT_SOFT_TEXT};font-weight:650;"
            if verdict["meaningful"] else
            f"background:{theme.WARN_SOFT_BG};border-left:4px solid {theme.WARN};padding:10px;"
            f"color:{theme.WARN_TEXT};font-weight:650;"
        )
        rehabilitation = self.summary.get("rehabilitation") or {}
        repeatability_tie = (
            rehabilitation.get("verdict") == "no_meaningful_improvement"
        )
        for index, card in enumerate(metric_card_data(baseline, best_components)):
            delta = card["delta_db"]
            percent = card["percent"]
            if delta is None or percent is None:
                value, detail = "Not available", "Older summary has no comparable metric"
            elif repeatability_tie:
                value = "No meaningful change"
                detail = f"{abs(delta):.2f} dB modelled difference; within repeatability"
            else:
                value = f"{abs(percent):.0f}% {'better' if delta >= 0 else 'worse'}"
                detail = f"{abs(delta):.2f} dB {'less' if delta >= 0 else 'more'} error"
            self.result_metric_values[index].setText(value)
            self.result_metric_values[index].setStyleSheet({
                "good": f"color:{theme.ACCENT_LINE};",
                "warn": f"color:{theme.DANGER};",
                "neutral": f"color:{theme.TEXT_MUTED};",
            }[card["state"]])
            self.result_metric_details[index].setText(detail)

        operations = list(rehabilitation.get("accepted_operations") or [])
        if not operations:
            for group, bands in (best.get("added_filters") or {}).items():
                for band in bands or []:
                    if len(band) >= 3:
                        operations.append({
                            "operation": "append",
                            "group": str(group),
                            "old": None,
                            "new": {
                                "frequency_hz": float(band[0]),
                                "q": float(band[1]),
                                "gain_db": float(band[2]),
                            },
                            "reason": "Added by the final guided search.",
                        })
        self.result_operations = operations
        self.result_plots = load_response_comparisons(summary_path)
        self._show_result_operations("final")

        warning_tokens = list(self.summary.get("warnings") or [])
        if best.get("left_alone"):
            warning_tokens.append(best["left_alone"])
        warning_rows = []
        for token in warning_tokens:
            info = warning_info(token)
            warning_rows.append(
                '<p style="color:%s"><b>%s</b><br>%s</p>'
                % (
                    theme.severity_colour(info["severity"]), html.escape(info["severity"].upper()),
                    html.escape(info["text"]),
                )
            )
        self.results_warnings.setHtml(
            "".join(warning_rows)
            or "<p>No measurement or candidate warnings were reported.</p>"
        )
        checks = list(self.summary.get("remeasure") or [])
        if not checks:
            checks = [
                "A/B the candidate against the baseline at matched listening level.",
                "Confirm vocal image, tonal balance and bass integration from the listening position.",
                "Keep the baseline if the audible change is negligible or less natural.",
            ]
        self.results_remeasure.setHtml(
            "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in checks) + "</ul>"
        )
        self.results_notice.setText(
            "The candidate is still a prediction. Keep the current tune available until the "
            "candidate has passed listening and any requested re-measurement checks."
        )
        mode = str((self.summary.get("search") or {}).get("mode") or "peq")
        plot = self.result_plots.get("final") or load_response_plot(summary_path)
        frequencies = plot.get("frequency_hz") or []
        if mode != "phase" and frequencies:
            self.results_chart.set_series(
                response_chart_series(plot),
                markers=[
                    frequency
                    for _group, frequency, _q, _gain in self.result_filters
                ],
                fallback="Response graph data was unavailable.",
            )
            self.results_chart_note.setText(
                "The 0 dB line is the target. Before and candidate share one fixed anchor; "
                "per-driver toggles show predicted change. Orange markers are added-filter centres. "
                "Hover for values or click to enlarge."
            )
        elif mode == "phase":
            self.results_chart.clear_chart(
                "Phase runs are summarized by crossover confidence in the PDF report."
            )
            self.results_chart_note.setText(
                "PEQ and Retarget runs show tonal before/after curves here."
            )
        else:
            self.results_chart.clear_chart(
                "This older run does not contain full response-plot data."
            )
            self.results_chart_note.setText(
                "New PEQ and Retarget runs automatically save the required plot data."
            )
        self.open_results_button.setEnabled(True)
        self._start_report_generation(summary_path)
        if rows:
            self.result_table.selectRow(1 if len(rows) > 1 else 0)

    def _start_report_generation(self, summary_path: Path):
        if self.report_task:
            return
        self.report_path = None
        self.open_report_button.setEnabled(False)
        self.phase_label.setText("Writing report")
        self._set_run_progress(990)
        self._set_badge("WRITING REPORT")
        self._show_busy("Writing tuning report")
        task = BackgroundTask(lambda _cancelled: generate_tuning_report(summary_path))
        self.report_task = task
        task.signals.result.connect(self._report_generated)
        task.signals.error.connect(self._report_error)
        task.signals.finished.connect(self._report_finished)
        self.thread_pool.start(task)

    def _report_generated(self, path: Path):
        self.report_path = Path(path)
        if self.config:
            run_root = Path(self.config.run_root)
            for name in ("preparing", "searching", "merging", "verifying", "reporting"):
                (run_root / f".phase_{name}").unlink(missing_ok=True)
            (run_root / ".phase_complete").touch()
        self.open_report_button.setEnabled(self.report_path.exists())
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.phase_label.setText("Complete")
        self._set_badge(
            "STOPPED - RESULTS SAVED" if self.stop_requested_reason else "COMPLETE"
        )

    def _report_error(self, error: str):
        self.report_path = None
        if self.config:
            run_root = Path(self.config.run_root)
            for name in ("preparing", "searching", "merging", "verifying", "reporting"):
                (run_root / f".phase_{name}").unlink(missing_ok=True)
            (run_root / ".phase_complete").touch()
        self.open_report_button.setEnabled(False)
        self.results_notice.setText(
            self.results_notice.text() + f"\nPDF report could not be generated: {error}"
        )
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.phase_label.setText("Complete")
        self._set_badge("COMPLETE - REPORT FAILED")

    def _report_finished(self):
        self.report_task = None
        self._hide_busy()

    @staticmethod
    def _format_operation_band(value):
        if not isinstance(value, dict):
            return "-"
        return (
            f"{float(value.get('frequency_hz', 0.0)):.1f} Hz | "
            f"Q {float(value.get('q', 0.0)):.2f} | "
            f"{float(value.get('gain_db', 0.0)):+.2f} dB"
        )

    def _show_result_operations(self, stage: str):
        all_operations = list(getattr(self, "result_operations", []))
        if stage == "supplied":
            operations = []
        elif stage == "rehabilitated":
            operations = [
                row for row in all_operations
                if str(row.get("operation")) != "append"
            ]
        else:
            operations = all_operations
        self.filter_table.setRowCount(len(operations))
        self.result_filters = []
        for row_index, operation in enumerate(operations):
            action = str(operation.get("operation") or "change").title()
            role = str(
                operation.get("channel_role")
                or GROUP_LABELS.get(
                    str(operation.get("group") or ""),
                    str(operation.get("group") or "DSP"),
                )
            )
            slot = (
                str(int(operation["slot"]))
                if operation.get("slot") is not None else "-"
            )
            old = self._format_operation_band(operation.get("old"))
            new = self._format_operation_band(operation.get("new"))
            reason = str(operation.get("reason") or "")
            for column, value in enumerate((action, role, slot, old, new, reason)):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.filter_table.setItem(row_index, column, item)
            band = operation.get("new")
            if isinstance(band, dict):
                self.result_filters.append((
                    str(operation.get("group") or role),
                    float(band.get("frequency_hz", 0.0)),
                    float(band.get("q", 0.0)),
                    float(band.get("gain_db", 0.0)),
                ))
        self.displayed_result_operations = operations
        self.filter_table.resizeRowsToContents()
        self.copy_filters_button.setEnabled(bool(operations))
    def _result_selected(self):
        rows = self.result_table.selectionModel().selectedRows()
        item = self.result_table.item(rows[0].row(), 0) if rows else None
        self.export_button.setEnabled(bool(item and item.data(Qt.UserRole + 1)))
        stage = str(item.data(Qt.UserRole + 3) or "final") if item else "final"
        self._show_result_operations(stage)
        plot = getattr(self, "result_plots", {}).get(stage)
        if (
            plot
            and str((self.summary.get("search") or {}).get("mode") or "peq")
            != "phase"
        ):
            self.results_chart.set_series(
                response_chart_series(plot),
                markers=[
                    frequency
                    for _group, frequency, _q, _gain in self.result_filters
                ],
                fallback="Response graph data was unavailable.",
            )
        if item and item.data(Qt.UserRole + 2):
            self.results_notice.setText(
                "Supplied tune selected. It remains the safe fallback and requires no export."
            )
        elif item and stage == "rehabilitated":
            self.results_notice.setText(
                "Existing tune improved selected. This contains only supported edits to "
                "existing PEQ bands; the Final row may add residual corrections."
            )
        elif item:
            self.results_notice.setText(
                "Final candidate selected. Export it as a new AFPX, keep the supplied tune "
                "available, then complete the listening and re-measurement checks above."
            )

    def _selected_candidate(self) -> Path | None:
        rows = self.result_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.result_table.item(rows[0].row(), 0)
        path = str(item.data(Qt.UserRole) or "") if item else ""
        return Path(path) if path and Path(path).is_file() else None

    def _copy_filters(self):
        operations = list(getattr(self, "displayed_result_operations", []))
        if not operations:
            return
        lines = ["AudioFischer Optimizer - DSP changes"]
        for operation in operations:
            role = str(
                operation.get("channel_role")
                or GROUP_LABELS.get(
                    str(operation.get("group") or ""),
                    str(operation.get("group") or "DSP"),
                )
            )
            slot = (
                f" slot {int(operation['slot'])}"
                if operation.get("slot") is not None else ""
            )
            lines.append(
                f"{str(operation.get('operation') or 'change').upper()}: "
                f"{role}{slot}"
            )
            lines.append(
                f"  {self._format_operation_band(operation.get('old'))} -> "
                f"{self._format_operation_band(operation.get('new'))}"
            )
            if operation.get("reason"):
                lines.append(f"  Why: {operation['reason']}")
        best = self.summary.get("best") or {}
        trims = best.get("output_volume_changes_db") or {}
        if trims:
            lines.extend(["", "Protective output trims"])
            lines.extend(
                f"  {channel}: {float(value):+.2f} dB"
                for channel, value in trims.items()
            )
        QApplication.clipboard().setText("\n".join(lines))
        self.copy_filters_button.setText("Changes Copied")
        QTimer.singleShot(
            1600, lambda: self.copy_filters_button.setText("Copy Filters as Text"),
        )

    def _export_selected(self):
        source = self._selected_candidate()
        if not source:
            return
        selected_rows = self.result_table.selectionModel().selectedRows()
        role_item = self.result_table.item(selected_rows[0].row(), 0)
        role = role_item.text() if role_item else source.stem
        folder = QFileDialog.getExistingDirectory(self, "Export candidate to")
        if folder:
            started_at = self.config.started_at if self.config else ""
            filename = default_export_name(source, role, started_at)
            target = Path(folder) / filename
            overwrite = False
            if target.exists():
                reply = QMessageBox.question(
                    self, "Replace existing export?",
                    f"The export already exists:\n{target}\n\nReplace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
                overwrite = True
            target = export_candidate(
                source, Path(folder), filename=filename, overwrite=overwrite,
            )
            message = QMessageBox(self)
            message.setWindowTitle("Exported")
            message.setText(f"Candidate exported to:\n{target}")
            open_folder = message.addButton("Open Containing Folder", QMessageBox.ActionRole)
            message.addButton(QMessageBox.Ok)
            message.exec()
            if message.clickedButton() is open_folder:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    def _open_run_folder(self):
        if self.config:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.config.run_root))

    def _loaded_run_root(self) -> Path | None:
        if not self.summary_path:
            return None
        folder = self.summary_path.parent
        return folder.parent if folder.name == "_merged_top" else folder

    def _record_listening_decision(self):
        root = self._loaded_run_root()
        if root is None:
            return
        row = self.result_table.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "Choose a candidate",
                "Select the tune you actually loaded before recording the decision.",
            )
            return
        role_item = self.result_table.item(row, 0)
        file_item = self.result_table.item(row, 2)
        candidate = file_item.text() if file_item else (
            role_item.text() if role_item else "Unknown candidate"
        )
        verdict, accepted = QInputDialog.getItem(
            self, "Listening decision", "What was the result?",
            [
                "Kept - clearly better",
                "Kept - small improvement",
                "Rejected - tonal balance",
                "Rejected - imaging",
                "Rejected - other",
                "Not yet evaluated",
            ],
            0, False,
        )
        if not accepted:
            return
        notes, accepted = QInputDialog.getMultiLineText(
            self, "Listening notes",
            "Optional notes (music, image, tonality, measurement conditions):",
        )
        if not accepted:
            return
        path = record_run_decision(root, candidate, verdict, notes)
        QMessageBox.information(
            self, "Decision saved",
            f"The listening decision was added to:\n{path}",
        )

    def _repeat_loaded_run(self):
        root = self._loaded_run_root()
        if root is None or not (root / "gui_job.json").is_file():
            QMessageBox.warning(
                self, "Run configuration unavailable",
                "This result does not have a reusable gui_job.json.",
            )
            return
        if self._run_is_active():
            QMessageBox.information(
                self, "Optimizer already running",
                "Stop or finish the active run before repeating another one.",
            )
            return
        config = RunConfig.load(root)
        new_root = timestamped_run_root(root.parent)
        config.run_root = str(new_root)
        config.status = "ready"
        config.summary_path = ""
        config.error = ""
        config.started_at = ""
        config.completed_at = ""
        config.save()
        self.start_run(new_root)

    def _open_results_folder(self):
        if self.summary_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.summary_path.parent)))

    def _open_report(self):
        if self.report_path and self.report_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path)))

    def closeEvent(self, event):
        if self._run_is_active():
            message = QMessageBox(self)
            message.setWindowTitle("Optimizer is running")
            message.setText(
                "The run can continue safely without this window.\n\n"
                "You can reopen it later from Recent Runs."
            )
            keep_running = message.addButton(
                "Keep Running and Close", QMessageBox.AcceptRole,
            )
            stop_and_close = message.addButton(
                "Stop Safely and Close", QMessageBox.DestructiveRole,
            )
            cancel = message.addButton(QMessageBox.Cancel)
            message.setDefaultButton(keep_running)
            message.exec()
            if message.clickedButton() is cancel:
                event.ignore()
                return
            if message.clickedButton() is stop_and_close:
                self.close_after_stop = True
                self.cancel_run()
                event.ignore()
                return
            if self.config:
                self.config.status = "running_detached"
                self.config.save()
                self._remember_run(self.config)
            if message.clickedButton() is keep_running:
                event.accept()
                return
        event.accept()


def run_gui() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AudioFischer Optimizer")
    app.setFont(QFont("Segoe UI", 9))
    app.setStyle("Fusion")
    theme.apply_palette(app)
    app_icon = theme.make_app_icon()
    app.setWindowIcon(app_icon)
    window = OptimizerWindow()
    window.setWindowIcon(app_icon)
    window.show()
    return app.exec()
