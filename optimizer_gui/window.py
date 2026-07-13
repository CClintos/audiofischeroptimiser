from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QStyle, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from .backend import (
    APP_NAME, RunConfig, candidate_files, collect_progress, default_target,
    discover_baseline, export_candidate, load_summary, locate_summary,
    powershell_command, process_tree_memory, timestamped_run_root, validate_config,
)
from .reporting import generate_tuning_report


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


class OptimizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1120, 760)
        self.setMinimumSize(920, 650)
        self.process: QProcess | None = None
        self.config: RunConfig | None = None
        self.summary_path: Path | None = None
        self.report_path: Path | None = None
        self.summary: dict = {}
        self.started_monotonic = 0.0
        self.memory_limit_hits = 0
        self.stop_requested_reason = ""
        self.active_mode = "peq"
        self._build_ui()
        self._apply_style()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_run)
        self.data_edit.textChanged.connect(self._input_changed)
        self.baseline_edit.textChanged.connect(self._input_changed)
        self.target_edit.textChanged.connect(self._input_changed)
        self.phase_data_edit.textChanged.connect(self._input_changed)
        self.phase_baseline_edit.textChanged.connect(self._input_changed)
        self.phase_target_edit.textChanged.connect(self._input_changed)
        self._set_defaults()

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 20)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("AudioFischer Optimizer")
        title.setObjectName("title")
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

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_inputs_tab(), "1  PEQ / RTA")
        self.tabs.addTab(self._build_phase_tab(), "2  Sweeps / Phase")
        self.tabs.addTab(self._build_run_tab(), "3  Run")
        self.tabs.addTab(self._build_results_tab(), "4  Results")
        self.tabs.addTab(self._build_about_tab(), "5  About")
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _path_row(self, mode: str, browse_slot):
        edit = DropLineEdit(mode)
        edit.setPlaceholderText("Drop a folder here" if mode == "folder" else "Drop a file here")
        button = QToolButton()
        button.setIcon(self.style().standardIcon(
            QStyle.SP_DirOpenIcon if mode == "folder" else QStyle.SP_FileIcon
        ))
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
        self.validate_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.validate_button.clicked.connect(self.validate_inputs)
        self.resume_button = QPushButton("Open Existing Run")
        self.resume_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.resume_button.clicked.connect(self._open_existing_run)
        actions.addWidget(self.validate_button)
        actions.addWidget(self.resume_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setPlaceholderText("Validation results appear here.")
        layout.addWidget(self.validation_text, 1)
        return page

    def _build_phase_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(14)

        intro = QLabel(
            "Second stage: load the PEQ result into the DSP, take fresh phase-valid sweeps, "
            "then use that PEQ result as the baseline here. Existing PEQ is preserved."
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
        form.addRow("PEQ result AFPX", base_row)
        form.addRow("Target curve", target_row)
        layout.addLayout(form)

        action_line = QHBoxLayout()
        self.validate_phase_button = QPushButton("Validate Sweeps / Prepare Phase")
        self.validate_phase_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.validate_phase_button.clicked.connect(self.validate_phase_inputs)
        self.phase_use_peq_button = QPushButton("Use Latest PEQ Result")
        self.phase_use_peq_button.clicked.connect(self._use_latest_peq_result)
        action_line.addWidget(self.validate_phase_button)
        action_line.addWidget(self.phase_use_peq_button)
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
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(14)

        grid = QGridLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Quick check - 2 minutes", 120)
        self.preset_combo.addItem("Normal - 20 minutes", 1200)
        self.preset_combo.addItem("Thorough - 40 minutes", 2400)
        self.preset_combo.addItem("Custom", 0)
        self.preset_combo.setCurrentIndex(1)
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
        cpu_box = QHBoxLayout()
        cpu_box.addWidget(self.cpu_slider, 1)
        cpu_box.addWidget(self.cpu_label)

        self.ram_slider = QSlider(Qt.Horizontal)
        self.ram_slider.setRange(20, 70)
        self.ram_slider.setValue(50)
        self.ram_label = QLabel("50% of RAM")
        self.ram_slider.valueChanged.connect(lambda value: self.ram_label.setText(f"{value}% of RAM"))
        ram_box = QHBoxLayout()
        ram_box.addWidget(self.ram_slider, 1)
        ram_box.addWidget(self.ram_label)

        self.workflow_value = QLabel("PEQ / RTA - Beam search")
        self.workflow_value.setObjectName("metricValue")

        grid.addWidget(QLabel("Run length"), 0, 0)
        grid.addWidget(self.preset_combo, 0, 1)
        grid.addWidget(self.seconds_spin, 0, 2)
        grid.addWidget(QLabel("CPU target"), 1, 0)
        grid.addLayout(cpu_box, 1, 1, 1, 2)
        grid.addWidget(QLabel("Optimizer RAM limit"), 2, 0)
        grid.addLayout(ram_box, 2, 1, 1, 2)
        grid.addWidget(QLabel("Workflow"), 3, 0)
        grid.addWidget(self.workflow_value, 3, 1, 1, 2)
        layout.addLayout(grid)

        option_line = QHBoxLayout()
        self.voicing_check = QCheckBox("Create voicing audition files")
        self.sub_blend_check = QCheckBox("Report sub level recommendation")
        self.sub_blend_check.toggled.connect(lambda checked: self.headroom_spin.setEnabled(checked))
        self.headroom_spin = QDoubleSpinBox()
        self.headroom_spin.setRange(0.0, 12.0)
        self.headroom_spin.setValue(3.0)
        self.headroom_spin.setSuffix(" dB headroom")
        self.headroom_spin.setEnabled(False)
        option_line.addWidget(self.voicing_check)
        option_line.addSpacing(16)
        option_line.addWidget(self.sub_blend_check)
        option_line.addWidget(self.headroom_spin)
        option_line.addStretch()
        layout.addLayout(option_line)

        self.phase_warning = QLabel("PEQ stage uses Beam and cannot write phase changes. Phase stage preserves PEQ and writes only changes that pass the evidence gates. The baseline is never overwritten.")
        self.phase_warning.setObjectName("warning")
        self.phase_warning.setWordWrap(True)
        layout.addWidget(self.phase_warning)

        action_line = QHBoxLayout()
        self.start_button = QPushButton("Start Optimizer")
        self.start_button.setObjectName("primary")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self.start_run)
        self.start_button.setEnabled(False)
        self.cancel_button = QPushButton("Stop Safely")
        self.cancel_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setEnabled(False)
        self.open_run_button = QPushButton("Open Run Folder")
        self.open_run_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.open_run_button.clicked.connect(self._open_run_folder)
        self.open_run_button.setEnabled(False)
        action_line.addWidget(self.start_button)
        action_line.addWidget(self.cancel_button)
        action_line.addWidget(self.open_run_button)
        action_line.addStretch()
        layout.addLayout(action_line)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        status_grid = QGridLayout()
        self.elapsed_value = QLabel("00:00")
        self.worker_value = QLabel("0")
        self.trial_value = QLabel("0")
        self.best_value = QLabel("-")
        self.memory_value = QLabel("-")
        for col, (label, widget) in enumerate((
            ("Elapsed", self.elapsed_value), ("Workers", self.worker_value),
            ("Candidates checked", self.trial_value), ("Best objective", self.best_value),
            ("Optimizer memory", self.memory_value),
        )):
            cell = QVBoxLayout()
            name = QLabel(label)
            name.setObjectName("metricName")
            widget.setObjectName("metricValue")
            cell.addWidget(name)
            cell.addWidget(widget)
            status_grid.addLayout(cell, 0, col)
        layout.addLayout(status_grid)

        self.run_log = QTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("Run status appears here. Full worker logs remain in the run folder.")
        layout.addWidget(self.run_log, 1)
        return page

    def _build_results_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)
        self.result_heading = QLabel("No completed run loaded")
        self.result_heading.setObjectName("sectionTitle")
        layout.addWidget(self.result_heading)

        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Candidate", "Objective", "File"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_table.itemSelectionChanged.connect(self._result_selected)
        layout.addWidget(self.result_table, 1)

        result_actions = QHBoxLayout()
        self.export_button = QPushButton("Export Selected AFPX")
        self.export_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.export_button.clicked.connect(self._export_selected)
        self.export_button.setEnabled(False)
        self.open_results_button = QPushButton("Open Results Folder")
        self.open_results_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.open_results_button.clicked.connect(self._open_results_folder)
        self.open_results_button.setEnabled(False)
        self.open_report_button = QPushButton("Open Tuning Report")
        self.open_report_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.open_report_button.clicked.connect(self._open_report)
        self.open_report_button.setEnabled(False)
        result_actions.addWidget(self.export_button)
        result_actions.addWidget(self.open_report_button)
        result_actions.addWidget(self.open_results_button)
        result_actions.addStretch()
        layout.addLayout(result_actions)

        self.result_details = QTextEdit()
        self.result_details.setReadOnly(True)
        self.result_details.setPlaceholderText("Named score components, phase actions and warnings appear here.")
        layout.addWidget(self.result_details, 1)
        return page

    def _build_about_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        about = QTextEdit()
        about.setReadOnly(True)
        about.setHtml("""
        <h1>AudioFischer Optimizer</h1>
        <p>A local, conservative tuning tool for Helix and Audiotec Fischer AFPX files. It reads REW measurements, predicts supported changes, writes new candidate tunes, and never overwrites the baseline.</p>
        <h2>Two-stage workflow</h2>
        <p><b>1. PEQ / RTA:</b> Uses fresh magnitude or moving-mic RTA measurements to improve tonal balance and L/R consistency. Delay, polarity, APF and crossovers remain untouched.</p>
        <p><b>2. Sweeps / Phase:</b> After the PEQ result is loaded, fresh phase-valid sweeps are used to test crossover polarity, bounded relative delay and residual all-pass correction. Existing PEQ remains unchanged.</p>
        <h2>How PEQ is judged</h2>
        <ul>
          <li>ERB-smoothed tonal error against the supplied target.</li>
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
          <li>Changing crossover frequency, slope, shelves or output levels automatically.</li>
          <li>Claiming that a predicted tune is verified before it is loaded and re-measured.</li>
        </ul>
        <h2>Objective</h2>
        <p>Lower is better. The displayed objective is a weighted decision score made from the named components above, not a single raw flatness number. Candidate reports show why a result won, what changed, what was left alone and what must be checked in-car.</p>
        """)
        layout.addWidget(about)
        return page

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f6f7; color: #202327; font-size: 13px; }
            QLabel#title { font-size: 25px; font-weight: 700; color: #14171a; }
            QLabel#subtitle { color: #626a73; }
            QLabel#badge { background: #e1e5e9; color: #394047; padding: 6px 11px; border-radius: 4px; font-weight: 700; }
            QLabel#warning { background: #fff4d9; border-left: 4px solid #d08a00; padding: 9px; color: #5e470f; }
            QLabel#metricName { color: #68717a; font-size: 11px; }
            QLabel#metricValue { color: #15191d; font-size: 18px; font-weight: 650; }
            QLabel#sectionTitle { font-size: 18px; font-weight: 650; }
            QTabWidget::pane { border: 1px solid #d9dde1; background: white; }
            QTabBar::tab { background: #e9ecef; border: 1px solid #d9dde1; padding: 9px 18px; }
            QTabBar::tab:selected { background: white; border-bottom-color: white; font-weight: 650; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
                background: white; border: 1px solid #cbd1d6; border-radius: 3px; padding: 6px;
            }
            QPushButton, QToolButton { background: #edf0f2; border: 1px solid #c5cbd0; border-radius: 4px; padding: 7px 12px; }
            QPushButton:hover, QToolButton:hover { background: #e2e7ea; }
            QPushButton#primary { background: #176b4d; color: white; border-color: #12563e; font-weight: 650; }
            QPushButton#primary:hover { background: #125b41; }
            QPushButton:disabled { color: #9ca3aa; background: #f0f1f2; }
            QProgressBar { border: 1px solid #cbd1d6; background: white; height: 16px; text-align: center; }
            QProgressBar::chunk { background: #23805f; }
            QHeaderView::section { background: #edf0f2; border: 0; border-bottom: 1px solid #cbd1d6; padding: 7px; font-weight: 650; }
        """)

    def _set_defaults(self):
        target = default_target()
        if target.exists():
            self.target_edit.setText(str(target))
            self.phase_target_edit.setText(str(target))

    def _input_changed(self):
        self.start_button.setEnabled(False)
        self.run_badge.setText("NEEDS VALIDATION")

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
        mode = mode or self.active_mode
        if mode == "phase":
            data_root = self.phase_data_edit.text().strip()
            baseline = self.phase_baseline_edit.text().strip()
            target = self.phase_target_edit.text().strip()
        else:
            data_root = self.data_edit.text().strip()
            baseline = self.baseline_edit.text().strip()
            target = self.target_edit.text().strip()
        return RunConfig(
            data_root=data_root, baseline=baseline, target=target,
            run_root=str(run_root or timestamped_run_root()), mode=mode,
            seconds=30 if mode == "phase" else self.seconds_spin.value(),
            cpu_percent=20 if mode == "phase" else self.cpu_slider.value(),
            ram_percent=self.ram_slider.value(), proposal="beam",
            phase_writes="auto" if mode == "phase" else "off",
            voicing_variants=("audition" if self.voicing_check.isChecked() else "off") if mode == "peq" else "off",
            sub_blend=("recommend" if self.sub_blend_check.isChecked() else "off") if mode == "peq" else "off",
            headroom_db=(self.headroom_spin.value() if self.sub_blend_check.isChecked() else None) if mode == "peq" else None,
            level_calibration="",
        )

    def validate_inputs(self):
        return self._validate_workflow("peq", self.validation_text)

    def validate_phase_inputs(self):
        return self._validate_workflow("phase", self.phase_validation_text)

    def _validate_workflow(self, mode: str, output: QTextEdit):
        config = self._current_config(mode=mode)
        result = validate_config(config)
        if result["compact"]:
            compact = result["compact"]
            lines = [
                f"Layout: {compact['detected_layout']}",
                f"Mode: {compact['safe_mode']}",
                f"Measurements found: {compact['measurement_count']}",
                f"Phase files: {compact['phase_file_count']}  |  Coherence: {compact['coherence_file_count']}  |  Impulses: {compact['impulse_file_count']}",
                f"Spatial positions: {', '.join(compact['spatial_positions']) or 'centre only'}",
            ]
            if compact["missing"]:
                lines.append("\nMissing:\n- " + "\n- ".join(compact["missing"]))
            if compact["warnings"]:
                lines.append("\nWarnings:\n- " + "\n- ".join(compact["warnings"]))
            preflight = result.get("preflight") or {}
            for row in preflight.get("pair_validation", []):
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
            if result["errors"]:
                lines.append("\nBlocked:\n- " + "\n- ".join(result["errors"]))
            if result["valid"]:
                lines.append("\nPASS: the optimizer can start with this input set.")
            output.setPlainText("\n".join(lines))
        else:
            output.setPlainText("\n".join(result["errors"]))
        self.start_button.setEnabled(bool(result["valid"]))
        self.run_badge.setText("VALIDATED" if result["valid"] else "INPUT BLOCKED")
        if result["valid"]:
            self.active_mode = mode
            phase_mode = mode == "phase"
            self.workflow_value.setText(
                "Sweeps / Phase - preserve PEQ, gated phase writes only"
                if phase_mode else "PEQ / RTA - Beam search, no phase writes"
            )
            self.preset_combo.setEnabled(not phase_mode)
            self.seconds_spin.setEnabled(not phase_mode)
            self.cpu_slider.setEnabled(not phase_mode)
            self.voicing_check.setEnabled(not phase_mode)
            self.sub_blend_check.setEnabled(not phase_mode)
            self.headroom_spin.setEnabled(not phase_mode and self.sub_blend_check.isChecked())
            self.tabs.setCurrentIndex(3)
        return result["valid"]

    def _preset_changed(self):
        seconds = self.preset_combo.currentData()
        if seconds:
            self.seconds_spin.setValue(seconds)

    def start_run(self, resume_root: Path | None = None):
        if self.process and self.process.state() != QProcess.NotRunning:
            return
        if resume_root is None:
            validator = self.validate_phase_inputs if self.active_mode == "phase" else self.validate_inputs
            if not validator():
                return
        if resume_root is not None:
            config = RunConfig.load(resume_root)
            config.status = "resuming"
            config.error = ""
        else:
            config = self._current_config()
        config.started_at = datetime.now().isoformat(timespec="seconds")
        config.status = "running"
        config.save()
        self.config = config
        self.started_monotonic = time.monotonic()
        self.memory_limit_hits = 0
        self.stop_requested_reason = ""
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(Path(config.run_root).parent))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        program, args = powershell_command(config)
        self.process.start(program, args)
        if not self.process.waitForStarted(5000):
            config.status = "failed"
            config.error = self.process.errorString()
            config.save()
            QMessageBox.critical(self, "Could not start", config.error)
            return
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.open_run_button.setEnabled(True)
        self.run_badge.setText("RUNNING")
        self.run_log.setPlainText(
            f"Run folder: {config.run_root}\nWorkers: {config.workers}\n"
            f"Memory stop limit: {config.ram_percent}% of physical RAM\n"
        )
        self.progress.setValue(0)
        self.poll_timer.start(1000)
        self.tabs.setCurrentIndex(2)

    def _read_process_output(self):
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace").strip()
        if text:
            self.run_log.append(text)

    def _poll_run(self):
        if not self.config or not self.process:
            return
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        self.elapsed_value.setText(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        self.progress.setValue(min(1000, round(1000 * elapsed / max(self.config.seconds, 1))))
        progress = collect_progress(Path(self.config.run_root))
        self.worker_value.setText(str(progress["workers_reporting"]))
        self.trial_value.setText(f"{progress['trials']:,}")
        objective = progress["best_objective"]
        self.best_value.setText("-" if objective is None else f"{objective:.5f}")
        pid = int(self.process.processId())
        rss, total = process_tree_memory(pid)
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

    def cancel_run(self, memory_stop: bool = False):
        if not self.process or self.process.state() == QProcess.NotRunning:
            return
        pid = int(self.process.processId())
        self.stop_requested_reason = "memory" if memory_stop else "user"
        stop_file = Path(self.config.run_root) / "stop_requested" if self.config else None
        if stop_file:
            stop_file.write_text(self.stop_requested_reason, encoding="ascii")
        self.run_badge.setText("STOPPING SAFELY")
        QApplication.processEvents()
        if not self.process.waitForFinished(20000):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, creationflags=creationflags)
            self.process.kill()
            self.process.waitForFinished(5000)
        if self.config:
            self.config.status = "memory_stopped" if memory_stop else "stopped"
            self.config.error = "Memory safety limit reached" if memory_stop else "Stopped by user"
            self.config.save()
        self.run_badge.setText("STOPPED - RESUMABLE")

    def _process_finished(self, exit_code: int, _status):
        self.poll_timer.stop()
        self.cancel_button.setEnabled(False)
        if not self.config:
            return
        summary = locate_summary(Path(self.config.run_root))
        if exit_code == 0 and summary:
            stopped = bool(self.stop_requested_reason)
            self.config.status = "stopped_complete" if stopped else "complete"
            self.config.summary_path = str(summary)
            self.config.completed_at = datetime.now().isoformat(timespec="seconds")
            self.config.error = ""
            self.config.save()
            self.progress.setValue(1000)
            self.run_badge.setText("STOPPED - RESULTS SAVED" if stopped else "COMPLETE")
            self.load_results(summary)
            self.tabs.setCurrentIndex(2)
        elif self.config.status not in ("stopped", "memory_stopped"):
            self.config.status = "failed"
            self.config.error = f"Optimizer exited with code {exit_code}"
            self.config.save()
            self.run_badge.setText("FAILED")
            self.run_log.append(self.config.error)
        self.start_button.setEnabled(True)

    def _open_existing_run(self):
        folder = QFileDialog.getExistingDirectory(self, "Open optimizer run folder")
        if not folder:
            return
        root = Path(folder)
        try:
            config = RunConfig.load(root)
        except Exception as exc:
            QMessageBox.warning(self, "Not a GUI run", str(exc))
            return
        self.config = config
        self.active_mode = getattr(config, "mode", "peq")
        if self.active_mode == "phase":
            self.phase_data_edit.setText(config.data_root)
            self.phase_baseline_edit.setText(config.baseline)
            self.phase_target_edit.setText(config.target)
        else:
            self.data_edit.setText(config.data_root)
            self.baseline_edit.setText(config.baseline)
            self.target_edit.setText(config.target)
        summary = locate_summary(root)
        if summary:
            self.load_results(summary)
            self.tabs.setCurrentIndex(3)
        else:
            reply = QMessageBox.question(self, "Resume run", "No merged result exists. Resume this run from its checkpoints?")
            if reply == QMessageBox.Yes:
                self.start_run(root)

    def load_results(self, summary_path: Path):
        self.summary_path = summary_path
        self.summary = load_summary(summary_path)
        rows = candidate_files(self.summary, summary_path)
        self.result_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            role = QTableWidgetItem(str(row["role"]))
            role.setData(Qt.UserRole, row["path"])
            objective = "-" if row["objective"] is None else f"{float(row['objective']):.6f}"
            self.result_table.setItem(index, 0, role)
            self.result_table.setItem(index, 1, QTableWidgetItem(objective))
            self.result_table.setItem(index, 2, QTableWidgetItem(str(row["file"])))
        best = self.summary.get("best") or {}
        self.result_heading.setText(
            f"Completed: {len(rows)} exportable candidates  |  Best objective: {best.get('objective', '-')}"
        )
        core = {
            "baseline": self.summary.get("baseline"),
            "best": best,
            "phase_actions": self.summary.get("phase_actions"),
            "sub_blend_recommendation": self.summary.get("sub_blend_recommendation"),
            "warnings": self.summary.get("warnings"),
            "remeasure": self.summary.get("remeasure"),
        }
        self.result_details.setPlainText(json.dumps(core, indent=2))
        self.open_results_button.setEnabled(True)
        try:
            self.report_path = generate_tuning_report(summary_path)
            self.open_report_button.setEnabled(self.report_path.exists())
        except Exception as exc:
            self.report_path = None
            self.open_report_button.setEnabled(False)
            self.result_details.append(f"\nPDF report could not be generated: {exc}")
        if rows:
            self.result_table.selectRow(0)

    def _result_selected(self):
        self.export_button.setEnabled(bool(self.result_table.selectedItems()))

    def _selected_candidate(self) -> Path | None:
        rows = self.result_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.result_table.item(rows[0].row(), 0)
        return Path(item.data(Qt.UserRole)) if item else None

    def _export_selected(self):
        source = self._selected_candidate()
        if not source:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export candidate to")
        if folder:
            target = export_candidate(source, Path(folder))
            QMessageBox.information(self, "Exported", f"Candidate exported to:\n{target}")

    def _open_run_folder(self):
        if self.config:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.config.run_root))

    def _open_results_folder(self):
        if self.summary_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.summary_path.parent)))

    def _open_report(self):
        if self.report_path and self.report_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path)))

    def closeEvent(self, event):
        if self.process and self.process.state() != QProcess.NotRunning:
            reply = QMessageBox.question(
                self, "Optimizer is running",
                "Stop the optimizer and preserve its checkpoints before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                self.cancel_run()
            else:
                QMessageBox.information(self, "Keep this window open", "The GUI owns the worker process tree, so it must stay open while the run continues.")
                event.ignore()
                return
        event.accept()


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("AudioFischer Optimizer")
    app.setFont(QFont("Segoe UI", 9))
    window = OptimizerWindow()
    window.show()
    return app.exec()
