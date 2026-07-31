# Changelog

Notable changes to the GUI and optimizer, newest first. This is a plain
human/AI-readable log, not auto-generated; keep entries short and specific
enough that Codex or Claude can pick up context without rereading the diff.

## 2026-07-31 - Native-chrome polish, status badge, app icon

Second pass on top of the dark theme below. UI/styling only.

- Forced `QApplication.setStyle("Fusion")` plus a matching dark `QPalette`
  (`theme.apply_palette()`) in `run_gui()`. The native "windowsvista" style
  mostly ignores QSS/palette for sliders, checkboxes, and combo/spin arrows -
  Fusion is what actually makes the QSS below take effect, and it also makes
  Qt-drawn dialogs (`QMessageBox`, `QInputDialog`) pick up the dark palette
  instead of popping up light. Native OS dialogs (`QFileDialog`'s file/folder
  picker) are unaffected by design - that's the real Windows picker, not Qt's.
- Re-themed the actual widget chrome in `theme.build_stylesheet()`: `QSlider`
  groove/handle, `QCheckBox` indicator (filled square, no image asset needed),
  `QComboBox` drop-down arrow, `QSpinBox`/`QDoubleSpinBox` up/down buttons,
  `QMenu`. All done with QSS border-triangle tricks / solid fills - no new
  image or font dependency.
- Added `theme.badge_style()`: keyword-matches the run_badge's own text
  (`RUNNING`, `FAILED`, `VALIDATED`, `NEEDS ...`, etc.) to a good/warn/danger/
  info colour, via a new `self._set_badge()` wrapper in `window.py` that
  replaced all 17 raw `self.run_badge.setText(...)` call sites. One status
  string is the only source of truth; no separate state enum to keep in sync.
- Added `theme.make_app_icon()`: a small EQ-bars monogram (drawn with
  `QPainter`, no asset file), wired into `setWindowIcon()` on both the
  `QApplication` and the main window. Replaces the default Python/Qt icon in
  the title bar and taskbar. Not yet wired into `AudioFischerOptimizer.spec`
  for the packaged `.exe`'s file icon - that's a separate follow-up if wanted.
- Added `setAlternatingRowColors(True)` to all four `QTableWidget`s
  (`result_table`, `filter_table`, `recent_runs_table`, `convergence_table`).
- Tried `QGraphicsDropShadowEffect` on the card frames for elevation and
  **reverted it** - applying a graphics effect to a QFrame with several plain
  child QLabels forces per-child compositing and left visible rectangular
  seams around each line of text (very noticeable in the Results tab's metric
  cards). Do not reintroduce `QGraphicsDropShadowEffect` on any card/frame that
  contains multiple plain child widgets without checking for this artifact
  first. Replaced it with a one-line `border-top` highlight
  (`theme.CARD_HIGHLIGHT`) on `QFrame#card`/`#cardAccent`/`#metricCard` - a
  much cheaper elevation cue with no compositing involved.
- Verified by re-rendering the affected tabs via `QWidget.grab()` before and
  after the shadow revert, and running the full test suite (97/97 pass).

## 2026-07-31 - Dark pro-audio GUI theme

UI/styling only. No changes to `objective_module/`, `_optimizer.py`, or
`_optimizer_stream.py` in this pass.

- Added `optimizer_gui/theme.py`: a single source of truth for every color used
  by the GUI (palette tokens, contrast-checked against WCAG AA), a hand-drawn
  flat icon set (folder, arrow, check, play, stop, export, file, report) that
  replaces every `QStyle.SP_*` system dialog icon, and a step-badge icon
  generator for the numbered workflow tabs.
- `optimizer_gui/window.py`: switched to the dark theme via
  `theme.build_stylesheet()`; rebuilt the Home tab into distinct cards (with an
  accent-striped "start here" card for PEQ/RTA); re-themed the live
  `_ChartCanvas` (Results/Verify/Retarget charts) to the dark palette; added
  circular step-number badges to the PEQ/Phase/Run/Results tabs, dimmed when
  disabled, filled when reachable; added the HiDPI
  `HighDpiScaleFactorRoundingPolicy.PassThrough` policy in `run_gui()`.
- `optimizer_gui/reporting.py`: re-themed `response_chart_series()` (the
  GUI-only chart series builder) to use the same tokens. Deliberately did NOT
  touch `_line_chart`, `_paired_bar_chart`, or `build_report_html` - the
  `SQ_Tuning_Report.pdf` output stays light/white for printing, on purpose.
  `warning_text.SEVERITY_COLOURS` also stays untouched for the same reason
  (it feeds the PDF); the GUI now reads warning color via the new
  `theme.severity_colour()` instead.
- Fixed two bugs found while touching this code, unrelated to theming:
  - The Verify tab's predicted-vs-achieved chart built series dicts with
    `frequency_hz`/`db` keys, but `_ChartCanvas._clean_series()` reads `x`/`y` -
    the chart was silently rendering empty. Fixed the key names.
  - The chart's "Frequency (Hz)" axis caption overlapped the last frequency
    tick label (e.g. "10k"/"20k"). Moved it to the top-right corner, mirroring
    the existing "dB" label convention at top-left.
- Verified by rendering all 8 tabs via `QWidget.grab()` (not just code review),
  running the full test suite (97/97 pass), and generating a real PDF report
  to confirm it is still unaffected.

If restyling further: add new colors to `optimizer_gui/theme.py` only, never as
a bare hex literal in `window.py`. Anything touching `SQ_Tuning_Report.pdf`
colors belongs in `optimizer_gui/reporting.py`'s `_line_chart` /
`_paired_bar_chart` / `warning_text.SEVERITY_COLOURS`, and should stay
light/print-appropriate regardless of the app's own theme.
