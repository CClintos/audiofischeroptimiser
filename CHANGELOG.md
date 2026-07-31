# Changelog

Notable changes to the GUI and optimizer, newest first. This is a plain
human/AI-readable log, not auto-generated; keep entries short and specific
enough that Codex or Claude can pick up context without rereading the diff.

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
