# Changelog

Notable changes to the GUI and optimizer, newest first. This is a plain
human/AI-readable log, not auto-generated; keep entries short and specific
enough that Codex or Claude can pick up context without rereading the diff.

## 2026-08-01 - Fix two more bugs found preparing the real v9 re-run

Found while running the fixed optimizer against real measurements and a
real baseline tune for the first time, not in the original audit.

- `scripts/verify_written_tune.py` never added the repo root to `sys.path`
  before its deferred `import _optimizer`, unlike every other script in
  `scripts/` (compare `prepare_phase_cache.py`). Invoked the normal way
  (`python scripts/verify_written_tune.py ...` from the repo root),
  `sys.path[0]` is `scripts/`, not the repo root where `_optimizer.py`
  lives, so the import always raised `ModuleNotFoundError` - the
  verification step of every real run has been silently broken. Fixed with
  the same `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
  line the other scripts already use.
- DEFECT 6's persistence gate (see below) compared each extra session's raw
  System Sum directly against `target` without re-anchoring it to that
  session's own broadband level first. Two REW sessions captured on
  different days rarely share the exact same source volume or mic gain; a
  session that was simply captured a few dB louder or quieter overall (but
  otherwise flat) could look like it "confirms" a positive deviation
  everywhere purely from that broadband offset, or a real small deviation
  could have its sign flipped by an unrelated level difference. Fixed in
  `_optimizer_stream.py`'s `find_guided_candidates()`: each session is now
  anchored to `target` with its own `target_anchor_offset()` median (over
  the same wide bands `target` itself was anchored to the primary session
  with) before its deviation is compared - the persistence vote is now
  shape-only, the way it needs to be. New regression:
  `tests/test_p1_features.py::test_persistence_gate_anchors_each_session_before_comparing`.

## 2026-08-01 - Fix append-chaining crash in band resolution

Found by a smoke test against the real v9 baseline before the full re-run
(not in the original 6-defect audit) - `_optimizer.py` only, no GUI changes.

- `_resolve_group_bands()` (DEFECT 1's edit/append/remove resolver) could
  crash with `expected exactly one active PEQ ... found 0` whenever a
  single group's guided pool proposed several new bands within
  `EDIT_MATCH_TOLERANCE_OCT` of each other but of any real baseline band -
  a real case against v9 (three fr_low proposals at 448.0/451.4/452.5 Hz).
  The second and third proposals matched the FIRST proposal's own
  in-memory tuple via `_find_edit_target` and were resolved as "edits" of
  it, but `write_candidate`/`apply_band_actions` only materializes appends
  into the XML in one batch *after* every edit/remove for that channel has
  already run - so the "edit" tried to find a filter slot in the file that
  did not exist yet and raised.
- Fixed by tracking, per channel, which `band_sets[channel]` entries still
  have a real AFPX slot behind them (baseline, or already edited from
  baseline) versus a same-pass append with no slot yet
  (`from_baseline` in `_resolve_group_bands`); `_find_edit_target()` now
  takes an `eligible` mask and only matches the former. A later proposal
  chaining onto an *edited* baseline band still works (that slot is real);
  only chaining onto a fresh append is now excluded, since only that case
  has nothing in the file yet to edit.
- Verified against the exact failing real-world `groups` dict
  (`_optimizer._resolve_group_bands`/`write_candidate` both re-run
  directly, succeeded) plus two new regression tests in
  `tests/test_objective_hardening.py::BandEditAndRemovalTests`.

## 2026-08-01 - Cross-session persistence gate (DEFECT 6)

Optimizer only, no GUI changes. Continuation of the guardrail pass below -
this is the 6th and final defect from the same audit.

- A single MMM (Moving Mic Method) session can't tell a genuine deviation
  from run-to-run capture noise apart. `_optimizer.py` gained
  `load_persistence_sessions()`: a deliberately lightweight loader for extra
  REW session folders that only needs a `System Sum.txt` export each (no
  baseline `.afpx`, target, or full manifest) - light enough to cover even a
  sparse extra session that otherwise only captured two of eight files.
- `objective_module/_tunefit.py` gained `cross_session_persistence()`: votes
  a candidate frequency's deviation across every session that had coverage
  there. Eligible only when all of them agree in sign AND every one
  individually clears the local noise floor - the weakest session sets the
  bar, not the average, so two strong sessions can't outvote one marginal
  one. A session with no coverage at that frequency is excluded from the
  vote, never counted as disagreement.
- `_optimizer_stream.py`'s `find_guided_candidates()` gained an opt-in
  `persistence_sessions` parameter: when supplied, a tonal candidate must
  pass this check (using each session's own System-Sum-vs-target deviation)
  before it's proposed at all, not merely scored differently - candidate
  generation itself is gated, matching the "no just go in whatever priority"
  scope of this defect ("a deviation is only eligible if it holds
  sign/magnitude across all supplied sessions"). Wired to a new
  `--persistence-sessions` CLI flag on the worker script
  (`_optimizer_stream.py`). Omitted, this is a complete no-op - every
  existing single-session run behaves exactly as before.
- Surviving candidates carry `persistence_session_count` forward into the
  "problem census" (`worth_fixing` list) so the report can name the
  cross-session evidence per band, not just assert it.
- Deliberately scoped to the tonal/target-matching candidate path only (not
  the L/R balance path, which already has its own frequency-domain
  consistency check via `signed_offset_evidence`) - this covers the actual
  defect pattern the audit found (sub/mids/tweeter tonal corrections), and
  not wired into the GUI's worker orchestration
  (`optimizer_gui/backend.py`) yet - CLI-only for now, GUI wiring (session
  folder pickers, etc.) is a separate follow-up if wanted.

## 2026-08-01 - Optimizer guardrail hardening (DEFECTS 1-4b)

Fixes for six defects found by an independent second-opinion audit of two
optimizer runs. Optimizer/objective only, no GUI changes.

- **DEFECT 1 (search could only append, never edit/remove a filter)** -
  `_optimizer.py` gained `_resolve_group_bands()`/`groups_to_band_sets()`,
  which now match a proposed band against the existing baseline slot by
  frequency (within `EDIT_MATCH_TOLERANCE_OCT`) and classify it as an edit or
  removal (`G == REMOVE_BAND_GAIN`), not just an append. `write_candidate()`
  writes these via new `_make_v3.edit_band()`/`remove_band()`/
  `apply_band_actions()`, which find the one matching active `<Fil>` slot by
  F/Q/G and mutate it in place (edit) or flip it back to `T="1"` (remove)
  instead of always adding a new slot. `_optimizer_stream.candidate_peaks()`
  now also proposes edits/removals: `_group_existing_targets()` feeds each
  baseline band in as a forced candidate target, which gets first pick in the
  min-separation selection loop so it can't lose its slot to a nearby organic
  point with marginally higher smoothed strength.
- **DEFECT 2 (`worst_position_error` double-counted with 1 measured
  position)** - `objective_module/afpx_objective.py`'s `objective()` now
  zeroes `W['worst']`'s contribution when `spatial_position_count < 2`, since
  at one position that term is the same signal as the tonal term it's added
  next to, not independent evidence.
- **DEFECT 3 (headroom was a soft, tradeable penalty)** - added a hard
  feasibility gate: a candidate that raises a channel's real peak output
  above the baseline's own peak, and leaves less than
  `HEADROOM_REQUIRED_MARGIN_DB` (1.5 dB) of margin (or the baseline was
  already clip-risky), now pays a fixed `HEADROOM_VIOLATION_PENALTY` (1000.0,
  same scale as the other hard guardrails) - never a tradeable cost. A run
  once paid +1.598 on the old soft term to buy a 0.026/7.5 objective "win" on
  a channel already clip-risky; that shape of trade is now infeasible, not
  merely expensive. The old soft `SOFT_CAP_DB` term stays as a tiebreaker
  only. `target_rms_null_excluded_db`/`target_rms_null_included_db` are also
  now both reported, so a null-masked win can never hide an unmasked loss.
- **DEFECT 4a (null classification ignored nearfield evidence)** -
  `scripts/make_measurement_manifest.py` recognizes optional
  `Front L/R Nearfield.txt` captures. When both sides are present,
  `objective_module/_tunefit.py`'s new `nearfield_null_evidence()` compares
  each already-flagged null's local depth (relative to its own 1/3-octave
  baseline, so absolute level offset between the loud close-mic capture and
  the seat doesn't matter) against the nearfield sum's depth at the same
  spot. Nearfield depth under half the at-seat depth confirms the null as a
  room/summation artifact and returns a -3dB-down guard band around it.
  `objective()` hard-rejects (same 1000.0-scale penalty) any newly added or
  edited positive-gain band whose own -3dB skirt (computed from the real
  biquad response via `peaking_db`, not an approximation) still reaches into
  that guard band - not just the exact null bin, and not merely the existing
  soft `null_boost` discouragement.
- **DEFECT 4b (the pre-search "problem census" was reported but never
  enforced)** - `_merge_stream_results.py` gained
  `census_found_nothing_eligible()`/`apply_census_gate()`: when every
  worker's census reports zero eligible correction centres, only the
  baseline entry may survive into the merged output, regardless of what any
  individual worker's search still explored. Also fixed a pre-existing bug
  found while wiring this in: the `argparse.Namespace` passed to
  `write_report()` never carried `proposal_audit` forward from `args`, so
  the merged `assistant_summary.json`'s census was silently always empty
  regardless of what workers actually found.
- DEFECTS 4c (comb-vs-systematic L/R misclassification) and 5 (crossover
  deviations tested against only the lower driver's noise floor) were
  checked against current `main` and found already fixed by an earlier pass
  - no changes needed for those two.
- Verified with dedicated regression tests per defect (108 -> 110 tests,
  all passing) rather than manual checks alone; see
  `tests/test_p0_integrity.py`, `tests/test_objective_hardening.py`, and
  `tests/test_p1_features.py` for the specific failure mode each one guards
  against.

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
