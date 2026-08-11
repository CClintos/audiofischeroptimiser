# Changelog

Notable changes to the GUI and optimizer, newest first. This is a plain
human/AI-readable log, not auto-generated; keep entries short and specific
enough that Codex or Claude can pick up context without rereading the diff.

## 2026-08-11 - v0.9.3 out-of-passband baseline hotfix

- Fixed existing-tune preparation crashing when a matched baseline filter sat
  outside the configured driver passband. Such filters remain eligible for
  removal but are no longer modified or allowed to abort the run.
- Hardened correction attribution so an ineligible paired owner is reported as
  outside its limits instead of raising an exception. Verified with the exact
  August 10 measurements and baseline that failed in v0.9.2: preparation
  completed 12,579 evaluations.

## 2026-08-10 - v0.9.2 two-way role-map hotfix

- Fixed PEQ runs misclassifying a declared two-way system as three-way when
  midbass measurements used names such as `Front L Mid.txt`. The manifest and
  optimizer now preserve the explicit layout stored in `role_map.json`.
- Added a regression for two-way role maps whose `FL Low` and `FR Low` roles
  point to mid-named files. Verified the previously failing real session through
  existing-tune preparation with 11,352 filter evaluations.

## 2026-08-10 - v0.9.1 validation deadlock hotfix

- Fixed Windows validation hanging indefinitely when the preflight diagnostic
  JSON filled the child-process output pipe. The GUI now drains stdout/stderr
  while the validator runs and retains responsive cancellation.
- Added a regression that drains 200 KB from a validator child without
  deadlocking. Verified against a real mapped measurement session that
  previously remained stuck on `VALIDATING`.
## 2026-08-10 - v0.9.0 automatic baseline rehabilitation

- The PEQ workflow now audits every eligible existing filter before adding new
  ones. It can remove harmful bands, adjust gain/Q/frequency in place, combine
  overlapping corrections, retain the unchanged baseline, and continue Beam
  search for residual problems.
- Candidate plans preserve exact AFPX slot identity through scoring,
  checkpoints, resume, worker merging, refinement, reporting, and final file
  writing. Delay, polarity, crossover, and APF protections remain enforced.
- Multiworker PEQ runs build one atomic, fingerprinted rehabilitation cache
  instead of repeating the same baseline analysis in every worker. Cache
  invalidation includes measurements, target, calibration, repeatability data,
  role mapping, and scoring context; preparation supports safe cancellation.
- Result summaries distinguish the supplied tune, improved existing tune, and
  final candidate when they are genuinely different. Reports include concrete
  filter operations, component deltas, headroom, guardrail outcomes, and
  repeatability-aware wording rather than claiming numerical ties as wins.
- Added extensive regression coverage for slot edits, removals, interaction
  Beam behavior, fixed-baseline scoring, stale resume/cache rejection, phase
  workflow separation, bounded runtime, reporting, and Windows launcher paths.

## 2026-08-01 - Platform fixes from a repo review ("Batch C")

Final tier of the same review - packaging, parsing, and CI, not acoustic
behaviour. No changes to search/scoring numerics. 139 tests passing (up
from 135 after Batch B).

- **`DeviceProfile`** (`objective_module/device_profile.py`): consolidates
  the DSP's sample rate and PEQ hardware range (frequency/Q/gain), which
  were hardcoded literals repeated in `_tunefit.py` (`FS = 96000.0`) and
  `_make_v3.py` (`validate_peq_band`'s 20-20000 Hz/0.5-15 Q/-15..+6 dB).
  Value-preserving - both call sites now read from one documented
  `HELIX_P_SIX_MK2` profile instead of repeating the numbers, with the
  model-verification caveat attached directly to it. Deliberately does NOT
  fold `GROUPS`'s own `gain_range`/`q_range` (`_optimizer.py`) into this -
  those are conservative ACOUSTIC SEARCH POLICY ("should we"), not a
  hardware limit ("can the hardware"), and stay separate on purpose. Found
  and fixed a real bug while wiring this in: an early version added
  `objective_module/` directly to `sys.path` from inside `_tunefit.py` (and
  from `_make_v3.py`), which let a bare `import _tunefit` elsewhere resolve
  straight to `objective_module/_tunefit.py` instead of the root
  `_tunefit.py` compatibility shim - silently loading the file TWICE under
  two different module identities, so every DSP function in it existed as
  two different objects. `CanonicalDspTests` (pre-existing) caught it
  immediately; fixed with a relative import in the package case and a
  fully-qualified `objective_module.device_profile` import from
  `_make_v3.py`, no sys.path changes needed either way.
- **REW parser hardening** (`_load_txt_rich` in `afpx_objective.py`): used
  to blindly `.replace(',', ' ')` on every line, which corrupts a European-
  locale decimal comma (e.g. "3,295898" -> "3 295898", split into two
  garbage tokens) even though REW itself supports comma/tab/space/
  semicolon-delimited exports with a selectable decimal convention. New
  `_detect_delimiter_and_decimal()` inspects a handful of real data lines
  and decides delimiter and decimal convention TOGETHER: tab or semicolon
  found first wins outright (semicolon implies comma-decimal); otherwise,
  if whitespace already splits the line into 2+ tokens, a comma living
  inside one of them is a decimal point; only when there's no whitespace
  at all is a comma treated as the field separator. (An initial version
  used "4+ digits after a comma implies decimal" instead, which misread a
  perfectly ordinary "100,5 60,0" SPL row as comma-delimited - caught by
  the tests before it shipped.) Also: exact-duplicate frequency rows are
  now dropped explicitly (keep the first, report the count in the trace's
  new `format` dict) instead of failing the generic "not strictly
  increasing" check with no indication of why; a genuinely out-of-order
  (non-duplicate) row still fails that check as before. Verified against
  all 22 real REW export files across all three of the user's sessions -
  zero behaviour change for real data, this only changes what happens on
  input that previously would have been silently corrupted or opaquely
  rejected.
- **Packaging**: `pyproject.toml`'s `packages.find` only ever included
  `optimizer_gui*`, even though most of the actual optimizer/DSP logic
  lives in root-level modules and `objective_module`/`scripts`. A plain
  `pip install` of this project silently produced a GUI-only, non-
  functional package. `objective_module/` and `scripts/` gained a minimal
  `__init__.py` (regular packages now, not implicit namespace packages, so
  `packages.find` can discover them); root-level modules are listed
  explicitly under a new `[tool.setuptools] py-modules` (setuptools'
  package finder only discovers directories, not standalone `.py` files).
  The empty `test` extra now pins `PySide6` (tests/test_gui_backend.py
  already imports it). Verified with a real wheel build + install into a
  clean venv + import from outside the source tree - every core module
  resolves from the installed package, not a stale source-tree fallback.
- **PR CI** (`.github/workflows/tests.yml`): the only existing workflow
  was the tag-triggered Windows release build - a plain commit or pull
  request never ran anything. New workflow runs the full suite on
  `pull_request` and push-to-`main`, matrixed across Python 3.11/3.12/3.14
  (3.14 matching the release build) and Ubuntu/Windows (Ubuntu needs a
  handful of headless-Qt system libraries; `test_gui_backend.py` already
  sets `QT_QPA_PLATFORM=offscreen`), plus a separate job that builds a real
  wheel and smoke-tests importing every core module from the installed
  package - the same check used to verify the packaging fix above, now
  running on every PR instead of only when someone remembers to check by
  hand.

## 2026-08-01 - Audible-quality improvements from a repo review ("Batch B")

Continuation of Batch A below - the same review's higher-effort, more
architectural recommendations. Optimizer only, no GUI changes. 133 tests
passing (up from 122 after Batch A).

- **Full-chain headroom now includes shelves.** Headroom only ever looked at
  the T=17 PEQ cascade; a T=3/4 shelf carries real gain but was invisible to
  every headroom check. The search never touches shelves (only PEQ), so this
  is a fixed, baseline-only addition to the real per-channel chain -
  `_BASE_SHELF_DB` (built once in `_init()` from `low_shelf_db`/
  `high_shelf_db`, already existing in `_tunefit.py` but unused for this)
  folded into the headroom peak everywhere PEQ cascade + output trim used to
  stand in for "the whole chain" alone. Verified with a synthetic baseline
  (a +6 dB/9 kHz shelf on a channel trimmed to -9.1 dB output): pre-fix, a
  +4 dB/12 kHz PEQ candidate reported a comfortable 5.1 dB margin; the real
  combined peak was actually within a fraction of a dB of 0 dBFS. Crossovers
  (T=9/15/16) and a separate "routing gain" stage were deliberately NOT
  added - crossovers are pure attenuation (never raise the peak, so omitting
  them is conservative, not unsafe) and this hardware's AFPX format has no
  routing/mix stage distinct from per-channel PEQ + output level.
- **Continuous boost/cut correction confidence.** The objective's null
  classification, repeatability, and driver-authority checks were real but
  separate binary gates. `correction_confidence()` (`_tunefit.py`) combines
  them into one continuous [0,1] score per frequency, split asymmetrically:
  boosting needs stronger evidence than cutting (a missed cut just leaves a
  peak in place; an unjustified boost can audibly worsen the exact spot it
  targeted). Wired into `_optimizer_stream.py`'s candidate generation as a
  gain-scaling refinement on top of (never instead of) every existing gate:
  a proposed boost scales by confidence², a cut by confidence's square root,
  matching the asymmetry. A near-miss on the design: the first version
  defaulted every missing-evidence factor to 0.5 and multiplied them all
  together, so a single-session run with no position/phase data (the common
  case) compounded three or four 0.5s into rejecting ~98% of every boost's
  gain even with strong evidence on the factors that WERE available. Caught
  before it shipped and fixed: missing evidence is neutral (1.0, no effect on
  a product of independent factors), not partial-credit.
- **Global target anchor + bounded per-position offsets.** A secondary
  listening position (left/right ear) used to get a FULLY INDEPENDENT target
  re-anchor, which can silently hide a genuine broad level difference
  between positions by always re-centering to its own local median. Now: one
  confidence-weighted, multi-band-fallback global anchor
  (`target_anchor_offset()` - already existed, was never used) stays fixed
  for the whole search, and each position gets only a small bounded
  "nuisance" offset (`POSITION_ANCHOR_NUISANCE_BOUND_DB = 1.5`) around it. A
  position whose raw independent anchor would have differed by more than
  that keeps the excess visible as real deviation instead of it vanishing.
  Offsets are recorded in `_MASK_AUDIT['target_anchor']` for the report.
- **Post-quantization filter reduction.** After the search picks its
  finalists, `simplify_removable_bands()` drops any PURELY APPENDED band
  (never an edit or removal - both already-deliberate actions, never
  touched) whose removal changes that channel's own cascade by less than
  0.1 dB everywhere, then checks near-cancelling PAIRS of whatever survives
  alone. Wired into `write_candidate()` only - deliberately NOT into
  `_resolve_group_bands`/`groups_to_band_sets`, which run for every
  candidate the beam search evaluates (thousands per run); this is a
  refinement on the small number of already-selected finalists, not a
  search-time cost.

## 2026-08-01 - Correctness fixes from a repo review ("Batch A")

An independent architecture review (static read of `main`, no test run) found
four correctness issues. All four verified against real code and real v9 data
before fixing, all four now have dedicated regression tests. Optimizer only.

- **Sub-channel guardrail gap (the highest-priority finding).** Headroom,
  null-boost exposure, and the parsimony filter count in
  `objective_module/afpx_objective.py`'s `objective()` all iterated
  `range(len(CH_KEYS))` - front channels only. The subwoofer lives at
  physical channels 6/7 (`GROUPS['sub']['channels']` in `_optimizer.py`,
  always, regardless of front layout), so a filter placed there was scored
  acoustically but bypassed every safety and parsimony penalty entirely.
  Verified against the real v9 baseline: a +6 dB/Q6 boost that gets
  hard-rejected (objective ~2048, guardrail penalty ~41) on a front channel
  scored *better than doing nothing* (objective 6.2359 vs baseline 6.2370)
  when placed on the sub instead. Fixed with an explicit
  `GUARDRAIL_CHANNEL_INDICES` covering CH_KEYS plus channels 6/7, used by
  the headroom loop, null-boost accumulation, and the parsimony band count.
  `_added_bands_by_channel()` gained an optional `channels` parameter so the
  one call site that should see sub additions (the DEFECT-4a nearfield-skirt
  guardrail) can ask for them, while `_guardrail_score()`'s own front-only
  L/R-pair/imaging/measurement-noise checks stay front-scoped on purpose -
  sub has no L/R pair or calibrated noise floor of its own yet; folding it
  into that per-band logic is a separate, larger design task, not this fix.
- **48 kHz/96 kHz minimum-phase inconsistency.**
  `objective_module/_tunefit.py`'s `minphase_from_mag()` hardcoded
  `fs=48000.0` for its cepstral reconstruction's internal FFT grid
  (`lin_f = linspace(0, fs/2, ...)`), and `excess_gd_mask()` (the EQ-ability
  classifier) called it without overriding that. Harmless for most car-audio
  REW exports (~20-24 kHz max), but any measured content at or above 24 kHz
  never made it into the reconstruction at all - `np.interp`'s default flat
  extrapolation just froze the output at whatever value sat right at 24 kHz.
  Not really a "should be 96 kHz instead" fix (this operates on measured
  acoustic data, not the DSP's own biquad math, so there's no reason it
  needs to match the 96 kHz `FS` constant) - now derived from the data
  itself (2.2x headroom above the highest measured frequency), so it's
  always correct for whatever `freqs` actually contains. `fs` is still
  overridable on both functions.
- **AFPX decoded with `.decode('utf-8', 'replace')`,** including on the
  write path (`_make_v3.decode_afpx`, used by every `write_candidate` call)
  and the verification path (`scripts/verify_written_tune.py`). A malformed
  or non-UTF-8 byte would silently become U+FFFD, which then re-encodes to
  DIFFERENT bytes than the original on write - corrupting whatever attribute
  held it, even outside the one value a write actually intends to change.
  There was already a weak indirect check (compare the AFPX header's
  declared length against the re-encoded length) but it only printed a
  warning and continued. Switched every decode site to strict UTF-8 (fails
  loudly instead). Verified empirically against 24 real AFPX files spanning
  several tuning iterations (v4 through v9, plus variants) - all decode as
  clean UTF-8 already, so this costs nothing for real files; it only closes
  the silent-corruption path for a malformed one. A full byte-preserving
  rewrite (never decoding to `str` at all) would be safer still but is a
  much larger change to every regex call site in `_make_v3.py` and
  `afpx_objective.py` - not done here.
- **Modal-null width used the whole search window, not the connected
  component.** In `modal_null_evidence()`'s single-position fallback,
  `contiguous = around` took every "dip" bin within 1/3 octave of the
  candidate minimum - not the connected region actually containing that
  minimum. Two separate narrow notches close together got silently merged
  into one inflated width estimate, which could push a genuinely narrow
  (and otherwise-qualifying, >=8 dB deep) null over the 1/6-octave modal
  cutoff and hide it. Verified with a synthetic two-notch signal: it
  returned `MASK_CLEAR` (found nothing) before the fix. Now walks outward
  from the confirmed minimum while the bin stays a "dip" bin, giving the
  true width of just that null.

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
