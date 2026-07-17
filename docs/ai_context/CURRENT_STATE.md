# Current State

Use this as the normal context checkpoint for `audiofischeroptimiser`. Source
code, reports, git history, and the original Codex task remain the full-detail
record.

## Scope

- Offline optimizer for Helix / Audiotec Fischer tunes.
- AFPX is the primary automated write path; PCT6 support is beta and must remain
  byte-preserving.
- GitHub: `CClintos/audiofischeroptimiser`.
- Treat the checked-out repository root as the workspace; do not hard-code a
  user's local path.
- This Codex task is exclusively for this optimizer and its DSP/file support.

## Context Route

1. Read `AGENTS.md` and this file.
2. Use `REPO_MAP.md` to locate only the relevant implementation.
3. Read `assistant_summary.json` before the full summary or raw reports.
4. Open deeper docs, measurements, logs, or task history only when a compact
   source leaves a real ambiguity.

## Current Optimizer

- Long runs use the constant-memory streaming multi-worker optimizer.
- When active, the authoritative score is only
  `afpx_objective.score_bands(band_sets)['objective']`; never add a second
  flatness objective.
- Dense REW exports are normalized once to the scorer's 96-points-per-octave
  grid, and phase is unwrapped before interpolation.
- Top candidates receive hardware-step coordinate refinement against the same
  named objective.
- Optional left-ear/right-ear system sums are discovered automatically. The
  scalar combines centre-weighted median, 80th-percentile, and worst-position
  error; narrow/asymmetric filters are penalized when they fail at an ear.
- The one-command runner defaults to deterministic seed-sharded beam search.
  Guided/CMA remain available for comparisons and fallback runs.
- Console output is compact; complete JSON/Markdown/CSV stays local.
- `assistant_summary.json` contains only the decision core; `optimizer_summary.json`
  retains full settings, validation, phase confidence, components, and refinement.
- Candidate summaries include an anchor-free response audit at each changed filter
  centre and half-octave shoulders. It exposes baseline error, candidate error,
  raw system delta, pair dilution, and absolute L/R balance change from the same
  objective prediction, preventing independent re-anchoring during review.
- Beam includes a low-Q `front_voicing` group written identically to every front
  output. The objective separately reports 1.3-5 kHz target-contour error relative
  to a fixed 1.0-1.4 kHz reference, so a bass-only change cannot masquerade as a
  successful retarget.
- Matched positive front voicing can trigger a calculated uniform protective
  attenuation. AFPX `<Vol>` changes are modelled, limited to `0..-6 dB`, rounded
  to 0.25 dB, and accepted only on every front output by both verifiers.
- Optional P2 choices are explicit: sub blend is recommendation-only and needs
  calibrated level plus declared headroom; voicing files are generated only on
  request and never identify a preferred tonal balance.
- A native PySide6 Windows GUI now provides drag/drop inputs, authoritative
  preflight, bounded CPU/RAM controls, durable stop/resume, compact progress,
  results review, and AFPX export without Codex.
- The GUI exposes Beam only. Developer CLI methods remain available for
  benchmarking. Its PEQ/RTA stage disables phase writes; its Sweeps/Phase stage
  uses one baseline-only candidate so existing PEQ remains byte-identical while
  gated polarity/delay/APF changes are considered.
- The GUI assumes one fresh, consistently leveled measurement session and does
  not expose level-calibration JSON. Explicit calibration remains CLI-only.
- Results automatically emit a local `SQ_Tuning_Report.pdf`. PEQ reports lead
  with a plain-language verdict, fixed-anchor before/after response graph,
  component-improvement bars, L/R plots, expected audible effects, filter changes,
  restraint and verification. Phase reports use crossover confidence and ladder
  before/after visuals instead of presenting phase as ordinary tonal EQ.
- The GUI About tab documents the two-stage workflow, named objective, phase
  ladder, guardrails, and deliberate non-changes.
- Users with an already-dialled PEQ tune can enter Sweeps/Phase directly. The
  phase run is a short one-worker baseline diagnostic, not a timed PEQ search.
- Packaged builds contain a windowed GUI plus a console worker companion so
  PowerShell can wait for the bundled runtime without displaying a GUI console.

## Objective And Guardrails

- Perceptually weighted tonal error reports distinct tonal, presence, and
  positive-peak components; peaks carry extra objective cost.
- `target_shape_error_db` is anchor-independent and prevents the full-range
  objective from overlooking a deliberate local target contour.
- L/R evidence comes from solo traces and combines signed bias with weighted
  absolute/RMS mismatch, so opposite errors cannot cancel in the score.
- Destructive summation/nulls earn no tonal reward.
- Penalize positive gain/headroom, wasted or inert filters, unsupported
  asymmetry, deep/narrow corrections, and filter count.
- EQ only inside a driver's useful passband. Leave crossover skirts and physical
  edges alone. Prefer fewer, wider, symmetric, shallower filters.
- Hardware PEQ limits remain `G=-15..+6 dB`, `Q=0.5..15`; active search spaces
  are intentionally tighter.
- The proposed special `+4..5 dB` broad-LF boost exception was rejected. Do not
  add it unless the user reverses that decision.

- Internal objective values retain full precision; rounding is report-only.
- A baseline candidate is always retained, so a short run cannot recommend a
  generated PEQ candidate that scores worse than the loaded tune.
- Immutable baseline cascades, ERB windows, solo references, contribution
  totals, and hardware-rounded PEQ responses are cached without changing the
  golden objective.

## Crossover Ladder

- Active order: polarity, relative delay, then residual APF.
- A solo complex sum must reproduce the measured together trace before a
  complex-phase correction may write.
- Reject weak improvements, bad reference locks, ambiguous polarity, and
  disagreement with available impulse evidence.
- Mid/tweeter delay search is bounded to `+/-0.5 ms`; sub/front to `+/-3 ms`.
- APF is searched only after polarity/delay leaves a supported residual.
- One-sided APFs above 1 kHz are report-only and require live verification.
- Sub changes preserve front-stage internal offsets where possible.
- AFPX polarity uses `<T PM="1|4">`; delay uses `<T T="samples">`. Do not
  substitute `CINV` in the normal writer.
- Candidate writers never overwrite the baseline.
- Phase-valid sessions model complete RBJ PEQ magnitude/phase and jointly verify
  PEQ with polarity, delay, and APF using measured-plus-complex-model delta.
- Sessions without valid phase retain the conservative 0.5 dB crossover PEQ veto.
- Lint and external verification independently check PEQ, PM polarity, delay,
  APF, crossover, output volume/attributes, and other time-alignment attributes.

## Optional Impulses

- Accept companion PCM/IEEE-float WAV or two-column time/amplitude text.
- Names use the measurement stem, for example `Front L High.wav`,
  `Front L High Impulse.wav`, or `Front L High IR.txt`.
- Files may be beside measurements or under `impulses`, `Impulse`, or `IR`.
- `--impulse-root` and PowerShell `-ImpulseRoot` select another folder.
- Band-limited cross-correlation estimates arrival and polarity.
- Strong impulse evidence plus measured destructive summation can replace an
  invalid complex reference; conflicting evidence vetoes a write.

## Measurements

- REW rows may include phase, coherence, and position ID columns.
- Solos, together pairs, and system sum must correspond to the loaded baseline
  and timing-reference session for coherent phase prediction.
- Aliases cover common `Front L/R`, `Front Left/Right`, Mid/Low/Tweeter,
  `Both Mids/Tweeters`, and `Sub/Subwoofer` forms.
- Source-level changes can still support timing, but not raw level comparison
  unless an explicit role/file dB calibration JSON is supplied.
- Position bundles use `Left Ear ` / `Right Ear ` filename prefixes or matching
  subfolders. Centre measurements remain required; spatial bundles are optional.
- Three-position measurements improve spatial confidence; fixed-position solos
  and together pairs remain necessary for coherent prediction.
- The manifest reports missing inputs, level/reference changes, phase/coherence,
  grid mismatches, and companion impulses.

## AFPX And PCT6

- Preserve crossovers unless explicitly requested.
- AFPX polarity/delay/APF writes require the active evidence gates and warnings.
- PCT6 must not assume generic channel numbering. In the previously decoded
  Helix DSP PRO MK3 example, visible Output A-J mapped to `ch12-ch21`; that is
  tune-specific evidence, not a universal hard-coded map.
- Preserve unknown bytes/fields and verify PCT6 round trips.

## Verified State

- Forty-two regression tests pass, including objective/target-shape invariants,
  session gates, crossover PEQ vetoes, protective volume-write safety, and a
  modern five-column TXT/AFPX golden benchmark.
- Python compilation and `git diff --check` pass.
- Historical real-data smoke testing rejected the stale-reference sub polarity
  flip and an ambiguous left polarity result; it allowed only a warning-level
  six-sample right-tweeter delay on that dataset.
- A synthetic inverted impulse arriving `0.500 ms` late was recovered correctly.
- A controlled AFPX polarity-plus-delay write changed only PM polarity and the
  intended delay value.
- A historical real-data smoke candidate changed only the supported six-sample
  delay; independent verification found no crossover, PEQ, polarity, output,
  time-alignment-attribute, or unknown-field changes.
- P1 equal-budget benchmark (10 seconds, seed `20260712`): beam `10.530787`,
  guided/CMA `10.917834`; beam also completed with lower wall time.
- Repeated exact scoring on the three-position historical set measured about
  `214.6 scores/s`; the P0 snapshot was about `17.9 scores/s`.
- The P1 beam candidate passed independent PEQ-only AFPX verification.
- The July 14 Harman retarget regression selected a matched whole-front
  `+5.0 dB @ 2985 Hz, Q1.30` transfer plus uniform `-1.0 dB` front attenuation;
  anchor-independent target-contour error improved from `4.21` to `1.97 dB`,
  and external AFPX verification found no crossover, delay, polarity, APF, or
  existing-filter changes.
- Complex RBJ magnitude matches the existing PEQ dB model to numerical precision;
  phase-valid combined candidates use the canonical phase-session schema.
- The packaged Windows worker completed a real AFPX/measurement run, merged 20
  candidates, emitted `assistant_summary.json`, and passed family verification.
- A phase-only smoke produced one candidate with one gated delay action; external
  verification reported zero added/removed PEQ filters and no crossover changes.
- Historical test results are not assumptions about future measurements.

## Deliberate Non-Changes

- No automatic target choice or preferred voicing. The supplied target remains
  authoritative; opt-in audition files remain neutral listening choices.
- No broad-LF boost exception.
- No direct `fit_peq()` post-pass because it uses a different objective.
- No sub output-level writes; sub blend remains a recommendation.
- No automated crossover-frequency/slope writing.
- No live capture app; measurement capture remains external, normally REW.

## Resource And Token Policy

- Prefer local execution; model tokens do not accelerate numerical search.
- For long runs: launch locally, stay quiet, merge once, and return the best
  files plus a short named-component summary.
- Do not poll CPU/RAM unless diagnosing a problem.
- Keep memory bounded and checkpoint to disk. Historical crashes coincided with
  memory reaching about 95 percent.
- Roughly 60 percent CPU and up to 50 percent RAM are acceptable if the PC stays
  usable and memory cannot spiral.
- Do not reread raw worker logs, every candidate, the whole task, or research
  reports for routine questions.
- Prefer `run_optimizer.ps1` for validate/run/merge/verify. It prints only the
  final `assistant_summary.json` path.

## Working Tree

- The repo may contain uncommitted optimizer and documentation changes.
- Never discard unrelated user changes.
- Check `git status` before commits or pushes.
- Do not claim GitHub is current unless changes were committed and pushed.
