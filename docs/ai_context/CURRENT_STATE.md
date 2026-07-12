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
3. Read `optimizer_summary.json` or a compact summarizer before raw reports.
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
- Console output is compact; complete JSON/Markdown/CSV stays local.
- `optimizer_summary.json` includes input hashes, settings, validation, phase
  confidence, named score components, and refinement results.

## Objective And Guardrails

- Perceptually weighted tonal error has extra vocal/presence importance.
- L/R evidence comes from solo traces, not only the mono system sum.
- Destructive summation/nulls earn no tonal reward.
- Penalize positive gain/headroom, wasted or inert filters, unsupported
  asymmetry, deep/narrow corrections, and filter count.
- EQ only inside a driver's useful passband. Leave crossover skirts and physical
  edges alone. Prefer fewer, wider, symmetric, shallower filters.
- Hardware PEQ limits remain `G=-15..+6 dB`, `Q=0.5..15`; active search spaces
  are intentionally tighter.
- The proposed special `+4..5 dB` broad-LF boost exception was rejected. Do not
  add it unless the user reverses that decision.

Audit correction: the active external objective still lacks a distinct
peak-vs-dip term, reduces L/R balance to a signed median, and aliases one tonal
value into several report component names. These are P0 roadmap defects, not
capabilities to assume are already solved. See `docs/AUDIT_2026-07-12.md`.

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
- Lint and external verification independently check PEQ, PM polarity, delay,
  APF, crossover, output attributes, and other time-alignment attributes.

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
  unless normalized.
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

- Four crossover-ladder regression tests pass.
- Python compilation and `git diff --check` pass.
- Historical real-data smoke testing rejected the stale-reference sub polarity
  flip and an ambiguous left polarity result; it allowed only a warning-level
  six-sample right-tweeter delay on that dataset.
- A synthetic inverted impulse arriving `0.500 ms` late was recovered correctly.
- A controlled AFPX polarity-plus-delay write changed only PM polarity and the
  intended delay value.
- Historical test results are not assumptions about future measurements.

## Deliberate Non-Changes

- No automatic voicing/target-family layer; the supplied target is authoritative.
- No broad-LF boost exception.
- No direct `fit_peq()` post-pass because it uses a different objective.
- No complex PEQ summation until filter phase is modeled consistently.
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

## Working Tree

- The repo may contain uncommitted optimizer and documentation changes.
- Never discard unrelated user changes.
- Check `git status` before commits or pushes.
- Do not claim GitHub is current unless changes were committed and pushed.
