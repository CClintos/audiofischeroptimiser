# AudioFischer Optimiser

AudioFischer Optimiser is a local Windows tuning app for Helix and Audiotec
Fischer DSP systems. Give it a baseline `.afpx` tune, REW measurement exports,
and a target curve; it searches for conservative improvements and produces
ranked AFPX candidates to load and test in the car.

The app runs entirely on your PC. Codex or Claude can help with development and
advanced analysis, but neither is required to use it.

## What It Does

- **Optimises PEQ from real measurements.** It considers tonal accuracy, vocal
  balance, peaks, left/right matching, filter cost, driver passbands, and
  optional centre/left-ear/right-ear measurements.
- **Supports 2-way and 3-way front stages plus subwoofer.** Common REW filenames
  are recognised automatically, and unfamiliar names can be mapped to speaker
  roles inside the app.
- **Handles phase and timing as a separate measured workflow.** With coherent
  solo/together sweeps, it can evaluate polarity, bounded relative delay, and a
  residual all-pass filter around a crossover.
- **Retargets an existing tune.** Take fresh measurements of the current tune,
  select a different target curve, and optimise supported PEQ toward the new
  tonal balance.
- **Produces usable results.** Compare the current tune with the recommended
  candidate, inspect the response chart and exact filters, export AFPX files,
  and generate a local PDF tuning report with in-car checks.
- **Keeps risky areas conservative.** It avoids treating destructive nulls as EQ
  problems, prefers fewer/wider/shallower filters, never overwrites the baseline,
  and does not rewrite crossover frequencies or slopes.

AFPX is the primary and verified write path. The repository also includes beta
`.pct6` container inspection and round-trip tools for newer DSP PC-Tool 6 saves;
PCT6 output should still be verified in PC-Tool before use.

## Desktop Workflows

1. **PEQ / RTA** - use fresh magnitude or moving-mic measurements to create
   tonal-correction candidates. Phase controls remain untouched.
2. **Sweeps / Phase** - after loading the chosen PEQ tune, take fresh
   phase-valid sweeps and test only evidence-supported polarity, delay, or APF
   changes while preserving PEQ.
3. **Retarget** - measure the current tune again and optimise PEQ toward a
   different supplied target.

Results show improvement versus the current tune, the exact added filters,
before/candidate/target response curves, optional per-driver predicted changes,
warnings, and what to verify in the car. Every completed run writes
`assistant_summary.json` and `SQ_Tuning_Report.pdf` beside the candidate files.

## Download or Run

For the normal Windows app, download the
[latest Windows package](https://github.com/CClintos/audiofischeroptimiser/releases/latest/download/AudioFischerOptimizer-windows-x64.zip),
extract the ZIP, and run `AudioFischerOptimizer.exe`. Keep the extracted
`_internal` folder beside the EXE. No Python installation is required.

All published versions and checksums are available on the
[GitHub Releases page](https://github.com/CClintos/audiofischeroptimiser/releases).

Developers can run the app from source:

```powershell
.\setup_gui.ps1
.\start_gui.ps1
```

To build the Windows package locally:

```powershell
.\build_gui.ps1
```

The executable is written to:

```text
dist\AudioFischerOptimizer\AudioFischerOptimizer.exe
```

Keep the complete `AudioFischerOptimizer` folder together because the EXE uses
the bundled `_internal` directory. `install_gui.ps1` can install that package
for the current Windows user and create a desktop shortcut.

See [docs/GUI.md](./docs/GUI.md) for the measurement workflow and
[PCT6_SUPPORT.md](./PCT6_SUPPORT.md) for PCT6 limitations.

## How It Decides

The scoring system is designed to reward candidates that are more likely to
sound better, not simply look flatter on one graph.

It scores candidates by:

- how close the full system response is to the target, with distinct tonal and vocal/presence terms and extra cost for peaks
- whether it reproduces the supplied target's 1.3-5 kHz contour in an anchor-independent shape check
- how well left and right match each other, using signed bias plus weighted absolute/RMS mismatch from the solo drivers
- whether the tune avoids boosting into destructive cancellation nulls
- whether each filter is on a driver that is actually contributing at that frequency
- whether it avoids wasting filters, using unnecessary gain, or adding deep/narrow one-seat corrections
- whether corrections hold across optional centre/left-ear/right-ear system sums

The math layer also includes confidence and timing helpers for phase-valid measurements:

- coherence weighting, so low-trust bins count less
- band-limited phase-delay estimation around crossover regions
- gated impulse helpers that estimate how low a time window can be trusted

The optimizer first proves that the solo complex responses reproduce the measured together trace. It then tests polarity and delay, and searches an APF only if a meaningful residual remains. Invalid reference locks, weak predicted improvements, ambiguous polarity, and conflicting impulse evidence block automatic writes. Written AFPX candidates are linted so only the intended PEQ, `PM` polarity, delay values, APF slots, and declared uniform protective front attenuation may change.

Optional companion impulse exports can be WAV or two-column time/amplitude text. Put them beside the measurements, or in an `impulses` folder, using the measurement stem, for example `Front L High.wav`, `Front L High Impulse.wav`, or `Front L High IR.txt`. Use `--impulse-root` / `-ImpulseRoot` when they live elsewhere. Band-limited cross-correlation supplies arrival and polarity evidence; disagreement with the complex-phase solution vetoes the write.

Supported REW text export rows:

- `freq spl`
- `freq spl phase`
- `freq spl phase coherence`
- `freq spl phase coherence position_id`

If you know the impulse/window gate length, pass `-GateMs` to the PowerShell launcher or `--gate-ms` to the Python scripts. The report will warn when a gated response should not be trusted below its lowest valid frequency.

The measurement-session gate checks REW source volume, sweep level, and timing
reference metadata. Mixed or missing level provenance requires an explicit JSON
map of role/file names to dB corrections; phase writes require one shared timing
reference plus measured-together validation.

It can also be used through Claude or Codex:

1. Drag in your REW measurement text exports.
2. Drag in your baseline `.afpx` tune.
3. Optionally drag in your target curve text file.
4. Ask Claude or Codex to use this repo as a local AFPX optimizer and run it.

The normal local entry point is `run_optimizer.ps1`. It validates the session,
uses a bounded worker count, prepares phase diagnostics once, runs/resumes the
search, merges and verifies family candidates, then prints only the path to
`assistant_summary.json`.

Optional audible extensions are never enabled silently:

```powershell
.\run_optimizer.ps1 -DataRoot ".\my measurements" `
  -SubBlend recommend -HeadroomDb 3 `
  -VoicingVariants audition
```

`SubBlend` reports a same-level sub output-trim suggestion only when the session
is calibrated and declared headroom is available; it never creates a broad PEQ
boost. `VoicingVariants` writes labelled warm, reference, and clear audition
files while leaving the supplied target untouched and declaring no winner.

For phase-valid solo/together sessions, candidate PEQ is evaluated as a full
complex RBJ transfer together with polarity, delay, and residual APF. Invalid or
missing phase data keeps the conservative crossover-band PEQ veto. The tonal
objective also complex-sums candidate PEQ from measured solo phase when those
solos reproduce the measured together/system trace within 2.5 dB RMS. Failed or
placeholder phase data automatically uses the measured-residual magnitude model.
Routine phase
analysis uses `analyze_phase_session()` and the stable
`audiofischer-phase-session-v1` schema; specialist multinull tools remain
experimental.

Suggested prompt:

```text
Use this repo as a local AFPX tuning tool.

I have attached:
- my REW measurement text exports
- my baseline .afpx tune
- optionally my target curve

Please verify the files, run the optimizer locally, merge the results, and give me the best AFPX candidates with a short summary of what improved.
Polarity/delay/APF changes may be written only when the crossover ladder clears its evidence gates. Crossovers remain untouched.
```

## Measurement Inputs

Canonical measurement filenames are listed below. The names themselves are not
mandatory: if a folder uses names such as `FL High Sweep.txt`, the GUI opens a
fuzzy-prefilled role-mapping dialog and can remember that naming for later runs.

Required for a PEQ run:

- `System Sum.txt`
- `Sub.txt`
- `Front L High.txt` or `Front L Tweeter.txt`
- `Front R High.txt` or `Front R Tweeter.txt`
- `Front L Low.txt` or `Front L Mid.txt`
- `Front R Low.txt` or `Front R Mid.txt`

Recommended pair evidence, but optional for PEQ:

- `Tweeters Together.txt` or `Both Tweeters.txt`
- `Mid Bass Together.txt` or `Both Mids.txt`

When a pair trace is absent, the optimizer uses the two measured solo drivers
plus the measured System Sum for conservative magnitude scoring. It clearly
marks pair-summation/null validation as unavailable and disables phase, delay,
polarity, and APF writes. Supplying the pair trace restores those evidence
checks.

For a true front 3-way system, separate left/right mid and low branch
measurements are required so the optimizer can detect and score
`high + mid + low + sub`. The corresponding `Mids Together` and
`Mid Bass Together` traces remain recommended pair evidence.

Expected tune file:

- `baseline.afpx`

## Main Files

- [optimizer_gui/window.py](./optimizer_gui/window.py): native Windows workflow,
  run monitoring, interactive Results, and export interface
- [optimizer_gui/backend.py](./optimizer_gui/backend.py): durable jobs,
  validation, detached process ownership, run claims, and exports
- [optimizer_gui/reporting.py](./optimizer_gui/reporting.py): shared result
  metrics, charts, and local PDF report
- [optimizer_gui/_version.py](./optimizer_gui/_version.py): single application
  and package version source
- [_optimizer.py](./_optimizer.py): core scoring, prediction, AFPX writing, reporting
- [_optimizer_stream.py](./_optimizer_stream.py): constant-memory multi-worker optimizer
- [run_optimizer.ps1](./run_optimizer.ps1): one-command validate/run/merge/verify wrapper
- [_merge_stream_results.py](./_merge_stream_results.py): merges worker archives into final outputs
- [run_guided_stream_workers.ps1](./run_guided_stream_workers.ps1): launches long local runs
- [merge_guided_stream_results.ps1](./merge_guided_stream_results.ps1): safe merge wrapper
- [objective_module/afpx_objective.py](./objective_module/afpx_objective.py): independent scalar objective used by the optimizer
- [objective_module/_tunefit.py](./objective_module/_tunefit.py): canonical DSP/math helpers used by both optimizer and objective
- [objective_module/session.py](./objective_module/session.py): isolated multi-session scoring API
- [_tunefit.py](./_tunefit.py): compatibility import for the canonical DSP module
- [afpx.py](./afpx.py): generic `.afpx` inspector and channel-role helper
- [pct6.py](./pct6.py): beta `.pct6` decode / encode utility for no-password PC-Tool 6 saves
- [PCT6_SUPPORT.md](./PCT6_SUPPORT.md): caveats and safe usage notes for `.pct6`

Required measurements fail fast when missing, malformed, non-monotonic, or
truncated. Optional pair traces and phase, coherence, and position data extend
the available diagnostics without blocking a magnitude-only PEQ run.

The optimizer normalizes REW exports to the 96-points-per-octave grid used by its
ERB and perceptual scoring math. The streaming search then applies a small
hardware-step coordinate refinement to its best candidates using the same named
scalar objective; it does not add a second flatness target.

## Compact Local Summaries

These scripts are for Claude/Codex efficiency. They produce small JSON files so an assistant does not need to read raw logs, every candidate, or full measurement exports.

- Every optimiser run writes `assistant_summary.json` as the first file for Claude/Codex to read. It contains fingerprints, gates, baseline/best component deltas, family files, phase actions, rejected PEQ/phase conflicts, warnings, and re-measure instructions.
- `optimizer_summary.json`, `optimizer_report.md`, and `optimizer_results.csv` retain the full local detail when the compact decision core is insufficient.
- Console helpers default to compact output while retaining full JSON/Markdown/CSV files locally. Use `--print-mode full` only when the extra detail is needed.
- [scripts/make_measurement_manifest.py](./scripts/make_measurement_manifest.py): resolves common REW filename aliases, detects 2-way/3-way layout and phase/coherence columns, and warns about inconsistent source level, timing reference, or frequency grids.
- [scripts/prepare_phase_cache.py](./scripts/prepare_phase_cache.py): fingerprints and prepares the crossover audit once per session.
- [scripts/benchmark_search_methods.py](./scripts/benchmark_search_methods.py): equal-seed/equal-time guided, beam, and CMA comparison.
- [scripts/summarise_optimizer_run.py](./scripts/summarise_optimizer_run.py): summarizes an optimizer output folder, preferring `assistant_summary.json`, then the full JSON, with CSV fallback.
- [scripts/summarise_candidate_filters.py](./scripts/summarise_candidate_filters.py): summarizes one candidate's added filters and risk flags.
- [scripts/verify_written_tune.py](./scripts/verify_written_tune.py): verifies candidate AFPX files only changed intended fields.

Useful examples:

```powershell
python .\scripts\make_measurement_manifest.py "C:\path\to\measurements"
python .\scripts\summarise_optimizer_run.py ".\Optimizer_Run\_merged_top"
python .\scripts\summarise_candidate_filters.py ".\Optimizer_Run\_merged_top\family_balanced.afpx" --baseline "C:\path\to\baseline.afpx"
python .\scripts\verify_written_tune.py "C:\path\to\baseline.afpx" ".\Optimizer_Run\_merged_top\family_balanced.afpx" --allow-output-trim --allow-delay --allow-apf
```

## Safety / Scope

This tool is intentionally conservative.

- It optimizes PEQ from magnitude data.
- It can edit polarity, delay tags, and residual APFs when the crossover ladder clears its gates.
- It does not change crossovers.
- It can add conservative APF filters when the phase report shows residual crossover uncertainty.
- It treats destructive summing regions as not EQ-fixable.

For `.pct6`, the repo currently provides careful container decode / encode support and inspection helpers. AFPX writing is still the primary automated output path.

That means it is primarily a PEQ optimizer, with conservative polarity/delay/APF writes only when crossover evidence clears the active ladder gates.
