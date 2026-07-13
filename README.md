# AudioFischer Optimiser

This repo is a local AFPX tuning tool for Helix / Audiotec Fischer DSP systems.

The normal user interface is the Windows desktop app. AI-assisted and command-line
work remain available, but neither Codex nor Claude is required to run a tune.

## Windows App

For a packaged release, open `AudioFischerOptimizer.exe`. From source:

```powershell
.\setup_gui.ps1
.\start_gui.ps1
```

After building, `install_gui.ps1` installs the app for the current Windows user
and creates a desktop shortcut; it does not require administrator access.

The app has separate **PEQ / RTA** and **Sweeps / Phase** stages. PEQ always uses
the recommended Beam search with phase writes disabled. The second stage uses
fresh sweeps and the PEQ result as its baseline, preserves PEQ, and writes only
gated phase/delay/APF changes. It also controls CPU and optimizer RAM use,
preserves checkpoints, safely stops/resumes runs, and loads verified candidates
from `assistant_summary.json`. See [docs/GUI.md](./docs/GUI.md).

For AI-assisted development, start with `AGENTS.md` and
`docs/ai_context/CURRENT_STATE.md`, then use `REPO_MAP.md`.

It takes your baseline `.afpx` tune plus REW measurement exports, tries many possible PEQ combinations, scores them, and gives you back ranked AFPX tune candidates to test.

The goal is to improve the tune conservatively. PEQ is handled from magnitude data. Crossover correction follows a gated polarity -> delay -> residual APF ladder, and the report warns when confidence is not full.

It also includes beta `.pct6` container support for newer DSP PC-Tool 6 tunes, so the repo can inspect or round-trip those files too. That path is less proven than `.afpx`, and should be treated as a careful utility rather than a blindly trusted writer.

Its scoring system is built to reward tunes that are more likely to sound better, not just look flatter on one graph.

It scores candidates by:

- how close the full system response is to the target, with distinct tonal and vocal/presence terms and extra cost for peaks
- how well left and right match each other, using signed bias plus weighted absolute/RMS mismatch from the solo drivers
- whether the tune avoids boosting into destructive cancellation nulls
- whether each filter is on a driver that is actually contributing at that frequency
- whether it avoids wasting filters, using unnecessary gain, or adding deep/narrow one-seat corrections
- whether corrections hold across optional centre/left-ear/right-ear system sums

The math layer also includes confidence and timing helpers for phase-valid measurements:

- coherence weighting, so low-trust bins count less
- band-limited phase-delay estimation around crossover regions
- gated impulse helpers that estimate how low a time window can be trusted

The optimizer first proves that the solo complex responses reproduce the measured together trace. It then tests polarity and delay, and searches an APF only if a meaningful residual remains. Invalid reference locks, weak predicted improvements, ambiguous polarity, and conflicting impulse evidence block automatic writes. Written AFPX candidates are linted so only the intended PEQ, `PM` polarity, delay values, and APF slots may change.

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
missing phase data keeps the conservative crossover-band PEQ veto. Routine phase
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

## How It Scores

It does not just chase a flat mono sum. Its scoring is designed to:

- improve tonal accuracy, especially through the vocal band
- improve left/right balance using the solo driver traces
- penalize boosting into destructive nulls
- penalize unsupported asymmetric EQ
- penalize unnecessary gain, wasted filters, and deep/narrow corrections
- write polarity/delay/APF changes only from gated crossover evidence, with warnings when confidence is not full
- leave crossovers alone

The optimizer is designed to:

- improve tonal accuracy
- improve left/right balance
- avoid boosting into destructive nulls
- keep PEQ conservative and phase writes independently auditable
- prefer fewer, wider, symmetric, shallower filters unless the solo measurements justify otherwise

Expected measurement files:

- `System Sum.txt`
- `Sub.txt`
- `Front L High.txt` or `Front L Tweeter.txt`
- `Front R High.txt` or `Front R Tweeter.txt`
- `Front L Low.txt` or `Front L Mid.txt`
- `Front R Low.txt` or `Front R Mid.txt`
- `Tweeters Together.txt` or `Both Tweeters.txt`
- `Mid Bass Together.txt` or `Both Mids.txt`

For a true front 3-way system, also provide separate mid and low branch measurements so the optimizer can detect and score `high + mid + low + sub` instead of the simpler 2-way front layout.

Expected tune file:

- `baseline.afpx`

## Main Files

- [_optimizer.py](./_optimizer.py): core scoring, prediction, AFPX writing, reporting
- [_optimizer_stream.py](./_optimizer_stream.py): constant-memory multi-worker optimizer
- [run_optimizer.ps1](./run_optimizer.ps1): one-command validate/run/merge/verify wrapper
- [_merge_stream_results.py](./_merge_stream_results.py): merges worker archives into final outputs
- [run_guided_stream_workers.ps1](./run_guided_stream_workers.ps1): launches long local runs
- [merge_guided_stream_results.ps1](./merge_guided_stream_results.ps1): safe merge wrapper
- [objective_module/afpx_objective.py](./objective_module/afpx_objective.py): independent scalar objective used by the optimizer
- [objective_module/_tunefit.py](./objective_module/_tunefit.py): DSP/math helpers used by the objective module
- [afpx.py](./afpx.py): generic `.afpx` inspector and channel-role helper
- [pct6.py](./pct6.py): beta `.pct6` decode / encode utility for no-password PC-Tool 6 saves
- [PCT6_SUPPORT.md](./PCT6_SUPPORT.md): caveats and safe usage notes for `.pct6`

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
python .\scripts\verify_written_tune.py "C:\path\to\baseline.afpx" ".\Optimizer_Run\_merged_top\family_balanced.afpx" --allow-delay --allow-apf
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
