# AudioFischer Optimiser

This repo is a local AFPX tuning tool for Helix / Audiotec Fischer DSP systems.

It takes your baseline `.afpx` tune plus REW measurement exports, tries many possible PEQ combinations, scores them, and gives you back ranked AFPX tune candidates to test.

The goal is to improve the tune conservatively. PEQ is always handled from magnitude data; delay/APF edits are written when phase-valid crossover measurements are present, and the report warns when the phase confidence is not full.

It also includes beta `.pct6` container support for newer DSP PC-Tool 6 tunes, so the repo can inspect or round-trip those files too. That path is less proven than `.afpx`, and should be treated as a careful utility rather than a blindly trusted writer.

Its scoring system is built to reward tunes that are more likely to sound better, not just look flatter on one graph.

It scores candidates by:

- how close the full system response is to the target, with extra weight through the vocal and midrange region
- how well left and right match each other, using the individual driver measurements
- whether the tune avoids boosting into destructive cancellation nulls
- whether each filter is on a driver that is actually contributing at that frequency
- whether it avoids wasting filters, using unnecessary gain, or adding deep/narrow one-seat corrections

The math layer also includes optional confidence and timing helpers for future phase-valid measurements:

- coherence weighting, so low-trust bins count less
- band-limited phase-delay estimation around crossover regions
- gated impulse helpers that estimate how low a time window can be trusted

Those helpers are now used in candidate reports and output writing when the measurement exports contain enough data. The optimizer inspects crossover bands such as sub-to-midbass and mid-to-tweeter for delay, polarity, phase stability, summation quality, and acoustic-sum agreement. It writes the delay/APF changes alongside the PEQ candidates, then explains confidence and re-measure checks in the report.

Supported REW text export rows:

- `freq spl`
- `freq spl phase`
- `freq spl phase coherence`
- `freq spl phase coherence position_id`

If you know the impulse/window gate length, pass `-GateMs` to the PowerShell launcher or `--gate-ms` to the Python scripts. The report will warn when a gated response should not be trusted below its lowest valid frequency.

It is meant to be used through Claude or Codex:

1. Drag in your REW measurement text exports.
2. Drag in your baseline `.afpx` tune.
3. Optionally drag in your target curve text file.
4. Ask Claude or Codex to use this repo as a local AFPX optimizer and run it.

Suggested prompt:

```text
Use this repo as a local AFPX tuning tool.

I have attached:
- my REW measurement text exports
- my baseline .afpx tune
- optionally my target curve

Please verify the files, run the optimizer locally, merge the results, and give me the best AFPX candidates with a short summary of what improved.
Delay/APF changes may be written when phase-valid measurements support them. Crossovers and polarity are still left alone.
```

## How It Scores

It does not just chase a flat mono sum. Its scoring is designed to:

- improve tonal accuracy, especially through the vocal band
- improve left/right balance using the solo driver traces
- penalize boosting into destructive nulls
- penalize unsupported asymmetric EQ
- penalize unnecessary gain, wasted filters, and deep/narrow corrections
- write delay/APF changes only from crossover-band phase evidence, with warnings when confidence is not full
- leave crossovers and polarity alone

The optimizer is designed to:

- improve tonal accuracy
- improve left/right balance
- avoid boosting into destructive nulls
- keep the tune conservative and PEQ-only
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
- [_merge_stream_results.py](./_merge_stream_results.py): merges worker archives into final outputs
- [run_guided_stream_workers.ps1](./run_guided_stream_workers.ps1): launches long local runs
- [merge_guided_stream_results.ps1](./merge_guided_stream_results.ps1): safe merge wrapper
- [objective_module/afpx_objective.py](./objective_module/afpx_objective.py): independent scalar objective used by the optimizer
- [objective_module/_tunefit.py](./objective_module/_tunefit.py): DSP/math helpers used by the objective module
- [afpx.py](./afpx.py): generic `.afpx` inspector and channel-role helper
- [pct6.py](./pct6.py): beta `.pct6` decode / encode utility for no-password PC-Tool 6 saves
- [PCT6_SUPPORT.md](./PCT6_SUPPORT.md): caveats and safe usage notes for `.pct6`

## Compact Local Summaries

These scripts are for Claude/Codex efficiency. They produce small JSON files so an assistant does not need to read raw logs, every candidate, or full measurement exports.

- Every optimiser run now writes `optimizer_summary.json` beside `optimizer_report.md` and `optimizer_results.csv`.
- [scripts/make_measurement_manifest.py](./scripts/make_measurement_manifest.py): checks which expected measurements exist, detects 2-way/3-way layout, and notes phase/coherence columns.
- [scripts/summarise_optimizer_run.py](./scripts/summarise_optimizer_run.py): summarizes an optimizer output folder, preferring `optimizer_summary.json` with CSV fallback.
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
- It can edit delay tags when phase-valid crossover data is present.
- It does not change crossovers.
- It can add conservative APF filters when the phase report shows residual crossover uncertainty.
- It treats destructive summing regions as not EQ-fixable.

For `.pct6`, the repo currently provides careful container decode / encode support and inspection helpers. AFPX writing is still the primary automated output path.

That means it is primarily a PEQ optimizer, with conservative delay/APF writes only when the measurement exports contain usable phase evidence.
