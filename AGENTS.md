# AGENTS.md

## Repository Role

This repository is a local conservative optimiser for Helix / Audiotec Fischer AFPX tune files.

## Scope

- Optimise PEQ from REW magnitude measurements.
- Write delay/APF changes only when explicitly requested or when phase-valid crossover data supports them.
- Never change crossovers or polarity unless the user explicitly asks for that exact operation.
- Do not overwrite the baseline tune.
- Write candidate files only.
- Treat destructive summing/null regions as not EQ-fixable.
- Prefer fewer, wider, shallower filters.
- Prefer cuts over boosts.
- Penalise high-Q, high-gain, unnecessary, and unsupported asymmetric filters.

## Main Files

- `_optimizer.py`: main optimiser, scoring, AFPX writing, reports.
- `_optimizer_stream.py`: constant-memory worker optimiser.
- `_merge_stream_results.py`: merges worker archives.
- `_benchmark.py`: benchmark/check script.
- `objective_module/afpx_objective.py`: scalar objective.
- `objective_module/_tunefit.py`: DSP/math helpers.
- `afpx.py`: AFPX inspector/helper.
- `pct6.py`: beta PCT6 decode/encode helper.
- `PCT6_SUPPORT.md`: PCT6 caveats.
- `scripts/`: compact local summaries and output verification.

## Validation Rules

When changing optimiser scoring:
- Run a benchmark or an equivalent before/after score comparison.
- Explain the audible trade-off, not just the numeric change.

When changing AFPX/PCT6 writing:
- Decode the original.
- Write the new file.
- Decode the output.
- Verify only intended fields changed.

## Output Style

- Do not paste huge logs.
- Prefer generated JSON summaries over raw logs.
- Summarise commands, outputs, failures, and generated files.
- Inspect raw measurements, raw logs, or every candidate AFPX only when the compact summaries do not answer the question.
