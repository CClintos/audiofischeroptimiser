# PROJECT_MAP.md

Use this before opening code when the task is not already tightly scoped.

## Purpose

`audiofischeroptimiser` is a local conservative optimiser for Helix / Audiotec Fischer tune files.

It is built to:
- optimise PEQ from REW-style measurement exports
- preserve safe AFPX/PCT6 read-write behaviour
- make delay/APF changes only when phase-valid crossover evidence supports them

## Main Domains

- measurement import/parsing
- target curve handling
- optimiser/scoring
- filter generation
- AFPX writing
- PCT6 decode/encode support
- run summaries and write verification

## Main Areas

- `_optimizer.py`: main optimiser, reporting, candidate writing
- `_optimizer_stream.py`: long-running constant-memory optimiser
- `_merge_stream_results.py`: merge/rank worker outputs
- `_tunefit.py`: DSP/prediction math
- `objective_module/afpx_objective.py`: scoring objective
- `afpx.py`: AFPX parsing/helper logic
- `pct6.py`: beta no-password PCT6 decode/encode support
- `scripts/`: compact summaries and output verification

## Read Order

1. `AGENTS.md`
2. `REPO_MAP.md`
3. This file
4. Only then open the smallest relevant code files

## Do Not Read By Default

- raw optimiser logs
- every candidate AFPX
- large output folders
- unrelated run artifacts

Use the compact summary scripts first.
