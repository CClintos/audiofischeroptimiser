# TESTING.md

Use this file to choose the right validation path quickly.

## First Choice

Read `docs/TEST_COMMANDS.md` for exact commands.

## Validation By Task Type

- Measurement inventory or run triage:
  use `scripts/make_measurement_manifest.py` or `scripts/summarise_optimizer_run.py`
- Scoring or candidate-selection changes:
  run `_benchmark.py` or an equivalent before/after comparison
- AFPX/PCT6 write-path changes:
  decode original, write output, decode output, verify only intended fields changed
- Candidate safety review:
  use `scripts/verify_written_tune.py`

## Current Reality

- There is no dedicated `tests/` tree yet in this repo.
- Treat the benchmark and verification scripts as the current regression layer.
