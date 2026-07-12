# FILE_OWNERSHIP.md

Use this to narrow file reads before editing.

## Subsystem Routing

- optimiser/scoring changes:
  `_optimizer.py`, `_tunefit.py`, `objective_module/afpx_objective.py`, `objective_module/_tunefit.py`
- long-run worker flow:
  `_optimizer_stream.py`, `_merge_stream_results.py`, `run_guided_stream_workers.ps1`, `merge_guided_stream_results.ps1`
- AFPX parsing or write safety:
  `afpx.py`, `scripts/verify_written_tune.py`
- PCT6 decode/encode:
  `pct6.py`, `PCT6_SUPPORT.md`
- run summaries / compact review:
  `scripts/summarise_optimizer_run.py`, `scripts/summarise_candidate_filters.py`
- measurement inventory / expected inputs:
  `scripts/make_measurement_manifest.py`, `README.md`

## Do Not Pull In By Default

- historical optimiser output folders
- unrelated run artifacts
- every generated candidate file
- raw logs when a JSON summary exists
