# REPO_MAP.md

Use this file to navigate quickly before scanning the repo.
Current decisions and verified behavior live in `docs/ai_context/CURRENT_STATE.md`.

## Primary Entry Points

- `_optimizer.py`: main single-process optimiser, scoring, candidate generation, AFPX writing, reports.
- `_optimizer_stream.py`: long-run constant-memory optimiser for larger searches.
- `_merge_stream_results.py`: merges worker outputs into ranked merged results.
- `run_guided_stream_workers.ps1`: preferred launcher for long local stream runs.
- `merge_guided_stream_results.ps1`: preferred merge wrapper after worker runs complete.

## Core Logic Areas

- `objective_module/afpx_objective.py`: scalar objective used to score candidate tunes.
- `_tunefit.py`: DSP and prediction math used by the optimiser.
- `objective_module/_tunefit.py`: objective-layer DSP/math helpers.
- `afpx.py`: AFPX parsing, inspection, role/channel helpers.
- `pct6.py`: beta PCT6 decode/encode support for no-password files.

## Validation And Summaries

- `_benchmark.py`: benchmark/check script for scoring changes.
- `scripts/make_measurement_manifest.py`: fast measurement inventory and layout detection.
- `scripts/summarise_optimizer_run.py`: compact run summary, prefers `optimizer_summary.json`.
- `scripts/summarise_candidate_filters.py`: compact filter/risk summary for one candidate.
- `scripts/verify_written_tune.py`: verifies that written tune files changed only intended fields.

## Reference Docs

- `AGENTS.md`: repo behaviour, scope, validation rules, output style.
- `README.md`: user-facing usage and repo overview.
- `docs/AUDIT_2026-07-12.md`: evidence-backed audible/efficiency audit.
- `docs/ROADMAP.md`: ranked implementation order from that audit.
- `docs/TEST_COMMANDS.md`: exact run/validation commands.
- `docs/AUDIO_DSP_RULES.md`: DSP-domain guardrails and safe-change rules.
- `PCT6_SUPPORT.md`: PCT6 caveats and round-trip safety notes.

## Common Task Routing

- "Run the optimiser locally": start with `scripts/make_measurement_manifest.py`, then `run_guided_stream_workers.ps1` or `_optimizer.py` depending on run size.
- "Review a completed run": start with `scripts/summarise_optimizer_run.py`, then inspect the report or candidate files only if needed.
- "Check whether a candidate write is safe": use `scripts/verify_written_tune.py`.
- "Change scoring": inspect `_optimizer.py`, `_tunefit.py`, and `objective_module/afpx_objective.py`, then run `_benchmark.py`.
- "Change AFPX/PCT6 writing": inspect `afpx.py` or `pct6.py`, then round-trip and verify only intended fields changed.
- "Investigate phase/delay/APF behaviour": inspect `_optimizer.py`, `_tunefit.py`, and `docs/AUDIO_DSP_RULES.md` before editing.
