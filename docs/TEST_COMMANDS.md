# TEST_COMMANDS.md

Use these commands instead of guessing run/validation steps.

## Environment

Run from the repo root in PowerShell.

## Fast Inventory

```powershell
py -3 .\scripts\make_measurement_manifest.py "C:\path\to\measurements"
```

Use this first to confirm expected measurements, 2-way vs 3-way layout, and whether phase/coherence columns exist.

## PCT6 Self-Test

```powershell
py -3 .\pct6.py selftest
```

Use before editing `pct6.py` or when validating a fresh environment.

## Single-Process Optimiser

Use when the task is a smaller direct run and not a long multi-worker search.

```powershell
py -3 .\_optimizer.py --help
```

Inspect available arguments before running; avoid inventing command flags from memory.

## Stream Optimiser

Use for longer local searches.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_guided_stream_workers.ps1
```

After workers finish:

```powershell
powershell -ExecutionPolicy Bypass -File .\merge_guided_stream_results.ps1
```

## One-Command Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_optimizer.ps1 -DataRoot "C:\path\to\measurements"
```

This validates inputs, chooses a bounded worker count, reuses the fingerprinted
phase audit, runs/resumes, merges, verifies family AFPX files, and prints only
the final `assistant_summary.json` path. Beam is the default proposal.

## Benchmark For Scoring Changes

```powershell
py -3 .\_benchmark.py
```

Required when changing scoring or candidate selection logic. Record before/after behaviour, not just whether it ran.

For equal-budget architecture comparisons:

```powershell
py -3 .\scripts\benchmark_search_methods.py --data-root "C:\path\to\measurements" --baseline "C:\path\to\baseline.afpx" --target ".\ResoNix Target Curve 2026.txt"
```

## Regression Suite

```powershell
py -3 -m unittest discover -s tests -v
```

This includes objective invariants, measurement-session gates, PEQ/phase
conflict protection, AFPX write safety, and the modern TXT/AFPX golden fixture.

## Summarise Completed Runs

```powershell
py -3 .\scripts\summarise_optimizer_run.py ".\Optimizer_Run\_merged_top"
```

Start here before opening large reports or CSVs.
The helper reads `assistant_summary.json` first.

## Summarise One Candidate

```powershell
py -3 .\scripts\summarise_candidate_filters.py ".\Optimizer_Run\_merged_top\family_balanced.afpx" --baseline "C:\path\to\baseline.afpx"
```

Use to inspect added filters and risk flags without reading the full file diff manually.

## Verify Written Tune Safety

```powershell
py -3 .\scripts\verify_written_tune.py "C:\path\to\baseline.afpx" ".\Optimizer_Run\_merged_top\family_balanced.afpx" --allow-delay --allow-apf --allow-polarity
```

Use after AFPX writes. If the task should be PEQ-only, omit all three allowances.

## PCT6 Round-Trip Safety

```powershell
py -3 .\pct6.py decode .\baseline.pct6
```

When editing PCT6 handling, also verify round-trip equality with the byte-preserving workflow described in `PCT6_SUPPORT.md`.
