# Task 6 Report: Shared Multiworker Cache And Bounded Resources

## Commit

- `01ef474 Cache rehabilitation across optimizer workers`
- Base: `2b556d4`

## Implemented

- Added `scripts/build_rehabilitation_cache.py` to validate the session, build deterministic existing-PEQ rehabilitation once, and atomically publish `rehabilitation_cache.json`.
- Added schema and strict fingerprint validation covering baseline, target, resolved measurement files/manifest, objective settings, rehabilitation config, measurement-session audit, and phase-scoring context.
- Malformed, incomplete, unsupported, or mismatched caches fail clearly. They never trigger silent per-worker rescoring.
- Stored full-precision census, retained operation candidates, scored interaction candidates, config, and fingerprint inputs in the shared disk cache.
- PEQ workers load the same cache path; phase workers receive no rehabilitation cache and reject one if supplied directly.
- Workers retain only compact cache metadata and resume state after validation, releasing full census/candidate diagnostics from process memory.
- The PowerShell launcher runs one bounded cache-builder before PEQ workers, deducts its elapsed time from the remaining search time, preserves worker count/RAM controls and resume semantics, and aborts before worker launch on cache failure.
- Added concise stage output: `Preparing existing tune`, `Searching remaining response`, and `Merging candidates`.
- GUI commands provide one deterministic cache path under the run folder for PEQ and omit it for phase.

## TDD Evidence

RED failures were observed for:

- missing shared cache build/load API;
- missing PEQ GUI cache argument;
- worker retention of full cache diagnostics;
- phase-scoring context omitted from the fingerprint.

Each test passed after the corresponding minimal implementation.

## Verification

- `python -m unittest discover -s tests`: **202 passed**
- `python -m compileall`: passed using an external pycache directory
- `scripts/gui_preflight.py` against the 14 July RTA session: `valid=true`
- PowerShell parser checks for both launchers: passed
- `git diff --check`: passed
- Real builder/worker smoke test: cache schema valid, fingerprints matched, worker loaded the cached plan, and full cache diagnostics were absent from worker state
- Real launcher smoke test: one cache built before one PEQ worker, shared fingerprint matched, and remaining wall time was passed to search

## Scope

Only Task 6 shared-cache, launcher, backend-contract, progress, and regression-test changes were committed. Phase optimization behavior, objective math, fixed-baseline scoring, worker-count policy, and GUI RAM guard behavior were not changed.
## Fix Round 1

### Review Findings Resolved

- Cache fingerprints now include the calibration source path and file content plus the normalized calibration values actually loaded by the scorer.
- Cache fingerprints now include every file in the repeatability folder plus the derived repeatability/noise model used by the objective.
- Cache preparation accepts the shared stop file and checks it before and between preparation stages, during rehabilitation scoring, before writing, and before atomic publication.
- Cancelled preparation publishes no cache and removes its lock and temporary file. An exclusive PID lock serializes concurrent builders; abandoned locks and orphaned temporary files are reclaimed before rebuilding.
- Added a formal `preparing` runner phase. The GUI distinguishes PEQ existing-tune preparation from phase-diagnostic preparation, then transitions to search after workers launch.
- PEQ workers still receive one identical cache path. Phase workers receive no rehabilitation cache. Worker-count and RAM-stop controls are unchanged.

### Added Regression Coverage

- changed calibration file content and changed normalized calibration values;
- changed repeatability-folder content and changed derived repeatability model;
- cooperative cancellation before scoring and before publication;
- concurrent one-build atomic publication;
- abandoned lock and temporary-file recovery;
- shared launcher stop-file propagation;
- GUI preparing-to-searching progress;
- Windows paths containing spaces;
- PEQ/phase cache argument separation.

### Fix Verification

- Focused cache and GUI suite: **43 passed**.
- Full suite: **210 passed**.
- GUI preflight against the 14 July RTA fixture: `valid=true`.
- Real cancelled-builder smoke test with Windows paths containing spaces: clear cancellation, no published cache, no cache artifacts.
- Python compilation, PowerShell parser checks, and `git diff --check`: passed.
