# Task 5 Report

## Result

Integrated automatic baseline rehabilitation into PEQ streaming Beam while preserving phase-only behavior.

## Files

- _optimizer_stream.py
  - Added plan-aware BeamEntry records.
  - Added CandidatePlan JSON serialization and audiofischer-stream-state-v2.
  - Added input fingerprints, stale-state rejection, exact resume, bounded rehabilitation budgeting, PEQ-only rehabilitation, guided Beam seeding, plan-preserving continuation/refinement, and CandidatePlan output writing.
- _optimizer.py
  - Added authoritative CandidatePlan component scoring from resolved full band sets.
  - Added slot-edit crossover phase conflict checks at score and write boundaries.
  - Updated family aliases to write CandidatePlans.
- _merge_stream_results.py
  - Loads, deduplicates, rescores, gates, ranks, and writes CandidatePlans.
  - Always includes the unchanged baseline and preserves rehabilitated candidates when no guided centres remain.
- tests/test_baseline_rehabilitation.py
  - Added PEQ-first stage, phase bypass, plan-aware Beam, v1 compatibility, v2 on-disk resume, stale fingerprint, merge preservation, wall-time, and slot-edit phase-veto coverage.

## Verification

- Focused pipeline/safety suite: 91 tests passed.
- Full suite: 188 tests passed.
- Python compileall: passed.
- Git diff check: passed.
- Expected existing warning: one test fixture uses a +4 dB boost above the app safety cap.

## Invariants

- Every CandidatePlan is resolved and scored against the original fixed measurement baseline.
- Phase mode does not run PEQ rehabilitation.
- Baseline, target, measurement, role-map, objective, or config changes invalidate resume state.
- Slot edits survive checkpoints, resume, worker merge, family selection, refinement, and AFPX writing.
- Attached phase writes retain coarse and complex crossover conflict protection.
## Review Fix Round 1

- Coordinate refinement now accepts and scores complete CandidatePlans, so rehabilitation slot edits remain active during every F/Q/G move.
- Rehabilitation interaction Beam now observes the same hard deadline as the census and cannot consume the guided-search budget.
- Merge recomputes the current worker fingerprint from baseline, target, measurement manifest/files, role map, objective weights, mode, profile, and rehabilitation config; missing or mismatched worker fingerprints stop the merge.
- Candidate predictions, tune scorecards, per-channel headroom, fixed-anchor audits, and response plots now resolve complete plans rather than appended groups alone.
- Timed guided continuation evaluates both unchanged-baseline and rehabilitated-baseline lineages, while non-Beam modes keep their existing single lineage.

### Review Regression Coverage

- Full-plan coordinate refinement.
- Rehabilitation Beam deadline enforcement.
- Missing and stale merge fingerprints.
- Numeric slot-edit prediction plus plan-aware scorecard/headroom/audit/plot routing.
- Dual-lineage guided continuation.

### Review Verification

- Focused review suite: 7 tests passed.
- Full suite: 195 tests passed.
- Python compileall: passed.
- Git diff check: passed.
