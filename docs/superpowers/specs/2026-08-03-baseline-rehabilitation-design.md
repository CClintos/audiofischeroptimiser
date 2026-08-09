# Baseline Rehabilitation Design

Date: 2026-08-03

## Purpose

Make existing AFPX tunes a first-class part of the PEQ search space. Before the
optimizer adds new filters, it must be able to remove, re-centre, reshape, or
retune every eligible existing front-stage and subwoofer PEQ filter. This closes
the main coverage gap found when a manually improved tune required mostly edits
to existing filters rather than appended filters.

The change must preserve the optimizer's current strengths: one authoritative
objective, fixed-anchor comparisons, hard safety gates, bounded local resource
use, resumable runs, byte-preserving AFPX writes, and an unchanged-baseline
fallback.

## Scope

Included:

- Automatic baseline rehabilitation as the first stage of every PEQ and
  Retarget run.
- Front-stage and subwoofer PEQ filters supported by the detected layout.
- Existing-filter keep, remove, frequency, Q, and gain operations.
- Matched left/right operations and evidence-gated asymmetric operations.
- Interaction-aware combination of promising baseline edits.
- Conservative consolidation of overlapping filters.
- Clear operation-level reports and compact assistant summaries.
- Privacy-safe synthetic and golden regression tests.

Excluded:

- Rear-fill optimization.
- General output-level optimization. Existing protective attenuation and
  recommendation-only sub blend remain unchanged.
- Delay, polarity, APF, crossover, shelf, routing, or input-EQ changes.
- Relaxing any existing measurement, null, headroom, or phase safety gate.
- Committing personal AFPX files or measurement exports.

## Architectural Placement

Add a focused baseline-rehabilitation module rather than expanding
`_optimizer_stream.py` further. The module owns filter identity, generation of
baseline-edit candidates, interaction search, consolidation, and operation
reporting. It depends on existing AFPX decoding and the authoritative scoring
session but does not own either.

The normal PEQ flow becomes:

1. Decode and validate the measurement session and baseline AFPX.
2. Build stable references for eligible existing PEQ filter slots.
3. Run deterministic baseline rehabilitation.
4. Pass the best meaningful rehabilitated candidates into the existing guided
   problem census and Beam search.
5. Refine, merge, verify, and export finalists using existing AFPX linting.

Retarget uses the same flow with the selected target. Sweeps/Phase remains a
separate workflow and does not run baseline PEQ rehabilitation.

## Data Model

### Filter reference

An existing filter is addressed by stable file identity, not inferred later
from frequency proximity:

- output channel index;
- AFPX filter slot/index or another stable decoder-provided slot identifier;
- original filter type;
- original frequency, Q, and gain;
- detected driver role and passband;
- logical pairing identifier when the corresponding L/R filters match.

Stable identity is required because a legitimate re-centre can move farther
than the current frequency-matching tolerance. For example, moving 97 Hz to
100 Hz currently falls outside the `1/24`-octave edit threshold and can be
misclassified as an append.

### Filter operation

Every proposed baseline operation is explicit:

- `keep(ref)`;
- `remove(ref)`;
- `modify(ref, frequency, q, gain)`;
- `modify_pair(left_ref, right_ref, shared_settings)`;
- `merge(refs, replacement)` for the later consolidation pass.

Scoring resolves these operations into complete per-channel band sets. Writing
applies the same operations to the same slots. There must be one resolution
path shared by scoring, reporting, and AFPX writing.

### Rehabilitation candidate

A candidate contains:

- the ordered filter operations;
- full-precision named objective components;
- headroom and guardrail results;
- operation counts;
- deterministic signature;
- provenance, including the parent candidate and search stage;
- meaningful-improvement classification relative to the original baseline.

## Search Algorithm

### 1. Baseline census

Score the immutable baseline once. For every eligible existing filter, measure:

- its cascade contribution in its driver's passband;
- whether it overlaps a masked/null or crossover-protected region;
- whether it has a matching filter on the opposite side;
- its positive-gain and headroom effect;
- the objective change caused by removing it.

Unlike new-filter generation, this census must not require an exposed residual
peak at the filter's current frequency. An existing filter can create or hide
the very error being diagnosed. All operations are still judged by the full
objective and hard gates.

### 2. Single-filter coarse search

For every eligible filter, evaluate `keep` and `remove`, then search a bounded
coarse grid around the original settings. The grid must be wide enough to
reach materially different shapes, including examples such as Q 3.0 to Q 1.2
and gain -2.0 dB to -3.5 dB. It must not rely on repeated tiny coordinate moves.

Frequency search is logarithmic and bounded by the driver's validated
passband. Q and gain use hardware-valid ranges and the existing frequency-
dependent safety caps. The exact grid belongs in configuration so benchmarks
can compare it without changing algorithm code.

Matched filters on paired channels are searched symmetrically first. A
one-sided alternative is generated only when repeatable solo or spatial L/R
evidence justifies the direction and frequency region.

Retain each operation only when it is feasible. Rank operations by the exact
authoritative scalar while recording tonal, presence, peak, balance, headroom,
masked/unmasked, and parsimony effects separately.

### 3. Coarse-to-fine refinement

Refine promising coarse settings on AFPX/hardware-valid frequency, Q, and gain
steps. Refinement is bounded by evaluation count and deadline. It may use
larger moves when a coordinate remains monotonic, then reduce to final hardware
steps near a local optimum.

### 4. Interaction beam

Single-filter wins are not simply applied greedily. Build a compact beam over
the strongest operations so interactions can be evaluated. Keep the unchanged
baseline in every generation.

The archive remains diverse by retaining candidates that lead on:

- overall objective;
- tonal and presence accuracy;
- L/R balance;
- headroom;
- filter count;
- near-tied combinations that differ in affected channel or frequency region.

A bounded near-tie allowance may retain an individually neutral operation long
enough to test a supported interaction, but final export still requires a
meaningful full-candidate improvement.

Conflicting operations against the same slot cannot coexist. Symmetric and
one-sided variants of the same logical pair are mutually exclusive.

### 5. Consolidation

After rehabilitation, inspect overlapping same-channel PEQ bands. Fit a single
replacement to the combined complex biquad response where available, otherwise
to the magnitude cascade. Accept a merge only when:

- maximum cascade mismatch is at most 0.1 dB across the relevant passband;
- the authoritative objective does not worsen beyond numerical tolerance;
- no hard gate changes from pass to fail;
- headroom is no worse;
- filter count is reduced.

Consolidation is optional per candidate and cannot remove protected non-PEQ
filters or alter AFPX filter types outside the supported PEQ type.

### 6. Existing guided Beam

The original baseline and the best rehabilitated candidates seed the existing
guided Beam search. New-filter proposals remain measurement-derived. Search
budgets are shared so rehabilitation cannot consume the entire requested run;
short runs reserve enough time for both stages, while a no-problem census may
end without adding filters.

### 7. Final tie-breaking

Objective comparisons retain full precision. Measurement repeatability defines
whether a delta is meaningful. Among acoustically tied feasible candidates,
choose lexicographically:

1. fewer active filters;
2. less total positive cascade gain;
3. greater minimum headroom;
4. less unsupported L/R asymmetry;
5. smaller total parameter movement from the supplied baseline.

This tie-break does not let parsimony hide an acoustic regression.

## Driver Attribution

The existing power-share estimate remains useful for initial pruning but is not
authoritative. For each promising correction region, probe a small supported
filter on each eligible driver or logical pair and pass it through the normal
prediction model. Use the resulting system-response and balance deltas to rank
which output can actually correct the problem.

This sensitivity probe prevents filters being assigned to a driver that is
buried at the requested frequency and improves decisions near crossover
boundaries. With phase-valid compatible measurements it naturally uses the
validated complex predictor; otherwise it uses the existing magnitude-residual
fallback and keeps crossover protections active.

## Safety And Failure Behaviour

- The supplied AFPX is immutable and always remains an exportable fallback.
- Any unidentified, unsupported, or ambiguously mapped slot is skipped and
  reported; it is never guessed.
- Existing hard headroom feasibility remains non-tradeable.
- Null, nearfield, driver-authority, measurement-noise, crossover, and
  asymmetric-EQ gates remain active.
- Candidate-specific target re-anchoring is forbidden.
- Phase-valid prediction is used only after existing validation passes.
- If rehabilitation finds no meaningful improvement, normal Beam starts from
  the original baseline and the report says so plainly.
- Stop requests and memory limits checkpoint the current beam and leave valid
  completed candidates available.
- An AFPX write or round-trip lint failure rejects only that finalist and is
  included in the report.

## Reporting And GUI Behaviour

Reports compare three states when distinct:

- supplied baseline;
- rehabilitated baseline;
- final candidate after new-filter Beam search.

Each state shows objective components, fixed-anchor response, headroom, L/R
confidence, filter count, and meaningful-improvement status. An operation table
lists channel/driver, slot, operation type, old settings, new settings, reason,
and objective/component delta.

The compact `assistant_summary.json` adds:

- rehabilitation status and evaluation count;
- operation counts by type;
- accepted operations;
- representative rejected operations and gate reasons;
- original-to-rehabilitated and rehabilitated-to-final deltas;
- consolidation results;
- whether the final result exceeded repeatability.

Normal GUI use does not expose search-method choices. Baseline rehabilitation
is automatic and appears as progress text within PEQ and Retarget runs.

## Performance And Resource Bounds

- Cache immutable baseline cascades and per-slot transfer responses.
- Cache scores by deterministic operation signature.
- Share a wall-time budget between rehabilitation and guided Beam.
- Bound beam width, retained operations per slot, and interaction depth.
- Use the existing bounded worker and RAM-limit infrastructure.
- Do not repeat identical rehabilitation independently in every worker. Run the
  deterministic census once per session, cache it by input fingerprint, then
  distribute distinct interaction orders or refinements where useful.
- Keep routine terminal and GUI output compact; full diagnostics remain in the
  run folder.

## Verification Plan

### Unit tests

- A 97 Hz existing filter can be re-centred to 100 Hz as an edit to the same
  slot, never an append.
- An existing Q 3.0 filter can reach Q 1.2 without requiring many tiny passes.
- A sub filter can change from approximately 33 Hz, Q 2.0, -2.0 dB to a nearby
  Q/gain setting such as Q 2.2, -3.5 dB.
- Every existing eligible filter receives a removal test.
- Matched L/R filters remain symmetric without qualifying asymmetry evidence.
- Conflicting operations cannot coexist.
- Neutral acoustic candidates prefer fewer filters.
- Merging respects the 0.1 dB cascade limit.
- Slot mapping, scoring, reporting, and writing resolve the same operation.

### Synthetic invariants

- Flat-on-target baseline produces no changes.
- A deliberately harmful existing filter is removed.
- A misplaced broad filter is re-centred rather than duplicated.
- Two weak interacting edits are retained and selected when their combination
  is meaningfully better.
- A narrow boost into a null remains infeasible.
- A one-sided edit without L/R evidence remains infeasible.
- A candidate below the repeatability floor is reported as a tie.

### Integration and regression tests

- Add a privacy-safe fixture that reproduces the operation categories from the
  external v7-to-v10 audit: modify gain, re-centre frequency, change Q, remove,
  append, and leave unsupported operations untouched.
- Confirm AFPX round-trip byte preservation outside explicitly permitted PEQ
  slots.
- Compare current Beam and rehabilitation-plus-Beam at equal seed and wall
  time. The new flow must never lose the unchanged baseline and must reach the
  synthetic known solution.
- Verify checkpoint/resume, RAM stopping, and deterministic cache reuse.
- Run the full existing P0, P1, P2, objective-hardening, GUI backend, and AFPX
  tests before release.

## Success Criteria

The implementation is complete when:

- every eligible existing front/sub PEQ filter is genuinely reachable by
  keep/remove/F/Q/G operations;
- a frequency move beyond the old proximity tolerance remains an edit to the
  original slot;
- the known synthetic re-centre, Q/gain, and removal examples are found;
- combined baseline edits are evaluated rather than applied only greedily;
- the unchanged baseline always survives;
- no existing safety gate is weakened;
- reports clearly distinguish rehabilitation from newly added corrections;
- equal-wall-time benchmarks show broader solution coverage without an
  unacceptable reduction in candidate throughput;
- no private user measurements or tune files enter the repository.
