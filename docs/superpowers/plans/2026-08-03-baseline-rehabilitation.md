# Baseline Rehabilitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an automatic, deterministic first stage that can safely keep, remove, re-centre, reshape, and retune every eligible existing front/sub PEQ filter before the normal measured-problem Beam search.

**Architecture:** Introduce stable AFPX slot references and an explicit `CandidatePlan` containing baseline slot edits plus ordinary Beam groups. Resolve and score the complete band sets against the original measured baseline, then use a bounded coarse-to-fine rehabilitation beam before the existing guided Beam. Scoring, reporting, and writing share one resolver so a re-centre can never become an accidental append.

**Tech Stack:** Python 3.11+, NumPy, existing AFPX regex/codec helpers, existing authoritative `objective_module.afpx_objective`, `unittest`, PowerShell worker launcher, PySide6 GUI.

## Global Constraints

- The supplied AFPX is immutable and must always remain selectable.
- Scope is existing front-stage and subwoofer PEQ plus normal PEQ additions; rear fill is excluded.
- Do not add general output-level optimization.
- Do not change delay, polarity, APF, crossover, shelf, routing, or input EQ.
- Use the existing authoritative scalar and all named components; do not add a second tonal objective.
- Candidate target anchoring is computed once from the supplied baseline and never recomputed per candidate.
- Existing hard headroom, null, driver-authority, crossover, measurement-noise, spatial, and asymmetric-EQ gates remain authoritative.
- Filter consolidation requires at most 0.1 dB maximum cascade mismatch, no objective regression, no headroom regression, and fewer filters.
- A delta below measurement repeatability is a tie; tie order is fewer filters, less positive gain, more headroom, less unsupported asymmetry, then less parameter movement.
- No private measurements, AFPX files, or personal absolute paths may be committed.
- Preserve the current unrelated GUI worktree changes; stage only files belonging to each task.

## File Structure

- Create `baseline_rehabilitation.py`: immutable slot/edit/candidate types, coarse candidate generation, deterministic interaction beam, tie-breaking, cache serialization, and consolidation.
- Modify `_make_v3.py`: enumerate active PEQ slots and apply verified edits by slot identity while keeping legacy tuple-based APIs compatible.
- Modify `_optimizer.py`: resolve `CandidatePlan` to complete band sets, expose direct band-set scoring, write planned candidates, and produce operation reports.
- Modify `_optimizer_stream.py`: run/load rehabilitation before guided candidate generation, seed Beam with candidate plans, checkpoint plan state, and divide the wall-time budget.
- Modify `_merge_stream_results.py`: merge and rescore candidate plans, preserve baseline fallback, and export operation-aware finalists.
- Modify `run_guided_stream_workers.ps1`: create one shared fingerprinted rehabilitation cache before workers start.
- Modify `optimizer_gui/backend.py` and `optimizer_gui/window.py`: pass the shared-cache path and show rehabilitation progress/results without adding a search-method control.
- Modify `optimizer_gui/reporting.py`: render baseline/rehabilitated/final comparisons and operation tables.
- Create `tests/test_baseline_rehabilitation.py`: unit, synthetic, resolver, and interaction tests.
- Modify `tests/test_objective_hardening.py`, `tests/test_p1_features.py`, `tests/test_gui_backend.py`: scoring, integration, and GUI regressions.
- Create `scripts/benchmark_rehabilitation.py`: privacy-safe equal-budget coverage benchmark.
- Modify `README.md`, `CHANGELOG.md`, and `docs/ai_context/CURRENT_STATE.md`: user-facing behavior and compact AI context.

---

### Task 1: Stable AFPX Slot Identity And Slot Writes

**Files:**
- Create: `baseline_rehabilitation.py`
- Modify: `_make_v3.py:27-120,300-365`
- Test: `tests/test_baseline_rehabilitation.py`

**Interfaces:**
- Produces: `FilterRef`, `SlotEdit`, `CandidatePlan`, `active_peq_slot_refs(xml, channel_roles)`, and `apply_slot_edits(xml, edits, protected_channels=())`.
- Consumes later: `_optimizer.resolve_candidate_plan()` and `_optimizer.write_candidate_plan()` use these exact types.

- [ ] **Step 1: Write failing slot-identity tests**

```python
class SlotIdentityTests(unittest.TestCase):
    def test_recentre_beyond_old_frequency_tolerance_keeps_same_slot(self):
        xml = fixture_afpx_xml({2: [(7, 97.0, 3.0, -1.5)]})
        refs = rehab.active_peq_slot_refs(xml, {2: "FL Low"})
        edit = rehab.SlotEdit.modify(refs[0], (100.0, 1.2, -1.5))
        written = rehab.apply_slot_edits(xml, (edit,))
        before_slots = filter_slots(xml, channel=2)
        after_slots = filter_slots(written, channel=2)
        self.assertEqual(after_slots[7]["F"], "100.00")
        self.assertEqual(after_slots[7]["Q"], "1.2")
        self.assertEqual(before_slots[:7] + before_slots[8:], after_slots[:7] + after_slots[8:])

    def test_remove_frees_exact_duplicate_slot_without_frequency_guessing(self):
        xml = fixture_afpx_xml({2: [(4, 100.0, 1.0, -2.0), (9, 100.0, 1.0, -2.0)]})
        refs = rehab.active_peq_slot_refs(xml, {2: "FL Low"})
        written = rehab.apply_slot_edits(xml, (rehab.SlotEdit.remove(refs[1]),))
        slots = filter_slots(written, channel=2)
        self.assertEqual(slots[4]["T"], "17")
        self.assertEqual(slots[9]["T"], "1")
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.SlotIdentityTests -v`

Expected: import or attribute failures because the new module and slot APIs do not exist.

- [ ] **Step 3: Add immutable operation types**

```python
Band = tuple[float, float, float]

@dataclass(frozen=True)
class FilterRef:
    channel: int
    slot: int
    role: str
    filter_type: str
    original: Band
    pair_key: str | None = None

@dataclass(frozen=True)
class SlotEdit:
    ref: FilterRef
    replacement: Band | None

    @classmethod
    def modify(cls, ref: FilterRef, replacement: Band) -> "SlotEdit":
        return cls(ref=ref, replacement=replacement)

    @classmethod
    def remove(cls, ref: FilterRef) -> "SlotEdit":
        return cls(ref=ref, replacement=None)

@dataclass(frozen=True)
class CandidatePlan:
    slot_edits: tuple[SlotEdit, ...] = ()
    groups: tuple[tuple[str, tuple[Band, ...]], ...] = ()
```

- [ ] **Step 4: Implement slot enumeration and guarded slot writes**

Enumerate `<Fil>` tags by their ordinal inside each `<OC>`. Accept only active `T="17"` PEQ filters. Before modifying, verify that channel, slot, type, F, Q, and G still equal the `FilterRef`; otherwise raise `ValueError("AFPX slot changed since rehabilitation census")`. Modify only F/Q/G for replacement and only T/G for removal.

- [ ] **Step 5: Run slot and existing writer tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.SlotIdentityTests tests.test_objective_hardening -v`

Expected: PASS, including duplicate-frequency slot selection and byte preservation outside the addressed slot.

- [ ] **Step 6: Commit the slot identity foundation**

```powershell
git add baseline_rehabilitation.py _make_v3.py tests/test_baseline_rehabilitation.py
git commit -m "Add stable AFPX filter slot operations"
```

---

### Task 2: One Resolver For Scoring, Reporting, And Writing

**Files:**
- Modify: `baseline_rehabilitation.py`
- Modify: `_optimizer.py:960-1090,2198-2290,2395-2470`
- Modify: `_make_v3.py:330-365`
- Test: `tests/test_baseline_rehabilitation.py`
- Test: `tests/test_objective_hardening.py`

**Interfaces:**
- Consumes: `CandidatePlan`, `SlotEdit`, and `FilterRef` from Task 1.
- Produces: `resolve_candidate_plan(plan) -> ResolvedPlan`, `candidate_plan_signature(plan)`, `make_band_set_component_scorer(...)`, and `write_candidate_plan(base_xml, path, plan, phase_plan=None)`.

- [ ] **Step 1: Write failing resolver-equivalence tests**

```python
def test_plan_resolution_is_identical_for_score_report_and_write(self):
    plan = CandidatePlan(
        slot_edits=(SlotEdit.modify(self.ref_97, (100.0, 1.2, -1.5)),),
        groups=freeze_groups({"low_sym": [(184.0, 0.63, -4.0)]}),
    )
    resolved = optimizer.resolve_candidate_plan(plan)
    score = self.score_band_sets(resolved.band_sets)
    written = optimizer.write_candidate_plan(self.xml, self.out, plan)
    parsed = bands_from_afpx(self.out)
    self.assertEqual(parsed, resolved.band_sets)
    self.assertEqual(written["operation_signature"], resolved.signature)
    self.assertEqual(score["filter_count"], sum(map(len, parsed)))

def test_unchanged_plan_resolves_to_exact_baseline(self):
    resolved = optimizer.resolve_candidate_plan(CandidatePlan())
    self.assertEqual(resolved.band_sets, optimizer.baseline_band_sets())
```

- [ ] **Step 2: Run resolver tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.PlanResolverTests -v`

Expected: FAIL because `ResolvedPlan` and the shared resolver do not exist.

- [ ] **Step 3: Implement `ResolvedPlan` and optional baseline resolution**

```python
@dataclass(frozen=True)
class ResolvedPlan:
    band_sets: tuple[tuple[Band, ...], ...]
    slot_edits: tuple[SlotEdit, ...]
    group_actions: tuple[tuple[int, tuple[tuple[str, Band | None, Band | None], ...]], ...]
    signature: tuple
```

Refactor `_resolve_group_bands(groups)` to accept `starting_band_sets`. `resolve_candidate_plan` applies slot edits to baseline band sets first, then resolves ordinary groups against those edited sets. Reject duplicate edits against one slot and reject group actions that try to edit/remove a slot already removed by rehabilitation.

- [ ] **Step 4: Expose direct complete-band-set scoring**

Refactor only the external-objective branch of `make_component_scorer` so normalization of named components is shared:

```python
band_set_score = make_band_set_component_scorer(freqs, traces, target, filter_cost_scale, worst_weight)

def group_score(groups):
    return band_set_score(groups_to_band_sets(groups))
```

The new scorer must call `AFPX_OBJECTIVE.score_bands(band_sets)` directly and preserve full precision. Do not reconstruct another objective.

- [ ] **Step 5: Implement plan writing through the shared resolution**

Apply slot edits by identity first, then apply the resolver's ordinary append/edit/remove actions against the resulting XML. Run the existing `afpx_roundtrip_lint`; include permitted edited/removed slots in the lint audit rather than broadening allowed change categories.

- [ ] **Step 6: Run focused and P0 tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.PlanResolverTests tests.test_p0_integrity tests.test_objective_hardening -v`

Expected: PASS with scoring/writing equivalence and unchanged legacy group behavior.

- [ ] **Step 7: Commit the shared resolver**

```powershell
git add baseline_rehabilitation.py _optimizer.py _make_v3.py tests/test_baseline_rehabilitation.py tests/test_objective_hardening.py
git commit -m "Resolve rehabilitation plans consistently"
```

---

### Task 3: Exhaustive Census And Coarse-To-Fine Existing-Filter Search

**Files:**
- Modify: `baseline_rehabilitation.py`
- Modify: `_optimizer.py:300-430`
- Test: `tests/test_baseline_rehabilitation.py`

**Interfaces:**
- Consumes: `FilterRef`, `CandidatePlan`, `resolve_candidate_plan`, direct band-set scorer, detected roles/passbands, and existing measurement guardrails.
- Produces: `RehabilitationConfig`, `FilterCensusRow`, `OperationCandidate`, `build_filter_census(...)`, and `search_filter_operations(...)`.

- [ ] **Step 1: Write failing exhaustive-coverage tests**

```python
def test_every_eligible_existing_filter_gets_removal_trial(self):
    result = rehab.build_filter_census(self.refs, self.score_plan, self.config)
    self.assertEqual({row.ref for row in result}, set(self.refs))
    self.assertTrue(all(row.removal_components is not None for row in result))

def test_coarse_search_reaches_known_recentre_and_q_change(self):
    candidates = rehab.search_filter_operations(self.ref_97_q3, self.score_plan, self.config)
    settings = {item.edit.replacement for item in candidates if item.edit.replacement}
    self.assertIn((100.0, 1.2, -1.5), settings)

def test_sub_search_reaches_gain_and_q_change(self):
    candidates = rehab.search_filter_operations(self.ref_sub_33, self.score_plan, self.config)
    self.assertTrue(any(
        abs(item.edit.replacement[0] - 33.0) <= 0.1
        and abs(item.edit.replacement[1] - 2.2) <= 0.01
        and item.edit.replacement[2] <= -3.25
        for item in candidates if item.edit.replacement
    ))
```

- [ ] **Step 2: Verify coverage tests fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.FilterSearchTests -v`

Expected: FAIL because census and multiscale search are absent.

- [ ] **Step 3: Implement explicit bounded search configuration**

```python
@dataclass(frozen=True)
class RehabilitationConfig:
    frequency_octaves: tuple[float, ...] = (-1/3, -1/6, -1/12, -1/24, 0.0, 1/24, 1/12, 1/6, 1/3)
    q_multipliers: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.25, 1.6)
    gain_offsets_db: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    retained_per_slot: int = 6
    refinement_passes: int = 4
```

Always include hardware-rounded values near Q `0.5, 0.7, 1.0, 1.2, 1.4, 2.0, 2.2, 2.5` when they fall within the role's cap. Always test removal separately. Bound frequency to the role passband and preserve all existing boost/cut restrictions.

- [ ] **Step 4: Add paired-filter detection and symmetry-first search**

Pair opposite-side filters when role, frequency, Q, and gain match within hardware precision. Generate paired edits as one operation. Generate one-sided variants only through an injected `asymmetry_eligible(ref, replacement) -> bool` callback backed by existing L/R evidence and guardrails.

- [ ] **Step 5: Implement sensitivity-based driver attribution**

For each retained correction region, score one small safe probe on each eligible role/pair through `resolve_candidate_plan` and the authoritative scorer. Store system, balance, and headroom deltas. Use the probe only to rank ownership; the full candidate score remains authoritative.

- [ ] **Step 6: Add monotonic coarse-to-fine refinement**

Starting from retained coarse candidates, try larger hardware-rounded moves while objective improvement remains monotonic, then reduce to frequency `1/96` octave, Q `0.1`, and gain `0.25 dB` moves. Stop on no improvement, deadline, or evaluation cap.

- [ ] **Step 7: Run filter-search and existing P1 tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.FilterSearchTests tests.test_p1_features -v`

Expected: PASS; the 97-to-100, Q 3.0-to-1.2, sub Q/gain, removal, symmetry, and attribution tests all succeed.

- [ ] **Step 8: Commit exhaustive baseline search**

```powershell
git add baseline_rehabilitation.py _optimizer.py tests/test_baseline_rehabilitation.py
git commit -m "Search every existing PEQ operation"
```

---

### Task 4: Interaction Beam, Meaningful Ties, And Consolidation

**Files:**
- Modify: `baseline_rehabilitation.py`
- Modify: `objective_module/afpx_objective.py:900-975`
- Test: `tests/test_baseline_rehabilitation.py`
- Test: `tests/test_p0_integrity.py`

**Interfaces:**
- Consumes: ranked `OperationCandidate` objects and authoritative plan scorer.
- Produces: `rehabilitation_beam(...) -> RehabilitationResult`, `compare_candidates(...)`, and `consolidate_candidate(...)`.

- [ ] **Step 1: Write failing interaction and tie tests**

```python
def test_interaction_beam_keeps_two_edits_that_win_only_together(self):
    result = rehab.rehabilitation_beam(
        self.baseline_plan, self.synergistic_operations, self.score_plan,
        beam_width=16, max_depth=4,
    )
    self.assertEqual(set(result.best.slot_edits), set(self.synergistic_operations))

def test_acoustic_tie_prefers_removal(self):
    winner = rehab.compare_candidates(
        self.same_acoustics_79_filters,
        self.same_acoustics_78_filters,
        repeatability_db=0.05,
    )
    self.assertIs(winner, self.same_acoustics_78_filters)

def test_merge_rejected_above_point_one_db(self):
    result = rehab.consolidate_candidate(self.overlapping_plan, self.score_plan)
    self.assertFalse(result.accepted)
    self.assertGreater(result.max_cascade_error_db, 0.1)
```

- [ ] **Step 2: Verify beam tests fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.InteractionBeamTests -v`

Expected: FAIL because the beam and lexicographic tie comparator do not exist.

- [ ] **Step 3: Implement conflict-safe operation combination**

Use `(channel, slot)` as the conflict key. A paired operation reserves both keys. Build generations from the unchanged baseline, expanding only non-conflicting operations. Cache every full-precision score by deterministic plan signature.

- [ ] **Step 4: Implement diversity retention**

Retain objective leaders plus leaders for tonal, presence, balance, headroom, and filter count. Fill remaining beam capacity by objective while ensuring different channel/frequency-region signatures. Permit a partial candidate within the measured repeatability allowance to survive one generation; never export it unless the complete candidate meaningfully wins.

- [ ] **Step 5: Implement lexicographic tie comparison**

```python
def tie_key(candidate):
    c = candidate.components
    return (
        c["filter_count"],
        c["positive_gain_penalty_db"],
        -candidate.minimum_headroom_db,
        c["asymmetric_eq_penalty_db"],
        candidate.parameter_movement,
    )
```

Use the installed empirical repeatability model when supplied; otherwise use the existing frequency-aware model summarized into the candidate's affected bands. A scalar delta alone must not override the meaningful-tie classification.

- [ ] **Step 6: Implement conservative same-channel consolidation**

Fit one replacement to each overlapping pair's combined `cascade_complex` response using bounded SciPy least squares or the existing local fitting utilities. Verify maximum passband magnitude mismatch, full objective, headroom, hard gates, and filter count before accepting.

- [ ] **Step 7: Run interaction, simplification, and integrity tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.InteractionBeamTests tests.test_p0_integrity -v`

Expected: PASS; interaction wins are found, acoustic ties simplify, and all existing hard-gate invariants remain intact.

- [ ] **Step 8: Commit interaction search and consolidation**

```powershell
git add baseline_rehabilitation.py objective_module/afpx_objective.py tests/test_baseline_rehabilitation.py tests/test_p0_integrity.py
git commit -m "Combine and simplify baseline PEQ edits"
```

---

### Task 5: Integrate Rehabilitation Into Streaming Beam And Resume State

**Files:**
- Modify: `_optimizer_stream.py:724-1060,1343-1578`
- Modify: `_optimizer.py:2198-2470`
- Modify: `_merge_stream_results.py:1-220`
- Test: `tests/test_baseline_rehabilitation.py`
- Test: `tests/test_objective_hardening.py`

**Interfaces:**
- Consumes: `RehabilitationResult`, `CandidatePlan`, plan resolver/scorer/writer.
- Produces: plan-aware Beam entries, JSON state schema `audiofischer-stream-state-v2`, and merged operation-aware AFPX candidates.

- [ ] **Step 1: Write failing pipeline tests**

```python
def test_peq_runs_rehabilitation_before_guided_beam(self):
    result = run_stream_fixture(mode="peq", seconds=2)
    self.assertEqual(result.events[1]["phase"], "baseline_rehabilitation_complete")
    self.assertGreater(result.rehabilitation["evaluations"], 0)

def test_phase_mode_does_not_run_peq_rehabilitation(self):
    result = run_stream_fixture(mode="phase", seconds=2)
    self.assertEqual(result.rehabilitation["status"], "not_applicable")

def test_resume_round_trips_candidate_plan(self):
    loaded = stream.load_state(stream.save_state(self.plan_with_slot_edit))
    self.assertEqual(loaded.best_plan, self.plan_with_slot_edit)
```

- [ ] **Step 2: Verify pipeline tests fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.StreamIntegrationTests -v`

Expected: FAIL because streaming state stores groups only and has no rehabilitation phase.

- [ ] **Step 3: Make Beam entries plan-aware**

Replace `(objective, group_signature, groups)` internally with a small dataclass containing objective, plan signature, `CandidatePlan`, and components. Preserve JSON compatibility when loading v1 group-only states by wrapping groups in `CandidatePlan(slot_edits=(), groups=...)`.

- [ ] **Step 4: Divide the wall-time budget**

Use a bounded policy:

```python
rehab_seconds = min(max(total_seconds * 0.25, 5.0), 180.0)
guided_deadline = run_start + total_seconds
```

For runs shorter than 20 seconds, cap rehabilitation by evaluation count and reserve at least half the budget for guided Beam. If no eligible guided centres remain after rehabilitation, retain the rehabilitated and original candidates and finish without random filler.

- [ ] **Step 5: Seed guided Beam with original and rehabilitated plans**

Adapt deterministic combination expansion so each partial plan keeps its slot edits while group options are added. Every score resolves the complete plan against the original measured baseline. Do not rebase the objective or measurement traces onto a temporary AFPX.

- [ ] **Step 6: Persist and resume rehabilitation state**

Store config, input fingerprint, census, retained operation candidates, candidate plans, evaluation count, and completion status. Reject stale cache/state when baseline, target, role map, measurement manifest, objective weights, or rehabilitation config changes.

- [ ] **Step 7: Merge and write candidate plans**

Update worker loading, signature deduplication, rescoring, census gating, family selection, and `write_candidate_plan`. Always append the unchanged baseline plan before ranking. Preserve phase-conflict checks for any future CLI combination, while GUI PEQ remains PEQ-only.

- [ ] **Step 8: Run streaming and merge regression tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.StreamIntegrationTests tests.test_objective_hardening tests.test_p0_integrity -v`

Expected: PASS for PEQ-first-stage behavior, phase bypass, v1 state compatibility, resume, merge, and baseline fallback.

- [ ] **Step 9: Commit pipeline integration**

```powershell
git add _optimizer.py _optimizer_stream.py _merge_stream_results.py baseline_rehabilitation.py tests/test_baseline_rehabilitation.py tests/test_objective_hardening.py tests/test_p0_integrity.py
git commit -m "Run baseline rehabilitation before Beam search"
```

---

### Task 6: Shared Multiworker Cache And Bounded Resources

**Files:**
- Create: `scripts/build_rehabilitation_cache.py`
- Modify: `run_guided_stream_workers.ps1`
- Modify: `_optimizer_stream.py:1343-1480`
- Modify: `optimizer_gui/backend.py`
- Test: `tests/test_baseline_rehabilitation.py`
- Test: `tests/test_gui_backend.py`

**Interfaces:**
- Consumes: deterministic rehabilitation runner and cache serializer.
- Produces: one shared `rehabilitation_cache.json` per run fingerprint and worker argument `--rehabilitation-cache PATH`.

- [ ] **Step 1: Write failing cache and launcher tests**

```python
def test_shared_cache_reused_without_rescoring(self):
    first = build_cache(self.inputs, scorer=self.counting_scorer)
    second = build_cache(self.inputs, scorer=self.fail_if_called_scorer)
    self.assertEqual(first.fingerprint, second.fingerprint)

def test_cache_invalidates_when_target_changes(self):
    a = rehab.session_fingerprint(self.inputs)
    b = rehab.session_fingerprint(replace(self.inputs, target_sha256="different"))
    self.assertNotEqual(a, b)
```

Add a backend command test asserting every PEQ worker receives the same cache path and phase workers receive none.

- [ ] **Step 2: Verify cache tests fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.CacheTests tests.test_gui_backend -v`

Expected: FAIL because no shared cache command exists.

- [ ] **Step 3: Implement atomic cache creation**

The cache-builder command validates inputs, builds rehabilitation once, writes to a sibling temporary file, then uses `Path.replace()` for atomic publication. Store schema, fingerprint inputs, config, census, candidates, and full-precision components. A worker treats malformed or mismatched cache as a clear error, not permission to silently rerun twelve copies.

- [ ] **Step 4: Update the PowerShell launcher**

Before starting workers in PEQ mode, invoke the project interpreter once to build the shared cache. Pass its path to all workers. Include this process in graceful stop handling and fail before worker launch if it exits nonzero.

- [ ] **Step 5: Keep GUI resource controls authoritative**

The cache-builder is one bounded process. Worker count and memory-stop rules remain unchanged. Progress messages distinguish `Preparing existing tune`, `Searching remaining response`, and `Merging candidates` without streaming full logs into the GUI.

- [ ] **Step 6: Run cache, GUI backend, and preflight tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.CacheTests tests.test_gui_backend -v`

Run: `.\.venv\Scripts\python.exe scripts/gui_preflight.py`

Expected: all tests pass and preflight reports success.

- [ ] **Step 7: Commit shared-cache integration**

```powershell
git add scripts/build_rehabilitation_cache.py run_guided_stream_workers.ps1 _optimizer_stream.py optimizer_gui/backend.py tests/test_baseline_rehabilitation.py tests/test_gui_backend.py
git commit -m "Cache rehabilitation across optimizer workers"
```

---

### Task 7: Operation-Level Reports And GUI Results

**Files:**
- Modify: `_optimizer.py:2700-3190`
- Modify: `_optimizer_stream.py:1200-1340`
- Modify: `_merge_stream_results.py`
- Modify: `optimizer_gui/reporting.py`
- Modify: `optimizer_gui/window.py`
- Test: `tests/test_baseline_rehabilitation.py`
- Test: `tests/test_gui_backend.py`

**Interfaces:**
- Consumes: resolved candidate plans and rehabilitation provenance.
- Produces: `rehabilitation` section in `assistant_summary.json`, report operation tables, and selectable baseline/rehabilitated/final result curves.

- [ ] **Step 1: Write failing report-schema tests**

```python
def test_summary_reports_all_operation_types_and_deltas(self):
    summary = build_summary(self.result)
    rehab = summary["rehabilitation"]
    self.assertEqual(rehab["operation_counts"], {
        "modify": 2, "remove": 1, "merge": 1, "append": 0,
    })
    self.assertIn("baseline_to_rehabilitated", rehab["component_deltas"])
    self.assertIn("rehabilitated_to_final", rehab["component_deltas"])
    self.assertTrue(all("reason" in row for row in rehab["accepted_operations"]))

def test_no_meaningful_improvement_is_not_reported_as_percentage_win(self):
    summary = build_summary(self.repeatability_tie)
    self.assertEqual(summary["rehabilitation"]["verdict"], "no_meaningful_improvement")
    self.assertNotIn("percent_improvement", summary["rehabilitation"])
```

- [ ] **Step 2: Verify report tests fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.ReportingTests tests.test_gui_backend -v`

Expected: FAIL because operation-level rehabilitation fields are absent.

- [ ] **Step 3: Extend compact JSON and markdown/HTML reports**

Report supplied baseline, rehabilitated baseline, and final candidate only when distinct. Include old/new F/Q/G, channel role, slot, operation type, rationale, objective/component deltas, headroom, gate rejections, consolidation mismatch, evaluation count, cache source, and repeatability verdict.

- [ ] **Step 4: Add GUI progress and result comparison**

Keep rehabilitation automatic. Do not add an advanced search selector. Add result rows or segmented curve selection for `Supplied`, `Existing tune improved`, and `Final` when available. Selecting a row updates the projected graph and operation explanation.

- [ ] **Step 5: Preserve readable white/dark themes and existing layouts**

Use existing widgets and theme tokens. Ensure long filter reasons wrap without text blocks obscuring labels. Validate at the current minimum window size and 1920x1080.

- [ ] **Step 6: Run reporting and GUI tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.ReportingTests tests.test_gui_backend -v`

Run: `.\.venv\Scripts\python.exe scripts/gui_preflight.py`

Expected: PASS, with reports readable for changed and unchanged rehabilitation outcomes.

- [ ] **Step 7: Commit reporting and GUI integration**

```powershell
git add _optimizer.py _optimizer_stream.py _merge_stream_results.py optimizer_gui/reporting.py optimizer_gui/window.py tests/test_baseline_rehabilitation.py tests/test_gui_backend.py
git commit -m "Report baseline rehabilitation decisions"
```

---

### Task 8: Privacy-Safe Golden Benchmark, Documentation, And Release Verification

**Files:**
- Create: `scripts/benchmark_rehabilitation.py`
- Create: `tests/fixtures/rehabilitation_case.json`
- Modify: `tests/test_baseline_rehabilitation.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/ai_context/CURRENT_STATE.md`

**Interfaces:**
- Consumes: complete rehabilitation pipeline.
- Produces: deterministic equal-budget benchmark JSON and final user/AI documentation.

- [ ] **Step 1: Add a failing privacy-safe golden test**

Create a synthetic fixture with generated frequencies/responses and an AFPX-like slot inventory containing:

- a 97 Hz Q3.0 filter whose known optimum is 100 Hz Q1.2;
- a 33 Hz Q2.0 -2.0 dB sub filter whose known optimum changes Q and gain;
- one harmful removable filter;
- one useful protected filter that must remain;
- one genuine new residual requiring an append in the later guided Beam.

```python
def test_golden_case_reaches_every_required_operation_type(self):
    result = run_golden_fixture("tests/fixtures/rehabilitation_case.json")
    self.assertEqual(result.slot_setting(channel=2, slot=7), (100.0, 1.2, -1.5))
    self.assertFalse(result.slot_active(channel=3, slot=9))
    self.assertTrue(result.has_operation("append"))
    self.assertTrue(result.meaningfully_better_than_baseline)
```

- [ ] **Step 2: Verify the golden test fails before fixture support**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_baseline_rehabilitation.GoldenBenchmarkTests -v`

Expected: FAIL because the fixture runner and benchmark do not exist.

- [ ] **Step 3: Implement equal-wall-time benchmark output**

Run current guided Beam and rehabilitation-plus-Beam with the same seed, wall time, objective, and synthetic inputs. Emit JSON containing evaluations, operation coverage, best full-precision components, runtime, peak memory, and baseline-survival status. Assert the new pipeline reaches the known operations and never scores worse than the unchanged baseline.

- [ ] **Step 4: Update documentation**

Document automatic first-stage behavior in plain language, explicit exclusions, unchanged-baseline fallback, operation reports, runtime sharing, and the distinction between meaningful improvement and a numerical tie. Update compact AI context so future agents open the new module and summary fields without rereading raw logs.

- [ ] **Step 5: Run the complete test suite and benchmark**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Run: `.\.venv\Scripts\python.exe scripts/benchmark_rehabilitation.py --seconds 10 --seed 20260803 --out rehabilitation_benchmark.json`

Run: `.\.venv\Scripts\python.exe -m compileall baseline_rehabilitation.py _optimizer.py _optimizer_stream.py _merge_stream_results.py _make_v3.py optimizer_gui scripts tests`

Run: `git diff --check`

Expected: all tests pass; benchmark reports all required operation types, baseline survival, and deterministic repeat output; compilation and diff checks are clean.

- [ ] **Step 6: Audit publish safety**

Run:

```powershell
git status --short
rg -n "ClintonPC|Adroit|New folder|\.afpx$|System Sum\.txt" README.md CHANGELOG.md docs baseline_rehabilitation.py scripts tests
```

Expected: no personal path, measurement, or tune artifacts in tracked changes. Inspect every untracked file before any later publish; do not stage unrelated GUI work automatically.

- [ ] **Step 7: Commit benchmark and documentation**

```powershell
git add scripts/benchmark_rehabilitation.py tests/fixtures/rehabilitation_case.json tests/test_baseline_rehabilitation.py README.md CHANGELOG.md docs/ai_context/CURRENT_STATE.md
git commit -m "Benchmark and document baseline rehabilitation"
```

- [ ] **Step 8: Final verification record**

Record the exact test count, benchmark result path, benchmark operation coverage, commit hashes, and any residual risks in the implementation handoff. Do not claim GitHub publication until `git ls-remote origin refs/heads/main` matches the local release commit.

