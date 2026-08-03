from __future__ import annotations

import math
import time

from dataclasses import dataclass, replace

from _make_v3 import (
    active_peq_slots,
    at,
    canonical_peq_band,
    edit_filter_slot,
)


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


@dataclass(frozen=True)
class ResolvedPlan:
    band_sets: tuple[tuple[Band, ...], ...]
    slot_edits: tuple[SlotEdit, ...]
    group_actions: tuple[
        tuple[int, tuple[tuple[str, Band | None, Band | None], ...]], ...
    ]
    signature: tuple


@dataclass(frozen=True)
class RehabilitationConfig:
    frequency_octaves: tuple[float, ...] = (
        -1 / 3, -1 / 6, -1 / 12, -1 / 24, 0.0,
        1 / 24, 1 / 12, 1 / 6, 1 / 3,
    )
    q_multipliers: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.25, 1.6)
    gain_offsets_db: tuple[float, ...] = (
        -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
    )
    retained_per_slot: int = 6
    refinement_passes: int = 4
    max_evaluations_per_slot: int = 2500
    role_limits: tuple[
        tuple[str, float, float, float, float, float, float], ...
    ] = ()
    paired_role_limits: tuple[
        tuple[str, str, float, float, float, float, float, float], ...
    ] = ()

    def limits_for(self, ref: FilterRef):
        for (
            role, f_min, f_max, q_min, q_max, gain_min, gain_max
        ) in self.role_limits:
            if role == ref.role:
                return (f_min, f_max, q_min, q_max, gain_min, gain_max)
        frequency, _q, gain = ref.original
        gain_min = max(-6.0, gain - 2.0)
        gain_max = 0.0 if gain <= 0.0 else min(3.0, gain + 2.0)
        return (
            frequency * 2 ** (-1 / 3),
            frequency * 2 ** (1 / 3),
            0.5,
            2.5 if gain <= 0.0 else 2.0,
            gain_min,
            gain_max,
        )

    def limits_for_refs(self, refs):
        refs = tuple(refs)
        if len(refs) == 1:
            return self.limits_for(refs[0])
        roles = tuple(sorted(ref.role for ref in refs))
        frequency = float(refs[0].original[0])
        matches = []
        for (
            left_role, right_role,
            f_min, f_max, q_min, q_max, gain_min, gain_max,
        ) in self.paired_role_limits:
            if tuple(sorted((left_role, right_role))) != roles:
                continue
            if f_min <= frequency <= f_max:
                centre = math.sqrt(f_min * f_max)
                matches.append((
                    abs(math.log2(frequency / centre)),
                    (f_min, f_max, q_min, q_max, gain_min, gain_max),
                ))
        if matches:
            return min(matches, key=lambda item: item[0])[1]

        raise ValueError(
            "paired rehabilitation roles have no configured symmetric limits"
        )


@dataclass(frozen=True)
class OperationCandidate:
    edit: SlotEdit
    plan: CandidatePlan
    components: dict[str, object]
    baseline_components: dict[str, object]
    paired_edit: SlotEdit | None = None
    region: CorrectionRegion | None = None
    owner_attributions: tuple[DriverAttribution, ...] = ()

    @property
    def edits(self) -> tuple[SlotEdit, ...]:
        return self.plan.slot_edits

    @property
    def objective(self) -> float:
        return float(self.components["objective"])

    @property
    def parameter_movement(self) -> float:
        movement = 0.0
        for edit in self.edits:
            if edit.replacement is None:
                movement += 1.0
                continue
            old_f, old_q, old_g = edit.ref.original
            new_f, new_q, new_g = edit.replacement
            movement += (
                abs(math.log2(new_f / old_f))
                + abs(new_q - old_q)
                + abs(new_g - old_g)
            )
        return movement


@dataclass(frozen=True)
class ScoredCandidate:
    plan: CandidatePlan
    components: dict[str, object]
    applied_operations: tuple[tuple, ...] = ()
    depth: int = 0
    bridge_only: bool = False
    export_eligible: bool = True
    requires_meaningful_export: bool = False

    @property
    def slot_edits(self) -> tuple[SlotEdit, ...]:
        return self.plan.slot_edits

    @property
    def objective(self) -> float:
        return float(self.components["objective"])

    @property
    def minimum_headroom_db(self) -> float:
        margins = self.components.get("headroom_margin_db", {})
        if isinstance(margins, dict) and margins:
            return min(float(value) for value in margins.values())
        if isinstance(margins, (int, float)):
            return float(margins)
        return float(self.components.get("minimum_headroom_db", float("inf")))

    @property
    def parameter_movement(self) -> float:
        movement = 0.0
        for edit in self.slot_edits:
            if edit.replacement is None:
                movement += 1.0
                continue
            old_f, old_q, old_g = edit.ref.original
            new_f, new_q, new_g = edit.replacement
            movement += (
                abs(math.log2(new_f / old_f))
                + abs(new_q - old_q)
                + abs(new_g - old_g)
            )
        return movement


@dataclass(frozen=True)
class RehabilitationResult:
    baseline: ScoredCandidate
    best: ScoredCandidate
    candidates: tuple[ScoredCandidate, ...]
    generations: tuple[tuple[ScoredCandidate, ...], ...]
    score_count: int


@dataclass(frozen=True)
class ConsolidationResult:
    accepted: bool
    original: ScoredCandidate
    candidate: ScoredCandidate
    max_cascade_error_db: float
    reason: str

@dataclass(frozen=True)
class CorrectionRegion:
    frequency: float
    q: float
    gain_delta_db: float = -0.25


@dataclass(frozen=True)
class DriverAttribution:
    region: CorrectionRegion
    owner_refs: tuple[FilterRef, ...]
    probe_band: Band | None
    components: dict[str, object] | None
    system_delta: float | None
    balance_delta: float | None
    headroom_delta: float | None
    skip_reason: str | None = None
    rank: int = 0


@dataclass(frozen=True)
class FilterCensusRow:
    ref: FilterRef
    paired_ref: FilterRef | None
    baseline_components: dict[str, object]
    removal_components: dict[str, object]
    probe_band: Band | None
    probe_components: dict[str, object] | None
    probe_skip_reason: str | None
    candidates: tuple[OperationCandidate, ...]
    system_delta: float | None
    balance_delta: float | None
    headroom_delta: float | None


def _component_delta(current, baseline, key):
    return float(current.get(key, 0.0)) - float(baseline.get(key, 0.0))


def _operation_plan(refs, replacement):
    edits = tuple(
        SlotEdit.remove(ref)
        if replacement is None
        else SlotEdit.modify(ref, canonical_peq_band(replacement))
        for ref in refs
    )
    return CandidatePlan(slot_edits=edits)


def _score_operation(refs, replacement, score_plan, baseline_components):
    plan = _operation_plan(refs, replacement)
    components = dict(score_plan(plan))
    if "objective" not in components:
        raise ValueError("rehabilitation scorer must return an objective component")
    return OperationCandidate(
        edit=plan.slot_edits[0],
        paired_edit=plan.slot_edits[1] if len(plan.slot_edits) > 1 else None,
        plan=plan,
        components=components,
        baseline_components=dict(baseline_components),
    )


def _bounded_band(refs, band, config):
    refs = tuple(refs)
    f_min, f_max, q_min, q_max, gain_min, gain_max = (
        config.limits_for_refs(refs)
    )
    frequency, q, gain = canonical_peq_band((
        round(band[0], 1),
        round(band[1], 2),
        round(band[2] * 4) / 4,
    ))
    if not (f_min <= frequency <= f_max):
        return None
    if not (q_min <= q <= q_max):
        return None
    if not (gain_min <= gain <= gain_max):
        return None
    return (frequency, q, gain)


def _coarse_bands(refs, config):
    refs = tuple(refs)
    ref = refs[0]
    frequency, q, gain = ref.original
    f_min, f_max, q_min, q_max, gain_min, gain_max = (
        config.limits_for_refs(refs)
    )
    frequencies = {canonical_peq_band(ref.original)[0]}
    for octave in config.frequency_octaves:
        moved = frequency * 2 ** octave
        if f_min <= moved <= f_max:
            frequencies.add(round(moved, 1))
            frequencies.add(float(round(moved)))
    preferred_q = (0.5, 0.7, 1.0, 1.2, 1.4, 2.0, 2.2, 2.5)
    q_values = {
        round(q * multiplier, 2)
        for multiplier in config.q_multipliers
        if q_min <= q * multiplier <= q_max
    }
    q_values.update(value for value in preferred_q if q_min <= value <= q_max)
    q_values.add(min(max(q, q_min), q_max))
    gains = {
        round((gain + offset) * 4) / 4
        for offset in config.gain_offsets_db
        if gain_min <= gain + offset <= gain_max
    }
    gains.add(round(min(max(gain, gain_min), gain_max) * 4) / 4)
    return {
        band
        for moved_f in frequencies
        for moved_q in q_values
        for moved_gain in gains
        if (band := _bounded_band(
            refs, (moved_f, moved_q, moved_gain), config
        )) is not None
        and band != canonical_peq_band(ref.original)
    }


def _search_for_refs(
    refs, score_plan, config, deadline, eligibility=None
):
    baseline_components = dict(score_plan(CandidatePlan()))

    def eligible(replacement):
        return eligibility is None or eligibility(replacement)

    scored = {}
    evaluations = 0

    def evaluate(replacement):
        nonlocal evaluations
        key = None if replacement is None else canonical_peq_band(replacement)
        if key in scored:
            return scored[key]
        if evaluations >= config.max_evaluations_per_slot:
            return None
        if deadline is not None and time.monotonic() >= deadline:
            return None
        candidate = _score_operation(
            refs, key, score_plan, baseline_components
        )
        scored[key] = candidate
        evaluations += 1
        return candidate

    removal = evaluate(None) if eligible(None) else None
    for band in sorted(_coarse_bands(refs, config)):
        if not eligible(band):
            continue
        if evaluate(band) is None:
            break

    coarse = sorted(
        (candidate for key, candidate in scored.items() if key is not None),
        key=lambda candidate: (
            candidate.objective,
            candidate.parameter_movement,
            candidate.edit.replacement,
        ),
    )[:config.retained_per_slot]

    refined = list(coarse)
    for pass_index in range(config.refinement_passes):
        frequency_step = 1 / (24 * 2 ** min(pass_index, 2))
        q_step = max(0.1, 0.4 / 2 ** pass_index)
        gain_step = max(0.25, 1.0 / 2 ** pass_index)
        next_refined = []
        for seed in refined:
            current = seed
            for _ in range(12):
                frequency, q, gain = current.edit.replacement
                neighbours = []
                for multiplier in (-1.0, 1.0):
                    neighbours.extend((
                        (
                            frequency * 2 ** (multiplier * frequency_step),
                            q,
                            gain,
                        ),
                        (frequency, q + multiplier * q_step, gain),
                        (frequency, q, gain + multiplier * gain_step),
                    ))
                trials = []
                for neighbour in neighbours:
                    band = _bounded_band(refs, neighbour, config)
                    if band is None or not eligible(band):
                        continue
                    candidate = evaluate(band)
                    if candidate is not None:
                        trials.append(candidate)
                if not trials:
                    break
                best = min(
                    trials,
                    key=lambda candidate: (
                        candidate.objective,
                        candidate.parameter_movement,
                    ),
                )
                if best.objective >= current.objective - 1e-12:
                    break
                current = best
            next_refined.append(current)
        refined = next_refined

    retained = sorted(
        {
            candidate.edit.replacement: candidate
            for candidate in (*coarse, *refined)
        }.values(),
        key=lambda candidate: (
            candidate.objective,
            candidate.parameter_movement,
            candidate.edit.replacement,
        ),
    )[:config.retained_per_slot]
    return tuple(([removal] if removal is not None else []) + retained)


def search_filter_operations(
    ref,
    score_plan,
    config=None,
    *,
    paired_ref=None,
    asymmetry_eligible=None,
    deadline=None,
):
    config = config or RehabilitationConfig()
    refs = (ref, paired_ref) if paired_ref is not None else (ref,)
    candidates = list(_search_for_refs(refs, score_plan, config, deadline))
    if paired_ref is not None and asymmetry_eligible is not None:
        asymmetric = _search_for_refs(
            (ref,),
            score_plan,
            config,
            deadline,
            eligibility=lambda replacement: asymmetry_eligible(
                ref, replacement
            ),
        )
        candidates.extend(asymmetric)
    unique = {}
    for candidate in candidates:
        key = tuple(
            (edit.ref.channel, edit.ref.slot, edit.replacement)
            for edit in candidate.edits
        )
        unique[key] = candidate
    return tuple(sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.objective,
            len(candidate.edits),
            candidate.parameter_movement,
        ),
    ))


def _role_side(role):
    lowered = role.casefold()
    for token, side in (
        ("fl ", "left"), ("fr ", "right"),
        ("left ", "left"), ("right ", "right"),
    ):
        if lowered.startswith(token):
            return side, lowered[len(token):]
    return None, lowered


def _matched_pair(ref, candidate):
    if ref.channel == candidate.channel:
        return False
    if canonical_peq_band(ref.original) != canonical_peq_band(candidate.original):
        return False
    if ref.pair_key is not None or candidate.pair_key is not None:
        return ref.pair_key is not None and ref.pair_key == candidate.pair_key
    ref_side, ref_role = _role_side(ref.role)
    candidate_side, candidate_role = _role_side(candidate.role)
    if ref_side is not None or candidate_side is not None:
        return (
            ref_role == candidate_role
            and {ref_side, candidate_side} == {"left", "right"}
        )
    return (
        ref_role == candidate_role == "sub"
        and {ref.channel, candidate.channel} == {6, 7}
    )


def _pair_map(refs):
    pairs = {}
    for index, ref in enumerate(refs):
        if ref in pairs:
            continue
        for candidate in refs[index + 1:]:
            if candidate not in pairs and _matched_pair(ref, candidate):
                pairs[ref] = candidate
                pairs[candidate] = ref
                break
    return pairs


def _system_delta(components, baseline_components):
    for key in ("sum_rms_db", "system_rms_db", "objective"):
        if key in components and key in baseline_components:
            return _component_delta(components, baseline_components, key)
    raise ValueError("rehabilitation scorer has no system or objective component")


def _probe_band(refs, config):
    refs = tuple(refs)
    original = canonical_peq_band(refs[0].original)
    f_min, f_max, q_min, q_max, gain_min, gain_max = (
        config.limits_for_refs(refs)
    )
    base = (
        min(max(original[0], f_min), f_max),
        min(max(original[1], q_min), q_max),
        min(max(original[2], gain_min), gain_max),
    )
    proposals = (
        (base[0], base[1], base[2] - 0.25),
        (base[0], base[1], base[2] + 0.25),
        (base[0], base[1] - 0.1, base[2]),
        (base[0], base[1] + 0.1, base[2]),
        (base[0] * 2 ** (-1 / 96), base[1], base[2]),
        (base[0] * 2 ** (1 / 96), base[1], base[2]),
        base,
    )
    for proposal in proposals:
        band = _bounded_band(refs, proposal, config)
        if band is not None and band != original:
            return band, None
    return None, "no valid bounded perturbation"


def attribute_correction_region(
    region,
    eligible_owners,
    score_plan,
    config=None,
):
    config = config or RehabilitationConfig()
    baseline_components = dict(score_plan(CandidatePlan()))
    rows = []
    for owner in eligible_owners:
        owner_refs = (
            (owner,) if isinstance(owner, FilterRef) else tuple(owner)
        )
        original_gain = canonical_peq_band(owner_refs[0].original)[2]
        probe_band = _bounded_band(
            owner_refs,
            (
                float(region.frequency),
                float(region.q),
                original_gain + float(region.gain_delta_db),
            ),
            config,
        )
        if probe_band is None:
            rows.append(DriverAttribution(
                region=region,
                owner_refs=owner_refs,
                probe_band=None,
                components=None,
                system_delta=None,
                balance_delta=None,
                headroom_delta=None,
                skip_reason="correction region outside owner limits",
            ))
            continue
        probe = _score_operation(
            owner_refs, probe_band, score_plan, baseline_components
        )
        rows.append(DriverAttribution(
            region=region,
            owner_refs=owner_refs,
            probe_band=probe_band,
            components=dict(probe.components),
            system_delta=_system_delta(
                probe.components, baseline_components
            ),
            balance_delta=_component_delta(
                probe.components, baseline_components, "balance_penalty_db"
            ),
            headroom_delta=_component_delta(
                probe.components,
                baseline_components,
                "positive_gain_penalty_db",
            ),
        ))

    ordered = sorted(
        rows,
        key=lambda item: (
            item.probe_band is None,
            float("inf") if item.system_delta is None else item.system_delta,
            float("inf") if item.balance_delta is None else item.balance_delta,
            float("inf") if item.headroom_delta is None else item.headroom_delta,
            tuple((ref.channel, ref.slot) for ref in item.owner_refs),
        ),
    )
    return tuple(
        replace(item, rank=rank)
        for rank, item in enumerate(ordered, start=1)
    )


def _candidate_region(candidate):
    replacement = candidate.edit.replacement
    if replacement is None:
        return None
    gain_change = replacement[2] - candidate.edit.ref.original[2]
    gain_delta_db = 0.25 if gain_change > 0.0 else -0.25
    return CorrectionRegion(
        frequency=replacement[0],
        q=replacement[1],
        gain_delta_db=gain_delta_db,
    )


def _eligible_region_owners(
    refs,
    pairs,
    region,
    config,
    asymmetry_eligible,
):
    owners = []
    seen = set()
    for ref in refs:
        paired_ref = pairs.get(ref)
        if paired_ref is None:
            key = ((ref.channel, ref.slot),)
            if key not in seen:
                owners.append((ref,))
                seen.add(key)
            continue

        pair = tuple(sorted((ref, paired_ref), key=lambda item: (
            item.channel, item.slot
        )))
        pair_key = tuple((item.channel, item.slot) for item in pair)
        if pair_key not in seen:
            owners.append(pair)
            seen.add(pair_key)

        if asymmetry_eligible is None:
            continue
        replacement = _bounded_band(
            (ref,),
            (
                region.frequency,
                region.q,
                ref.original[2] + region.gain_delta_db,
            ),
            config,
        )
        side_key = ((ref.channel, ref.slot),)
        if (
            replacement is not None
            and side_key not in seen
            and asymmetry_eligible(ref, replacement)
        ):
            owners.append((ref,))
            seen.add(side_key)
    return tuple(owners)


def build_filter_census(
    refs,
    score_plan,
    config=None,
    *,
    asymmetry_eligible=None,
    deadline=None,
):
    config = config or RehabilitationConfig()
    refs = tuple(refs)
    baseline_components = dict(score_plan(CandidatePlan()))
    pairs = _pair_map(refs)
    rows = []
    for ref in refs:
        paired_ref = pairs.get(ref)
        operation_refs = (ref, paired_ref) if paired_ref is not None else (ref,)
        removal = _score_operation(
            operation_refs, None, score_plan, baseline_components
        )
        probe_band, probe_skip_reason = _probe_band(operation_refs, config)
        probe = (
            None
            if probe_band is None
            else _score_operation(
                operation_refs,
                probe_band,
                score_plan,
                baseline_components,
            )
        )
        candidates = search_filter_operations(
            ref,
            score_plan,
            config,
            paired_ref=paired_ref,
            asymmetry_eligible=asymmetry_eligible,
            deadline=deadline,
        )
        probe_components = (
            None if probe is None else dict(probe.components)
        )
        rows.append(FilterCensusRow(
            ref=ref,
            paired_ref=paired_ref,
            baseline_components=dict(baseline_components),
            removal_components=dict(removal.components),
            probe_band=probe_band,
            probe_components=probe_components,
            probe_skip_reason=probe_skip_reason,
            candidates=candidates,
            system_delta=(
                None
                if probe is None
                else _system_delta(probe.components, baseline_components)
            ),
            balance_delta=(
                None
                if probe is None
                else _component_delta(
                    probe.components,
                    baseline_components,
                    "balance_penalty_db",
                )
            ),
            headroom_delta=(
                None
                if probe is None
                else _component_delta(
                    probe.components,
                    baseline_components,
                    "positive_gain_penalty_db",
                )
            ),
        ))

    regions = {
        region
        for row in rows
        for candidate in row.candidates
        if (region := _candidate_region(candidate)) is not None
    }
    ownership_by_region = {}
    for region in sorted(
        regions,
        key=lambda item: (item.frequency, item.q, item.gain_delta_db),
    ):
        owners = _eligible_region_owners(
            refs,
            pairs,
            region,
            config,
            asymmetry_eligible,
        )
        ownership_by_region[region] = attribute_correction_region(
            region,
            owners,
            score_plan,
            config,
        )

    attributed_rows = []
    for row in rows:
        candidates = []
        for candidate in row.candidates:
            region = _candidate_region(candidate)
            candidates.append(replace(
                candidate,
                region=region,
                owner_attributions=(
                    ()
                    if region is None
                    else ownership_by_region[region]
                ),
            ))
        attributed_rows.append(replace(
            row,
            candidates=tuple(candidates),
        ))
    return tuple(attributed_rows)


def freeze_groups(groups) -> tuple[tuple[str, tuple[Band, ...]], ...]:
    return tuple(
        (str(group), tuple(canonical_peq_band(band) for band in bands))
        for group, bands in groups.items()
    )


def thaw_groups(groups) -> dict[str, list[Band]]:
    return {
        group: [canonical_peq_band(band) for band in bands]
        for group, bands in groups
    }


def candidate_plan_signature(plan: CandidatePlan) -> tuple:
    slot_signature = tuple(
        sorted(
            [
                (
                    edit.ref.channel,
                    edit.ref.slot,
                    edit.ref.role,
                    edit.ref.filter_type,
                    canonical_peq_band(edit.ref.original),
                    edit.ref.pair_key,
                    None
                    if edit.replacement is None
                    else canonical_peq_band(edit.replacement),
                )
                for edit in plan.slot_edits
            ],
            key=lambda item: item[:2],
        )
    )
    groups_signature = tuple(
        (group, tuple(canonical_peq_band(band) for band in bands))
        for group, bands in plan.groups
    )
    return ("candidate-plan-v1", slot_signature, groups_signature)


_ACOUSTIC_TIE_COMPONENTS = (
    "tonal_masked", "spatial_tonal_db", "presence_error_db",
    "target_shape_error_db", "peak_penalty_db", "spatial_peak_db",
    "balance_penalty_db",
)


def _component_value(candidate, *keys, default=0.0):
    for key in keys:
        if key in candidate.components:
            return float(candidate.components[key])
    return float(default)


def _affected_frequencies(candidate):
    frequencies = []
    roles = []
    for edit in candidate.slot_edits:
        frequencies.append(float(
            edit.ref.original[0] if edit.replacement is None else edit.replacement[0]
        ))
        roles.append(edit.ref.role)
    return tuple(frequencies), tuple(roles)


def _repeatability_allowance(first, second, repeatability_db):
    if repeatability_db is not None:
        if callable(repeatability_db):
            frequencies = _affected_frequencies(first)[0] + _affected_frequencies(second)[0]
            return float(repeatability_db(frequencies))
        return float(repeatability_db)

    from _tunefit import measurement_noise_floor_db

    values = []
    for candidate in (first, second):
        frequencies, roles = _affected_frequencies(candidate)
        for frequency, role in zip(frequencies, roles):
            role_key = role.casefold()
            branch = "high" if "high" in role_key or "tweet" in role_key else "low"
            values.append(float(measurement_noise_floor_db([frequency], branch)[0]))
    return max(values, default=0.1)


def _acoustically_tied(first, second, repeatability_db=None):
    allowance = _repeatability_allowance(first, second, repeatability_db)
    compared = False
    for key in _ACOUSTIC_TIE_COMPONENTS:
        if key not in first.components or key not in second.components:
            continue
        compared = True
        if abs(float(first.components[key]) - float(second.components[key])) > allowance:
            return False
    return compared


def _filter_count(candidate):
    for key in ("filter_count", "n_front_bands"):
        if key in candidate.components:
            return float(candidate.components[key])
    return float(sum(edit.replacement is not None for edit in candidate.slot_edits))


def tie_key(candidate):
    return (
        _filter_count(candidate),
        _component_value(candidate, "positive_gain_penalty_db", default=0.0),
        -candidate.minimum_headroom_db,
        _component_value(
            candidate, "asymmetric_eq_penalty_db", "asymmetric_eq_penalty",
            default=0.0,
        ),
        candidate.parameter_movement,
        candidate_plan_signature(candidate.plan),
    )


def compare_candidates(first, second, repeatability_db=None):
    if _acoustically_tied(first, second, repeatability_db):
        return min((first, second), key=tie_key)
    return min(
        (first, second),
        key=lambda candidate: (candidate.objective, tie_key(candidate)),
    )


def _acoustic_component_key(candidate):
    return tuple(
        float(candidate.components[key])
        for key in _ACOUSTIC_TIE_COMPONENTS
        if key in candidate.components
    )


def _acoustic_reference(candidates):
    return min(
        candidates,
        key=lambda candidate: (
            candidate.objective,
            _acoustic_component_key(candidate),
            candidate_plan_signature(candidate.plan),
        ),
    )


def _meaningfully_better(candidate, reference, repeatability_db=None):
    return (
        candidate.objective < reference.objective - 1e-12
        and not _acoustically_tied(candidate, reference, repeatability_db)
    )


def select_best_candidate(
    candidates, repeatability_db=None, *, acoustic_reference=None
):
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one rehabilitation candidate is required")
    reference = acoustic_reference or _acoustic_reference(candidates)
    equivalence_class = tuple(
        candidate
        for candidate in candidates
        if _acoustically_tied(candidate, reference, repeatability_db)
    )
    return min(equivalence_class or (reference,), key=tie_key)


def _operation_keys(operation):
    return frozenset((edit.ref.channel, edit.ref.slot) for edit in operation.edits)


def _region_signature(candidate):
    regions = []
    for edit in candidate.slot_edits:
        frequency = float(
            edit.ref.original[0] if edit.replacement is None else edit.replacement[0]
        )
        regions.append((
            edit.ref.channel,
            int(math.floor(math.log2(max(frequency, 20.0) / 20.0) * 3.0)),
        ))
    return tuple(sorted(set(regions)))


def _beam_sort_key(candidate):
    return (candidate.objective, tie_key(candidate))


def _retain_diverse(
    candidates, baseline, beam_width, generation, repeatability_db=None
):
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    by_signature = {
        candidate_plan_signature(candidate.plan): candidate
        for candidate in sorted(candidates, key=_beam_sort_key, reverse=True)
    }
    baseline_signature = candidate_plan_signature(baseline.plan)
    retained = [by_signature[baseline_signature]]
    chosen = {baseline_signature}

    def add(candidate, *, bridge_only=False):
        signature = candidate_plan_signature(candidate.plan)
        if signature not in chosen and len(retained) < beam_width:
            if bridge_only:
                candidate = replace(
                    candidate,
                    bridge_only=True,
                    export_eligible=False,
                    requires_meaningful_export=True,
                )
            retained.append(candidate)
            chosen.add(signature)

    leaders = (
        lambda item: item.objective,
        lambda item: _component_value(item, "tonal_masked", "spatial_tonal_db", default=float("inf")),
        lambda item: _component_value(item, "presence_error_db", default=float("inf")),
        lambda item: _component_value(item, "balance_penalty_db", default=float("inf")),
        lambda item: -item.minimum_headroom_db,
        _filter_count,
    )
    pool = list(by_signature.values())
    for key in leaders:
        add(min(pool, key=lambda item: (key(item), _beam_sort_key(item))))

    near_ties = [
        item for item in pool
        if item.depth == generation
        and _acoustically_tied(item, baseline, repeatability_db)
    ]
    if near_ties:
        add(min(near_ties, key=_beam_sort_key), bridge_only=True)

    seen_regions = {_region_signature(item) for item in retained}
    for candidate in sorted(pool, key=_beam_sort_key):
        region = _region_signature(candidate)
        if region not in seen_regions:
            add(candidate)
            seen_regions.add(region)
    for candidate in sorted(pool, key=_beam_sort_key):
        add(candidate)
    return tuple(retained)


def rehabilitation_beam(
    baseline_plan,
    operations,
    score_plan,
    *,
    beam_width=16,
    max_depth=4,
    repeatability_db=None,
    deadline=None,
):
    score_cache = {}
    bridge_signatures = set()

    def score(
        plan,
        applied_operations=(),
        depth=0,
        requires_meaningful_export=False,
    ):
        signature = candidate_plan_signature(plan)
        if signature not in score_cache:
            components = dict(score_plan(plan))
            if "objective" not in components:
                raise ValueError("rehabilitation scorer must return an objective component")
            score_cache[signature] = components
        return ScoredCandidate(
            plan=plan,
            components=dict(score_cache[signature]),
            applied_operations=tuple(applied_operations),
            depth=depth,
            bridge_only=signature in bridge_signatures,
            export_eligible=not (
                signature in bridge_signatures or requires_meaningful_export
            ),
            requires_meaningful_export=requires_meaningful_export,
        )

    baseline = score(baseline_plan)
    unique_operations = {
        candidate_plan_signature(operation.plan): operation
        for operation in operations
    }
    ranked_operations = tuple(unique_operations.values())
    beam = (baseline,)
    generations = [beam]
    all_candidates = {candidate_plan_signature(baseline.plan): baseline}

    for generation in range(1, max_depth + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        deadline_reached = False
        expanded = {candidate_plan_signature(baseline.plan): baseline}
        for parent in beam:
            expanded[candidate_plan_signature(parent.plan)] = parent
            reserved = {
                (edit.ref.channel, edit.ref.slot) for edit in parent.slot_edits
            }
            applied = set(parent.applied_operations)
            for operation in ranked_operations:
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_reached = True
                    break
                operation_signature = candidate_plan_signature(operation.plan)
                if operation_signature in applied:
                    continue
                if reserved & _operation_keys(operation):
                    continue
                plan = CandidatePlan(
                    slot_edits=parent.slot_edits + operation.edits,
                    groups=parent.plan.groups + operation.plan.groups,
                )
                needs_meaningful_export = (
                    parent.bridge_only or parent.requires_meaningful_export
                )
                candidate = score(
                    plan,
                    parent.applied_operations + (operation_signature,),
                    generation,
                    needs_meaningful_export,
                )
                if needs_meaningful_export and _meaningfully_better(
                    candidate, baseline, repeatability_db
                ):
                    candidate = replace(
                        candidate,
                        export_eligible=True,
                        requires_meaningful_export=False,
                    )
                expanded[candidate_plan_signature(plan)] = candidate
            if deadline_reached:
                break
        if deadline_reached:
            break
        beam = _retain_diverse(
            tuple(expanded.values()), baseline, beam_width, generation,
            repeatability_db,
        )
        generations.append(beam)
        all_candidates.update({
            candidate_plan_signature(candidate.plan): candidate
            for candidate in expanded.values()
        })
        for retained_candidate in beam:
            signature = candidate_plan_signature(retained_candidate.plan)
            if retained_candidate.bridge_only:
                bridge_signatures.add(signature)
            all_candidates[signature] = retained_candidate
        if not any(candidate.depth == generation for candidate in beam):
            break

    export_candidates = tuple(
        candidate
        for candidate in all_candidates.values()
        if candidate.export_eligible and not candidate.bridge_only
    )
    best = select_best_candidate(export_candidates, repeatability_db)
    return RehabilitationResult(
        baseline=baseline,
        best=best,
        candidates=tuple(sorted(all_candidates.values(), key=_beam_sort_key)),
        generations=tuple(generations),
        score_count=len(score_cache),
    )


def hard_gate_regressed(original_components, trial_components):
    original_counts = {
        key: float(value)
        for key, value in original_components.items()
        if key.endswith("violation_count")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    trial_counts = {
        key: float(value)
        for key, value in trial_components.items()
        if key.endswith("violation_count")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    if any(
        trial_counts.get(key, 0.0) > original_counts.get(key, 0.0)
        for key in original_counts.keys() | trial_counts.keys()
    ):
        return True
    return any(
        value is True and trial_components.get(key) is not True
        for key, value in original_components.items()
        if isinstance(value, bool)
    )


def consolidate_candidate(candidate, score_plan, tolerance_db=0.1):
    from objective_module.afpx_objective import fit_consolidated_peq_pair

    original_plan = candidate.plan if isinstance(candidate, ScoredCandidate) else candidate
    original = (
        candidate
        if isinstance(candidate, ScoredCandidate)
        else ScoredCandidate(original_plan, dict(score_plan(original_plan)))
    )
    attempted_errors = []
    rejection = "no overlapping same-channel PEQ pair"
    edits = list(original_plan.slot_edits)

    for first_index, first in enumerate(edits):
        if first.replacement is None:
            continue
        for second_index in range(first_index + 1, len(edits)):
            second = edits[second_index]
            if second.replacement is None or first.ref.channel != second.ref.channel:
                continue
            fit = fit_consolidated_peq_pair(first.replacement, second.replacement)
            if not fit.overlap:
                continue
            attempted_errors.append(fit.max_cascade_error_db)
            if fit.max_cascade_error_db > tolerance_db + 1e-12:
                rejection = "maximum cascade mismatch exceeds tolerance"
                continue
            trial_edits = list(edits)
            trial_edits[first_index] = SlotEdit.modify(first.ref, fit.replacement)
            trial_edits[second_index] = SlotEdit.remove(second.ref)
            trial_plan = CandidatePlan(tuple(trial_edits), original_plan.groups)
            trial = ScoredCandidate(trial_plan, dict(score_plan(trial_plan)))
            if _filter_count(trial) >= _filter_count(original):
                rejection = "filter count did not decrease"
                continue
            if trial.objective > original.objective + 1e-12:
                rejection = "objective regressed"
                continue
            original_margins = original.components.get("headroom_margin_db", {})
            trial_margins = trial.components.get("headroom_margin_db", {})
            if isinstance(original_margins, dict) and original_margins:
                headroom_regressed = any(
                    float(trial_margins.get(channel, float("-inf")))
                    < float(margin) - 1e-12
                    for channel, margin in original_margins.items()
                )
            else:
                headroom_regressed = (
                    trial.minimum_headroom_db
                    < original.minimum_headroom_db - 1e-12
                )
            if headroom_regressed:
                rejection = "headroom regressed"
                continue
            if hard_gate_regressed(original.components, trial.components):
                rejection = "hard gate regressed"
                continue
            return ConsolidationResult(
                True, original, trial, fit.max_cascade_error_db, "accepted"
            )

    max_error = min(attempted_errors) if attempted_errors else float("inf")
    return ConsolidationResult(False, original, original, max_error, rejection)

def active_peq_slot_refs(xml, channel_roles):
    refs = []
    for channel, slot, tag in active_peq_slots(xml, channel_roles):
        refs.append(
            FilterRef(
                channel=channel,
                slot=slot,
                role=channel_roles[channel],
                filter_type=at(tag, "T"),
                original=canonical_peq_band(
                    tuple(float(at(tag, key)) for key in ("F", "Q", "G"))
                ),
            )
        )
    return tuple(refs)


def apply_slot_edits(xml, edits, protected_channels=()):
    protected = set(protected_channels)
    written = xml
    for edit in edits:
        ref = edit.ref
        written = edit_filter_slot(
            written,
            channel=ref.channel,
            slot=ref.slot,
            expected_type=ref.filter_type,
            expected_band=ref.original,
            replacement=edit.replacement,
            protected_boost=ref.channel in protected,
        )
    return written
