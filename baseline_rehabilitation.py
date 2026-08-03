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
