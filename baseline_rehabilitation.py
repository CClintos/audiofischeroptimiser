from __future__ import annotations

from dataclasses import dataclass

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
