from __future__ import annotations

from dataclasses import dataclass

from _make_v3 import active_peq_slots, at, edit_filter_slot


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


def active_peq_slot_refs(xml, channel_roles):
    refs = []
    for channel, slot, tag in active_peq_slots(xml, channel_roles):
        refs.append(
            FilterRef(
                channel=channel,
                slot=slot,
                role=channel_roles[channel],
                filter_type=at(tag, "T"),
                original=tuple(float(at(tag, key)) for key in ("F", "Q", "G")),
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
