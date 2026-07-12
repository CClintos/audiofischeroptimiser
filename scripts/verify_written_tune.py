from __future__ import annotations

import argparse
import json
import re
import struct
import zlib
from pathlib import Path


def decode_afpx(path: Path) -> str:
    raw = path.read_bytes()
    declared = struct.unpack(">I", raw[:4])[0]
    xml = zlib.decompress(raw[4:]).decode("utf-8", "replace")
    if declared != len(xml.encode("utf-8")):
        raise ValueError("Header length mismatch in %s" % path)
    return xml


def attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'([A-Za-z]+)="([^"]*)"', tag))


def active_filters(xml: str) -> list[str]:
    return re.findall(r"<Fil\b[^>]*/?>", xml)


def delay_tags(xml: str) -> list[tuple[tuple[str, str], ...]]:
    return [tuple(sorted(attrs(t).items())) for t in re.findall(r"<T [^>]*/?>", xml)]


def delay_values(xml: str) -> list[str | None]:
    return [attrs(tag).get("T") for tag in re.findall(r"<T [^>]*/?>", xml)]


def delay_polarities(xml: str) -> list[str | None]:
    return [attrs(tag).get("PM") for tag in re.findall(r"<T [^>]*/?>", xml)]


def delay_other_attributes(xml: str) -> list[tuple[tuple[str, str], ...]]:
    rows = []
    for tag in re.findall(r"<T [^>]*/?>", xml):
        rows.append(tuple(sorted((key, value) for key, value in attrs(tag).items() if key not in {"T", "PM"})))
    return rows


def output_attributes(xml: str, exclude: set[str] | None = None) -> list[tuple[tuple[str, str], ...]]:
    excluded = exclude or set()
    rows = []
    for block in re.findall(r"<OC\b.*?</OC>", xml, re.S):
        opening = re.match(r"<OC\b[^>]*>", block)
        values = attrs(opening.group()) if opening else {}
        rows.append(tuple(sorted((key, value) for key, value in values.items() if key not in excluded)))
    return rows


def output_polarities(xml: str) -> list[str | None]:
    values = []
    for block in re.findall(r"<OC\b.*?</OC>", xml, re.S):
        opening = re.match(r"<OC\b[^>]*>", block)
        values.append(attrs(opening.group()).get("CINV") if opening else None)
    return values


def filter_key(tag: str) -> tuple[tuple[str, str | None], ...]:
    a = attrs(tag)
    return tuple((k, a.get(k)) for k in ("T", "F", "Q", "G", "dF", "I", "FilBy"))


def filter_keys(xml: str, types: set[str] | None = None) -> list[tuple[tuple[str, str | None], ...]]:
    keys = []
    for tag in active_filters(xml):
        a = attrs(tag)
        if types is None or a.get("T") in types:
            keys.append(filter_key(tag))
    return keys


def multiset_delta(old_items: list[object], new_items: list[object]) -> tuple[list[object], list[object]]:
    old_counts: dict[object, int] = {}
    new_counts: dict[object, int] = {}
    for item in old_items:
        old_counts[item] = old_counts.get(item, 0) + 1
    for item in new_items:
        new_counts[item] = new_counts.get(item, 0) + 1
    added: list[object] = []
    removed: list[object] = []
    for item, count in new_counts.items():
        added.extend([item] * max(0, count - old_counts.get(item, 0)))
    for item, count in old_counts.items():
        removed.extend([item] * max(0, count - new_counts.get(item, 0)))
    return added, removed


def verify(baseline: Path, candidate: Path, allow_delay: bool, allow_apf: bool,
           allow_polarity: bool = False) -> dict[str, object]:
    old_xml = decode_afpx(baseline)
    new_xml = decode_afpx(candidate)
    old_all = filter_keys(old_xml)
    new_all = filter_keys(new_xml)
    added, removed = multiset_delta(old_all, new_all)
    added_types = sorted({dict(item).get("T", "") for item in added})
    removed_types = sorted({dict(item).get("T", "") for item in removed})
    removed_nonfree = [item for item in removed if dict(item).get("T") != "1"]

    delay_changed = delay_values(old_xml) != delay_values(new_xml)
    polarity_changed = (
        delay_polarities(old_xml) != delay_polarities(new_xml)
        or output_polarities(old_xml) != output_polarities(new_xml)
    )
    delay_attributes_changed = delay_other_attributes(old_xml) != delay_other_attributes(new_xml)
    output_attributes_changed = output_attributes(old_xml, {"CINV"}) != output_attributes(new_xml, {"CINV"})
    crossover_changed = filter_keys(old_xml, {"15", "16", "9"}) != filter_keys(new_xml, {"15", "16", "9"})
    apf_added = any(dict(item).get("T") in ("19", "20") for item in added)
    forbidden_added = [
        dict(item) for item in added
        if dict(item).get("T") not in ({"17", "19", "20"} if allow_apf else {"17"})
    ]
    errors = []
    if delay_changed and not allow_delay:
        errors.append("delay_changed")
    if polarity_changed and not allow_polarity:
        errors.append("polarity_changed")
    if output_attributes_changed:
        errors.append("unrelated_output_attributes_changed")
    if delay_attributes_changed:
        errors.append("unrelated_time_alignment_attributes_changed")
    if crossover_changed:
        errors.append("crossover_changed")
    if apf_added and not allow_apf:
        errors.append("apf_added")
    if removed_nonfree:
        errors.append("existing_filter_removed_or_changed")
    if forbidden_added:
        errors.append("forbidden_filter_type_added")

    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "pass": not errors,
        "errors": errors,
        "peq_only": not delay_changed and not crossover_changed and not apf_added and not forbidden_added and not removed_nonfree,
        "delay_changed": delay_changed,
        "polarity_changed": polarity_changed,
        "output_attributes_changed": output_attributes_changed,
        "time_alignment_attributes_changed": delay_attributes_changed,
        "crossover_changed": crossover_changed,
        "apf_changed": apf_added,
        "added_filter_types": added_types,
        "removed_filter_types": removed_types,
        "added_filter_count": len(added),
        "removed_filter_count": len(removed),
        "removed_nonfree_filter_count": len(removed_nonfree),
        "unknown_field_changes": forbidden_added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a written AFPX candidate only changed intended fields.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-delay", action="store_true")
    parser.add_argument("--allow-apf", action="store_true")
    parser.add_argument("--allow-polarity", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("latest_verify_written_tune.json"))
    args = parser.parse_args()

    payload = verify(
        args.baseline.resolve(), args.candidate.resolve(), args.allow_delay,
        args.allow_apf, args.allow_polarity,
    )
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
