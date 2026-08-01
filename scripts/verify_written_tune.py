from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# Invoked as `python scripts/verify_written_tune.py`, so sys.path[0] is this
# file's own "scripts" directory, not the repo root where _optimizer.py
# lives - without this, the deferred `import _optimizer` below always fails.
# Same pattern as scripts/prepare_phase_cache.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def decode_afpx(path: Path) -> str:
    raw = path.read_bytes()
    declared = struct.unpack(">I", raw[:4])[0]
    # Strict, not "replace" - see _make_v3.decode_afpx for why.
    xml = zlib.decompress(raw[4:]).decode("utf-8", "strict")
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


def output_volumes_db(xml: str) -> list[float]:
    values = []
    for block in re.findall(r"<OC\b.*?</OC>", xml, re.S):
        tag = re.search(r"<Vol\b[^>]*/?>", block)
        linear = float(attrs(tag.group()).get("L", "1")) if tag else 1.0
        values.append(20.0 * math.log10(max(linear, 1e-30)))
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


def channel_filter_keys(xml: str, types: set[str] | None = None) -> list[list[tuple[tuple[str, str | None], ...]]]:
    rows = []
    for block in re.findall(r"<OC\b.*?</OC>", xml, re.S):
        rows.append([
            filter_key(tag) for tag in active_filters(block)
            if types is None or attrs(tag).get("T") in types
        ])
    return rows


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


def _added_peq_by_channel(old_xml: str, new_xml: str) -> dict[int, list[dict[str, str | None]]]:
    old_channels = channel_filter_keys(old_xml, {"17"})
    new_channels = channel_filter_keys(new_xml, {"17"})
    result = {}
    for index in range(max(len(old_channels), len(new_channels))):
        old = old_channels[index] if index < len(old_channels) else []
        new = new_channels[index] if index < len(new_channels) else []
        added, _ = multiset_delta(old, new)
        result[index] = [dict(item) for item in added]
    return result


def measurement_guardrail_errors(freqs, traces, target, added_by_channel, groups,
                                 pair_defs, synthetic_pairs) -> list[dict[str, object]]:
    import _tunefit as tunefit

    log_f = np.log10(freqs)
    system_dev = tunefit.erb_smooth(freqs, traces["System Sum"] - target)
    errors = []
    side_groups = [
        (name, cfg) for name, cfg in groups.items()
        if cfg.get("pair") and cfg.get("side") and len(cfg.get("channels", ())) == 1
    ]
    for group_name, cfg in side_groups:
        channel = int(cfg["channels"][0])
        pair = pair_defs[cfg["pair"]]
        peer_group = next(
            (peer for peer, peer_cfg in side_groups
             if peer_cfg.get("pair") == cfg.get("pair")
             and peer_cfg.get("side") != cfg.get("side")),
            None,
        )
        peer_channel = int(groups[peer_group]["channels"][0]) if peer_group else -1
        peer_keys = {
            (round(float(item.get("F") or 0.0), 1),
             round(float(item.get("Q") or 0.0), 2),
             round(float(item.get("G") or 0.0), 2))
            for item in added_by_channel.get(peer_channel, [])
        }
        evidence_state = tunefit.interference_mask_evidence(
            freqs,
            traces[pair["left"]],
            traces[pair["right"]],
            traces.get(pair["together"]),
            synthetic=pair["together"] in synthetic_pairs,
            band=pair["branch_band"],
        )["state"]
        diff = tunefit.erb_smooth(freqs, traces[pair["left"]] - traces[pair["right"]])
        for item in added_by_channel.get(channel, []):
            center = float(item.get("F") or 0.0)
            gain = float(item.get("G") or 0.0)
            key = (
                round(center, 1),
                round(float(item.get("Q") or 0.0), 2),
                round(gain, 2),
            )
            if gain >= 0.0 or key in peer_keys:
                continue
            reasons = []
            if evidence_state == tunefit.MASK_UNKNOWN:
                reasons.append("interference_evidence_unknown")
            evidence = tunefit.signed_offset_evidence(
                freqs, diff, center, cfg.get("branch", "low")
            )
            if not evidence["eligible"]:
                reasons.append(str(evidence["reason"]))
            deviation = float(np.interp(np.log10(center), log_f, system_dev))
            if deviation < -0.5:
                reasons.append("summed_response_already_below_target")
            if float(tunefit.imaging_balance_weight([center])[0]) < 0.5:
                reasons.append("imaging_frequency_outside_authority")
            if reasons:
                errors.append({
                    "group": group_name,
                    "channel": channel,
                    "frequency_hz": center,
                    "gain_db": gain,
                    "system_deviation_db": deviation,
                    "reasons": list(dict.fromkeys(reasons)),
                })
    for channel, items in added_by_channel.items():
        role = next((
            str(cfg.get("branch", "low")) for cfg in groups.values()
            if channel in cfg.get("channels", ()) and len(cfg.get("channels", ())) == 1
        ), "low")
        for item in items:
            center = float(item.get("F") or 0.0)
            floor = float(tunefit.measurement_noise_floor_db([center], role)[0])
            required = tunefit.MEASUREMENT_NOISE_MULTIPLIER * floor
            deviation = abs(float(np.interp(np.log10(center), log_f, traces["System Sum"] - target)))
            if deviation < required:
                errors.append({
                    "channel": channel,
                    "frequency_hz": center,
                    "reasons": ["below_measurement_noise_floor"],
                    "deviation_db": deviation,
                    "required_deviation_db": required,
                })
    return errors


def _measurement_lint(data_root: Path, baseline: Path, target_path: Path,
                      role_map: Path | None, repeatability_folder: Path | None,
                      old_xml: str, new_xml: str) -> list[dict[str, object]]:
    os.environ["AFPX_DATA_ROOT"] = str(data_root)
    os.environ["AFPX_BASELINE"] = str(baseline)
    os.environ["AFPX_TARGET"] = str(target_path)
    if role_map:
        os.environ["AFPX_ROLE_MAP"] = str(role_map)
    import _optimizer as optimizer

    optimizer.configure_repeatability_floor(repeatability_folder)
    freqs, traces, _ = optimizer.load_measurements()
    raw_target = optimizer.load_target(target_path, freqs)
    target = raw_target + optimizer.target_anchor_offset(freqs, traces["System Sum"], raw_target)
    return measurement_guardrail_errors(
        freqs,
        traces,
        target,
        _added_peq_by_channel(old_xml, new_xml),
        optimizer.GROUPS,
        optimizer.PAIR_DEFS,
        optimizer.SYNTHETIC_PAIR_ROLES,
    )


def verify(baseline: Path, candidate: Path, allow_delay: bool, allow_apf: bool,
           allow_polarity: bool = False, allow_output_trim: bool = False,
           data_root: Path | None = None, target: Path | None = None,
           role_map: Path | None = None,
           repeatability_folder: Path | None = None) -> dict[str, object]:
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
    old_volumes = output_volumes_db(old_xml)
    new_volumes = output_volumes_db(new_xml)
    output_volume_changes = {
        index: new_volumes[index] - old_volumes[index]
        for index in range(min(len(old_volumes), len(new_volumes)))
        if abs(new_volumes[index] - old_volumes[index]) >= 0.001
    }
    trim_values = list(output_volume_changes.values())
    output_trim_valid = not output_volume_changes
    if output_volume_changes and allow_output_trim:
        changed = set(output_volume_changes)
        valid_front_set = changed in ({0, 1, 2, 3}, {0, 1, 2, 3, 4, 5})
        uniform = max(trim_values) - min(trim_values) <= 0.02
        attenuation_only = all(-6.001 <= value <= -0.001 for value in trim_values)
        hardware_steps = all(abs(value * 4.0 - round(value * 4.0)) <= 0.04 for value in trim_values)
        output_trim_valid = valid_front_set and uniform and attenuation_only and hardware_steps
    crossover_changed = filter_keys(old_xml, {"15", "16", "9"}) != filter_keys(new_xml, {"15", "16", "9"})
    apf_added = any(dict(item).get("T") in ("19", "20") for item in added)
    forbidden_added = [
        dict(item) for item in added
        if dict(item).get("T") not in ({"17", "19", "20"} if allow_apf else {"17"})
    ]
    errors = []
    measurement_lint = []
    if data_root is not None and target is not None:
        measurement_lint = _measurement_lint(
            data_root.resolve(), baseline, target.resolve(),
            role_map.resolve() if role_map else None,
            repeatability_folder.resolve() if repeatability_folder else None,
            old_xml, new_xml,
        )
    if delay_changed and not allow_delay:
        errors.append("delay_changed")
    if polarity_changed and not allow_polarity:
        errors.append("polarity_changed")
    if output_attributes_changed:
        errors.append("unrelated_output_attributes_changed")
    if not output_trim_valid:
        errors.append("unapproved_output_volume_changed")
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
    if measurement_lint:
        errors.append("measurement_guardrail_failed")

    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "pass": not errors,
        "errors": errors,
        "peq_only": not delay_changed and not crossover_changed and not apf_added and not forbidden_added and not removed_nonfree,
        "delay_changed": delay_changed,
        "polarity_changed": polarity_changed,
        "output_attributes_changed": output_attributes_changed,
        "output_volume_changes_db": {str(key): round(value, 4) for key, value in output_volume_changes.items()},
        "protective_output_trim_valid": output_trim_valid,
        "time_alignment_attributes_changed": delay_attributes_changed,
        "crossover_changed": crossover_changed,
        "apf_changed": apf_added,
        "added_filter_types": added_types,
        "removed_filter_types": removed_types,
        "added_filter_count": len(added),
        "removed_filter_count": len(removed),
        "removed_nonfree_filter_count": len(removed_nonfree),
        "unknown_field_changes": forbidden_added,
        "measurement_guardrail_errors": measurement_lint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a written AFPX candidate only changed intended fields.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-delay", action="store_true")
    parser.add_argument("--allow-apf", action="store_true")
    parser.add_argument("--allow-polarity", action="store_true")
    parser.add_argument("--allow-output-trim", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--role-map", type=Path)
    parser.add_argument("--repeatability-folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("latest_verify_written_tune.json"))
    args = parser.parse_args()

    payload = verify(
        args.baseline.resolve(), args.candidate.resolve(), args.allow_delay,
        args.allow_apf, args.allow_polarity, args.allow_output_trim,
        args.data_root, args.target, args.role_map, args.repeatability_folder,
    )
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
