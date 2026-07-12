from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
import struct
from pathlib import Path


def decode_afpx(path: Path) -> str:
    raw = path.read_bytes()
    return zlib.decompress(raw[4:]).decode("utf-8", "replace")


def attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'([A-Za-z]+)="([^"]*)"', tag))


def channel_blocks(xml: str) -> list[str]:
    return re.findall(r"<OC\b.*?</OC>", xml, re.S)


def filters(block: str) -> list[str]:
    return re.findall(r"<Fil\b[^>]*/?>", block)


def sem_filter(tag: str) -> tuple[tuple[str, str | None], ...]:
    a = attrs(tag)
    return tuple((k, a.get(k)) for k in ("T", "F", "Q", "G", "dF", "I"))


def multiset_added(old: list[object], new: list[object]) -> list[object]:
    counts: dict[object, int] = {}
    for item in old:
        counts[item] = counts.get(item, 0) + 1
    added = []
    for item in new:
        if counts.get(item, 0):
            counts[item] -= 1
        else:
            added.append(item)
    return added


def added_filters(baseline_xml: str | None, candidate_xml: str) -> list[dict[str, object]]:
    base_blocks = channel_blocks(baseline_xml) if baseline_xml else []
    cand_blocks = channel_blocks(candidate_xml)
    out: list[dict[str, object]] = []
    for ch, block in enumerate(cand_blocks):
        old = [sem_filter(f) for f in filters(base_blocks[ch])] if ch < len(base_blocks) else []
        new_tags = filters(block)
        new = [sem_filter(f) for f in new_tags]
        added = multiset_added(old, new)
        for sig in added:
            tag = next((f for f in new_tags if sem_filter(f) == sig), "")
            a = attrs(tag)
            out.append({
                "channel": ch,
                "type": a.get("T", ""),
                "frequency_hz": float(a.get("F", "0") or 0),
                "q": float(a.get("Q", "0") or 0),
                "gain_db": float(a.get("G", "0") or 0),
                "dF": a.get("dF", ""),
            })
    return out


def risk(filters_out: list[dict[str, object]]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    peqs = [f for f in filters_out if f["type"] == "17"]
    boosts = [f for f in peqs if float(f["gain_db"]) > 2.5]
    high_q = [f for f in peqs if float(f["q"]) > 3.0]
    deep = [f for f in peqs if float(f["gain_db"]) < -5.0]
    apf = [f for f in filters_out if f["type"] in ("19", "20")]
    if boosts:
        warnings.append("large_boosts")
    if high_q:
        warnings.append("high_q_filters")
    if deep:
        warnings.append("deep_cuts")
    if apf:
        warnings.append("apf_present_verify_phase")
    if len(peqs) > 10:
        warnings.append("many_filters")
    level = "high" if len(warnings) >= 3 else "medium" if warnings else "low"
    return level, warnings


def summarise(candidate: Path, baseline: Path | None) -> dict[str, object]:
    candidate_xml = decode_afpx(candidate)
    baseline_xml = decode_afpx(baseline) if baseline else None
    added = added_filters(baseline_xml, candidate_xml)
    peqs = [f for f in added if f["type"] == "17"]
    level, warnings = risk(added)
    return {
        "candidate": str(candidate),
        "baseline": str(baseline) if baseline else "",
        "added_filter_count": len(added),
        "added_peq_count": len(peqs),
        "max_boost_db": max([float(f["gain_db"]) for f in peqs] + [0.0]),
        "max_cut_db": min([float(f["gain_db"]) for f in peqs] + [0.0]),
        "max_q": max([float(f["q"]) for f in peqs] + [0.0]),
        "warnings": warnings,
        "risk": level,
        "added_filters": added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise candidate AFPX filter changes as compact JSON.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("latest_candidate_filter_summary.json"))
    parser.add_argument("--print-mode", choices=("compact", "full", "none"), default="compact")
    args = parser.parse_args()

    payload = summarise(args.candidate.resolve(), args.baseline.resolve() if args.baseline else None)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.print_mode == "none":
        return
    if args.print_mode == "full":
        print(json.dumps(payload, indent=2))
        return
    print(json.dumps({
        "candidate": payload["candidate"],
        "baseline": payload["baseline"],
        "added_filter_count": payload["added_filter_count"],
        "risk": payload["risk"],
        "warnings": payload["warnings"],
        "added_filters": payload["added_filters"],
        "summary_file": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
