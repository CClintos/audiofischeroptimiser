from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HIGH_L = ("Front L High.txt", "Front L Tweeter.txt", "Front Left High.txt", "Front Left Tweeter.txt")
HIGH_R = ("Front R High.txt", "Front R Tweeter.txt", "Front Right High.txt", "Front Right Tweeter.txt")
MID_L = ("Front L Mid.txt", "Front L MID.txt", "Front L Midrange.txt", "Front Left Mid.txt")
MID_R = ("Front R Mid.txt", "Front R MID.txt", "Front R Midrange.txt", "Front Right Mid.txt")
LOW_L = ("Front L Low.txt", "Front L Midbass.txt", "Front L Mid Bass.txt", "Front Left Low.txt")
LOW_R = ("Front R Low.txt", "Front R Midbass.txt", "Front R Mid Bass.txt", "Front Right Low.txt")
MID_PAIR = ("Both Mids.txt", "Mids Together.txt", "Midrange Together.txt")
LOW_PAIR = ("Mid Bass Together.txt", "Both Midbass.txt", "Both Midbasses.txt", "Both Mid Bass.txt")
SUB = ("Sub.txt", "SUB.txt", "Subwoofer.txt")
SYSTEM = ("System Sum.txt", "SYSTEM SUM.txt")
HIGH_PAIR = ("Tweeters Together.txt", "Both Tweeters.txt")
POSITION_PREFIXES = {
    "left": ("Left Ear ", "Left "),
    "right": ("Right Ear ", "Right "),
}


def first_existing(root: Path, aliases: tuple[str, ...]) -> Path | None:
    for name in aliases:
        p = root / name
        if p.exists():
            return p
    return None


def first_position_existing(root: Path, prefixes: tuple[str, ...], aliases: tuple[str, ...]) -> Path | None:
    for prefix in prefixes:
        for alias in aliases:
            for path in (root / (prefix + alias), root / prefix.strip() / alias):
                if path.exists():
                    return path
    return None


def companion_impulse(root: Path, measurement: Path) -> Path | None:
    stem = measurement.stem
    names = (
        f"{stem}.wav", f"{stem} Impulse.wav", f"{stem} IR.wav",
        f"{stem} Impulse.txt", f"{stem} IR.txt",
    )
    for folder in (root, root / "impulses", root / "Impulse", root / "IR"):
        for name in names:
            path = folder / name
            if path.exists():
                return path
    return None


def rew_metadata(path: Path) -> dict[str, object]:
    """Read one export once and return only compact trust/provenance fields."""
    text = path.read_text(encoding="utf-8", errors="replace")
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    delay = re.search(r"\bDelay\s+" + number + r"\s+ms", text[:3000])
    volume = re.search(r"\bvolume:\s*" + number, text[:3000], re.I)
    sweep = re.search(r"\b(?:sweep|sweeps)\s+at\s+" + number + r"\s+dBFS", text[:3000], re.I)
    timing = re.search(r"reference played from\s+([^\r\n]+?)(?:\s+with|$)", text[:3000], re.I | re.M)
    rows = 0
    columns = 0
    first_frequency = None
    last_frequency = None
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("*"):
            continue
        numeric = []
        for part in value.replace(",", " ").split():
            try:
                numeric.append(float(part))
            except ValueError:
                break
        if len(numeric) < 2:
            continue
        rows += 1
        columns = max(columns, len(numeric))
        if first_frequency is None:
            first_frequency = numeric[0]
        last_frequency = numeric[0]
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns,
        "start_hz": first_frequency,
        "end_hz": last_frequency,
        "phase": columns >= 3,
        "coherence": columns >= 4,
        "position_id": columns >= 5,
        "delay_ms": float(delay.group(1)) if delay else None,
        "source_volume": float(volume.group(1)) if volume else None,
        "sweep_dbfs": float(sweep.group(1)) if sweep else None,
        "timing_reference": timing.group(1).strip() if timing else "",
    }


def measurement_spec(layout: str) -> dict[str, tuple[str, ...]]:
    spec = {
        "System Sum": SYSTEM,
        "Sub": SUB,
        "FL High": HIGH_L,
        "FR High": HIGH_R,
        "Tweeters Together": HIGH_PAIR,
    }
    if layout == "front_3way_plus_sub":
        spec.update({
            "FL Mid": MID_L,
            "FR Mid": MID_R,
            "Mids Together": MID_PAIR,
            "FL Low": LOW_L,
            "FR Low": LOW_R,
            "Mid Bass Together": LOW_PAIR,
        })
    else:
        spec.update({
            "FL Low": LOW_L + MID_L,
            "FR Low": LOW_R + MID_R,
            "Mid Bass Together": LOW_PAIR + MID_PAIR,
        })
    return spec


def detect_layout(root: Path) -> str:
    has_mid = all(first_existing(root, aliases) for aliases in (MID_L, MID_R, MID_PAIR))
    has_low = all(first_existing(root, aliases) for aliases in (LOW_L, LOW_R, LOW_PAIR))
    return "front_3way_plus_sub" if has_mid and has_low else "front_2way_plus_sub"


def build_manifest(root: Path, baseline: Path | None, target: Path | None) -> dict[str, object]:
    layout = detect_layout(root)
    spec = measurement_spec(layout)
    present: list[str] = []
    missing: list[str] = []
    resolved: dict[str, str] = {}
    phase_files: list[str] = []
    coherence_files: list[str] = []
    position_files: list[str] = []
    metadata: dict[str, dict[str, object]] = {}
    impulse_files: dict[str, str] = {}
    spatial_bundles: dict[str, dict[str, str]] = {}

    for role, aliases in spec.items():
        found = first_existing(root, aliases)
        if found is None:
            missing.append(aliases[0])
            continue
        present.append(found.name)
        resolved[role] = str(found)
        info = rew_metadata(found)
        metadata[role] = info
        impulse = companion_impulse(root, found)
        if impulse is not None:
            impulse_files[role] = str(impulse)
        if info["phase"]:
            phase_files.append(found.name)
        if info["coherence"]:
            coherence_files.append(found.name)
        if info["position_id"]:
            position_files.append(found.name)

    for position, prefixes in POSITION_PREFIXES.items():
        bundle = {}
        for role, aliases in spec.items():
            found = first_position_existing(root, prefixes, aliases)
            if found is not None:
                bundle[role] = str(found)
                if role == "System Sum":
                    key = f"{position}:System Sum"
                    resolved[key] = str(found)
                    metadata[key] = rew_metadata(found)
        if bundle:
            spatial_bundles[position] = bundle

    baseline_path = baseline or first_existing(root, ("baseline.afpx",))
    target_path = target or first_existing(root, ("ResoNix Target Curve 2026.txt", "target.txt"))
    phase_available = len(phase_files) >= 2
    impulse_available = len(impulse_files) >= 2

    warnings: list[str] = []
    if missing:
        warnings.append(f"missing_required_measurements:{len(missing)}")
    if not baseline_path or not baseline_path.exists():
        warnings.append("baseline_missing")
    if not target_path or not target_path.exists():
        warnings.append("target_missing")
    volumes = sorted({info["source_volume"] for info in metadata.values() if info["source_volume"] is not None})
    if len(volumes) > 1:
        warnings.append("measurement_source_volume_changed")
    timing_refs = sorted({
        str(info["timing_reference"])
        for role, info in metadata.items()
        if ":" not in role and info["timing_reference"]
    })
    if len(timing_refs) > 1:
        warnings.append("mixed_timing_references")
    grids = {
        (round(float(info["start_hz"]), 3), round(float(info["end_hz"]), 3), int(info["rows"]))
        for info in metadata.values()
        if info["start_hz"] is not None and info["end_hz"] is not None
    }
    if len(grids) > 1:
        warnings.append("measurement_frequency_grids_differ")
    if not phase_available and impulse_available:
        warnings.append("phase_unavailable_impulse_timing_only")
    elif not phase_available:
        warnings.append("phase_unavailable_peq_only")
    delays = [
        float(info["delay_ms"])
        for role, info in metadata.items()
        if ":" not in role and info["delay_ms"] is not None
    ]
    return {
        "measurement_folder": str(root),
        "baseline_afpx": str(baseline_path) if baseline_path else "",
        "baseline_exists": bool(baseline_path and baseline_path.exists()),
        "target_curve": str(target_path) if target_path else "",
        "target_exists": bool(target_path and target_path.exists()),
        "measurements_present": sorted(present),
        "measurements_missing": sorted(missing),
        "resolved_roles": resolved,
        "detected_layout": layout,
        "phase_available": phase_available,
        "phase_files": sorted(phase_files),
        "coherence_files": sorted(coherence_files),
        "position_id_files": sorted(position_files),
        "spatial_bundles": spatial_bundles,
        "impulse_files": impulse_files,
        "measurement_metadata": metadata,
        "measurement_conditions": {
            "source_volumes": volumes,
            "timing_references": timing_refs,
            "phase_delay_range_ms": [] if not delays else [round(min(delays), 4), round(max(delays), 4)],
            "frequency_grid_count": len(grids),
        },
        "warnings": warnings,
        "safe_mode": "crossover_ladder_available" if phase_available or impulse_available else "magnitude_only_peq",
    }


def compact_manifest(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "measurement_folder": manifest["measurement_folder"],
        "detected_layout": manifest["detected_layout"],
        "safe_mode": manifest["safe_mode"],
        "baseline_afpx": manifest["baseline_afpx"],
        "target_curve": manifest["target_curve"],
        "measurement_count": len(manifest["measurements_present"]),
        "missing": manifest["measurements_missing"],
        "phase_file_count": len(manifest["phase_files"]),
        "coherence_file_count": len(manifest["coherence_files"]),
        "impulse_file_count": len(manifest["impulse_files"]),
        "spatial_positions": sorted(manifest.get("spatial_bundles", {})),
        "warnings": manifest["warnings"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact JSON manifest for an AFPX optimiser input folder.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("latest_measurement_manifest.json"))
    parser.add_argument("--print-mode", choices=("compact", "full", "none"), default="compact")
    args = parser.parse_args()

    manifest = build_manifest(args.root.resolve(), args.baseline, args.target)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.print_mode != "none":
        payload = manifest if args.print_mode == "full" else compact_manifest(manifest)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
