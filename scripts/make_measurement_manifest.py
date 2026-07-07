from __future__ import annotations

import argparse
import json
from pathlib import Path


HIGH_L = ("Front L High.txt", "Front L Tweeter.txt")
HIGH_R = ("Front R High.txt", "Front R Tweeter.txt")
MID_L = ("Front L Mid.txt", "Front L MID.txt", "Front L Midrange.txt")
MID_R = ("Front R Mid.txt", "Front R MID.txt", "Front R Midrange.txt")
LOW_L = ("Front L Low.txt", "Front L Midbass.txt", "Front L Mid Bass.txt")
LOW_R = ("Front R Low.txt", "Front R Midbass.txt", "Front R Mid Bass.txt")
MID_PAIR = ("Both Mids.txt", "Mids Together.txt", "Midrange Together.txt")
LOW_PAIR = ("Mid Bass Together.txt", "Both Midbass.txt", "Both Midbasses.txt", "Both Mid Bass.txt")
SUB = ("Sub.txt", "SUB.txt")
SYSTEM = ("System Sum.txt", "SYSTEM SUM.txt")
HIGH_PAIR = ("Tweeters Together.txt", "Both Tweeters.txt")


def first_existing(root: Path, aliases: tuple[str, ...]) -> Path | None:
    for name in aliases:
        p = root / name
        if p.exists():
            return p
    return None


def has_numeric_column(path: Path, min_cols: int) -> bool:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("*") or s[0].isalpha():
                continue
            vals = []
            for part in s.replace(",", " ").split()[:min_cols]:
                try:
                    vals.append(float(part))
                except ValueError:
                    break
            if len(vals) >= min_cols:
                return True
    except OSError:
        return False
    return False


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

    for role, aliases in spec.items():
        found = first_existing(root, aliases)
        if found is None:
            missing.append(aliases[0])
            continue
        present.append(found.name)
        resolved[role] = str(found)
        if has_numeric_column(found, 3):
            phase_files.append(found.name)
        if has_numeric_column(found, 4):
            coherence_files.append(found.name)

    baseline_path = baseline or first_existing(root, ("baseline.afpx",))
    target_path = target or first_existing(root, ("ResoNix Target Curve 2026.txt", "target.txt"))
    phase_available = len(phase_files) >= 2

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
        "safe_mode": "phase_delay_apf_available" if phase_available else "magnitude_only_peq",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact JSON manifest for an AFPX optimiser input folder.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("latest_measurement_manifest.json"))
    args = parser.parse_args()

    manifest = build_manifest(args.root.resolve(), args.baseline, args.target)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
