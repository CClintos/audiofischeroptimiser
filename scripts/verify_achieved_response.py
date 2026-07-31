from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.make_measurement_manifest import load_role_map, mapped_measurement, measurement_spec


def _load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            frequency, level = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if frequency > 0.0 and math.isfinite(level):
            rows.append((frequency, level))
    if len(rows) < 16:
        raise ValueError(f"No usable REW magnitude trace found in {path}")
    rows.sort()
    return np.asarray([row[0] for row in rows]), np.asarray([row[1] for row in rows])


def _resolve_role(folder: Path, role: str, role_map: dict[str, str]) -> Path | None:
    mapped = mapped_measurement(folder, role, role_map)
    if mapped is not None:
        return mapped
    for filename in measurement_spec("front_3way_plus_sub").get(role, ()):
        path = folder / filename
        if path.is_file():
            return path
    return None


def _aligned_achieved(freqs: np.ndarray, predicted: np.ndarray,
                      measured_f: np.ndarray, measured_db: np.ndarray,
                      band: tuple[float, float]) -> tuple[np.ndarray, float]:
    achieved = np.interp(np.log10(freqs), np.log10(measured_f), measured_db)
    selected = (freqs >= band[0]) & (freqs <= band[1])
    offset = float(np.median(achieved[selected] - predicted[selected])) if np.any(selected) else 0.0
    return achieved - offset, offset


def _rms(values: np.ndarray, selected: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values[selected] ** 2))) if np.any(selected) else float("nan")


def verify_run(run_folder: Path, post_measurements: Path,
               role_map_path: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    run_folder = Path(run_folder).resolve()
    post_measurements = Path(post_measurements).resolve()
    merged = run_folder / "_merged_top" if (run_folder / "_merged_top").is_dir() else run_folder
    full_path = merged / "optimizer_summary.json"
    if not full_path.is_file():
        raise FileNotFoundError(f"optimizer_summary.json not found under {run_folder}")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    plot = dict(full.get("response_plot") or {})
    freqs = np.asarray(plot.get("frequency_hz") or [], dtype=float)
    baseline_error = np.asarray(plot.get("baseline_error_db") or [], dtype=float)
    candidate_error = np.asarray(plot.get("candidate_error_db") or [], dtype=float)
    if len(freqs) < 16 or len(candidate_error) != len(freqs):
        raise ValueError("The run does not contain a usable predicted response plot")

    original_folder = Path(str(full.get("data_root", "")))
    role_map = load_role_map(role_map_path)
    original_system_path = _resolve_role(original_folder, "System Sum", role_map)
    post_system_path = _resolve_role(post_measurements, "System Sum", role_map)
    if original_system_path is None or post_system_path is None:
        raise FileNotFoundError("System Sum is required in both the original and post-load folders")
    original_f, original_system = _load_trace(original_system_path)
    original_aligned = np.interp(np.log10(freqs), np.log10(original_f), original_system)
    anchored_target = original_aligned - baseline_error
    predicted_system = anchored_target + candidate_error
    post_f, post_system = _load_trace(post_system_path)
    achieved_system, system_offset = _aligned_achieved(
        freqs, predicted_system, post_f, post_system, (300.0, 3000.0)
    )
    system_difference = achieved_system - predicted_system
    full_band = (freqs >= 60.0) & (freqs <= 16000.0)
    vocal_band = (freqs >= 200.0) & (freqs <= 6000.0)

    drivers = {}
    for role, driver_plot in dict(plot.get("drivers") or {}).items():
        original_path = _resolve_role(original_folder, role, role_map)
        post_path = _resolve_role(post_measurements, role, role_map)
        if original_path is None or post_path is None:
            continue
        driver_f = np.asarray(driver_plot.get("frequency_hz") or [], dtype=float)
        change = np.asarray(driver_plot.get("change_db") or [], dtype=float)
        if len(driver_f) < 8 or len(change) != len(driver_f):
            continue
        first_f, first_db = _load_trace(original_path)
        after_f, after_db = _load_trace(post_path)
        predicted = np.interp(np.log10(driver_f), np.log10(first_f), first_db) + change
        branch_band = (1800.0, 16000.0) if role.endswith("High") else (80.0, 5000.0)
        achieved, offset = _aligned_achieved(driver_f, predicted, after_f, after_db, branch_band)
        selected = (driver_f >= branch_band[0]) & (driver_f <= branch_band[1])
        drivers[role] = {
            "frequency_hz": [round(float(value), 3) for value in driver_f],
            "predicted_db": [round(float(value), 3) for value in predicted],
            "achieved_db": [round(float(value), 3) for value in achieved],
            "difference_rms_db": round(_rms(achieved - predicted, selected), 3),
            "capture_alignment_db": round(offset, 3),
        }

    system_rms = _rms(system_difference, full_band)
    vocal_rms = _rms(system_difference, vocal_band)
    verdict = (
        "model_matched_measurement" if system_rms <= 0.75
        else "mixed_prediction_accuracy" if system_rms <= 1.50
        else "model_mismatch_remeasure_and_review"
    )
    payload = {
        "schema": "audiofischer-achieved-verification-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_folder": str(run_folder),
        "post_measurements": str(post_measurements),
        "candidate": str((full.get("top_candidates") or [{}])[0].get("file", "")),
        "verdict": verdict,
        "system": {
            "frequency_hz": [round(float(value), 3) for value in freqs],
            "predicted_db": [round(float(value), 3) for value in predicted_system],
            "achieved_db": [round(float(value), 3) for value in achieved_system],
            "difference_db": [round(float(value), 3) for value in system_difference],
            "difference_rms_db": round(system_rms, 3),
            "vocal_difference_rms_db": round(vocal_rms, 3),
            "capture_alignment_db": round(system_offset, 3),
        },
        "drivers": drivers,
        "note": (
            "Capture level was aligned over 300-3000 Hz. The comparison tests "
            "response shape; it does not claim absolute SPL calibration."
        ),
    }
    destination = out or (
        run_folder / "verification"
        / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["file"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predicted and post-load measured response.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("post_measurements", type=Path)
    parser.add_argument("--role-map", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(
        args.run_folder, args.post_measurements, args.role_map, args.out
    ), indent=2))


if __name__ == "__main__":
    main()
