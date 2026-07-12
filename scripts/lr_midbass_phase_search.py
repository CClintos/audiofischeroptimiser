"""Targeted L/R-midbass phase search for phase-valid REW measurements.

This stage exists because PEQ and mid/tweeter crossover searches must not try
to fill a low-frequency L/R cancellation. It validates the complex solo sum,
finds the strongest repeatable cancellation, and searches a paired all-pass
correction whose relative phase returns close to zero outside the problem band.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _make_v3 import afpx_roundtrip_lint, decode_afpx, encode_afpx
from _optimizer import add_allpass_to_oc, replace_oc_blocks
from _tunefit import allpass_H


ALIASES = {
    "fl_mid": ("Front L Mid.txt", "Front Left Mid.txt", "Front L Low.txt"),
    "fr_mid": ("Front R Mid.txt", "Front Right Mid.txt", "Front R Low.txt"),
    "together": ("Both Mids.txt", "Mid Bass Together.txt", "Mids Together.txt", "Front Stage.txt"),
    "left_ear": ("Left Ear Both Mids.txt",),
    "right_ear": ("Right Ear Both Mids.txt",),
}


def resolve(root: Path, key: str, required: bool = True) -> Path | None:
    for name in ALIASES[key]:
        path = root / name
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"missing {key} measurement; tried {ALIASES[key]}")
    return None


def load_rew(path: Path) -> Dict[str, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.replace(",", " ").split()
        try:
            row = [float(parts[0]), float(parts[1])]
            if len(parts) >= 3:
                row.append(float(parts[2]))
        except (ValueError, IndexError):
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"no numeric REW rows in {path}")
    width = max(len(row) for row in rows)
    arr = np.asarray([row + [float("nan")] * (width - len(row)) for row in rows])
    out = {"freq": arr[:, 0], "spl": arr[:, 1]}
    if width >= 3:
        out["phase"] = arr[:, 2]
    return out


def extract_delay_ms(path: Path) -> float | None:
    head = path.read_text(encoding="utf-8", errors="replace")[:1800]
    match = re.search(r"Delay\s+([-+0-9.]+)\s+ms", head)
    return float(match.group(1)) if match else None


def complex_on_grid(trace: Dict[str, np.ndarray], freqs: np.ndarray) -> np.ndarray:
    if "phase" not in trace:
        raise ValueError("phase column is required for L/R-midbass phase search")
    mag = np.interp(freqs, trace["freq"], trace["spl"])
    phase = np.unwrap(np.deg2rad(trace["phase"]))
    phase = np.interp(freqs, trace["freq"], phase)
    return 10.0 ** (mag / 20.0) * np.exp(1j * phase)


def smooth_db(values: np.ndarray, points_per_octave: int, octave_fraction: float) -> np.ndarray:
    sigma = (octave_fraction / 2.355) * points_per_octave
    power = gaussian_filter1d(10.0 ** (values / 10.0), sigma, mode="nearest")
    return 10.0 * np.log10(np.maximum(power, 1e-24))


def rms(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values[mask] ** 2))) if np.any(mask) else float("inf")


def local_min(freqs: np.ndarray, values: np.ndarray, band: Tuple[float, float]) -> Dict[str, float]:
    sel = (freqs >= band[0]) & (freqs <= band[1])
    index = int(np.argmin(values[sel]))
    return {
        "db": round(float(values[sel][index]), 2),
        "frequency_hz": round(float(freqs[sel][index]), 2),
    }


class MidbassPhaseModel:
    def __init__(self, data_root: Path, points_per_octave: int = 96):
        self.data_root = data_root
        self.points_per_octave = points_per_octave
        self.paths = {
            key: resolve(data_root, key, required=key in ("fl_mid", "fr_mid", "together"))
            for key in ALIASES
        }
        self.fl_trace = load_rew(self.paths["fl_mid"])  # type: ignore[arg-type]
        self.fr_trace = load_rew(self.paths["fr_mid"])  # type: ignore[arg-type]
        lo = max(60.0, float(self.fl_trace["freq"][0]), float(self.fr_trace["freq"][0]))
        hi = min(2500.0, float(self.fl_trace["freq"][-1]), float(self.fr_trace["freq"][-1]))
        self.freqs = 2.0 ** np.arange(
            math.log2(lo), math.log2(hi) + 1.0 / points_per_octave, 1.0 / points_per_octave
        )
        self.fl = complex_on_grid(self.fl_trace, self.freqs)
        self.fr = complex_on_grid(self.fr_trace, self.freqs)
        self.current = 20.0 * np.log10(np.abs(self.fl + self.fr) + 1e-12)
        self.ceiling = 20.0 * np.log10(np.abs(self.fl) + np.abs(self.fr) + 1e-12)
        self.current_12 = smooth_db(self.current, points_per_octave, 1.0 / 12.0)
        self.current_24 = smooth_db(self.current, points_per_octave, 1.0 / 24.0)
        self.ceiling_12 = smooth_db(self.ceiling, points_per_octave, 1.0 / 12.0)
        self.ceiling_24 = smooth_db(self.ceiling, points_per_octave, 1.0 / 24.0)
        self.problem_frequency_hz, self.problem_gap_db = self._detect_problem()
        # Keep the focus narrow enough to fix the detected cancellation instead
        # of letting healthy response on either side dilute it.
        half_octave = 0.16
        self.focus_band = (
            self.problem_frequency_hz / (2.0 ** half_octave),
            self.problem_frequency_hz * (2.0 ** half_octave),
        )
        self.broad_band = (
            max(80.0, self.problem_frequency_hz / 1.35),
            min(650.0, self.problem_frequency_hz * 3.0),
        )

    def _detect_problem(self) -> Tuple[float, float]:
        level_delta = np.abs(20.0 * np.log10(np.abs(self.fl) / np.maximum(np.abs(self.fr), 1e-12)))
        valid = (
            (self.freqs >= 100.0)
            & (self.freqs <= 600.0)
            & (level_delta <= 10.0)
        )
        gap = self.ceiling_24 - self.current_24
        peaks, _ = find_peaks(np.where(valid, gap, -99.0), distance=8, prominence=0.5)
        if len(peaks):
            index = int(peaks[np.argmax(gap[peaks])])
        else:
            index = int(np.argmax(np.where(valid, gap, -99.0)))
        return float(self.freqs[index]), float(gap[index])

    def validate(self, threshold_db: float = 2.5) -> Dict[str, object]:
        together = load_rew(self.paths["together"])  # type: ignore[arg-type]
        measured = np.interp(self.freqs, together["freq"], together["spl"])
        sel = (self.freqs >= 100.0) & (self.freqs <= 300.0)
        error = measured - self.current
        lock_fl = extract_delay_ms(self.paths["fl_mid"])  # type: ignore[arg-type]
        lock_fr = extract_delay_ms(self.paths["fr_mid"])  # type: ignore[arg-type]
        lock_delta = None if lock_fl is None or lock_fr is None else abs(lock_fl - lock_fr)
        error_rms = rms(error, sel)
        return {
            "together_file": str(self.paths["together"]),
            "predicted_sum_rms_db_100_300": round(error_rms, 3),
            "predicted_sum_level_bias_db": round(float(np.mean(error[sel])), 3),
            "solo_delay_lock_delta_ms": None if lock_delta is None else round(lock_delta, 4),
            "pass": bool(error_rms <= threshold_db and lock_delta is not None and lock_delta <= 0.75),
            "threshold_db": float(threshold_db),
        }

    def position_audit(self) -> Dict[str, object]:
        out = {}
        audit_band = (self.problem_frequency_hz / 1.18, self.problem_frequency_hz * 1.18)
        for key in ("left_ear", "right_ear"):
            path = self.paths.get(key)
            if path is None:
                continue
            trace = load_rew(path)
            values = np.interp(self.freqs, trace["freq"], trace["spl"])
            out[key] = {"file": str(path), "current_raw_min": local_min(self.freqs, values, audit_band)}
        return out

    def evaluate(self, params: Iterable[float], max_damage_12_db: float) -> Dict[str, object]:
        log_fl_f, fl_q, log_fr_f, fr_q = [float(value) for value in params]
        fl_f, fr_f = math.exp(log_fl_f), math.exp(log_fr_f)
        fl_h = allpass_H(self.freqs, fl_f, fl_q)
        fr_h = allpass_H(self.freqs, fr_f, fr_q)
        predicted = 20.0 * np.log10(np.abs(self.fl * fl_h + self.fr * fr_h) + 1e-12)
        pred_12 = smooth_db(predicted, self.points_per_octave, 1.0 / 12.0)
        pred_24 = smooth_db(predicted, self.points_per_octave, 1.0 / 24.0)
        gap_12 = np.maximum(self.ceiling_12 - pred_12, 0.0)
        damage_12 = np.maximum(self.current_12 - pred_12 - 0.25, 0.0)
        damage_24 = np.maximum(self.current_24 - pred_24 - 0.35, 0.0)
        focus = (self.freqs >= self.focus_band[0]) & (self.freqs <= self.focus_band[1])
        broad = (self.freqs >= self.broad_band[0]) & (self.freqs <= self.broad_band[1])
        guard = (self.freqs >= 80.0) & (self.freqs <= 1800.0)
        vocal = (self.freqs >= 600.0) & (self.freqs <= 1800.0)
        focus_gap = rms(gap_12, focus)
        broad_gap = rms(gap_12, broad)
        damage_rms = rms(damage_12, guard)
        max_damage = float(np.max(damage_12[guard]))
        max_vocal_damage = float(np.max(damage_24[vocal]))
        score = (
            1.4 * focus_gap
            + 0.7 * broad_gap
            + 1.5 * damage_rms
            + 0.5 * max_damage
            + 0.35 * max_vocal_damage
            + 20.0 * max(0.0, max_damage - max_damage_12_db) ** 2
            + 12.0 * max(0.0, max_vocal_damage - 1.5) ** 2
        )
        index = int(np.argmin(np.abs(self.freqs - self.problem_frequency_hz)))
        low_band = (self.problem_frequency_hz / 1.25, self.problem_frequency_hz * 1.25)
        return {
            "objective": float(score),
            "fl_apf_f": float(fl_f),
            "fl_apf_q": float(fl_q),
            "fr_apf_f": float(fr_f),
            "fr_apf_q": float(fr_q),
            "focus_gap_12_rms_db": focus_gap,
            "broad_gap_12_rms_db": broad_gap,
            "damage_12_rms_db": damage_rms,
            "max_damage_12_db": max_damage,
            "max_vocal_damage_24_db": max_vocal_damage,
            "raw_lift_at_problem_db": float(predicted[index] - self.current[index]),
            "current_raw_min": local_min(self.freqs, self.current, low_band),
            "candidate_raw_min": local_min(self.freqs, predicted, low_band),
            "current_12_min": local_min(self.freqs, self.current_12, low_band),
            "candidate_12_min": local_min(self.freqs, pred_12, low_band),
            "current_24_min": local_min(self.freqs, self.current_24, low_band),
            "candidate_24_min": local_min(self.freqs, pred_24, low_band),
        }

    def optimize(self, seed: int, max_damage_12_db: float) -> Dict[str, object]:
        f_lo = max(80.0, self.problem_frequency_hz / (2.0 ** 0.75))
        f_hi = min(800.0, self.problem_frequency_hz * (2.0 ** 0.75))
        bounds = [
            (math.log(f_lo), math.log(f_hi)), (0.5, 2.0),
            (math.log(f_lo), math.log(f_hi)), (0.5, 2.0),
        ]
        result = differential_evolution(
            lambda values: float(self.evaluate(values, max_damage_12_db)["objective"]),
            bounds,
            seed=seed,
            popsize=24,
            maxiter=220,
            tol=1e-8,
            polish=True,
            workers=1,
            updating="immediate",
        )
        return self.evaluate(result.x, max_damage_12_db)


def write_candidate(baseline: Path, output: Path, result: Dict[str, object]) -> Dict[str, object]:
    base_xml = decode_afpx(baseline)
    blocks = re.findall(r"<OC\b.*?</OC>", base_xml, re.S)
    if len(blocks) < 4:
        raise ValueError("AFPX has fewer than four output-channel blocks")
    blocks[2] = add_allpass_to_oc(blocks[2], float(result["fl_apf_f"]), float(result["fl_apf_q"]))
    blocks[3] = add_allpass_to_oc(blocks[3], float(result["fr_apf_f"]), float(result["fr_apf_q"]))
    candidate_xml = replace_oc_blocks(base_xml, blocks)
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_afpx(candidate_xml, output)
    written_xml = decode_afpx(output)
    lint = afpx_roundtrip_lint(base_xml, written_xml, allowed_added_types=("20",))
    if not lint["pass"]:
        raise AssertionError("AFPX lint failed: " + "; ".join(lint["errors"]))
    return lint


def run(args: argparse.Namespace) -> Dict[str, object]:
    args.out_root.mkdir(parents=True, exist_ok=True)
    model = MidbassPhaseModel(args.data_root, args.points_per_octave)
    validation = model.validate(args.validation_threshold)
    summary: Dict[str, object] = {
        "mode": "lr_midbass_paired_apf",
        "baseline": str(args.baseline),
        "data_root": str(args.data_root),
        "problem_frequency_hz": round(model.problem_frequency_hz, 2),
        "problem_gap_24_db": round(model.problem_gap_db, 2),
        "focus_band_hz": [round(value, 2) for value in model.focus_band],
        "broad_band_hz": [round(value, 2) for value in model.broad_band],
        "validation": validation,
        "position_audit": model.position_audit(),
    }
    if not validation["pass"]:
        summary["status"] = "blocked_bad_phase_validation"
        summary["reason"] = "solo complex sum did not reproduce the measured together trace"
    elif model.problem_gap_db < args.minimum_problem_gap:
        summary["status"] = "no_material_lr_midbass_cancellation"
    else:
        baseline_params = [
            math.log(model.problem_frequency_hz), 0.7,
            math.log(model.problem_frequency_hz), 0.7,
        ]
        baseline_result = model.evaluate(baseline_params, args.max_damage_12)
        result = model.optimize(args.seed, args.max_damage_12)
        rounded = dict(result)
        for key in ("fl_apf_f", "fr_apf_f"):
            rounded[key] = round(float(rounded[key]), 1)
        for key in ("fl_apf_q", "fr_apf_q"):
            rounded[key] = round(float(rounded[key]), 2)
        rounded_params = [
            math.log(float(rounded["fl_apf_f"])), float(rounded["fl_apf_q"]),
            math.log(float(rounded["fr_apf_f"])), float(rounded["fr_apf_q"]),
        ]
        rounded_result = model.evaluate(rounded_params, args.max_damage_12)
        rounded_result.update({key: rounded[key] for key in ("fl_apf_f", "fl_apf_q", "fr_apf_f", "fr_apf_q")})
        candidate = args.out_root / "lr_midbass_phase_candidate_01.afpx"
        lint = write_candidate(args.baseline, candidate, rounded_result)
        copied = None
        copied_verification = None
        if args.copy_best:
            args.copy_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, args.copy_best)
            copied = str(args.copy_best)
            verify_path = args.copy_best.with_suffix(".verify.json")
            verify_path.write_text(json.dumps({
                "baseline": str(args.baseline),
                "candidate": str(args.copy_best),
                "intended_change": (
                    f"Add FL Mid APF {rounded_result['fl_apf_f']:.1f} Hz Q{rounded_result['fl_apf_q']:.2f} "
                    f"and FR Mid APF {rounded_result['fr_apf_f']:.1f} Hz Q{rounded_result['fr_apf_q']:.2f}"
                ),
                "validation": validation,
                "result": rounded_result,
                "lint": lint,
            }, indent=2), encoding="utf-8")
            copied_verification = str(verify_path)
        summary.update({
            "status": "candidate_written",
            "candidate": str(candidate),
            "copied_best": copied,
            "copied_best_verification": copied_verification,
            "result": {
                key: round(float(value), 4) if isinstance(value, (float, np.floating)) else value
                for key, value in rounded_result.items()
            },
            "objective_before": round(float(baseline_result["objective"]), 4),
            "objective_after": round(float(rounded_result["objective"]), 4),
            "objective_improvement_pct": round(
                100.0 * (float(baseline_result["objective"]) - float(rounded_result["objective"]))
                / max(float(baseline_result["objective"]), 1e-9),
                2,
            ),
            "lint": lint,
            "left_alone": [
                "No PEQ, delay, crossover, polarity, or existing APF was changed.",
                "The paired APFs localize relative phase rotation around the detected cancellation.",
                "Re-measure centre, left-ear, and right-ear Both Mids before accepting the candidate.",
            ],
        })
    path = args.out_root / "lr_midbass_phase_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_file"] = str(path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--copy-best", type=Path, default=None)
    parser.add_argument("--points-per-octave", type=int, default=96)
    parser.add_argument("--validation-threshold", type=float, default=2.5)
    parser.add_argument("--minimum-problem-gap", type=float, default=4.0)
    parser.add_argument("--max-damage-12", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1752026)
    parser.add_argument("--print-mode", choices=("compact", "full", "none"), default="compact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    if args.print_mode == "none":
        return
    if args.print_mode == "full":
        print(json.dumps(summary, indent=2))
        return
    result = summary.get("result", {})
    compact = {
        "status": summary.get("status"),
        "problem_frequency_hz": summary.get("problem_frequency_hz"),
        "problem_gap_24_db": summary.get("problem_gap_24_db"),
        "validation_pass": summary.get("validation", {}).get("pass"),
        "candidate": summary.get("copied_best") or summary.get("candidate"),
        "verification": summary.get("copied_best_verification"),
        "apfs": {
            "fl": [result.get("fl_apf_f"), result.get("fl_apf_q")],
            "fr": [result.get("fr_apf_f"), result.get("fr_apf_q")],
        },
        "objective": result.get("objective"),
        "raw_lift_at_problem_db": result.get("raw_lift_at_problem_db"),
        "max_damage_12_db": result.get("max_damage_12_db"),
        "summary": summary.get("summary_file"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
