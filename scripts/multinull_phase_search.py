"""Time-bounded search for multiple phase-valid L/R-midbass cancellations."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, Iterable, Tuple

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _make_v3 import afpx_roundtrip_lint, decode_afpx, encode_afpx
from _optimizer import add_allpass_to_oc, replace_oc_blocks
from _tunefit import allpass_H, group_delay_ms_from_H
from scripts.lr_midbass_phase_search import (
    complex_on_grid,
    extract_delay_ms,
    load_rew,
    resolve,
    smooth_db,
)


@dataclass(frozen=True)
class Candidate:
    fl_f: float
    fl_q: float
    fr_f: float
    fr_q: float


class MultiNullModel:
    def __init__(self, data_root: Path, points_per_octave: int = 96):
        self.data_root = data_root
        self.ppo = points_per_octave
        self.paths = {
            "fl": resolve(data_root, "fl_mid"),
            "fr": resolve(data_root, "fr_mid"),
            "together": resolve(data_root, "together"),
        }
        fl_trace = load_rew(self.paths["fl"])
        fr_trace = load_rew(self.paths["fr"])
        together = load_rew(self.paths["together"])
        lo = max(60.0, float(fl_trace["freq"][0]), float(fr_trace["freq"][0]))
        hi = min(3000.0, float(fl_trace["freq"][-1]), float(fr_trace["freq"][-1]))
        self.freqs = 2.0 ** np.arange(
            math.log2(lo), math.log2(hi) + 1.0 / points_per_octave, 1.0 / points_per_octave
        )
        self.fl = complex_on_grid(fl_trace, self.freqs)
        self.fr = complex_on_grid(fr_trace, self.freqs)
        self.measured = np.interp(self.freqs, together["freq"], together["spl"])
        self.base_model = 20.0 * np.log10(np.abs(self.fl + self.fr) + 1e-12)
        self.ceiling_model = 20.0 * np.log10(np.abs(self.fl) + np.abs(self.fr) + 1e-12)
        self.estimated_ceiling = self.measured + self.ceiling_model - self.base_model
        self.measured_12 = smooth_db(self.measured, points_per_octave, 1.0 / 12.0)
        self.measured_24 = smooth_db(self.measured, points_per_octave, 1.0 / 24.0)
        self.ceiling_24 = smooth_db(self.estimated_ceiling, points_per_octave, 1.0 / 24.0)
        self.nulls = self._detect_nulls()
        if len(self.nulls) < 2:
            raise ValueError("fewer than two material L/R-midbass cancellations were detected")
        self.low_f, self.high_f = self.nulls[:2]
        if self.low_f > self.high_f:
            self.low_f, self.high_f = self.high_f, self.low_f
        self.validation = self._validation(fl_trace, fr_trace)

    def _detect_nulls(self) -> list[float]:
        level_delta = np.abs(20.0 * np.log10(np.abs(self.fl) / np.maximum(np.abs(self.fr), 1e-12)))
        valid = (self.freqs >= 100.0) & (self.freqs <= 600.0) & (level_delta <= 10.0)
        gap = self.ceiling_24 - self.measured_24
        peaks, props = find_peaks(np.where(valid, gap, -99.0), distance=20, prominence=2.0)
        ranked = sorted(peaks, key=lambda index: float(gap[index]), reverse=True)
        return [float(self.freqs[index]) for index in ranked if gap[index] >= 4.0]

    def _validation(self, fl_trace: Dict[str, np.ndarray], fr_trace: Dict[str, np.ndarray]) -> Dict[str, object]:
        sel = (self.freqs >= 180.0) & (self.freqs <= 500.0)
        error = self.measured - self.base_model
        fl_lock = extract_delay_ms(self.paths["fl"])
        fr_lock = extract_delay_ms(self.paths["fr"])
        lock_delta = None if fl_lock is None or fr_lock is None else abs(fl_lock - fr_lock)
        return {
            "rms_db_180_500": round(float(np.sqrt(np.mean(error[sel] ** 2))), 3),
            "mae_db_180_500": round(float(np.mean(np.abs(error[sel]))), 3),
            "correlation_180_500": round(float(np.corrcoef(self.measured[sel], self.base_model[sel])[0, 1]), 6),
            "solo_lock_delta_ms": None if lock_delta is None else round(float(lock_delta), 4),
            "pass": bool(lock_delta is not None and lock_delta <= 0.75 and np.sqrt(np.mean(error[sel] ** 2)) <= 1.5),
        }

    def _band(self, center: float, half_octave: float = 0.055) -> np.ndarray:
        return (
            (self.freqs >= center / (2.0 ** half_octave))
            & (self.freqs <= center * (2.0 ** half_octave))
        )

    def evaluate(self, cand: Candidate, risk: float = 1.0) -> Dict[str, float]:
        fl_h = allpass_H(self.freqs, cand.fl_f, cand.fl_q)
        fr_h = allpass_H(self.freqs, cand.fr_f, cand.fr_q)
        new_model = 20.0 * np.log10(np.abs(self.fl * fl_h + self.fr * fr_h) + 1e-12)
        predicted = self.measured + new_model - self.base_model
        pred_12 = smooth_db(predicted, self.ppo, 1.0 / 12.0)
        pred_24 = smooth_db(predicted, self.ppo, 1.0 / 24.0)
        gap_24 = np.maximum(self.ceiling_24 - pred_24, 0.0)
        damage_12 = np.maximum(self.measured_12 - pred_12 - 0.25, 0.0)
        damage_24 = np.maximum(self.measured_24 - pred_24 - 0.35, 0.0)
        low = self._band(self.low_f)
        high = self._band(self.high_f)
        guard = (self.freqs >= 80.0) & (self.freqs <= 3000.0)
        vocal = (self.freqs >= 600.0) & (self.freqs <= 1800.0)
        intended = low | high
        outside = guard & ~intended

        def rms(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.sqrt(np.mean(values[mask] ** 2)))

        low_gap = rms(gap_24, low)
        high_gap = rms(gap_24, high)
        damage_rms = rms(damage_12, guard)
        max_damage = float(np.max(damage_12[guard]))
        vocal_damage = float(np.max(damage_24[vocal]))
        outside_abs = float(np.mean(np.abs(pred_12[outside] - self.measured_12[outside])))
        gd_fl = float(np.max(group_delay_ms_from_H(self.freqs, fl_h)))
        gd_fr = float(np.max(group_delay_ms_from_H(self.freqs, fr_h)))
        gd_risk = max(0.0, gd_fl - 8.0) + max(0.0, gd_fr - 8.0)
        q_risk = max(0.0, cand.fl_q - 1.5) / 6.0 + max(0.0, cand.fr_q - 1.5) / 10.0
        objective = (
            1.45 * low_gap
            + 1.45 * high_gap
            + risk * (
                1.6 * damage_rms
                + 0.55 * max_damage
                + 0.45 * vocal_damage
                + 0.8 * outside_abs
                + 0.025 * gd_risk
                + 0.12 * q_risk
            )
        )
        low_raw = self._band(self.low_f, 0.045)
        high_raw = self._band(self.high_f, 0.045)
        return {
            "objective": float(objective),
            "low_gap_24_rms_db": low_gap,
            "high_gap_24_rms_db": high_gap,
            "damage_12_rms_db": damage_rms,
            "max_damage_12_db": max_damage,
            "max_vocal_damage_24_db": vocal_damage,
            "outside_mean_abs_12_db": outside_abs,
            "fl_group_delay_peak_ms": gd_fl,
            "fr_group_delay_peak_ms": gd_fr,
            "low_band_mean_db": float(np.mean(predicted[low_raw])),
            "low_band_worst_db": float(np.min(predicted[low_raw])),
            "high_band_mean_db": float(np.mean(predicted[high_raw])),
            "high_band_worst_db": float(np.min(predicted[high_raw])),
            "mean_180_300_db": float(np.mean(predicted[(self.freqs >= 180) & (self.freqs <= 300)])),
            "mean_300_500_db": float(np.mean(predicted[(self.freqs >= 300) & (self.freqs <= 500)])),
        }

    def bounds(self, low_q_max: float, high_q_max: float):
        return [
            (self.low_f / 1.10, self.low_f * 1.10), (1.0, low_q_max),
            (self.high_f / 1.10, self.high_f * 1.10), (1.0, high_q_max),
        ]


def rounded_candidate(values: Iterable[float]) -> Candidate:
    fl_f, fl_q, fr_f, fr_q = [float(value) for value in values]
    return Candidate(round(fl_f, 1), round(fl_q, 1), round(fr_f, 1), round(fr_q, 1))


def signature(cand: Candidate) -> Tuple[float, ...]:
    return cand.fl_f, cand.fl_q, cand.fr_f, cand.fr_q


def worker_main(worker_id: int, args: argparse.Namespace, deadline: float) -> None:
    model = MultiNullModel(args.data_root, args.points_per_octave)
    risk_levels = (0.25, 0.45, 0.7, 1.0, 1.35, 1.8)
    risk = risk_levels[(worker_id - 1) % len(risk_levels)]
    rng = random.Random(args.seed + worker_id * 1009)
    archive: list[tuple[float, Tuple[float, ...], Dict[str, object]]] = []
    seen = set()
    completed = 0
    worker_dir = args.out_root / f"worker_{worker_id:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    def consider(cand: Candidate) -> None:
        nonlocal completed
        sig = signature(cand)
        if sig in seen:
            return
        seen.add(sig)
        metrics = model.evaluate(cand, risk=1.0)
        row = {"candidate": asdict(cand), "metrics": metrics, "worker_risk": risk}
        item = (-float(metrics["objective"]), sig, row)
        if len(archive) < args.archive_size:
            heapq.heappush(archive, item)
        elif item[0] > archive[0][0]:
            heapq.heapreplace(archive, item)
        completed += 1

    consider(Candidate(174.0, 5.0, 402.0, 8.0))
    while time.time() < deadline:
        bounds = model.bounds(args.low_q_max, args.high_q_max)
        result = differential_evolution(
            lambda values: model.evaluate(rounded_candidate(values), risk=risk)["objective"],
            bounds,
            seed=rng.randrange(1, 2**31 - 1),
            popsize=12,
            maxiter=70,
            tol=1e-7,
            polish=True,
            workers=1,
            updating="immediate",
        )
        consider(rounded_candidate(result.x))
        # Add local hardware-step variations so the archive is useful rather
        # than hundreds of repeats of one continuous optimum.
        center = rounded_candidate(result.x)
        for _ in range(300):
            consider(Candidate(
                round(np.clip(center.fl_f + rng.gauss(0, 2.0), bounds[0][0], bounds[0][1]), 1),
                round(np.clip(center.fl_q + rng.gauss(0, 0.35), 1.0, args.low_q_max), 1),
                round(np.clip(center.fr_f + rng.gauss(0, 4.0), bounds[2][0], bounds[2][1]), 1),
                round(np.clip(center.fr_q + rng.gauss(0, 0.6), 1.0, args.high_q_max), 1),
            ))
        payload = {
            "completed": completed,
            "risk": risk,
            "validation": model.validation,
            "nulls_hz": model.nulls,
            "top": [item[2] for item in sorted(archive, key=lambda value: -value[0])],
        }
        tmp = worker_dir / "state.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(worker_dir / "state.json")


def write_candidate(base_xml: str, path: Path, cand: Candidate) -> Dict[str, object]:
    blocks = re.findall(r"<OC\b.*?</OC>", base_xml, re.S)
    blocks[2] = add_allpass_to_oc(blocks[2], cand.fl_f, cand.fl_q)
    blocks[3] = add_allpass_to_oc(blocks[3], cand.fr_f, cand.fr_q)
    xml = replace_oc_blocks(base_xml, blocks)
    encode_afpx(xml, path)
    lint = afpx_roundtrip_lint(base_xml, decode_afpx(path), allowed_added_types=("20",))
    if not lint["pass"]:
        raise AssertionError("AFPX lint failed: " + "; ".join(lint["errors"]))
    return lint


def merge(args: argparse.Namespace) -> Dict[str, object]:
    model = MultiNullModel(args.data_root, args.points_per_octave)
    rows = []
    completed = 0
    for state in sorted(args.out_root.glob("worker_*/state.json")):
        payload = json.loads(state.read_text(encoding="utf-8"))
        completed += int(payload.get("completed", 0))
        rows.extend(payload.get("top", []))
    dedup = {}
    for row in rows:
        cand = Candidate(**row["candidate"])
        sig = signature(cand)
        metrics = model.evaluate(cand, risk=1.0)
        row["metrics"] = metrics
        if sig not in dedup or metrics["objective"] < dedup[sig]["metrics"]["objective"]:
            dedup[sig] = row
    ranked = sorted(dedup.values(), key=lambda row: row["metrics"]["objective"])
    base_xml = decode_afpx(args.baseline)
    candidate_dir = args.out_root / "_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, row in enumerate(ranked[:args.top], 1):
        cand = Candidate(**row["candidate"])
        path = candidate_dir / f"candidate_{rank:02d}_objective_{row['metrics']['objective']:.4f}.afpx"
        lint = write_candidate(base_xml, path, cand)
        written.append({**row, "rank": rank, "file": str(path), "lint": lint})
    copied = copied_verify = None
    if written and args.copy_best:
        args.copy_best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(written[0]["file"], args.copy_best)
        copied = str(args.copy_best)
        verify = args.copy_best.with_suffix(".verify.json")
        verify.write_text(json.dumps({
            "baseline": str(args.baseline),
            "candidate": copied,
            "validation": model.validation,
            "nulls_hz": model.nulls,
            "selection": written[0],
            "hardware_warning": (
                "Audiotec Fischer states APF maximum Q depends on frequency and Q above 1.5 is rarely used. "
                "Confirm PC-Tool displays both Q values unchanged before loading to the DSP."
            ),
        }, indent=2), encoding="utf-8")
        copied_verify = str(verify)
    summary = {
        "mode": "time_bounded_multi_null_split_apf",
        "seconds": args.seconds,
        "workers": args.workers,
        "completed_candidates": completed,
        "unique_candidates": len(ranked),
        "baseline": str(args.baseline),
        "data_root": str(args.data_root),
        "validation": model.validation,
        "detected_nulls_hz": model.nulls,
        "copied_best": copied,
        "copied_best_verification": copied_verify,
        "top": written,
    }
    (args.out_root / "multinull_phase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--copy-best", type=Path, default=None)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--points-per-octave", type=int, default=96)
    parser.add_argument("--low-q-max", type=float, default=5.2)
    parser.add_argument("--high-q-max", type=float, default=12.0)
    parser.add_argument("--archive-size", type=int, default=300)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--print-mode", choices=("compact", "full", "none"), default="compact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if not args.merge_only:
        deadline = time.time() + args.seconds
        workers = []
        for worker_id in range(1, args.workers + 1):
            proc = Process(target=worker_main, args=(worker_id, args, deadline), daemon=False)
            proc.start()
            workers.append(proc)
        for proc in workers:
            proc.join()
            if proc.exitcode != 0:
                raise SystemExit(f"worker failed with exit code {proc.exitcode}")
    summary = merge(args)
    if args.print_mode == "none":
        return
    if args.print_mode == "full":
        print(json.dumps(summary, indent=2))
        return
    best = summary["top"][0] if summary["top"] else None
    print(json.dumps({
        "status": "complete",
        "completed_candidates": summary["completed_candidates"],
        "unique_candidates": summary["unique_candidates"],
        "copied_best": summary["copied_best"],
        "verification": summary["copied_best_verification"],
        "detected_nulls_hz": summary["detected_nulls_hz"],
        "best": None if best is None else {
            "candidate": best["candidate"],
            "objective": best["metrics"].get("objective"),
            "low_gap_24_rms_db": best["metrics"].get("low_gap_24_rms_db"),
            "high_gap_24_rms_db": best["metrics"].get("high_gap_24_rms_db"),
            "max_damage_12_db": best["metrics"].get("max_damage_12_db"),
            "max_vocal_damage_24_db": best["metrics"].get("max_vocal_damage_24_db"),
        },
        "summary": str(args.out_root / "multinull_phase_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
