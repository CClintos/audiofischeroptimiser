"""Phase/time search for verified REW phase sweeps and AFPX tunes.

This runner is intentionally separate from the PEQ optimizer. It searches only
mid/tweeter timing and optional APF moves using complex solo traces that share a
timing lock. Sub/front changes are reported as skipped unless their solo traces
are lock-consistent, because cross-lock phase was the trap that created the bad
sub APF/polarity recommendation.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _make_v3 import afpx_roundtrip_lint, decode_afpx, encode_afpx, set_attr
from _optimizer import add_allpass_to_oc, replace_oc_blocks, slot_attr
from _tunefit import allpass_H, group_delay_ms_from_H, prediction_confidence


PHASE_BAND = (1800.0, 4500.0)
PRESENCE_BAND = (2200.0, 3300.0)
DAMAGE_BAND = (900.0, 9000.0)


ALIASES = {
    "fl_high": ("Front Left High.txt", "Front L High.txt", "Front L Tweeter.txt"),
    "fr_high": ("Front Right High.txt", "Front R High.txt", "Front R Tweeter.txt"),
    "fl_mid": ("Front Left Mid.txt", "Front L Mid.txt", "Front L Low.txt"),
    "fr_mid": ("Front Right Mid.txt", "Front R Mid.txt", "Front R Low.txt"),
    "fl_pair": ("Front L Mid + Tweeter.txt", "Front L Mid+Tweeter.txt", "Front L Mid Tweeter Together.txt"),
    "fr_pair": ("Front R Mid + Tweeter.txt", "Front R Mid+Tweeter.txt", "Front R Mid Tweeter Together.txt"),
    "mids": ("Both Mids.txt", "Mid Bass Together.txt", "Mids Together.txt"),
    "sub": ("Subwoofer.txt", "Sub.txt", "SUB.txt"),
    "mids_sub": ("Mids and Sub.txt", "Sub + Mids.txt", "Sub and Mids.txt"),
}


@dataclass(frozen=True)
class Candidate:
    fl_high_samples: int
    fr_high_samples: int
    fl_apf_f: float = 0.0
    fl_apf_q: float = 0.0
    fr_apf_f: float = 0.0
    fr_apf_q: float = 0.0


def resolve(data_root: Path, key: str) -> Path:
    for name in ALIASES[key]:
        path = data_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing measurement for {key}: {ALIASES[key]}")


def load_rew(path: Path) -> Dict[str, np.ndarray]:
    cols: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.replace(",", " ").split()
        try:
            vals = [float(parts[0]), float(parts[1])]
            if len(parts) >= 3:
                vals.append(float(parts[2]))
        except (ValueError, IndexError):
            continue
        cols.append(vals)
    if not cols:
        raise ValueError(f"no numeric REW rows in {path}")
    max_len = max(len(row) for row in cols)
    arr = np.asarray([row + [float("nan")] * (max_len - len(row)) for row in cols], dtype=float)
    out = {"freq": arr[:, 0], "spl": arr[:, 1]}
    if max_len >= 3:
        out["phase"] = arr[:, 2]
    return out


def interp_trace(trace: Dict[str, np.ndarray], freqs: np.ndarray) -> Dict[str, np.ndarray]:
    out = {"freq": freqs}
    out["spl"] = np.interp(freqs, trace["freq"], trace["spl"])
    if "phase" in trace:
        out["phase"] = np.interp(freqs, trace["freq"], trace["phase"])
    return out


def complex_trace(trace: Dict[str, np.ndarray]) -> np.ndarray:
    if "phase" not in trace:
        raise ValueError("phase column is required")
    mag = 10.0 ** (trace["spl"] / 20.0)
    return mag * np.exp(1j * np.deg2rad(trace["phase"]))


def extract_delay_ms(path: Path) -> float | None:
    head = path.read_text(encoding="utf-8", errors="replace")[:1600]
    match = re.search(r"Delay\s+([-+0-9.]+)\s+ms", head)
    return float(match.group(1)) if match else None


def wrap_delay_ms(delta_samples: int, sample_rate: float) -> float:
    return 1000.0 * float(delta_samples) / float(sample_rate)


def delay_h(freqs: np.ndarray, delta_samples: int, sample_rate: float) -> np.ndarray:
    delay_ms = wrap_delay_ms(delta_samples, sample_rate)
    return np.exp(-1j * 2.0 * np.pi * freqs * delay_ms / 1000.0)


def wrms(freqs: np.ndarray, values: np.ndarray, band: Tuple[float, float]) -> float:
    sel = (freqs >= band[0]) & (freqs <= band[1]) & np.isfinite(values)
    if not np.any(sel):
        return float("inf")
    # Smoothly favor the vocal/presence part of the crossover without making a
    # single bin dominate the score.
    weight = np.ones(np.count_nonzero(sel))
    f = freqs[sel]
    weight[(f >= PRESENCE_BAND[0]) & (f <= PRESENCE_BAND[1])] = 1.35
    return float(np.sqrt(np.average(values[sel] ** 2, weights=weight)))


def sum_db(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(a + b) + 1e-12)


def score_pair(freqs: np.ndarray, mid: np.ndarray, high_base: np.ndarray, high_mod: np.ndarray,
               current_sum: np.ndarray) -> Dict[str, float]:
    coherent = 20.0 * np.log10(np.abs(mid) + np.abs(high_mod) + 1e-12)
    sdb = sum_db(mid, high_mod)
    gap = np.maximum(coherent - sdb, 0.0)
    damage = np.maximum(current_sum - sdb - 0.35, 0.0)
    presence_sel = (freqs >= PRESENCE_BAND[0]) & (freqs <= PRESENCE_BAND[1])
    lift = sdb - current_sum
    return {
        "gap": wrms(freqs, gap, PHASE_BAND),
        "presence_gap": wrms(freqs, gap, PRESENCE_BAND),
        "damage": wrms(freqs, damage, DAMAGE_BAND),
        "lift_2600": float(lift[int(np.argmin(np.abs(freqs - 2600.0)))]),
        "avg_presence_lift": float(np.mean(lift[presence_sel])) if np.any(presence_sel) else 0.0,
        "sum_peak": float(np.max(sdb[(freqs >= PHASE_BAND[0]) & (freqs <= PHASE_BAND[1])])),
    }


class PhaseSearch:
    def __init__(self, data_root: Path, sample_rate: float):
        self.data_root = data_root
        self.sample_rate = sample_rate
        base = load_rew(resolve(data_root, "fl_high"))
        self.freqs = base["freq"]
        traces = {"fl_high": base}
        for key in ("fr_high", "fl_mid", "fr_mid", "fl_pair", "fr_pair", "mids", "sub", "mids_sub"):
            try:
                traces[key] = interp_trace(load_rew(resolve(data_root, key)), self.freqs)
            except FileNotFoundError:
                pass
        self.traces = traces
        self.cx = {
            "fl_high": complex_trace(traces["fl_high"]),
            "fr_high": complex_trace(traces["fr_high"]),
            "fl_mid": complex_trace(traces["fl_mid"]),
            "fr_mid": complex_trace(traces["fr_mid"]),
        }
        self.current = {
            "left": sum_db(self.cx["fl_mid"], self.cx["fl_high"]),
            "right": sum_db(self.cx["fr_mid"], self.cx["fr_high"]),
        }
        self.meas_delay_ms = {}
        for key in traces:
            try:
                self.meas_delay_ms[key] = extract_delay_ms(resolve(data_root, key))
            except FileNotFoundError:
                pass

    def high_with_candidate(self, side: str, cand: Candidate) -> np.ndarray:
        key = "fl_high" if side == "left" else "fr_high"
        samples = cand.fl_high_samples if side == "left" else cand.fr_high_samples
        h = self.cx[key] * delay_h(self.freqs, samples, self.sample_rate)
        f = cand.fl_apf_f if side == "left" else cand.fr_apf_f
        q = cand.fl_apf_q if side == "left" else cand.fr_apf_q
        if f > 0.0 and q > 0.0:
            h = h * allpass_H(self.freqs, f, q)
        return h

    def evaluate(self, cand: Candidate) -> Dict[str, float]:
        fl_high = self.high_with_candidate("left", cand)
        fr_high = self.high_with_candidate("right", cand)
        left = score_pair(self.freqs, self.cx["fl_mid"], self.cx["fl_high"], fl_high, self.current["left"])
        right = score_pair(self.freqs, self.cx["fr_mid"], self.cx["fr_high"], fr_high, self.current["right"])
        apf_count = int(cand.fl_apf_f > 0.0) + int(cand.fr_apf_f > 0.0)
        apf_gd_penalty = 0.0
        for f, q in ((cand.fl_apf_f, cand.fl_apf_q), (cand.fr_apf_f, cand.fr_apf_q)):
            if f > 0.0 and q > 0.0:
                gd = group_delay_ms_from_H(self.freqs, allpass_H(self.freqs, f, q))
                sel = (self.freqs >= PHASE_BAND[0]) & (self.freqs <= PHASE_BAND[1])
                apf_gd_penalty += max(0.0, float(np.max(gd[sel])) - 0.75)
        delay_penalty = 0.009 * (abs(cand.fl_high_samples) + abs(cand.fr_high_samples))
        # Favor fixing the known left crossover null, but force the right side to
        # remain healthy. This keeps the search from simply "breaking both sides
        # equally" to win a narrow metric.
        score = (
            1.25 * left["gap"]
            + 0.75 * left["presence_gap"]
            + 1.15 * right["gap"]
            + 0.55 * right["presence_gap"]
            + 0.42 * (left["damage"] + right["damage"])
            + 0.22 * delay_penalty
            + 0.35 * apf_count
            + 0.45 * apf_gd_penalty
        )
        return {
            "score": float(score),
            "left_gap": left["gap"],
            "left_presence_gap": left["presence_gap"],
            "left_damage": left["damage"],
            "left_lift_2600": left["lift_2600"],
            "left_avg_presence_lift": left["avg_presence_lift"],
            "right_gap": right["gap"],
            "right_presence_gap": right["presence_gap"],
            "right_damage": right["damage"],
            "right_lift_2600": right["lift_2600"],
            "right_avg_presence_lift": right["avg_presence_lift"],
            "delay_penalty": delay_penalty,
            "apf_count": float(apf_count),
            "apf_gd_penalty": apf_gd_penalty,
        }

    def diagnostics(self) -> Dict[str, object]:
        out: Dict[str, object] = {"measurement_delay_ms": self.meas_delay_ms}
        for side, mid_key, high_key, pair_key in (
            ("left", "fl_mid", "fl_high", "fl_pair"),
            ("right", "fr_mid", "fr_high", "fr_pair"),
        ):
            pred = prediction_confidence(
                self.freqs,
                self.cx[mid_key],
                self.cx[high_key],
                self.traces[pair_key]["spl"],
                PHASE_BAND,
            )
            out[f"{side}_mid_tweeter_prediction"] = pred
        if "sub" in self.meas_delay_ms and "mids" in self.meas_delay_ms:
            sub = self.meas_delay_ms["sub"]
            mids = self.meas_delay_ms["mids"]
            out["sub_midbass_phase_status"] = {
                "usable_for_solo_phase_search": bool(sub is not None and mids is not None and abs(sub - mids) < 0.75),
                "reason": "sub and both-mids solo delay locks differ too much for coherent solo phase search",
                "sub_delay_ms": sub,
                "mids_delay_ms": mids,
                "lock_delta_ms": None if sub is None or mids is None else round(abs(sub - mids), 4),
            }
        return out


def random_candidate(rng: random.Random, args: argparse.Namespace) -> Candidate:
    # Delay samples are absolute deltas from the tune that was measured. Negative
    # means the tweeter is advanced; positive means delayed.
    fl = int(round(rng.triangular(args.fl_min_samples, args.fl_max_samples, args.fl_mode_samples)))
    fr = int(round(rng.triangular(args.fr_min_samples, args.fr_max_samples, args.fr_mode_samples)))
    fl_apf_f = fl_apf_q = fr_apf_f = fr_apf_q = 0.0
    if rng.random() < 0.28:
        fl_apf_f = float(math.exp(rng.uniform(math.log(1900.0), math.log(4400.0))))
        fl_apf_q = rng.uniform(0.5, 1.65)
    if rng.random() < 0.16:
        fr_apf_f = float(math.exp(rng.uniform(math.log(1900.0), math.log(4400.0))))
        fr_apf_q = rng.uniform(0.5, 1.65)
    return Candidate(
        fl_high_samples=fl,
        fr_high_samples=fr,
        fl_apf_f=round(fl_apf_f, 1),
        fl_apf_q=round(fl_apf_q, 2),
        fr_apf_f=round(fr_apf_f, 1),
        fr_apf_q=round(fr_apf_q, 2),
    )


def seed_candidates(args: argparse.Namespace) -> Iterable[Candidate]:
    for fl in sorted(set((
        0,
        args.fl_mode_samples,
        args.fl_min_samples,
        args.fl_max_samples,
        args.fl_mode_samples - 4,
        args.fl_mode_samples - 2,
        args.fl_mode_samples - 1,
        args.fl_mode_samples + 1,
        args.fl_mode_samples + 2,
        args.fl_mode_samples + 4,
    ))):
        if args.fl_min_samples <= fl <= args.fl_max_samples:
            yield Candidate(int(fl), 0)
    for fr in sorted(set((
        args.fr_mode_samples - 4,
        args.fr_mode_samples - 2,
        args.fr_mode_samples,
        args.fr_mode_samples + 2,
        args.fr_mode_samples + 4,
    ))):
        if args.fr_min_samples <= fr <= args.fr_max_samples:
            yield Candidate(0, int(fr))
    yield Candidate(args.fl_mode_samples, 0, 2600.0, 0.7, 0.0, 0.0)
    yield Candidate(args.fl_mode_samples, 0, 3200.0, 0.8, 0.0, 0.0)
    yield Candidate(args.fl_mode_samples, 0, 0.0, 0.0, 2600.0, 0.7)


def signature(cand: Candidate) -> Tuple[object, ...]:
    return tuple(asdict(cand).items())


def worker_main(worker_id: int, args: argparse.Namespace, deadline: float) -> None:
    rng = random.Random(args.seed + worker_id * 1009)
    search = PhaseSearch(args.data_root, args.sample_rate)
    worker_dir = args.out_root / f"worker_{worker_id:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    top: List[Tuple[float, Tuple[object, ...], Dict[str, object]]] = []
    seen = set()
    completed = 0

    def consider(cand: Candidate) -> None:
        nonlocal top, completed
        sig = signature(cand)
        if sig in seen:
            return
        seen.add(sig)
        metrics = search.evaluate(cand)
        row = {"candidate": asdict(cand), "metrics": metrics}
        # max-heap by negative score, bounded archive.
        item = (-metrics["score"], sig, row)
        if len(top) < args.archive_size:
            heapq.heappush(top, item)
        else:
            if item[0] > top[0][0]:
                heapq.heapreplace(top, item)
        completed += 1

    for cand in seed_candidates(args):
        consider(cand)

    last_save = 0.0
    while time.time() < deadline:
        consider(random_candidate(rng, args))
        now = time.time()
        if now - last_save >= args.checkpoint_seconds:
            save_worker_state(worker_dir / "state.json", top, completed, args, search.diagnostics())
            last_save = now
    save_worker_state(worker_dir / "state.json", top, completed, args, search.diagnostics())


def sorted_top(heap: List[Tuple[float, Tuple[object, ...], Dict[str, object]]]) -> List[Dict[str, object]]:
    return [row for _neg, _sig, row in sorted(heap, key=lambda item: -item[0])]


def save_worker_state(path: Path, top, completed: int, args: argparse.Namespace, diagnostics: Dict[str, object]) -> None:
    payload = {
        "version": 1,
        "mode": "phase_time_mid_tweeter_delay_apf",
        "completed": completed,
        "seconds": args.seconds,
        "archive_size": args.archive_size,
        "top": sorted_top(top),
        "diagnostics": diagnostics,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def modify_delay_tags(xml: str, delay_samples_by_channel: Dict[int, int]) -> str:
    tags = list(re.finditer(r"<T [^>]*/?>", xml))
    out = xml
    for ch, samples in sorted(delay_samples_by_channel.items(), reverse=True):
        if ch >= len(tags):
            raise IndexError(f"channel {ch} has no delay tag")
        tag = tags[ch].group()
        new_tag = set_attr(tag, "T", str(int(samples)))
        out = out[:tags[ch].start()] + new_tag + out[tags[ch].end():]
    return out


def add_apfs(xml: str, cand: Candidate) -> str:
    blocks = [m.group() for m in re.finditer(r"<OC\b.*?</OC>", xml, re.S)]
    if cand.fl_apf_f > 0.0 and cand.fl_apf_q > 0.0:
        blocks[0] = add_allpass_to_oc(blocks[0], cand.fl_apf_f, cand.fl_apf_q)
    if cand.fr_apf_f > 0.0 and cand.fr_apf_q > 0.0:
        blocks[1] = add_allpass_to_oc(blocks[1], cand.fr_apf_f, cand.fr_apf_q)
    return replace_oc_blocks(xml, blocks)


def baseline_delay_samples(xml: str) -> List[int]:
    tags = re.findall(r"<T [^>]*/?>", xml)
    out = []
    for tag in tags:
        raw = slot_attr(tag, "T")
        out.append(int(float(raw)) if raw is not None else 0)
    return out


def write_phase_candidate(base_xml: str, baseline_delays: List[int], path: Path, cand: Candidate) -> Dict[str, object]:
    xml = base_xml
    delay_targets = {
        0: baseline_delays[0] + cand.fl_high_samples,
        1: baseline_delays[1] + cand.fr_high_samples,
    }
    if any(v < 0 for v in delay_targets.values()):
        raise ValueError(f"candidate delay went negative: {delay_targets}")
    xml = modify_delay_tags(xml, delay_targets)
    xml = add_apfs(xml, cand)
    encode_afpx(xml, path)
    written = decode_afpx(path)
    lint = afpx_roundtrip_lint(
        base_xml,
        written,
        allow_delay_changes=True,
        allowed_added_types=("20",),
    )
    if not lint["pass"]:
        raise AssertionError("AFPX lint failed: " + "; ".join(lint["errors"]))
    return lint


def merge_and_write(args: argparse.Namespace) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {}
    completed = 0
    for state in sorted(args.out_root.glob("worker_*/state.json")):
        payload = json.loads(state.read_text(encoding="utf-8"))
        completed += int(payload.get("completed", 0))
        diagnostics = payload.get("diagnostics", diagnostics)
        rows.extend(payload.get("top", []))
    dedup: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        cand = Candidate(**row["candidate"])
        sig = signature(cand)
        if sig not in dedup or row["metrics"]["score"] < dedup[sig]["metrics"]["score"]:
            dedup[sig] = row
    ranked = sorted(dedup.values(), key=lambda row: row["metrics"]["score"])

    candidates_dir = args.out_root / "_phase_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    base_xml = decode_afpx(args.baseline)
    delays = baseline_delay_samples(base_xml)
    written = []
    for rank, row in enumerate(ranked[: args.top], 1):
        cand = Candidate(**row["candidate"])
        path = candidates_dir / f"phase_candidate_{rank:02d}_score_{row['metrics']['score']:.4f}.afpx"
        lint = write_phase_candidate(base_xml, delays, path, cand)
        item = dict(row)
        item["rank"] = rank
        item["file"] = str(path)
        item["lint"] = lint
        written.append(item)

    copied_best = None
    if written and args.copy_best:
        copied_best = args.copy_best
        shutil.copy2(written[0]["file"], copied_best)

    summary = {
        "run_root": str(args.out_root),
        "baseline": str(args.baseline),
        "data_root": str(args.data_root),
        "seconds": args.seconds,
        "workers": args.workers,
        "completed_candidates": completed,
        "unique_archived_candidates": len(ranked),
        "written_count": len(written),
        "copied_best": str(copied_best) if copied_best else None,
        "diagnostics": diagnostics,
        "top": written,
        "scope_notes": [
            "Searched front high-channel delay and optional APF for mid/tweeter crossover summation.",
            "Sub/front phase writes skipped because the sub solo and both-mids solo timing locks do not agree.",
            "Crossover filter writes are not performed by this runner; crossover changes need export-diff verification before automatic writes.",
        ],
    }
    (args.out_root / "phase_time_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="The AFPX that was loaded when the phase sweeps were measured.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=2400)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sample-rate", type=float, default=96000.0)
    parser.add_argument("--seed", type=int, default=260708)
    parser.add_argument("--fl-min-samples", type=int, default=-34)
    parser.add_argument("--fl-max-samples", type=int, default=8)
    parser.add_argument("--fl-mode-samples", type=int, default=-14)
    parser.add_argument("--fr-min-samples", type=int, default=-20)
    parser.add_argument("--fr-max-samples", type=int, default=20)
    parser.add_argument("--fr-mode-samples", type=int, default=0)
    parser.add_argument("--archive-size", type=int, default=500)
    parser.add_argument("--checkpoint-seconds", type=int, default=30)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--copy-best", type=Path, default=None)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if not args.merge_only:
        deadline = time.time() + args.seconds
        procs = []
        for worker_id in range(1, args.workers + 1):
            proc = Process(target=worker_main, args=(worker_id, args, deadline), daemon=False)
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise SystemExit(f"worker failed with exit code {proc.exitcode}")
    summary = merge_and_write(args)
    compact = {
        "completed_candidates": summary["completed_candidates"],
        "unique_archived_candidates": summary["unique_archived_candidates"],
        "copied_best": summary["copied_best"],
        "top3": [
            {
                "rank": row["rank"],
                "file": row["file"],
                "candidate": row["candidate"],
                "metrics": row["metrics"],
            }
            for row in summary["top"][:3]
        ],
        "scope_notes": summary["scope_notes"],
        "diagnostics": summary["diagnostics"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
