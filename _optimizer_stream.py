"""Constant-memory random-search optimizer.

Use this for long brute-force runs. It does not use Optuna's in-memory Study,
so RAM stays flat: each worker keeps only the best candidates it has seen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

import _optimizer as opt

try:
    from cmaes import CMA
except ImportError:  # Keep random/guided modes usable if the optional backend is absent.
    CMA = None


GroupBands = Dict[str, List[Tuple[float, float, float]]]


def configure_profile(profile: str) -> None:
    opt.GROUPS = {
        k: dict(v)
        for k, v in (opt.EXPLORE_GROUPS if profile == "explore" else opt.SAFE_GROUPS).items()
    }


def random_band(rng: np.random.Generator, cfg: Dict[str, object]):
    lo, hi = cfg["range"]
    qlo, qhi = cfg["q_range"]
    glo, ghi = cfg["gain_range"]
    F = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    Q = float(rng.uniform(qlo, qhi))
    G = float(rng.uniform(glo, ghi))
    return opt.rounded_band(F, Q, G)


def random_groups(rng: np.random.Generator, profile: str) -> GroupBands:
    groups: GroupBands = {}
    p_on = 0.30 if profile == "explore" else 0.22
    for group, cfg in opt.GROUPS.items():
        bands = []
        for _ in range(cfg["max_bands"]):
            if rng.random() > p_on:
                continue
            band = random_band(rng, cfg)
            if band is not None:
                bands.append(band)
        bands.sort(key=lambda b: b[0])
        groups[group] = bands
    return groups


def q_from_oct_width(width_oct: float, q_range: Tuple[float, float]) -> float:
    width_oct = max(float(width_oct), 1 / 12)
    n = 2 ** width_oct
    q = math.sqrt(n) / max(n - 1, 1e-9)
    return float(np.clip(q, q_range[0], q_range[1]))


def branch_contribution(freqs, traces, group: str) -> np.ndarray:
    """Approximate how much each branch can move the measured system sum.

    This is not phase-aware; it is just a magnitude power-share estimate used
    to decide where a branch is a plausible cause of an above-target excess.
    The final score still judges the predicted whole-system response.
    """
    cfg = opt.GROUPS[group]
    branch_name = cfg.get("branch", group)
    if cfg.get("system_transfer"):
        branch_power = np.zeros_like(freqs, dtype=float)
        for pair in opt.PAIR_DEFS.values():
            branch_power += 10 ** (traces[pair["together"]] / 10)
        branch = 10.0 * np.log10(np.maximum(branch_power, 1e-30))
    elif cfg.get("trace"):
        branch = traces[cfg["trace"]]
    else:
        branch = {"sub": traces["Sub"]}.get(branch_name)
        if branch is None:
            pair = opt.PAIR_DEFS[branch_name]
            branch = traces[pair["together"]]
    total = 10 ** (traces["Sub"] / 10)
    for pair in opt.PAIR_DEFS.values():
        total = total + 10 ** (traces[pair["together"]] / 10)
    share = 10 ** (branch / 10) / np.maximum(total, 1e-30)
    return np.clip(share, 0.0, 1.0)


def interference_masks(freqs, traces):
    masks = {group: np.zeros_like(freqs, dtype=bool) for group in opt.GROUPS}
    pair_masks = {name: np.zeros_like(freqs, dtype=bool) for name in opt.PAIR_DEFS}
    for name, pair in opt.PAIR_DEFS.items():
        try:
            pair_masks[name] |= opt.interference_audit(
                freqs, traces[pair["left"]], traces[pair["right"]], traces[pair["together"]]
            )[3]
        except Exception:
            pass
    for group, cfg in opt.GROUPS.items():
        branch = cfg.get("branch")
        if cfg.get("system_transfer"):
            for pair_mask in pair_masks.values():
                masks[group] |= pair_mask
        elif branch in pair_masks:
            masks[group] |= pair_masks[branch]
    return masks


def candidate_peaks(freqs, strength, desired_gain, lo, hi, q_range, gain_range, source, profile):
    strength = np.asarray(strength, dtype=float).copy()
    desired_gain = np.asarray(desired_gain, dtype=float)
    strength[(freqs < lo) | (freqs > hi)] = 0.0
    strength[~np.isfinite(strength)] = 0.0
    strength[np.abs(desired_gain) < 0.25] = 0.0
    strength = opt.erb_smooth(freqs, strength)

    thresh = 0.35 if profile == "explore" else 0.60
    idxs = []
    for i in range(1, len(freqs) - 1):
        if strength[i] < thresh:
            continue
        if strength[i] >= strength[i - 1] and strength[i] >= strength[i + 1]:
            idxs.append(i)
    idxs.sort(key=lambda i: -strength[i])

    chosen = []
    min_sep_oct = 1 / 5
    for i in idxs:
        if all(abs(math.log2(freqs[i] / freqs[j])) >= min_sep_oct for j in chosen):
            chosen.append(i)
        if len(chosen) >= 12:
            break

    candidates = []
    for i in chosen:
        half = max(thresh * 0.7, strength[i] * 0.5)
        l = i
        r = i
        while l > 0 and strength[l] > half and freqs[l] > lo:
            l -= 1
        while r < len(freqs) - 1 and strength[r] > half and freqs[r] < hi:
            r += 1
        width_oct = max(math.log2(freqs[r] / freqs[l]), 1 / 12)
        if freqs[i] >= 1000.0:
            width_oct = max(width_oct, 1 / 6)
        q_hint = q_from_oct_width(width_oct, q_range)
        gain_hint = float(np.clip(desired_gain[i], gain_range[0], gain_range[1]))
        band = opt.rounded_band(float(freqs[i]), q_hint, gain_hint)
        if band is None:
            continue
        rounded_f, rounded_q, rounded_gain = band
        candidates.append({
            "F": float(rounded_f),
            "Q": float(rounded_q),
            "G": float(rounded_gain),
            "strength": float(strength[i]),
            "width_oct": float(width_oct),
            "branch_share": 0.0,
            "source": source,
        })
    return candidates


def find_guided_candidates(freqs, traces, target, profile: str):
    """Find data-derived candidate PEQ centers before random search.

    Candidate centers come from two math-derived needs:
      - tonal target error in the predicted system sum, with stronger presence
        and peak weighting;
      - L/R solo imbalance for the per-side front groups.
    Destructive-summing zones from the together-vs-solo audit are masked from
    tonal candidate generation so PEQ is not asked to fix phase.
    """
    system_dev = opt.erb_smooth(freqs, traces["System Sum"] - target)
    masks = interference_masks(freqs, traces)
    audible = opt.audibility_weight(freqs)
    vocal = np.ones_like(freqs)
    vocal[(freqs >= 200.0) & (freqs <= 6000.0)] = 1.8
    peak_mult = np.where(system_dev > 0.0, 2.0, 0.75)
    balance_w = audible.copy()
    balance_w[(freqs >= 700.0) & (freqs <= 5000.0)] *= 1.8
    pools = {}
    for group, cfg in opt.GROUPS.items():
        lo, hi = cfg["range"]
        q_range = cfg["q_range"]
        gain_range = cfg["gain_range"]
        contribution = branch_contribution(freqs, traces, group)
        active_driver = contribution >= (10 ** (-6.0 / 10.0))
        candidates = []
        if cfg.get("system_transfer"):
            anchor_sel = (
                (freqs >= 1000.0) & (freqs <= 1400.0)
                & ~masks.get(group, np.zeros_like(freqs, dtype=bool))
            )
            shape_reference = float(np.median(system_dev[anchor_sel])) if np.any(anchor_sel) else 0.0
            shape_dev = system_dev - shape_reference
            shape_peak_mult = np.where(shape_dev > 0.0, 2.0, 0.85)
            tonal_strength = np.abs(shape_dev) * contribution * audible * vocal * shape_peak_mult
            tonal_gain = -0.90 * shape_dev / np.maximum(contribution, 0.55)
            tonal_source = "target_shape"
        else:
            tonal_strength = np.abs(system_dev) * contribution * audible * vocal * peak_mult
            tonal_gain = -0.65 * system_dev / np.maximum(contribution, 0.35)
            tonal_source = "tonal"
        tonal_strength[masks.get(group, False)] = 0.0
        tonal_strength[~active_driver] = 0.0
        candidates.extend(candidate_peaks(
            freqs, tonal_strength, tonal_gain, lo, hi, q_range, gain_range, tonal_source, profile
        ))

        if cfg.get("pair") and cfg.get("side"):
            pair = opt.PAIR_DEFS[cfg["pair"]]
            diff = opt.erb_smooth(freqs, traces[pair["left"]] - traces[pair["right"]])
            if cfg["side"] == "left":
                bal_gain = -0.85 * diff
            else:
                bal_gain = 0.85 * diff
            bal_strength = np.abs(bal_gain) * balance_w
            bal_strength[~active_driver] = 0.0
            blo, bhi = pair["balance_band"]
            bal_strength[(freqs < blo) | (freqs > bhi)] = 0.0
            candidates.extend(candidate_peaks(
                freqs, bal_strength, bal_gain, lo, hi, q_range, gain_range, "balance", profile
            ))

        candidates.sort(key=lambda c: -c["strength"])
        deduped = []
        for c in candidates:
            if all(abs(math.log2(c["F"] / d["F"])) >= 1 / 8 or c["source"] != d["source"] for d in deduped):
                c["branch_share"] = float(np.interp(np.log10(c["F"]), np.log10(freqs), contribution))
                if c["branch_share"] < 10 ** (-6.0 / 10.0):
                    continue
                deduped.append(c)
            if len(deduped) >= 14:
                break
        pools[group] = deduped
    return pools


def guided_band(rng: np.random.Generator, candidate, cfg: Dict[str, object]):
    qlo, qhi = cfg["q_range"]
    glo, ghi = cfg["gain_range"]
    flo, fhi = cfg["range"]
    sigma_oct = float(np.clip(candidate["width_oct"] / 3.0, 1 / 48, 1 / 5))
    F = candidate["F"] * (2 ** rng.normal(0.0, sigma_oct))
    Q = candidate["Q"] * math.exp(rng.normal(0.0, 0.28))
    gain_sigma = max(0.35, abs(candidate["G"]) * 0.22)
    G = rng.normal(candidate["G"], gain_sigma)
    band = opt.rounded_band(
        float(np.clip(F, flo, fhi)),
        float(np.clip(Q, qlo, qhi)),
        float(np.clip(G, glo, ghi)),
    )
    return band


def guided_groups(rng: np.random.Generator, profile: str, pools) -> GroupBands:
    groups: GroupBands = {}
    for group, cfg in opt.GROUPS.items():
        pool = pools.get(group, [])
        bands = []
        if pool:
            weights = np.array([max(c["strength"], 0.05) for c in pool], dtype=float)
            weights /= weights.sum()
            max_bands = min(int(cfg["max_bands"]), len(pool))
            n = int(rng.integers(0, max_bands + 1))
            if profile == "explore" and max_bands and rng.random() < 0.50:
                n = max(1, n)
            if n:
                picked = rng.choice(len(pool), size=n, replace=False, p=weights)
                for idx in np.atleast_1d(picked):
                    band = guided_band(rng, pool[int(idx)], cfg)
                    if band is not None:
                        bands.append(band)

        # A small wildcard rate keeps the search capable of finding a missed
        # broad region, but the run is dominated by data-derived centers.
        wildcard_rate = 0.04 if profile == "explore" else 0.02
        while len(bands) < int(cfg["max_bands"]) and rng.random() < wildcard_rate:
            band = random_band(rng, cfg)
            if band is not None:
                bands.append(band)

        bands.sort(key=lambda b: b[0])
        groups[group] = bands
    return groups


def deterministic_beam_combinations(pools, component_score, beam_width: int = 24,
                                    pool_limit: int = 6, deadline: float | None = None,
                                    order_seed: int | None = None, stop_requested=None):
    """Build exact guided-band combinations while retaining best partial tunes."""
    empty = {group: [] for group in opt.GROUPS}
    beam = [(float(component_score(empty)["objective"]), opt.bands_signature(empty), empty)]
    evaluations = 1
    score_cache = {beam[0][1]: beam[0][0]}
    group_names = list(opt.GROUPS)
    if order_seed is not None and len(group_names) > 1:
        order_rng = np.random.default_rng(int(order_seed))
        group_names = [group_names[index] for index in order_rng.permutation(len(group_names))]
    for group in group_names:
        cfg = opt.GROUPS[group]
        candidates = sorted(pools.get(group, []), key=lambda item: -item["strength"])[:max(0, pool_limit)]
        bands = [opt.rounded_band(item["F"], item["Q"], item["G"]) for item in candidates]
        bands = [band for band in bands if band is not None]
        options = [()]
        max_active = min(int(cfg["max_bands"]), len(bands), 2)
        for count in range(1, max_active + 1):
            options.extend(itertools.combinations(bands, count))
        expanded = []
        for _value, _signature, partial in beam:
            for option in options:
                if stop_requested is not None and stop_requested():
                    candidates_now = beam + expanded
                    return sorted(candidates_now, key=lambda item: (item[0], item[1]))[:max(1, int(beam_width))], evaluations
                if deadline is not None and time.monotonic() >= deadline:
                    candidates_now = beam + expanded
                    unique_now = {}
                    for entry in sorted(candidates_now, key=lambda item: (item[0], item[1])):
                        unique_now.setdefault(entry[1], entry)
                    return list(unique_now.values())[:max(1, int(beam_width))], evaluations
                groups = _copy_groups(partial)
                groups[group] = sorted(option, key=lambda band: band[0])
                signature = opt.bands_signature(groups)
                value = score_cache.get(signature)
                if value is None:
                    value = float(component_score(groups)["objective"])
                    score_cache[signature] = value
                    evaluations += 1
                expanded.append((value, signature, groups))
        unique = {}
        for entry in sorted(expanded, key=lambda item: (item[0], item[1])):
            unique.setdefault(entry[1], entry)
        beam = list(unique.values())[:max(1, int(beam_width))]
    return beam, evaluations


def gain_to_unit(gain: float, cfg: Dict[str, object]) -> float:
    glo, ghi = cfg["gain_range"]
    return float(np.clip((float(gain) - glo) / max(float(ghi - glo), 1e-9), 0.0, 1.0))


def band_to_unit(band, cfg: Dict[str, object]) -> List[float]:
    F, Q, G = band
    flo, fhi = cfg["range"]
    qlo, qhi = cfg["q_range"]
    return [
        float(np.clip((math.log(float(F)) - math.log(flo)) / (math.log(fhi) - math.log(flo)), 0.0, 1.0)),
        float(np.clip((float(Q) - qlo) / max(float(qhi - qlo), 1e-9), 0.0, 1.0)),
        gain_to_unit(float(G), cfg),
    ]


def unit_to_band(values, cfg: Dict[str, object]):
    flo, fhi = cfg["range"]
    qlo, qhi = cfg["q_range"]
    glo, ghi = cfg["gain_range"]
    F = math.exp(math.log(flo) + float(values[0]) * (math.log(fhi) - math.log(flo)))
    Q = qlo + float(values[1]) * (qhi - qlo)
    G = glo + float(values[2]) * (ghi - glo)
    return opt.rounded_band(F, Q, G)


def cma_seed_vector(pools) -> np.ndarray:
    values = []
    for group, cfg in opt.GROUPS.items():
        pool = sorted(pools.get(group, []), key=lambda c: -c["strength"])
        max_bands = int(cfg["max_bands"])
        flo, fhi = cfg["range"]
        qlo, qhi = cfg["q_range"]
        off = [
            0.5,
            float(np.clip((1.4 - qlo) / max(float(qhi - qlo), 1e-9), 0.0, 1.0)),
            gain_to_unit(0.0, cfg),
        ]
        for idx in range(max_bands):
            if idx < len(pool):
                candidate = pool[idx]
                values.extend(band_to_unit((candidate["F"], candidate["Q"], candidate["G"]), cfg))
            else:
                # Center unused slots in the passband with zero gain; decoded
                # rounded_band then treats them as off.
                values.extend([
                    float(np.clip((math.log(math.sqrt(flo * fhi)) - math.log(flo)) / (math.log(fhi) - math.log(flo)), 0.0, 1.0)),
                    off[1],
                    off[2],
                ])
    return np.asarray(values, dtype=float)


def cma_decode_vector(vector: np.ndarray) -> GroupBands:
    groups: GroupBands = {}
    pos = 0
    for group, cfg in opt.GROUPS.items():
        bands = []
        for _idx in range(int(cfg["max_bands"])):
            band = unit_to_band(vector[pos:pos + 3], cfg)
            pos += 3
            if band is not None:
                bands.append(band)
        bands.sort(key=lambda b: b[0])
        groups[group] = bands
    return groups


class CmaProposal:
    def __init__(self, seed: int, pools, sigma: float = 0.18, population_size: int | None = None):
        if CMA is None:
            raise RuntimeError("cmaes is not installed; use --proposal guided or install cmaes")
        mean = cma_seed_vector(pools)
        bounds = np.tile(np.asarray([[0.0, 1.0]], dtype=float), (len(mean), 1))
        self.seed = int(seed)
        self.sigma = float(sigma)
        self.population_size = population_size
        self.bounds = bounds
        self.optimizer = CMA(
            mean=mean,
            sigma=self.sigma,
            bounds=self.bounds,
            seed=self.seed,
            population_size=self.population_size,
        )
        self.pending = []
        self.restart_count = 0

    def ask(self) -> Tuple[np.ndarray, GroupBands]:
        x = self.optimizer.ask()
        return x, cma_decode_vector(x)

    def tell(self, x: np.ndarray, value: float) -> None:
        self.pending.append((x, float(value)))
        if len(self.pending) >= self.optimizer.population_size:
            self.optimizer.tell(self.pending)
            self.pending = []
            if self.optimizer.should_stop():
                self.restart()

    def restart(self) -> None:
        self.restart_count += 1
        seed = self.seed + 1009 * self.restart_count
        mean = np.clip(self.optimizer.mean, 0.0, 1.0)
        self.optimizer = CMA(
            mean=mean,
            sigma=self.sigma,
            bounds=self.bounds,
            seed=seed,
            population_size=self.population_size,
        )
        self.pending = []


def write_guidance(path: Path, pools) -> None:
    lines = ["# Guided Candidate Centers", ""]
    for group in opt.GROUPS:
        lines.append(f"## {group}")
        if not pools.get(group):
            lines.append("- no tonal/balance candidate centers found")
            continue
        for c in pools[group]:
            lines.append(
                "- F={F:.1f} Hz Q~{Q:.2f} G~{G:+.2f} dB "
                "source={source} strength={strength:.2f} branch_share={branch_share:.2f} "
                "width={width_oct:.2f} oct".format(**c)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_group_objective(freqs, traces, target, filter_cost_scale, worst_weight, min_total_bands):
    component_score = opt.make_component_scorer(freqs, traces, target, filter_cost_scale, worst_weight)

    def score(groups: GroupBands) -> float:
        return component_score(groups)["objective"]

    return score


def _copy_groups(groups: GroupBands) -> GroupBands:
    return {name: list(bands) for name, bands in groups.items()}


def _band_neighbours(band, cfg: Dict[str, object]):
    """Yield hardware-rounded one-coordinate moves around an active band."""
    F, Q, G = band
    flo, fhi = cfg["range"]
    qlo, qhi = cfg["q_range"]
    glo, ghi = cfg["gain_range"]
    moves = []
    for octaves in (-1 / 48, -1 / 96, 1 / 96, 1 / 48):
        moves.append((float(np.clip(F * (2 ** octaves), flo, fhi)), Q, G))
    for dq in (-0.10, 0.10):
        moves.append((F, float(np.clip(Q + dq, qlo, qhi)), G))
    for dg in (-0.25, 0.25):
        moves.append((F, Q, float(np.clip(G + dg, glo, ghi))))
    seen = set()
    for raw in moves:
        candidate = opt.rounded_band(*raw)
        if candidate is None or candidate == band or candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def coordinate_refine(groups: GroupBands, component_score, passes: int = 2):
    """Refine F/Q/G on the authoritative scalar objective only.

    This deliberately uses hardware-sized coordinate moves instead of
    ``fit_peq``: fit_peq has its own tonal residual objective, while this pass
    must preserve every null, balance, headroom, and guardrail term returned by
    ``afpx_objective.score_bands``.
    """
    current = _copy_groups(groups)
    current_components = component_score(current)
    current_value = float(current_components["objective"])
    evaluations = 1
    for _pass in range(max(0, int(passes))):
        improved_this_pass = False
        for group in tuple(opt.GROUPS):
            index = 0
            while index < len(current.get(group, [])):
                original = current[group][index]
                best_groups = current
                best_components = current_components
                best_value = current_value
                variants = list(_band_neighbours(original, opt.GROUPS[group])) + [None]
                for replacement in variants:
                    trial = _copy_groups(current)
                    if replacement is None:
                        del trial[group][index]
                    else:
                        trial[group][index] = replacement
                    trial[group].sort(key=lambda band: band[0])
                    components = component_score(trial)
                    evaluations += 1
                    value = float(components["objective"])
                    if value + 1e-12 < best_value:
                        best_groups = trial
                        best_components = components
                        best_value = value
                if best_groups is not current:
                    removed = len(best_groups[group]) < len(current[group])
                    current = best_groups
                    current_components = best_components
                    current_value = best_value
                    improved_this_pass = True
                    if removed:
                        continue
                index += 1
        if not improved_this_pass:
            break
    return current, current_components, evaluations


def refine_entries(entries, component_score, top: int, passes: int):
    refined = []
    improved = 0
    evaluations = 0
    best_before = float(entries[0][0]) if entries else None
    for value, _signature, groups in entries[:max(0, int(top))]:
        new_groups, components, used = coordinate_refine(groups, component_score, passes=passes)
        evaluations += used
        new_value = float(components["objective"])
        if new_value + 1e-12 < float(value):
            improved += 1
        refined.append((new_value, opt.bands_signature(new_groups), new_groups))
    best_after = min((float(item[0]) for item in refined), default=best_before)
    report = {
        "enabled": bool(top > 0 and passes > 0),
        "seed_candidates": min(len(entries), max(0, int(top))),
        "passes": max(0, int(passes)),
        "evaluations": evaluations,
        "improved_candidates": improved,
        "best_before": best_before,
        "best_after": best_after,
    }
    return refined, report


ARCHIVE_KEYS = (
    "pareto_tonal_db",
    "peak_penalty_db",
    "balance_penalty_db",
    "positive_gain_penalty_db",
    "filter_count",
    "objective",
)


def entry_metric(item, score_map, key: str) -> float:
    if key == "objective":
        return float(item[0])
    return float(score_map[item[1]].get(key, float("inf")))


def combine_unique_entries(*collections):
    out = []
    seen = set()
    for collection in collections:
        for item in sorted(collection, key=lambda x: x[0]):
            sig = item[1]
            if sig in seen:
                continue
            seen.add(sig)
            out.append(item)
    out.sort(key=lambda x: x[0])
    return out


def prune_archive(archive, score_map, archive_size):
    if archive_size <= 0:
        return [], {}
    archive = combine_unique_entries(archive)
    if len(archive) <= archive_size:
        return archive, {item[1]: score_map[item[1]] for item in archive if item[1] in score_map}

    per_key = max(8, archive_size // len(ARCHIVE_KEYS))
    chosen = set()
    for key in ARCHIVE_KEYS:
        ranked = sorted(archive, key=lambda item: entry_metric(item, score_map, key))
        for item in ranked[:per_key]:
            chosen.add(item[1])

    if len(chosen) < archive_size:
        for item in sorted(archive, key=lambda x: x[0]):
            chosen.add(item[1])
            if len(chosen) >= archive_size:
                break

    new_archive = [item for item in archive if item[1] in chosen]
    new_archive.sort(key=lambda x: x[0])
    if len(new_archive) > archive_size:
        new_archive = new_archive[:archive_size]
    new_scores = {item[1]: score_map[item[1]] for item in new_archive if item[1] in score_map}
    return new_archive, new_scores


def insert_best(best, item, keep):
    value, signature, _groups = item
    if signature in {sig for _v, sig, _g in best}:
        return best
    best.append(item)
    best.sort(key=lambda x: x[0])
    if len(best) > keep:
        best.pop()
    return best


def insert_archive(archive, score_map, item, components, archive_size):
    if archive_size <= 0:
        return archive, score_map
    signature = item[1]
    if signature in score_map:
        return archive, score_map
    archive.append(item)
    score_map[signature] = dict(components)
    limit = max(archive_size, int(archive_size * 1.25))
    if len(archive) >= limit:
        archive, score_map = prune_archive(archive, score_map, archive_size)
    return archive, score_map


def build_rows(freqs, traces, target, best, component_score=None):
    if component_score is None:
        component_score = opt.make_component_scorer(freqs, traces, target)
    rows = []
    for rank, (value, signature, groups) in enumerate(best, start=1):
        pred = opt.predict_traces(freqs, traces, groups)
        score = opt.tune_scorecard(freqs, pred, target)
        components = component_score(groups)
        rows.append({
            "rank": rank,
            "file": f"candidate_{rank:02d}_objective_{value:.4f}.afpx",
            "objective": float(value),
            "score": score,
            "components": components,
            "groups": groups,
            "signature": signature,
            "lint": None,
            "headroom": {g: opt.headroom_report(freqs, b) for g, b in groups.items()},
            "left_alone": opt.left_alone_note(freqs, traces),
        })
    return rows


def serializable_groups(groups: GroupBands):
    return {
        group: [[float(F), float(Q), float(G)] for F, Q, G in bands]
        for group, bands in groups.items()
    }


def groups_from_json(data) -> GroupBands:
    groups: GroupBands = {}
    for group in opt.GROUPS:
        groups[group] = [
            (float(F), float(Q), float(G))
            for F, Q, G in data.get(group, [])
        ]
    return groups


def save_state(path: Path, best, rng: np.random.Generator, completed_trials: int,
               elapsed_seconds: float, args: argparse.Namespace, archive=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = archive or []
    payload = {
        "version": 6,
        "objective": "spatial_objective_beam_cached_phase_v6",
        "completed_trials": int(completed_trials),
        "elapsed_seconds": float(elapsed_seconds),
        "seed": int(args.seed),
        "profile": args.profile,
        "proposal": args.proposal,
        "mode": getattr(args, "mode", "peq"),
        "filter_cost_scale": float(args.filter_cost_scale),
        "worst_weight": float(args.worst_weight),
        "min_total_bands": int(args.min_total_bands),
        "archive_size": int(getattr(args, "archive_size", 0)),
        "rng_state": rng.bit_generator.state,
        "best": [
            {"objective": float(value), "groups": serializable_groups(groups)}
            for value, _signature, groups in best
        ],
        "archive": [
            {"objective": float(value), "groups": serializable_groups(groups)}
            for value, _signature, groups in archive
        ],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path, rng: np.random.Generator, component_score=None, archive_size: int = 0):
    if not path.exists():
        return [], [], {}, 0, 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in (4, 5, 6):
        return [], [], {}, 0, 0.0
    if "rng_state" in payload:
        rng.bit_generator.state = payload["rng_state"]
    best = []
    for item in payload.get("best", []):
        groups = groups_from_json(item.get("groups", {}))
        signature = opt.bands_signature(groups)
        value = (
            float(component_score(groups)["objective"])
            if component_score is not None else float(item["objective"])
        )
        best.append((value, signature, groups))
    best.sort(key=lambda x: x[0])
    archive = []
    score_map = {}
    for item in payload.get("archive", []):
        groups = groups_from_json(item.get("groups", {}))
        signature = opt.bands_signature(groups)
        value = (
            float(component_score(groups)["objective"])
            if component_score is not None else float(item["objective"])
        )
        entry = (value, signature, groups)
        archive.append(entry)
        if component_score is not None:
            score_map[signature] = component_score(groups)
    if component_score is not None:
        archive, score_map = prune_archive(archive, score_map, archive_size)
    else:
        archive = []
    return best, archive, score_map, int(payload.get("completed_trials", 0)), float(payload.get("elapsed_seconds", 0.0))


def interference_notes(freqs, traces):
    notes = []
    for name, pair in opt.PAIR_DEFS.items():
        try:
            audit = opt.interference_audit(
                freqs, traces[pair["left"]], traces[pair["right"]], traces[pair["together"]]
            )
        except Exception:
            continue
        label = {"low": "Midbass L/R", "mid": "Midrange L/R", "high": "Tweeter L/R"}.get(name, name)
        ranges = opt.mask_ranges(freqs, audit[3], pair["branch_band"])
        if ranges:
            pretty = ", ".join(f"{lo:.0f}-{hi:.0f} Hz" for lo, hi in ranges[:8])
            if len(ranges) > 8:
                pretty += ", ..."
            notes.append(f"{label} destructive-summing audit flagged: {pretty}.")
    return notes


def write_outputs(out_dir, base_xml, freqs, traces, rich_traces, target, best, baseline_score, args,
                  checkpoint=False, family_entries=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("candidate_*.afpx"):
        old.unlink()
    phase_plan = getattr(args, "phase_plan", [])
    crossover_rows = getattr(args, "crossover_rows", [])
    phase_valid = bool(dict(args.measurement_session.get("audit", {})).get("phase_valid"))
    def safe_entries(entries):
        kept = []
        for entry in entries or []:
            if phase_valid and phase_plan:
                verification = opt.complex_crossover_verification(freqs, rich_traces, entry[2], phase_plan)
                if verification["pass"]:
                    kept.append(entry)
            else:
                conflicts = opt.phase_peq_conflicts(freqs, entry[2], phase_plan)
                if conflicts:
                    args.phase_peq_rejections.extend(conflicts)
                    continue
                kept.append(entry)
        return kept
    best = safe_entries(best)
    family_entries = safe_entries(family_entries)
    component_score = opt.complex_phase_component_scorer(
        opt.make_component_scorer(
            freqs, traces, target, args.filter_cost_scale, args.worst_weight
        ),
        freqs,
        rich_traces,
        phase_plan,
        phase_valid,
    )
    rows = build_rows(freqs, traces, target, best, component_score)
    family_rows = build_rows(freqs, traces, target, family_entries, component_score) if family_entries else rows
    unique_rejections = {}
    for item in args.phase_peq_rejections:
        key = (
            str(item.get("source")), str(item.get("group")),
            float(item.get("filter", {}).get("F", 0.0)),
            float(item.get("filter", {}).get("Q", 0.0)),
            float(item.get("filter", {}).get("G", 0.0)),
        )
        unique_rejections.setdefault(key, item)
    args.phase_peq_rejections = list(unique_rejections.values())[:20]
    for row in rows:
        path = out_dir / row["file"]
        row["lint"] = opt.write_candidate(base_xml, path, row["groups"], phase_plan=phase_plan)
        row["path"] = str(path)
    opt.write_family_aliases(out_dir, family_rows, base_xml, phase_plan=phase_plan)
    args.trials = args._completed_trials
    opt.write_report(out_dir, rows, baseline_score, interference_notes(freqs, traces), args,
                     family_rows=family_rows, crossover_rows=crossover_rows, phase_plan=phase_plan)
    status = [
        f"checkpoint={checkpoint}",
        f"completed_trials={args._completed_trials}",
        f"elapsed_seconds={int(args._elapsed_seconds)}",
    ]
    if rows:
        status.append(f"best_objective={rows[0]['objective']:.6f}")
        comp = rows[0].get("components", {})
        if comp:
            status.append(
                "best_components="
                f"tonal:{comp['tonal_error_db']:.3f},"
                f"peak:{comp['peak_penalty_db']:.3f},"
                f"balance:{comp['balance_penalty_db']:.3f},"
                f"headroom:{comp['positive_gain_penalty_db']:.3f},"
                f"filters:{comp['filter_count']:.0f}"
            )
        status.append(opt.format_bands(rows[0]["groups"]))
    (out_dir / "stream_status.txt").write_text("\n".join(status) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Constant-memory random optimizer.")
    parser.add_argument("--baseline", type=Path, default=opt.DEFAULT_BASELINE)
    parser.add_argument("--target", type=Path, default=opt.DEFAULT_TARGET)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--keep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--profile", choices=("safe", "explore"), default="explore")
    parser.add_argument("--proposal", choices=("guided", "random", "mixed", "cmaes", "beam"), default="guided")
    parser.add_argument("--mode", choices=("peq", "phase"), default="peq",
                        help="Phase mode preserves PEQ and evaluates only the gated phase plan.")
    parser.add_argument("--filter-cost-scale", type=float, default=0.1)
    parser.add_argument("--worst-weight", type=float, default=0.10)
    parser.add_argument("--min-total-bands", type=int, default=0)
    parser.add_argument("--archive-size", type=int, default=4000)
    parser.add_argument("--refine-top", type=int, default=12,
                        help="Deterministically refine this many final candidates on the same scalar objective; 0 disables.")
    parser.add_argument("--refine-passes", type=int, default=2,
                        help="Maximum hardware-step coordinate passes per refined candidate.")
    parser.add_argument("--cma-sigma", type=float, default=0.18)
    parser.add_argument("--cma-population", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--beam-pool-limit", type=int, default=6)
    parser.add_argument("--max-positive-gain-penalty", type=float, default=0.0,
                        help="Reject candidates above this headroom penalty; 0 disables the hard gate.")
    parser.add_argument("--validation-threshold", type=float, default=2.5)
    parser.add_argument("--gate-ms", type=float, default=None,
                        help="Optional impulse/window gate length in milliseconds for confidence warnings.")
    parser.add_argument("--sample-rate", type=float, default=96000.0,
                        help="DSP internal sample rate used for delay writes.")
    parser.add_argument("--impulse-root", type=Path, default=None,
                        help="Optional folder containing companion WAV/text impulse exports.")
    parser.add_argument("--phase-cache", type=Path, default=None,
                        help="Shared fingerprinted crossover diagnostic cache.")
    parser.add_argument("--level-calibration", type=Path, default=None,
                        help="JSON role/file -> dB offsets for mixed-level measurement sessions.")
    parser.add_argument("--phase-writes", choices=("auto", "off"), default="auto",
                        help="Use 'off' to report the crossover ladder without writing polarity/delay/APF changes.")
    parser.add_argument("--checkpoint-seconds", type=int, default=60)
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="Optional shared stop-request file for graceful GUI cancellation.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from OUT\\stream_state.json if it exists.")
    parser.add_argument("--print-mode", choices=("compact", "full", "none"), default="compact",
                        help="Console detail only; full reports are always written to disk.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stop_requested = lambda: bool(args.stop_file and args.stop_file.exists())

    args.measurement_session, level_calibration = opt.prepare_measurement_session(
        args.baseline, args.target, args.level_calibration
    )
    opt.sync_external_objective(args.baseline, args.target, level_calibration)
    configure_profile(args.profile)
    rng = np.random.default_rng(args.seed)
    freqs, traces, rich_traces = opt.load_measurements(level_calibration)
    raw_target = opt.load_target(args.target, freqs)
    target = raw_target + opt.target_anchor_offset(freqs, traces["System Sum"], raw_target)
    base_xml = opt.decode_afpx(args.baseline)
    args.validation = opt.pair_sum_validation(freqs, traces, threshold=args.validation_threshold)
    failed_validation = [item for item in args.validation if not item["pass"]]
    if failed_validation:
        details = "; ".join(
            f"{item['pair']} {item['rms_db']} dB > {item['threshold_db']} dB"
            for item in failed_validation
        )
        raise SystemExit("Measurement validation gate failed: " + details)
    phase_session = opt.analyze_phase_session(
        freqs, traces, rich_traces, args.measurement_session, args.sample_rate,
        args.impulse_root, args.phase_cache, writes=args.phase_writes != "off"
    )
    args.crossover_rows = phase_session["diagnostics"]
    args.phase_diagnostic_cache = phase_session["cache"]
    args.phase_plan = phase_session["writes"]
    if args.mode == "phase":
        args.proposal = "beam"
        guided_pools = {group: [] for group in opt.GROUPS}
    else:
        guided_pools = find_guided_candidates(freqs, traces, target, args.profile)
    cma_proposal = None
    if args.proposal == "cmaes":
        cma_proposal = CmaProposal(
            args.seed,
            guided_pools,
            sigma=args.cma_sigma,
            population_size=args.cma_population or None,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_guidance(args.out / "guided_candidates.md", guided_pools)

    baseline_groups: GroupBands = {group: [] for group in opt.GROUPS}
    baseline_pred = opt.predict_traces(freqs, traces, baseline_groups)
    baseline_score = opt.tune_scorecard(freqs, baseline_pred, target)
    component_score = opt.complex_phase_component_scorer(
        opt.make_component_scorer(
            freqs, traces, target, args.filter_cost_scale, args.worst_weight
        ),
        freqs,
        rich_traces,
        args.phase_plan,
        bool(args.measurement_session["audit"].get("phase_valid")),
    )
    args.phase_peq_rejections = []
    baseline_score["components"] = component_score(baseline_groups)
    score_groups = make_group_objective(
        freqs,
        traces,
        target,
        args.filter_cost_scale,
        args.worst_weight,
        args.min_total_bands,
    )

    state_path = args.out / "stream_state.json"
    if args.resume:
        best, archive, archive_scores, completed_before, elapsed_before = load_state(
            state_path, rng, component_score, args.archive_size
        )
    else:
        best, archive, archive_scores, completed_before, elapsed_before = [], [], {}, 0, 0.0
    baseline_item = (
        float(baseline_score["components"]["objective"]),
        opt.bands_signature(baseline_groups),
        baseline_groups,
    )
    best = insert_best(best, baseline_item, args.keep)
    archive, archive_scores = insert_archive(
        archive, archive_scores, baseline_item, baseline_score["components"], args.archive_size
    )

    start = time.monotonic()
    next_checkpoint = start + max(10, args.checkpoint_seconds)
    trials = 0
    args.beam = None
    if args.proposal == "beam":
        beam_order_seed = args.seed + completed_before
        beam_entries, beam_evaluations = deterministic_beam_combinations(
            guided_pools,
            component_score,
            beam_width=args.beam_width,
            pool_limit=args.beam_pool_limit,
            deadline=(start + args.seconds) if args.seconds else None,
            order_seed=beam_order_seed,
            stop_requested=stop_requested,
        )
        for item in beam_entries:
            components = component_score(item[2])
            best = insert_best(best, item, args.keep)
            archive, archive_scores = insert_archive(
                archive, archive_scores, item, components, args.archive_size
            )
        trials += beam_evaluations
        args.beam = {
            "width": args.beam_width,
            "pool_limit": args.beam_pool_limit,
            "evaluations": beam_evaluations,
            "retained": len(beam_entries),
            "order_seed": beam_order_seed,
        }
    while True:
        now = time.monotonic()
        if stop_requested():
            break
        if args.seconds and now - start >= args.seconds:
            break
        if args.max_trials and trials >= args.max_trials:
            break
        if args.proposal == "beam":
            break
        cma_x = None
        if args.proposal == "cmaes":
            cma_x, groups = cma_proposal.ask()
        elif args.proposal == "random":
            groups = random_groups(rng, args.profile)
        elif args.proposal == "mixed" and rng.random() < 0.20:
            groups = random_groups(rng, args.profile)
        else:
            groups = guided_groups(rng, args.profile, guided_pools)
        components = component_score(groups)
        if components.get("phase_peq_conflict_count", 0.0) > 0.0:
            args.phase_peq_rejections.extend(opt.phase_peq_conflicts(freqs, groups, args.phase_plan))
        value = float(components["objective"])
        if args.max_positive_gain_penalty > 0 and components["positive_gain_penalty_db"] > args.max_positive_gain_penalty:
            value = 1e6 + float(components["positive_gain_penalty_db"])
        signature = opt.bands_signature(groups)
        item = (value, signature, groups)
        best = insert_best(best, item, args.keep)
        archive, archive_scores = insert_archive(archive, archive_scores, item, components, args.archive_size)
        if cma_x is not None:
            cma_proposal.tell(cma_x, value)
        trials += 1

        if best and args.checkpoint_seconds and now >= next_checkpoint:
            args._completed_trials = completed_before + trials
            args._elapsed_seconds = elapsed_before + (now - start)
            archive, archive_scores = prune_archive(archive, archive_scores, args.archive_size)
            save_state(
                state_path, best, rng, args._completed_trials, args._elapsed_seconds, args, archive=archive
            )
            output_entries = combine_unique_entries(best, archive)[: args.top]
            family_limit = max(args.top * 10, min(args.archive_size, 200))
            family_entries = combine_unique_entries(best, archive)[:family_limit]
            write_outputs(
                args.out / "_checkpoint",
                base_xml,
                freqs,
                traces,
                rich_traces,
                target,
                output_entries,
                baseline_score,
                args,
                checkpoint=True,
                family_entries=family_entries,
            )
            next_checkpoint = now + args.checkpoint_seconds

    args._completed_trials = completed_before + trials
    args._elapsed_seconds = elapsed_before + (time.monotonic() - start)
    archive, archive_scores = prune_archive(archive, archive_scores, args.archive_size)
    save_state(state_path, best, rng, args._completed_trials, args._elapsed_seconds, args, archive=archive)
    final_entries = combine_unique_entries(best, archive)
    if stop_requested():
        refined_entries = []
        args.refinement = {"enabled": False, "reason": "graceful stop requested"}
    else:
        refined_entries, args.refinement = refine_entries(
            final_entries,
            component_score,
            top=args.refine_top,
            passes=args.refine_passes,
        )
    final_entries = combine_unique_entries(final_entries, refined_entries)
    save_state(
        state_path,
        final_entries[:args.keep],
        rng,
        args._completed_trials,
        args._elapsed_seconds,
        args,
        archive=final_entries[:args.archive_size],
    )
    output_entries = final_entries[: args.top]
    family_limit = max(args.top * 10, min(args.archive_size, 200))
    family_entries = final_entries[:family_limit]
    write_outputs(
        args.out, base_xml, freqs, traces, rich_traces, target, output_entries, baseline_score, args, family_entries=family_entries
    )
    if args.print_mode != "none":
        best_entry = output_entries[0] if output_entries else None
        compact = {
            "status": "complete",
            "trials_this_run": trials,
            "trials_total": args._completed_trials,
            "elapsed_seconds": round(float(args._elapsed_seconds), 1),
            "best_objective": None if best_entry is None else round(float(best_entry[0]), 6),
            "best_file": None if best_entry is None else str(
                args.out / ("candidate_01_objective_%.4f.afpx" % best_entry[0])
            ),
            "refinement": args.refinement,
            "assistant_summary": str(args.out / "assistant_summary.json"),
        }
        print(json.dumps(compact, indent=2))
        if args.print_mode == "full" and best_entry is not None:
            print(opt.format_bands(best_entry[2]))


if __name__ == "__main__":
    main()
