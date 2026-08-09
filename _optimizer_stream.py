"""Constant-memory random-search optimizer.

Use this for long brute-force runs. It does not use Optuna's in-memory Study,
so RAM stays flat: each worker keeps only the best candidates it has seen.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass, replace
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
import baseline_rehabilitation as rehab

try:
    from cmaes import CMA
except ImportError:  # Keep random/guided modes usable if the optional backend is absent.
    CMA = None


GroupBands = Dict[str, List[Tuple[float, float, float]]]
LAST_PROPOSAL_AUDIT: Dict[str, object] = {}


@dataclass(frozen=True)
class BeamEntry:
    objective: float
    signature: tuple
    plan: rehab.CandidatePlan
    components: dict[str, object]

    @property
    def groups(self) -> GroupBands:
        groups = {group: [] for group in opt.GROUPS}
        groups.update(rehab.thaw_groups(self.plan.groups))
        return groups

    def __iter__(self):
        yield self.objective
        yield self.signature
        yield self.groups

    def __getitem__(self, index):
        return (self.objective, self.signature, self.groups)[index]


@dataclass(frozen=True)
class RehabilitationBudget:
    seconds: float
    max_evaluations: int


def rehabilitation_budget(total_seconds: float | None) -> RehabilitationBudget:
    total = max(0.0, float(total_seconds or 0.0))
    if total <= 0.0:
        return RehabilitationBudget(180.0, 2500)
    seconds = min(max(total * 0.25, 5.0), 180.0)
    if total < 20.0:
        seconds = min(seconds, total * 0.5)
        evaluations = max(16, int(max(total, 1.0) * 20))
    else:
        evaluations = 2500
    return RehabilitationBudget(float(seconds), int(evaluations))


def _band_to_json(band):
    return [float(value) for value in band]


def _filter_ref_to_json(ref):
    return {
        "channel": int(ref.channel),
        "slot": int(ref.slot),
        "role": str(ref.role),
        "filter_type": str(ref.filter_type),
        "original": _band_to_json(ref.original),
        "pair_key": ref.pair_key,
    }


def _filter_ref_from_json(payload):
    return rehab.FilterRef(
        channel=int(payload["channel"]),
        slot=int(payload["slot"]),
        role=str(payload["role"]),
        filter_type=str(payload["filter_type"]),
        original=tuple(float(value) for value in payload["original"]),
        pair_key=payload.get("pair_key"),
    )


def candidate_plan_to_json(plan):
    return {
        "slot_edits": [
            {
                "ref": _filter_ref_to_json(edit.ref),
                "replacement": (
                    None if edit.replacement is None
                    else _band_to_json(edit.replacement)
                ),
            }
            for edit in plan.slot_edits
        ],
        "groups": {
            group: [_band_to_json(band) for band in bands]
            for group, bands in plan.groups
        },
    }


def candidate_plan_from_json(payload):
    payload = dict(payload or {})
    edits = []
    for row in payload.get("slot_edits", []):
        replacement = row.get("replacement")
        edits.append(rehab.SlotEdit(
            ref=_filter_ref_from_json(row["ref"]),
            replacement=(
                None if replacement is None
                else tuple(float(value) for value in replacement)
            ),
        ))
    raw_groups = dict(payload.get("groups", {}))
    groups = {
        str(group): [
            tuple(float(value) for value in band)
            for band in bands
        ]
        for group, bands in raw_groups.items()
    }
    return rehab.CandidatePlan(
        slot_edits=tuple(edits),
        groups=rehab.freeze_groups(groups),
    )


def beam_entry_to_json(entry):
    if not isinstance(entry, BeamEntry):
        value, _signature, groups = entry
        plan = rehab.CandidatePlan(groups=rehab.freeze_groups(groups))
        entry = BeamEntry(
            float(value), rehab.candidate_plan_signature(plan), plan,
            {"objective": float(value)},
        )
    return {
        "objective": float(entry.objective),
        "plan": candidate_plan_to_json(entry.plan),
        "groups": serializable_groups(entry.groups),
        "components": _json_safe(entry.components),
    }


def beam_entry_from_json(payload, score_plan=None, component_score=None):
    if "plan" in payload:
        plan = candidate_plan_from_json(payload.get("plan"))
    else:
        groups = groups_from_json(payload.get("groups", {}))
        plan = rehab.CandidatePlan(groups=rehab.freeze_groups(groups))
    if score_plan is not None:
        components = dict(score_plan(plan))
    elif component_score is not None:
        components = dict(component_score(rehab.thaw_groups(plan.groups)))
    else:
        components = dict(payload.get("components", {}))
        components.setdefault("objective", float(payload["objective"]))
    return BeamEntry(
        objective=float(components["objective"]),
        signature=rehab.candidate_plan_signature(plan),
        plan=plan,
        components=components,
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _input_path_fingerprint(path):
    if path is None:
        return None
    path = Path(path).resolve()
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "content": opt.file_fingerprint(path),
        }
    if path.is_dir():
        return {
            "path": str(path),
            "kind": "folder",
            "files": [
                {
                    "relative_path": item.relative_to(path).as_posix(),
                    "content": opt.file_fingerprint(item),
                }
                for item in sorted(
                    (candidate for candidate in path.rglob("*") if candidate.is_file()),
                    key=lambda candidate: candidate.relative_to(path).as_posix().lower(),
                )
            ],
        }
    return {"path": str(path), "kind": "missing"}

def stream_input_fingerprint_payload(args, rehabilitation_config):
    manifest = dict(
        getattr(args, "measurement_session", {}).get("manifest", {})
    )
    return {
        "baseline": opt.file_fingerprint(args.baseline),
        "target": opt.file_fingerprint(args.target),
        "role_map": manifest.get("resolved_roles", {}),
        "measurement_manifest": manifest,
        "measurement_files": {
            str(role): opt.file_fingerprint(Path(str(path)))
            for role, path in dict(
                manifest.get("resolved_roles", {})
            ).items()
        },
        "objective": {
            "filter_cost_scale": float(args.filter_cost_scale),
            "worst_weight": float(args.worst_weight),
            "min_total_bands": int(args.min_total_bands),
        },
        "rehabilitation_config": _json_safe(
            rehabilitation_config.__dict__
        ),
        "measurement_session_audit": dict(
            getattr(args, "measurement_session", {}).get("audit", {})
        ),
        "level_calibration": {
            "source": _input_path_fingerprint(
                getattr(args, "level_calibration", None)
            ),
            "loaded_values": _json_safe(
                getattr(args, "loaded_level_calibration", {})
            ),
        },
        "repeatability": {
            "source": _input_path_fingerprint(
                getattr(args, "repeatability_folder", None)
            ),
            "model": _json_safe(
                getattr(args, "measurement_noise_guard", {})
            ),
        },
        "phase_context": {
            "writes": str(getattr(args, "phase_writes", "off")),
            "sample_rate": float(getattr(args, "sample_rate", 96000.0)),
            "cache": (
                opt.file_fingerprint(Path(args.phase_cache))
                if getattr(args, "phase_cache", None)
                else None
            ),
        },
        "mode": str(args.mode),
        "profile": str(args.profile),
    }


def stream_input_fingerprint(args, rehabilitation_config):
    payload = stream_input_fingerprint_payload(
        args, rehabilitation_config
    )
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rehabilitation_state_payload(stage, config):
    result = stage.get("result")
    census = tuple(stage.get("census", ()))
    operation_candidates = [
        candidate
        for row in census
        for candidate in row.candidates
    ]
    candidates = () if result is None else result.candidates
    baseline_components = (
        {} if result is None else dict(result.baseline.components)
    )
    best_components = (
        dict(baseline_components) if result is None else dict(result.best.components)
    )
    meaningful_improvement = bool(
        result is not None
        and rehab.meaningfully_better(result.best, result.baseline)
    )
    gate_rejections = [
        {
            "channel": int(row.ref.channel),
            "channel_role": str(row.ref.role),
            "slot": int(row.ref.slot),
            "reason": str(row.probe_skip_reason),
        }
        for row in census
        if row.probe_skip_reason
    ]
    return {
        "status": str(stage.get("status", "unknown")),
        "completion_status": str(stage.get("status", "unknown")),
        "evaluations": int(stage.get("evaluations", 0)),
        "meaningful_improvement": meaningful_improvement,
        "baseline_components": _json_safe(baseline_components),
        "best_components": _json_safe(best_components),
        "gate_rejections": gate_rejections,
        "config": _json_safe(config.__dict__),
        "census": [
            {
                "ref": _filter_ref_to_json(row.ref),
                "paired_ref": (
                    None
                    if row.paired_ref is None
                    else _filter_ref_to_json(row.paired_ref)
                ),
                "retained_candidates": len(row.candidates),
            }
            for row in census
        ],
        "retained_operation_candidates": [
            {
                "plan": candidate_plan_to_json(candidate.plan),
                "components": _json_safe(candidate.components),
            }
            for candidate in operation_candidates
        ],
        "candidate_plans": [
            candidate_plan_to_json(candidate.plan)
            for candidate in candidates
        ],
        "best_plan": candidate_plan_to_json(stage["best_plan"]),
    }



REHABILITATION_CACHE_SCHEMA = "audiofischer-rehabilitation-cache-v1"


def rehabilitation_cache_payload(
    stage, config, fingerprint, fingerprint_inputs
):
    state = rehabilitation_state_payload(stage, config)
    result = stage.get("result")
    state["census_detail"] = [
        {
            "ref": _filter_ref_to_json(row.ref),
            "paired_ref": (
                None
                if row.paired_ref is None
                else _filter_ref_to_json(row.paired_ref)
            ),
            "baseline_components": _json_safe(row.baseline_components),
            "removal_components": _json_safe(row.removal_components),
            "probe_band": (
                None if row.probe_band is None else _band_to_json(row.probe_band)
            ),
            "probe_components": _json_safe(row.probe_components),
            "probe_skip_reason": row.probe_skip_reason,
            "system_delta": row.system_delta,
            "balance_delta": row.balance_delta,
            "headroom_delta": row.headroom_delta,
            "retained_candidates": [
                {
                    "plan": candidate_plan_to_json(candidate.plan),
                    "components": _json_safe(candidate.components),
                }
                for candidate in row.candidates
            ],
        }
        for row in stage.get("census", ())
    ]
    state["scored_candidates"] = [
        {
            "plan": candidate_plan_to_json(candidate.plan),
            "components": _json_safe(candidate.components),
            "depth": int(candidate.depth),
            "export_eligible": bool(candidate.export_eligible),
        }
        for candidate in (() if result is None else result.candidates)
    ]
    return {
        "schema": REHABILITATION_CACHE_SCHEMA,
        "fingerprint": str(fingerprint),
        "fingerprint_inputs": _json_safe(fingerprint_inputs),
        "config": _json_safe(config.__dict__),
        "rehabilitation": state,
    }


def load_rehabilitation_cache(path, expected_fingerprint):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"rehabilitation cache is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"rehabilitation cache is malformed: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"rehabilitation cache is malformed: {path}")
    if payload.get("schema") != REHABILITATION_CACHE_SCHEMA:
        raise RuntimeError(
            f"rehabilitation cache schema is unsupported: {path}"
        )
    actual = str(payload.get("fingerprint", ""))
    if actual != str(expected_fingerprint):
        raise RuntimeError(
            "rehabilitation cache fingerprint mismatch: "
            f"expected {expected_fingerprint}, found {actual or 'missing'}"
        )
    state = payload.get("rehabilitation")
    if not isinstance(state, dict):
        raise RuntimeError(
            f"rehabilitation cache is malformed: missing rehabilitation state: {path}"
        )
    if state.get("completion_status") not in (
        "complete", "no_eligible_filters"
    ):
        raise RuntimeError(
            "rehabilitation cache is incomplete: "
            f"{state.get('completion_status', 'missing')}"
        )
    try:
        best_plan = candidate_plan_from_json(state["best_plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"rehabilitation cache is malformed: invalid best plan: {path}"
        ) from exc
    return {
        "fingerprint": actual,
        "fingerprint_inputs": payload.get("fingerprint_inputs", {}),
        "best_plan": best_plan,
        "rehabilitation": state,
        "payload": payload,
    }


def compact_rehabilitation_cache_state(loaded, path):
    state = dict(loaded["rehabilitation"])
    for key in (
        "census_detail",
        "scored_candidates",
        "retained_operation_candidates",
        "candidate_plans",
    ):
        state.pop(key, None)
    state["cache_path"] = str(Path(path).resolve())
    state["cache_fingerprint"] = str(loaded["fingerprint"])
    return state


def build_or_load_rehabilitation_cache(
    path,
    *,
    expected_fingerprint,
    fingerprint_inputs,
    config,
    build_stage,
    stop_requested=None,
    lock_timeout_seconds=7200.0,
):
    path = Path(path)
    stop_requested = stop_requested or (lambda: False)

    def check_cancelled():
        if stop_requested():
            raise RuntimeError("rehabilitation cache preparation cancelled")

    check_cancelled()
    if path.exists():
        return load_rehabilitation_cache(path, expected_fingerprint)

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    temporary = None
    owns_lock = False
    wait_deadline = time.monotonic() + max(0.1, float(lock_timeout_seconds))
    while not owns_lock:
        check_cancelled()
        if path.exists():
            return load_rehabilitation_cache(path, expected_fingerprint)
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            try:
                owner_pid = int(lock_path.read_text(encoding="ascii").strip())
                owner_alive = owner_pid == os.getpid()
                if not owner_alive:
                    try:
                        os.kill(owner_pid, 0)
                        owner_alive = True
                    except (OSError, OverflowError):
                        owner_alive = False
                if not owner_alive:
                    lock_path.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                pass
            if time.monotonic() >= wait_deadline:
                raise RuntimeError(
                    f"timed out waiting for rehabilitation cache preparation: {path}"
                )
            time.sleep(0.025)
            continue
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        finally:
            os.close(descriptor)
        owns_lock = True

    try:
        check_cancelled()
        if path.exists():
            return load_rehabilitation_cache(path, expected_fingerprint)
        for orphan in path.parent.glob(f"{path.name}.*.tmp"):
            orphan.unlink(missing_ok=True)
        stage = build_stage()
        check_cancelled()
        payload = rehabilitation_cache_payload(
            stage, config, expected_fingerprint, fingerprint_inputs
        )
        temporary = path.with_name(
            f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        check_cancelled()
        temporary.replace(path)
        temporary = None
        return load_rehabilitation_cache(path, expected_fingerprint)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if owns_lock:
            lock_path.unlink(missing_ok=True)


def saved_rehabilitation_plan(path, expected_fingerprint):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        payload.get("version") != 7
        or payload.get("input_fingerprint") != expected_fingerprint
    ):
        return None
    state = dict(payload.get("rehabilitation", {}))
    if state.get("status") not in (
        "complete", "no_eligible_filters", "not_applicable"
    ):
        return None
    try:
        return candidate_plan_from_json(state.get("best_plan", {})), state
    except (KeyError, TypeError, ValueError):
        return None

def run_rehabilitation_stage(
    *, mode, refs, score_plan, total_seconds, config=None,
    asymmetry_eligible=None, stop_requested=None,
):
    if mode != "peq":
        return {
            "status": "not_applicable",
            "evaluations": 0,
            "best_plan": rehab.CandidatePlan(),
            "result": None,
            "census": (),
        }
    refs = tuple(refs)
    if not refs:
        return {
            "status": "no_eligible_filters",
            "evaluations": 1,
            "best_plan": rehab.CandidatePlan(),
            "result": None,
            "census": (),
        }
    budget = rehabilitation_budget(total_seconds)
    cfg = config or rehab.RehabilitationConfig()
    cfg = replace(
        cfg,
        max_evaluations_per_slot=min(
            int(cfg.max_evaluations_per_slot), budget.max_evaluations
        ),
    )
    deadline = time.monotonic() + budget.seconds
    evaluations = 0
    stop_requested = stop_requested or (lambda: False)

    def check_cancelled():
        if stop_requested():
            raise RuntimeError("rehabilitation cache preparation cancelled")

    def counted_score_plan(plan):
        nonlocal evaluations
        check_cancelled()
        evaluations += 1
        return score_plan(plan)

    census = rehab.build_filter_census(
        refs,
        counted_score_plan,
        cfg,
        asymmetry_eligible=asymmetry_eligible,
        deadline=deadline,
    )
    operations = tuple(
        candidate
        for row in census
        for candidate in row.candidates
    )
    check_cancelled()

    result = rehab.rehabilitation_beam(
        rehab.CandidatePlan(),
        operations,
        counted_score_plan,
        beam_width=16,
        max_depth=4,
        deadline=deadline,
    )
    return {
        "status": "complete",
        "evaluations": int(evaluations),
        "best_plan": result.best.plan,
        "result": result,
        "census": census,
    }


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
    pair_states = {}
    for name, pair in opt.PAIR_DEFS.items():
        evidence = opt.interference_mask_evidence(
            freqs,
            traces[pair["left"]],
            traces[pair["right"]],
            traces.get(pair["together"]),
            synthetic=pair["together"] in opt.SYNTHETIC_PAIR_ROLES,
            band=pair["branch_band"],
        )
        pair_masks[name] = evidence["mask"]
        pair_states[name] = {
            "state": evidence["state"],
            "reason": evidence["reason"],
            "band_hz": list(pair["branch_band"]),
        }
    for group, cfg in opt.GROUPS.items():
        branch = cfg.get("branch")
        if cfg.get("system_transfer"):
            for pair_mask in pair_masks.values():
                masks[group] |= pair_mask
        elif branch in pair_masks:
            masks[group] |= pair_masks[branch]
    return masks, pair_states


def candidate_peaks(freqs, strength, desired_gain, lo, hi, q_range, gain_range, source, profile,
                     forced_targets=None):
    """forced_targets: optional {grid_index: (existing_F, existing_Q, existing_G)}
    - existing baseline bands on this group's own channels whose own
    frequency is always evaluated as a candidate edit/removal target, not
    only organically-detected local maxima of `strength`. Without this, the
    search could only ever append a new band into a free slot; it could
    never modify or retire one of its own earlier picks (DEFECT 1 - see
    CHANGELOG.md). A forced target that clears the same bars as any other
    candidate becomes a normal proposal; groups_to_band_sets()/
    _resolve_group_bands() then recognise it as an edit purely by frequency
    proximity to the existing band, and score/write it identically to an
    appended one."""
    strength = np.asarray(strength, dtype=float).copy()
    desired_gain = np.asarray(desired_gain, dtype=float)
    strength[(freqs < lo) | (freqs > hi)] = 0.0
    strength[~np.isfinite(strength)] = 0.0
    strength[np.abs(desired_gain) < 0.25] = 0.0
    strength = opt.erb_smooth(freqs, strength)
    forced_targets = forced_targets or {}

    thresh = 0.35 if profile == "explore" else 0.60
    idxs = []
    for i in range(1, len(freqs) - 1):
        if strength[i] < thresh:
            continue
        if strength[i] >= strength[i - 1] and strength[i] >= strength[i + 1]:
            idxs.append(i)
    idxs.sort(key=lambda i: -strength[i])

    def min_sep_for(i):
        strong = strength[i] >= max(2.5, thresh * 4.0)
        return 1 / 12 if strong else 1 / 5

    chosen = []
    # An existing band's own frequency is judged by whether it still carries
    # useful residual signal, not by whether it happens to be THE local
    # maximum of the residual-error surface - so it is checked against a
    # real (lower) floor rather than the strict peak-detection loop above.
    # It is also given first pick of the min-separation slot its region
    # would occupy: without this, a fresh, purely-organic point a fraction
    # of an octave away can register marginally higher smoothed strength
    # purely by chance and win that slot instead, so the search proposes a
    # redundant new band next to an existing one it could have edited
    # instead - defeating the point of DEFECT 1's fix (see CHANGELOG.md).
    forced_floor = max(0.05, thresh * 0.25)
    forced_order = sorted(
        (i for i, _t in forced_targets.items() if 0 <= i < len(freqs) and strength[i] >= forced_floor),
        key=lambda i: -strength[i],
    )
    for i in forced_order:
        if all(abs(math.log2(freqs[i] / freqs[j])) >= min_sep_for(i) for j in chosen):
            chosen.append(i)

    for i in idxs:
        if i in chosen:
            continue
        if all(abs(math.log2(freqs[i] / freqs[j])) >= min_sep_for(i) for j in chosen):
            chosen.append(i)
        if len(chosen) >= 18:
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
        target = forced_targets.get(i)
        band = opt.rounded_band(float(freqs[i]), q_hint, gain_hint)
        if band is None:
            if target is None:
                continue
            # The data-supported setting for this existing band is now
            # effectively "off" (too small to round to a real filter) -
            # propose retiring it instead of silently dropping the finding.
            existing_f, existing_q, _existing_g = target
            candidates.append({
                "F": float(existing_f),
                "Q": float(existing_q),
                "G": 0.0,
                "strength": float(strength[i]),
                "width_oct": float(width_oct),
                "branch_share": 0.0,
                "source": source + "_remove",
                "edit_target": tuple(float(v) for v in target),
            })
            continue
        rounded_f, rounded_q, rounded_gain = band
        entry = {
            "F": float(rounded_f),
            "Q": float(rounded_q),
            "G": float(rounded_gain),
            "strength": float(strength[i]),
            "width_oct": float(width_oct),
            "branch_share": 0.0,
            "source": source,
        }
        if target is not None:
            entry["source"] = source + "_edit"
            entry["edit_target"] = tuple(float(v) for v in target)
        candidates.append(entry)
    return candidates


def _group_existing_targets(cfg, freqs):
    """Existing baseline bands on this group's own channels, each mapped to
    its nearest frequency-grid index, for candidate_peaks' forced_targets -
    what lets the search find edit/removal opportunities on bands it (or a
    previous run) already placed, not only brand-new ones. Bands shared
    across a symmetric group's channels (e.g. front_voicing writes the same
    band to every front output) collapse to one target, since they are one
    logical filter, not several."""
    try:
        baseline = opt.baseline_band_sets()
    except Exception:
        # Existing-band evidence is an enhancement, not a requirement - fall
        # back to append-only candidate generation for this pass rather than
        # failing the whole search over it.
        return {}
    log_freqs = np.log10(freqs)
    targets = {}
    for channel in cfg["channels"]:
        if channel >= len(baseline):
            continue
        for f, q, g in baseline[channel]:
            index = int(np.argmin(np.abs(log_freqs - math.log10(float(f)))))
            targets.setdefault(index, (float(f), float(q), float(g)))
    return targets


def find_guided_candidates(freqs, traces, target, profile: str, persistence_sessions=None):
    """Find data-derived candidate PEQ centers before random search.

    Candidate centers come from two math-derived needs:
      - tonal target error in the predicted system sum, with stronger presence
        and peak weighting;
      - L/R solo imbalance for the per-side front groups.
    Destructive-summing zones from the together-vs-solo audit are masked from
    tonal candidate generation so PEQ is not asked to fix phase.

    DEFECT 6: when `persistence_sessions` is supplied (extra REW folders'
    System Sum traces, from opt.load_persistence_sessions), a tonal
    candidate's target deviation must hold sign and clear the noise floor in
    every one of them too, not just this primary session - a single MMM
    session cannot tell a real deviation from run-to-run capture noise.
    Deliberately opt-in: with no extra sessions this is a complete no-op, so
    every existing single-session run behaves exactly as before.

    Each session is re-anchored to `target` with its own broadband median
    offset before comparison (same convention `target` itself was anchored
    to this primary session's own level with, in the caller). Two REW
    sessions captured on different days/mic positions rarely share the same
    absolute source volume or mic gain; without this, a broadband level
    difference between sessions could flip a real deviation's sign or hide
    it below the noise floor for reasons that have nothing to do with
    whether the *shape* of the deviation actually repeats.
    """
    raw_system_dev = traces["System Sum"] - target
    system_dev = opt.erb_smooth(freqs, raw_system_dev)
    session_devs = []
    for session in (persistence_sessions or []):
        session_system_sum = session["system_sum"]
        anchor = opt.target_anchor_offset(freqs, session_system_sum, target)
        session_devs.append(opt.erb_smooth(freqs, session_system_sum - anchor - target))
    global LAST_PROPOSAL_AUDIT
    suppressions = []
    masks, pair_states = interference_masks(freqs, traces)
    audible = opt.audibility_weight(freqs)
    vocal = np.ones_like(freqs)
    vocal[(freqs >= 200.0) & (freqs <= 6000.0)] = 1.8
    peak_mult = np.where(system_dev > 0.0, 2.0, 0.75)
    balance_w = audible * opt.imaging_balance_weight(freqs)
    balance_w[(freqs >= 700.0) & (freqs <= 5000.0)] *= 1.8
    pools = {}
    for group, cfg in opt.GROUPS.items():
        lo, hi = cfg["range"]
        q_range = cfg["q_range"]
        gain_range = cfg["gain_range"]
        contribution = branch_contribution(freqs, traces, group)
        # Near a crossover, keep even a weak branch in the comparison so the
        # scorer can explicitly choose tweeters, mids/midbass, or both.  A
        # weak branch still pays the authoritative driver-share penalty.
        active_limit_db = -24.0 if cfg.get("crossover_scope") else -6.0
        active_driver = (
            np.ones_like(contribution, dtype=bool)
            if cfg.get("crossover_scope")
            else contribution >= (10 ** (active_limit_db / 10.0))
        )
        candidate_contribution = (
            np.maximum(contribution, 0.25) if cfg.get("crossover_scope") else contribution
        )
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
            tonal_strength = np.abs(system_dev) * candidate_contribution * audible * vocal * peak_mult
            tonal_gain = -0.65 * system_dev / np.maximum(contribution, 0.35)
            tonal_source = "tonal"
        tonal_strength[masks.get(group, False)] = 0.0
        tonal_strength[~active_driver] = 0.0
        existing_targets = _group_existing_targets(cfg, freqs)
        tonal_candidates = candidate_peaks(
            freqs, tonal_strength, tonal_gain, lo, hi, q_range, gain_range, tonal_source, profile,
            forced_targets=existing_targets,
        )
        guarded_tonal = []
        for candidate in tonal_candidates:
            center = float(candidate["F"])
            if (
                cfg.get("pair") and cfg.get("side")
                and pair_states[cfg["pair"]]["state"] == opt.MASK_UNKNOWN
            ):
                suppressions.append({
                    "group": group,
                    "frequency_hz": center,
                    "reason": "interference_evidence_unknown",
                })
                continue
            branch = str(cfg.get("branch", "low"))
            noise_branch = "high" if branch == "front" and center >= 1800.0 else branch
            deviation_curve = shape_dev if cfg.get("system_transfer") else system_dev
            target_deviation = float(np.interp(
                np.log10(center),
                np.log10(freqs),
                raw_system_dev if cfg.get("crossover_scope") else deviation_curve,
            ))
            floor = float(opt.measurement_noise_floor_db([center], noise_branch)[0])
            required = float(opt.MEASUREMENT_NOISE_MULTIPLIER * floor)
            if abs(target_deviation) < required:
                suppressions.append({
                    "group": group,
                    "frequency_hz": center,
                    "reason": "below_measurement_noise_floor",
                    "deviation_db": target_deviation,
                    "required_deviation_db": required,
                })
                continue
            if session_devs:
                session_values = [
                    float(np.interp(np.log10(center), np.log10(freqs), dev))
                    for dev in session_devs
                ]
                persistence = opt.cross_session_persistence(
                    [target_deviation] + session_values, floor,
                )
                if not persistence["eligible"]:
                    suppressions.append({
                        "group": group,
                        "frequency_hz": center,
                        "reason": "cross_session_" + persistence["reason"],
                        "session_count": persistence["session_count"],
                        "deviation_db": target_deviation,
                    })
                    continue
                candidate["persistence_session_count"] = persistence["session_count"]
            if cfg.get("pair") and cfg.get("side") and candidate["G"] < 0.0:
                pair = opt.PAIR_DEFS[cfg["pair"]]
                diff = opt.erb_smooth(
                    freqs, traces[pair["left"]] - traces[pair["right"]]
                )
                evidence = opt.signed_offset_evidence(
                    freqs, diff, center, cfg.get("branch", "low")
                )
                imaging_weight = float(opt.imaging_balance_weight([center])[0])
                hot_side_matches = (
                    (cfg["side"] == "left" and evidence["offset_db"] > 0.0)
                    or (cfg["side"] == "right" and evidence["offset_db"] < 0.0)
                )
                absolute_system_deviation = float(np.interp(
                    np.log10(center), np.log10(freqs), system_dev
                ))
                if (
                    not evidence["eligible"]
                    or imaging_weight < 0.5
                    or absolute_system_deviation < -0.5
                    or not hot_side_matches
                ):
                    reason = (
                        evidence["reason"] if not evidence["eligible"]
                        else "imaging_frequency_outside_authority" if imaging_weight < 0.5
                        else "summed_response_already_below_target"
                        if absolute_system_deviation < -0.5
                        else "cut_not_on_systematically_hotter_side"
                    )
                    suppressions.append({
                        "group": group,
                        "frequency_hz": center,
                        "reason": reason,
                        "system_deviation_db": absolute_system_deviation,
                    })
                    continue
            # Repo-review finding: scale the proposed gain by continuous
            # evidence-based confidence, on top of (never instead of) the
            # hard/soft gates already checked above. Boosting needs much
            # stronger evidence than cutting, so it scales with the SQUARE
            # of boost confidence while a cut only scales with its square
            # root - a candidate with mediocre evidence keeps most of its
            # proposed cut but loses most of its proposed boost. A null bin
            # is already excluded upstream (tonal_strength zeroed there);
            # this is a further, continuous refinement on what's left, not
            # a replacement for that hard exclusion.
            null_at_center = bool(masks.get(group, np.zeros_like(freqs, dtype=bool))[
                int(np.argmin(np.abs(np.log10(freqs) - np.log10(center))))
            ])
            authority_at_center = float(np.interp(np.log10(center), np.log10(freqs), contribution))
            session_agreement = session_required = None
            if session_devs:
                session_agreement = min(abs(target_deviation), *(abs(v) for v in session_values))
                session_required = required
            confidence = opt.correction_confidence(
                [center], null_fraction=1.0 if null_at_center else 0.0,
                driver_authority=authority_at_center,
                session_agreement_db=session_agreement, session_required_db=session_required,
            )
            conf_boost = float(confidence["boost"][0])
            conf_cut = float(confidence["cut"][0])
            scale = conf_boost ** 2 if candidate["G"] > 0.0 else math.sqrt(max(conf_cut, 0.0))
            scaled_gain = candidate["G"] * scale
            if abs(scaled_gain) < 0.3:
                suppressions.append({
                    "group": group,
                    "frequency_hz": center,
                    "reason": "low_correction_confidence",
                    "confidence_boost": round(conf_boost, 3),
                    "confidence_cut": round(conf_cut, 3),
                })
                continue
            candidate["G"] = scaled_gain
            candidate.update({
                "target_deviation_db": target_deviation,
                "noise_floor_db": floor,
                "required_deviation_db": required,
                "confidence_boost": round(conf_boost, 3),
                "confidence_cut": round(conf_cut, 3),
            })
            guarded_tonal.append(candidate)
        candidates.extend(guarded_tonal)

        if cfg.get("pair") and cfg.get("side"):
            pair = opt.PAIR_DEFS[cfg["pair"]]
            if pair_states[cfg["pair"]]["state"] == opt.MASK_UNKNOWN:
                suppressions.append({
                    "group": group,
                    "frequency_hz": None,
                    "band_hz": list(pair["branch_band"]),
                    "reason": "interference_evidence_unknown",
                })
                pools[group] = candidates
                continue
            diff = opt.erb_smooth(freqs, traces[pair["left"]] - traces[pair["right"]])
            if cfg["side"] == "left":
                bal_gain = -0.85 * diff
            else:
                bal_gain = 0.85 * diff
            bal_strength = np.abs(bal_gain) * balance_w
            bal_strength[~active_driver] = 0.0
            blo, bhi = pair["balance_band"]
            bal_strength[(freqs < blo) | (freqs > bhi)] = 0.0
            balance_candidates = candidate_peaks(
                freqs, bal_strength, bal_gain, lo, hi, q_range, gain_range, "balance", profile
            )
            guarded_balance = []
            for candidate in balance_candidates:
                evidence = opt.signed_offset_evidence(
                    freqs, diff, candidate["F"], cfg.get("branch", "low")
                )
                system_at_center = float(np.interp(
                    np.log10(candidate["F"]), np.log10(freqs), system_dev
                ))
                if not evidence["eligible"]:
                    suppressions.append({
                        "group": group,
                        "frequency_hz": float(candidate["F"]),
                        "reason": evidence["reason"],
                        "lr_sign_changes": int(evidence["sign_changes"]),
                    })
                    continue
                if candidate["G"] < 0.0 and system_at_center < -0.5:
                    suppressions.append({
                        "group": group,
                        "frequency_hz": float(candidate["F"]),
                        "reason": "summed_response_already_below_target",
                        "system_deviation_db": system_at_center,
                    })
                    continue
                candidate.update({
                    "lr_offset_db": float(evidence["offset_db"]),
                    "lr_sign_changes": int(evidence["sign_changes"]),
                    "noise_floor_db": float(evidence["noise_floor_db"]),
                    "required_deviation_db": float(evidence["required_deviation_db"]),
                    "system_deviation_db": system_at_center,
                })
                guarded_balance.append(candidate)
            candidates.extend(guarded_balance)

        candidates.sort(key=lambda c: -c["strength"])
        deduped = []
        for c in candidates:
            separation = 1 / 16 if c["strength"] >= 2.5 else 1 / 8
            if all(abs(math.log2(c["F"] / d["F"])) >= separation or c["source"] != d["source"] for d in deduped):
                c["branch_share"] = float(np.interp(np.log10(c["F"]), np.log10(freqs), contribution))
                if (
                    not cfg.get("crossover_scope")
                    and c["branch_share"] < 10 ** (active_limit_db / 10.0)
                ):
                    continue
                deduped.append(c)
            recoverable = sum(max(0.0, float(item["strength"])) for item in deduped)
            adaptive_cap = int(np.clip(6 + round(recoverable / 2.0), 8, 24))
            if len(deduped) >= adaptive_cap:
                break
        pools[group] = deduped

    # Use one common crossover centre to make the scope comparison real:
    # tweeters only, mids/midbass only, and the whole front stage.  Alternate
    # scopes remain candidates even when their branch-specific noise evidence
    # is weak; the authoritative scorer then rejects them rather than silently
    # skipping the requested comparison.
    high_scope = pools.get("high_crossover_sym", [])
    crossover_seeds = [
        candidate for candidate in high_scope
        if 1800.0 <= candidate["F"] <= 3500.0 and candidate["G"] < 0.0
    ]
    if crossover_seeds:
        seed = max(crossover_seeds, key=lambda item: item["strength"])
        for group in ("front_voicing", "mid_crossover_sym", "low_crossover_sym"):
            if group not in pools:
                continue
            cfg = opt.GROUPS[group]
            branch = str(cfg.get("branch", "low"))
            noise_branch = "high" if branch == "front" else branch
            qlo, qhi = cfg["q_range"]
            glo, ghi = cfg["gain_range"]
            band = opt.rounded_band(
                seed["F"],
                float(np.clip(seed["Q"], qlo, qhi)),
                float(np.clip(seed["G"], glo, ghi)),
            )
            if band is None:
                continue
            center, q_value, gain = band
            if any(abs(math.log2(center / item["F"])) < 1 / 48 for item in pools[group]):
                continue
            contribution = branch_contribution(freqs, traces, group)
            floor = float(opt.measurement_noise_floor_db([center], noise_branch)[0])
            pools[group].append({
                "F": float(center),
                "Q": float(q_value),
                "G": float(gain),
                "strength": float(seed["strength"]),
                "width_oct": float(seed["width_oct"]),
                "branch_share": float(np.interp(
                    np.log10(center), np.log10(freqs), contribution
                )),
                "source": "crossover_scope",
                "target_deviation_db": float(np.interp(
                    np.log10(center), np.log10(freqs), raw_system_dev
                )),
                "noise_floor_db": floor,
                "required_deviation_db": float(opt.MEASUREMENT_NOISE_MULTIPLIER * floor),
            })
            pools[group].sort(key=lambda item: -item["strength"])
    worth_fixing = sorted(
        (
            {
                "group": group,
                "frequency_hz": float(item["F"]),
                "gain_db": float(item["G"]),
                "source": str(item["source"]),
                "recoverable_error": float(item["strength"]),
                # DEFECT 6: how many sessions (this one plus any supplied via
                # --persistence-sessions) actually back this band, so the
                # report can name the evidence per band rather than assert
                # it. Absent when no extra sessions were supplied.
                **(
                    {"persistence_session_count": item["persistence_session_count"]}
                    if "persistence_session_count" in item else {}
                ),
            }
            for group, items in pools.items() for item in items
        ),
        key=lambda item: -item["recoverable_error"],
    )[:10]
    skipped = sorted(
        suppressions,
        key=lambda item: -abs(float(item.get("deviation_db", item.get("system_deviation_db", 0.0)) or 0.0)),
    )[:10]
    LAST_PROPOSAL_AUDIT = {
        "mask_evidence": pair_states,
        "blocking_pairs": [
            name for name, item in pair_states.items() if item["state"] == opt.MASK_UNKNOWN
        ],
        "suppressions": suppressions,
        "problem_census": {
            "worth_fixing": worth_fixing,
            "deliberately_skipped": skipped,
        },
    }
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


def search_budgets(pools, pool_limit: int, beam_width: int) -> dict[str, dict[str, int]]:
    recoverable = {
        group: sum(max(0.0, float(item.get("strength", 0.0))) for item in items)
        for group, items in pools.items()
    }
    total = sum(recoverable.values())
    active = max(1, sum(value > 0.0 for value in recoverable.values()))
    budgets = {}
    for group in opt.GROUPS:
        share = recoverable.get(group, 0.0) / total if total > 0.0 else 0.0
        budgets[group] = {
            "pool_limit": int(np.clip(round(2 + pool_limit * active * share), 2, max(2, pool_limit * 2))),
            "beam_width": int(np.clip(round(4 + beam_width * active * share), 4, max(4, beam_width * 2))),
        }
    return budgets


def deterministic_beam_combinations(
    pools,
    component_score=None,
    beam_width: int = 24,
    pool_limit: int | dict[str, int] = 6,
    beam_budgets: dict[str, dict[str, int]] | None = None,
    deadline: float | None = None,
    order_seed: int | None = None,
    stop_requested=None,
    *,
    score_plan=None,
    seed_plans=None,
):
    """Build exact guided combinations without losing baseline slot edits."""
    if score_plan is None:
        if component_score is None:
            raise TypeError("component_score or score_plan is required")

        def score_plan(plan):
            return component_score(rehab.thaw_groups(plan.groups))

    seeds = tuple(seed_plans or (rehab.CandidatePlan(),))
    beam = []
    score_cache = {}
    evaluations = 0

    def evaluate(plan):
        nonlocal evaluations
        signature = rehab.candidate_plan_signature(plan)
        components = score_cache.get(signature)
        if components is None:
            components = dict(score_plan(plan))
            score_cache[signature] = components
            evaluations += 1
        return BeamEntry(
            float(components["objective"]), signature, plan, dict(components)
        )

    for plan in seeds:
        entry = evaluate(plan)
        if all(entry.signature != existing.signature for existing in beam):
            beam.append(entry)
    beam.sort(key=lambda entry: (entry.objective, entry.signature))
    beam = beam[:max(1, int(beam_width))]

    group_names = list(opt.GROUPS)
    if order_seed is not None and len(group_names) > 1:
        order_rng = np.random.default_rng(int(order_seed))
        group_names = [
            group_names[index]
            for index in order_rng.permutation(len(group_names))
        ]
    for group in group_names:
        cfg = opt.GROUPS[group]
        group_pool_limit = (
            int(pool_limit.get(group, 0))
            if isinstance(pool_limit, dict)
            else int(
                (beam_budgets or {}).get(group, {}).get(
                    "pool_limit", pool_limit
                )
            )
        )
        group_beam_width = int(
            (beam_budgets or {}).get(group, {}).get(
                "beam_width", beam_width
            )
        )
        candidates = sorted(
            pools.get(group, []), key=lambda item: -item["strength"]
        )[:max(0, group_pool_limit)]
        bands = [
            opt.rounded_band(item["F"], item["Q"], item["G"])
            for item in candidates
        ]
        bands = [band for band in bands if band is not None]
        options = [()]
        max_active = min(int(cfg["max_bands"]), len(bands), 2)
        for count in range(1, max_active + 1):
            options.extend(itertools.combinations(bands, count))

        expanded = []
        for partial in beam:
            for option in options:
                if (
                    stop_requested is not None and stop_requested()
                ) or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    candidates_now = beam + expanded
                    unique = {
                        entry.signature: entry
                        for entry in sorted(
                            candidates_now,
                            key=lambda item: (
                                item.objective, item.signature
                            ),
                        )
                    }
                    return list(unique.values())[
                        :max(1, int(beam_width))
                    ], evaluations
                groups = partial.groups
                groups[group] = sorted(option, key=lambda band: band[0])
                plan = rehab.CandidatePlan(
                    slot_edits=partial.plan.slot_edits,
                    groups=rehab.freeze_groups(groups),
                )
                expanded.append(evaluate(plan))

        unique = {}
        for entry in sorted(
            expanded,
            key=lambda item: (item.objective, item.signature),
        ):
            unique.setdefault(entry.signature, entry)
        beam = list(unique.values())[:max(1, group_beam_width)]
    return beam, evaluations

def guided_continuation_plans(groups, lineages):
    frozen_groups = rehab.freeze_groups(groups)
    plans = []
    seen = set()
    for lineage in lineages:
        plan = rehab.CandidatePlan(
            slot_edits=lineage.slot_edits, groups=frozen_groups
        )
        signature = rehab.candidate_plan_signature(plan)
        if signature not in seen:
            seen.add(signature)
            plans.append(plan)
    return tuple(plans)

def beam_uses_timed_guided_continuation(proposal: str, mode: str) -> bool:
    """Whether a normal PEQ Beam pass should use the remaining run budget.

    Beam exhausts a finite, deterministic set of candidate combinations quickly.
    PEQ runs therefore continue from that seeded pass with small, data-guided
    variations until their requested deadline.  The phase diagnostic stays a
    deliberately bounded baseline-only pass.
    """
    return proposal == "beam" and mode == "peq"


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


def refine_entries(entries, score_plan, top: int, passes: int):
    refined = []
    improved = 0
    evaluations = 0
    best_before = float(entries[0][0]) if entries else None
    for entry in entries[:max(0, int(top))]:
        value, _signature, groups = entry
        base_plan = (
            entry.plan
            if isinstance(entry, BeamEntry)
            else rehab.CandidatePlan(groups=rehab.freeze_groups(groups))
        )

        def score_groups(candidate_groups):
            plan = rehab.CandidatePlan(
                slot_edits=base_plan.slot_edits,
                groups=rehab.freeze_groups(candidate_groups),
            )
            return score_plan(plan)

        new_groups, components, used = coordinate_refine(
            groups, score_groups, passes=passes
        )
        evaluations += used
        new_value = float(components["objective"])
        if new_value + 1e-12 < float(value):
            improved += 1
        new_plan = rehab.CandidatePlan(
            slot_edits=base_plan.slot_edits,
            groups=rehab.freeze_groups(new_groups),
        )
        refined.append(BeamEntry(
            new_value,
            rehab.candidate_plan_signature(new_plan),
            new_plan,
            dict(components),
        ))
    best_after = min(
        (float(item[0]) for item in refined), default=best_before
    )
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


def build_rows(
    freqs,
    traces,
    target,
    best,
    component_score=None,
    *,
    score_plan=None,
):
    if component_score is None:
        component_score = opt.make_component_scorer(freqs, traces, target)
    rows = []
    for rank, entry in enumerate(best, start=1):
        value, signature, groups = entry
        plan = (
            entry.plan
            if isinstance(entry, BeamEntry)
            else rehab.CandidatePlan(groups=rehab.freeze_groups(groups))
        )
        pred = opt.predict_candidate_plan(freqs, traces, plan)
        score = opt.tune_scorecard(freqs, pred, target)
        components = (
            dict(entry.components)
            if isinstance(entry, BeamEntry)
            else dict(
                score_plan(plan)
                if score_plan is not None
                else component_score(groups)
            )
        )
        rows.append({
            "rank": rank,
            "file": f"candidate_{rank:02d}_objective_{value:.4f}.afpx",
            "objective": float(value),
            "score": score,
            "components": components,
            "groups": groups,
            "plan": plan,
            "signature": signature,
            "lint": None,
            "headroom": opt.candidate_plan_headroom(freqs, plan),
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


def replace_checkpoint_with_retry(source: Path, destination: Path, attempts: int = 16) -> None:
    """Atomically replace a checkpoint despite brief Windows reader locks."""
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.025 * (2 ** attempt), 0.4))


def save_state(path: Path, best, rng: np.random.Generator, completed_trials: int,
               elapsed_seconds: float, args: argparse.Namespace, archive=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = archive or []
    payload = {
        "schema": "audiofischer-stream-state-v2",
        "version": 7,
        "objective": "spatial_objective_candidate_plan_v7",
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
        "best": [beam_entry_to_json(entry) for entry in best],
        "archive": [beam_entry_to_json(entry) for entry in archive],
        "rehabilitation": _json_safe(dict(getattr(args, "rehabilitation", {}) or {})),
        "input_fingerprint": getattr(args, "input_fingerprint", None),
        "proposal_audit": getattr(args, "proposal_audit", {}),
        "convergence": getattr(args, "convergence", {}),
    }
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        replace_checkpoint_with_retry(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def load_state(
    path: Path,
    rng: np.random.Generator,
    component_score=None,
    archive_size: int = 0,
    *,
    score_plan=None,
    expected_fingerprint=None,
):
    if not path.exists():
        return [], [], {}, 0, 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in (4, 5, 6, 7):
        return [], [], {}, 0, 0.0
    if (
        expected_fingerprint is not None
        and payload.get("input_fingerprint") != expected_fingerprint
    ):
        return [], [], {}, 0, 0.0
    if "rng_state" in payload:
        rng.bit_generator.state = payload["rng_state"]

    def load_entry(item):
        return beam_entry_from_json(
            item,
            score_plan=score_plan,
            component_score=component_score,
        )

    best = [load_entry(item) for item in payload.get("best", [])]
    best.sort(key=lambda entry: (entry.objective, entry.signature))
    archive = [load_entry(item) for item in payload.get("archive", [])]
    score_map = {}
    if score_plan is not None or component_score is not None:
        score_map = {
            entry.signature: dict(entry.components)
            for entry in archive
        }
        archive, score_map = prune_archive(
            archive, score_map, archive_size
        )
    else:
        archive = []
    return (
        best,
        archive,
        score_map,
        int(payload.get("completed_trials", 0)),
        float(payload.get("elapsed_seconds", 0.0)),
    )

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
            plan = (
                entry.plan
                if isinstance(entry, BeamEntry)
                else rehab.CandidatePlan(
                    groups=rehab.freeze_groups(entry[2])
                )
            )
            slot_conflicts = [
                item
                for item in opt.candidate_plan_phase_conflicts(
                    freqs, plan, phase_plan
                )
                if item.get("group") == "existing_slot"
            ]
            if slot_conflicts:
                args.phase_peq_rejections.extend(slot_conflicts)
                continue
            if phase_valid and phase_plan:
                verification = opt.complex_crossover_verification(
                    freqs, rich_traces, entry[2], phase_plan
                )
                if verification["pass"]:
                    kept.append(entry)
            else:
                conflicts = opt.phase_peq_conflicts(
                    freqs, entry[2], phase_plan
                )
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
    score_plan = opt.make_candidate_plan_component_scorer(
        opt.make_band_set_component_scorer(
            freqs,
            traces,
            target,
            args.filter_cost_scale,
            args.worst_weight,
        ),
        freqs,
        rich_traces,
        phase_plan,
        phase_valid,
    )
    rows = build_rows(
        freqs, traces, target, best, component_score,
        score_plan=score_plan,
    )
    family_rows = (
        build_rows(
            freqs, traces, target, family_entries, component_score,
            score_plan=score_plan,
        )
        if family_entries else rows
    )
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
        row["lint"] = opt.write_candidate_plan(base_xml, path, row["plan"], phase_plan=phase_plan)
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
    parser.add_argument("--rehabilitation-cache", type=Path, default=None,
                        help="Shared fingerprinted existing-PEQ rehabilitation cache.")
    parser.add_argument("--level-calibration", type=Path, default=None,
                        help="JSON role/file -> dB offsets for mixed-level measurement sessions.")
    parser.add_argument("--repeatability-folder", type=Path, default=None,
                        help="Second same-day session used to derive the measurement floor.")
    parser.add_argument("--persistence-sessions", type=Path, nargs="+", default=None,
                        help="Extra REW session folders (each needs a System Sum export). A tonal "
                             "candidate is only proposed if its deviation holds sign and clears the "
                             "noise floor in this primary session AND every one of these. Opt-in: "
                             "omitted, the search behaves exactly as a single-session run.")
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
    args.measurement_noise_guard = opt.configure_repeatability_floor(
        args.repeatability_folder, level_calibration
    )
    args.loaded_level_calibration = dict(level_calibration or {})
    opt.sync_external_objective(args.baseline, args.target, level_calibration)
    configure_profile(args.profile)
    rng = np.random.default_rng(args.seed)
    freqs, traces, rich_traces = opt.load_measurements(level_calibration)
    raw_target = opt.load_target(args.target, freqs)
    target = raw_target + opt.target_anchor_offset(freqs, traces["System Sum"], raw_target)
    base_xml = opt.decode_afpx(args.baseline)
    args.validation = opt.pair_sum_validation(freqs, traces, threshold=args.validation_threshold)
    failed_validation = [item for item in args.validation if item.get("pass") is False]
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
        persistence_sessions = opt.load_persistence_sessions(
            args.persistence_sessions or [], freqs,
        )
        guided_pools = find_guided_candidates(
            freqs, traces, target, args.profile, persistence_sessions=persistence_sessions,
        )
    args.proposal_audit = dict(LAST_PROPOSAL_AUDIT)
    (args.out / "problem_census.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "problem_census.json").write_text(
        json.dumps(args.proposal_audit, indent=2), encoding="utf-8"
    )
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

    score_plan = opt.make_candidate_plan_component_scorer(
        opt.make_band_set_component_scorer(
            freqs,
            traces,
            target,
            args.filter_cost_scale,
            args.worst_weight,
        ),
        freqs,
        rich_traces,
        args.phase_plan,
        bool(args.measurement_session["audit"].get("phase_valid")),
    )
    channel_roles = dict(opt.CH_TRACE)
    channel_roles.update({6: "Left Sub", 7: "Right Sub"})
    rehabilitation_config = opt.rehabilitation_config(
        channel_roles, explore=args.profile == "explore"
    )
    args.input_fingerprint = stream_input_fingerprint(
        args, rehabilitation_config
    )
    state_path = args.out / "stream_state.json"
    start = time.monotonic()
    if args.resume:
        best, archive, archive_scores, completed_before, elapsed_before = load_state(
            state_path,
            rng,
            archive_size=args.archive_size,
            score_plan=score_plan,
            expected_fingerprint=args.input_fingerprint,
        )
    else:
        best, archive, archive_scores, completed_before, elapsed_before = (
            [], [], {}, 0, 0.0
        )

    if args.mode == "phase" and args.rehabilitation_cache is not None:
        raise SystemExit(
            "Phase workers must not receive a PEQ rehabilitation cache."
        )
    cached_rehabilitation = None
    if args.rehabilitation_cache is not None:
        try:
            cached_rehabilitation = load_rehabilitation_cache(
                args.rehabilitation_cache, args.input_fingerprint
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    saved_rehabilitation = (
        saved_rehabilitation_plan(state_path, args.input_fingerprint)
        if args.resume else None
    )
    if cached_rehabilitation is not None:
        rehabilitation_plan = cached_rehabilitation["best_plan"]
        args.rehabilitation = compact_rehabilitation_cache_state(
            cached_rehabilitation, args.rehabilitation_cache
        )
        cached_rehabilitation = None
        rehabilitation_trials = 0
        rehabilitation_status = "cached"
    elif saved_rehabilitation is not None:
        rehabilitation_plan, args.rehabilitation = saved_rehabilitation
        rehabilitation_trials = 0
        rehabilitation_status = "resumed"
    else:
        refs = rehab.active_peq_slot_refs(base_xml, channel_roles)
        stage = run_rehabilitation_stage(
            mode=args.mode,
            refs=refs,
            score_plan=score_plan,
            total_seconds=args.seconds,
            config=rehabilitation_config,
        )
        rehabilitation_plan = stage["best_plan"]
        args.rehabilitation = rehabilitation_state_payload(
            stage, rehabilitation_config
        )
        rehabilitation_trials = int(stage["evaluations"])
        rehabilitation_status = str(stage["status"])
    baseline_plan = rehab.CandidatePlan()
    baseline_components = dict(score_plan(baseline_plan))
    baseline_item = BeamEntry(
        float(baseline_components["objective"]),
        rehab.candidate_plan_signature(baseline_plan),
        baseline_plan,
        baseline_components,
    )
    best = insert_best(best, baseline_item, args.keep)
    archive, archive_scores = insert_archive(
        archive,
        archive_scores,
        baseline_item,
        baseline_components,
        args.archive_size,
    )
    rehabilitation_components = dict(score_plan(rehabilitation_plan))
    args.rehabilitation_plan = rehabilitation_plan
    args.rehabilitation_components = rehabilitation_components
    if args.rehabilitation:
        baseline_candidate = rehab.ScoredCandidate(
            baseline_plan, dict(score_plan(baseline_plan))
        )
        rehabilitation_candidate = rehab.ScoredCandidate(
            rehabilitation_plan, rehabilitation_components
        )
        args.rehabilitation["baseline_components"] = dict(
            baseline_candidate.components
        )
        args.rehabilitation["best_components"] = dict(
            rehabilitation_candidate.components
        )
        args.rehabilitation["meaningful_improvement"] = rehab.meaningfully_better(
            rehabilitation_candidate, baseline_candidate
        )
    rehabilitation_item = BeamEntry(
        float(rehabilitation_components["objective"]),
        rehab.candidate_plan_signature(rehabilitation_plan),
        rehabilitation_plan,
        rehabilitation_components,
    )
    best = insert_best(best, rehabilitation_item, args.keep)
    archive, archive_scores = insert_archive(
        archive,
        archive_scores,
        rehabilitation_item,
        rehabilitation_components,
        args.archive_size,
    )

    next_checkpoint = start + max(10, args.checkpoint_seconds)
    trials = rehabilitation_trials
    args.beam = None
    baseline_value = float(baseline_components["objective"])
    convergence_events = [{
        "elapsed_seconds": 0.0,
        "objective": baseline_value,
        "phase": "baseline",
    }, {
        "elapsed_seconds": float(time.monotonic() - start),
        "objective": float(rehabilitation_components["objective"]),
        "phase": "baseline_rehabilitation_complete",
        "status": rehabilitation_status,
    }]
    best_value_seen = min(
        (item[0] for item in best), default=baseline_value
    )
    last_improvement_time = start
    if args.proposal == "beam":
        beam_order_seed = args.seed + completed_before
        adaptive_budgets = search_budgets(
            guided_pools, args.beam_pool_limit, args.beam_width
        )
        beam_entries, beam_evaluations = deterministic_beam_combinations(
            guided_pools,
            score_plan=score_plan,
            seed_plans=(baseline_plan, rehabilitation_plan),
            beam_width=args.beam_width,
            pool_limit=args.beam_pool_limit,
            beam_budgets=adaptive_budgets,
            deadline=(start + args.seconds) if args.seconds else None,
            order_seed=beam_order_seed,
            stop_requested=stop_requested,
        )
        for item in beam_entries:
            components = item.components
            best = insert_best(best, item, args.keep)
            archive, archive_scores = insert_archive(
                archive, archive_scores, item, components, args.archive_size
            )
        trials += beam_evaluations
        beam_best = min((item[0] for item in beam_entries), default=best_value_seen)
        if beam_best < best_value_seen:
            best_value_seen = beam_best
            last_improvement_time = time.monotonic()
        convergence_events.append({
            "elapsed_seconds": float(time.monotonic() - start),
            "objective": float(best_value_seen),
            "phase": "deterministic_beam_complete",
        })
        args.beam = {
            "width": args.beam_width,
            "pool_limit": args.beam_pool_limit,
            "evaluations": beam_evaluations,
            "retained": len(beam_entries),
            "order_seed": beam_order_seed,
            "adaptive_group_budgets": adaptive_budgets,
            "continuation": (
                "guided_until_deadline"
                if beam_uses_timed_guided_continuation(args.proposal, args.mode)
                else "none"
            ),
            "transition": (
                "deterministic_to_guided"
                if beam_uses_timed_guided_continuation(args.proposal, args.mode)
                else "deterministic_complete"
            ),
        }
    while True:
        now = time.monotonic()
        if stop_requested():
            break
        if args.seconds and now - start >= args.seconds:
            break
        if args.max_trials and trials >= args.max_trials:
            break
        if args.proposal == "beam" and not beam_uses_timed_guided_continuation(
            args.proposal, args.mode
        ):
            break
        if args.proposal == "beam" and not any(guided_pools.values()):
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
        continuation_lineages = (
            (baseline_plan, rehabilitation_plan)
            if args.proposal == "beam"
            else (rehabilitation_plan,)
        )
        cma_value = None
        for plan in guided_continuation_plans(
            groups, continuation_lineages
        ):
            now = time.monotonic()
            if stop_requested():
                break
            if args.seconds and now - start >= args.seconds:
                break
            if args.max_trials and trials >= args.max_trials:
                break
            components = dict(score_plan(plan))
            if components.get("phase_peq_conflict_count", 0.0) > 0.0:
                args.phase_peq_rejections.extend(
                    opt.candidate_plan_phase_conflicts(
                        freqs, plan, args.phase_plan
                    )
                )
            value = float(components["objective"])
            if (
                args.max_positive_gain_penalty > 0
                and components["positive_gain_penalty_db"]
                > args.max_positive_gain_penalty
            ):
                value = 1e6 + float(
                    components["positive_gain_penalty_db"]
                )
                components["objective"] = value
            signature = rehab.candidate_plan_signature(plan)
            item = BeamEntry(value, signature, plan, components)
            if value + 1e-12 < best_value_seen:
                best_value_seen = value
                last_improvement_time = now
                convergence_events.append({
                    "elapsed_seconds": float(now - start),
                    "objective": float(value),
                    "phase": "guided_improvement",
                })
            best = insert_best(best, item, args.keep)
            archive, archive_scores = insert_archive(
                archive, archive_scores, item, components,
                args.archive_size,
            )
            cma_value = (
                value if cma_value is None else min(cma_value, value)
            )
            trials += 1
        if cma_x is not None and cma_value is not None:
            cma_proposal.tell(cma_x, cma_value)

        if best and args.checkpoint_seconds and now >= next_checkpoint:
            args._completed_trials = completed_before + trials
            args._elapsed_seconds = elapsed_before + (now - start)
            stalled_seconds_now = max(0.0, now - last_improvement_time)
            args.convergence = {
                "events": list(convergence_events),
                "last_improvement_elapsed_seconds": float(
                    max(0.0, last_improvement_time - start)
                ),
                "stalled_seconds": float(stalled_seconds_now),
                "verdict": (
                    "stalled" if stalled_seconds_now >= 360.0
                    else "still_improving" if len(convergence_events) > 2
                    else "deterministic_plateau"
                ),
            }
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
    stalled_seconds = max(0.0, time.monotonic() - last_improvement_time)
    args.convergence = {
        "events": convergence_events,
        "last_improvement_elapsed_seconds": float(
            max(0.0, last_improvement_time - start)
        ),
        "stalled_seconds": float(stalled_seconds),
        "verdict": (
            "stalled" if stalled_seconds >= 360.0
            else "still_improving" if len(convergence_events) > 2
            else "deterministic_plateau"
        ),
    }
    archive, archive_scores = prune_archive(archive, archive_scores, args.archive_size)
    save_state(state_path, best, rng, args._completed_trials, args._elapsed_seconds, args, archive=archive)
    final_entries = combine_unique_entries(best, archive)
    if stop_requested():
        refined_entries = []
        args.refinement = {"enabled": False, "reason": "graceful stop requested"}
    else:
        refined_entries, args.refinement = refine_entries(
            final_entries,
            score_plan,
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
