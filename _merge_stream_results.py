"""Merge resumable streaming optimizer workers into one ranked AFPX folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _optimizer as opt
from _optimizer_stream import build_rows, groups_from_json, interference_notes


def load_worker_best(worker_dir: Path):
    state = worker_dir / "stream_state.json"
    if not state.exists():
        return []
    payload = json.loads(state.read_text(encoding="utf-8"))
    out = []
    for bucket in ("best", "archive"):
        for item in payload.get(bucket, []):
            groups = groups_from_json(item.get("groups", {}))
            out.append((float(item["objective"]), opt.bands_signature(groups), groups, worker_dir.name))
    return out


def unique_best(items, keep):
    out = []
    seen = set()
    for value, sig, groups, source in sorted(items, key=lambda x: x[0]):
        if sig in seen:
            continue
        seen.add(sig)
        out.append((value, sig, groups, source))
        if len(out) >= keep:
            break
    return out


def census_found_nothing_eligible(proposal_audits: list) -> bool:
    """DEFECT 4b: the pre-search census ("worth fixing" / "deliberately
    skipped") was computed and reported but never gated the run's actual
    output - a candidate could still be selected and written even when the
    census said nothing was eligible. True here means every worker that
    reported a census found zero eligible correction centres, and no
    candidate but the baseline may be selected. See CHANGELOG.md."""
    if not proposal_audits:
        return False
    first_worth_fixing = dict(proposal_audits[0]).get("problem_census", {}).get("worth_fixing", [])
    return not first_worth_fixing


def apply_census_gate(items, gate_active: bool):
    """When census_found_nothing_eligible() is True, only the "baseline"
    entry in `items` (always present - see main()) may survive, regardless
    of what any individual worker's search still explored."""
    if not gate_active:
        return items
    return [item for item in items if item[3] == "baseline"]


def main():
    parser = argparse.ArgumentParser(description="Merge streaming optimizer worker outputs.")
    parser.add_argument("root", type=Path, help="Folder containing worker_* folders.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--baseline", type=Path, default=opt.DEFAULT_BASELINE)
    parser.add_argument("--target", type=Path, default=opt.DEFAULT_TARGET)
    parser.add_argument("--filter-cost-scale", type=float, default=0.1)
    parser.add_argument("--worst-weight", type=float, default=0.10)
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
    parser.add_argument("--repeatability-folder", type=Path, default=None,
                        help="Second same-day session used to derive the measurement floor.")
    parser.add_argument("--phase-writes", choices=("auto", "off"), default="auto",
                        help="Use 'off' to report the crossover ladder without writing polarity/delay/APF changes.")
    parser.add_argument("--mode", choices=("peq", "phase"), default="peq")
    parser.add_argument("--sub-blend", choices=("off", "recommend"), default="off")
    parser.add_argument("--headroom-db", type=float, default=None)
    parser.add_argument("--voicing-variants", choices=("off", "audition"), default="off")
    args = parser.parse_args()

    measurement_session, level_calibration = opt.prepare_measurement_session(
        args.baseline, args.target, args.level_calibration
    )
    measurement_noise_guard = opt.configure_repeatability_floor(
        args.repeatability_folder, level_calibration
    )
    opt.sync_external_objective(args.baseline, args.target, level_calibration)
    worker_dirs = sorted(p for p in args.root.glob("worker_*") if p.is_dir())
    worker_state_payloads = []
    for worker in worker_dirs:
        state_path = worker / "stream_state.json"
        if state_path.is_file():
            try:
                worker_state_payloads.append(json.loads(state_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    proposal_audits = [
        dict(payload.get("proposal_audit", {}))
        for payload in worker_state_payloads if payload.get("proposal_audit")
    ]
    args.proposal_audit = proposal_audits[0] if proposal_audits else {}
    args.census_found_nothing_eligible = census_found_nothing_eligible(proposal_audits)
    convergence_rows = [
        dict(payload.get("convergence", {}))
        for payload in worker_state_payloads if payload.get("convergence")
    ]
    if convergence_rows:
        args.convergence = {
            "workers": convergence_rows,
            "verdict": (
                "still_improving"
                if any(row.get("verdict") == "still_improving" for row in convergence_rows)
                else "stalled"
                if all(row.get("verdict") == "stalled" for row in convergence_rows)
                else "deterministic_plateau"
            ),
            "stalled_seconds": max(
                float(row.get("stalled_seconds", 0.0)) for row in convergence_rows
            ),
        }
    else:
        args.convergence = {}
    items = []
    for worker in worker_dirs:
        items.extend(load_worker_best(worker))
    if not items:
        raise SystemExit("No stream_state.json best candidates found under " + str(args.root))

    freqs, traces, rich_traces = opt.load_measurements(level_calibration)
    raw_target = opt.load_target(args.target, freqs)
    target = raw_target + opt.target_anchor_offset(freqs, traces["System Sum"], raw_target)
    base_xml = opt.decode_afpx(args.baseline)
    validation = opt.pair_sum_validation(freqs, traces, threshold=args.validation_threshold)
    phase_session = opt.analyze_phase_session(
        freqs, traces, rich_traces, measurement_session, args.sample_rate,
        args.impulse_root, args.phase_cache, writes=args.phase_writes != "off"
    )
    crossover_rows = phase_session["diagnostics"]
    phase_diagnostic_cache = phase_session["cache"]
    phase_plan = phase_session["writes"]
    failed_validation = [item for item in validation if item.get("pass") is False]
    if failed_validation:
        details = "; ".join(
            f"{item['pair']} {item['rms_db']} dB > {item['threshold_db']} dB"
            for item in failed_validation
        )
        raise SystemExit("Measurement validation gate failed: " + details)
    phase_valid = bool(measurement_session["audit"].get("phase_valid"))
    component_score = opt.complex_phase_component_scorer(
        opt.make_component_scorer(
            freqs, traces, target, args.filter_cost_scale, args.worst_weight
        ),
        freqs,
        rich_traces,
        phase_plan,
        phase_valid,
    )
    baseline_groups = {group: [] for group in opt.GROUPS}
    items.append((
        float(component_score(baseline_groups)["objective"]),
        opt.bands_signature(baseline_groups),
        baseline_groups,
        "baseline",
    ))
    items = apply_census_gate(items, args.census_found_nothing_eligible)

    rescored_items = []
    phase_peq_rejections = []
    for _stored_value, sig, groups, source in items:
        if phase_valid and phase_plan:
            if not opt.complex_crossover_verification(freqs, rich_traces, groups, phase_plan)["pass"]:
                continue
        else:
            conflicts = opt.phase_peq_conflicts(freqs, groups, phase_plan)
            if conflicts:
                phase_peq_rejections.extend(conflicts)
                continue
        rescored_items.append((component_score(groups)["objective"], sig, groups, source))
    if not rescored_items:
        raise SystemExit("Every stored candidate conflicts with an attached crossover phase write")
    family_pool = unique_best(rescored_items, max(args.top * 10, 200))
    best = family_pool[: args.top]
    out_dir = args.out or (args.root / "_merged_top")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("candidate_*.afpx"):
        old.unlink()

    rows = []
    for rank, (value, sig, groups, source) in enumerate(best, start=1):
        pred = opt.predict_traces(freqs, traces, groups)
        score = opt.tune_scorecard(freqs, pred, target)
        components = component_score(groups)
        file_name = f"candidate_{rank:02d}_objective_{value:.4f}_{source}.afpx"
        path = out_dir / file_name
        lint = opt.write_candidate(base_xml, path, groups, phase_plan=phase_plan)
        rows.append({
            "rank": rank,
            "file": file_name,
            "objective": value,
            "score": score,
            "components": components,
            "groups": groups,
            "signature": sig,
            "lint": lint,
            "headroom": {g: opt.headroom_report(freqs, b) for g, b in groups.items()},
            "source": source,
            "left_alone": opt.left_alone_note(freqs, traces),
        })
    family_rows = []
    for rank, (value, sig, groups, source) in enumerate(family_pool, start=1):
        pred = opt.predict_traces(freqs, traces, groups)
        score = opt.tune_scorecard(freqs, pred, target)
        components = component_score(groups)
        family_rows.append({
            "rank": rank,
            "file": f"candidate_{rank:02d}_objective_{value:.4f}_{source}.afpx",
            "objective": value,
            "score": score,
            "components": components,
            "groups": groups,
            "signature": sig,
            "source": source,
            "left_alone": opt.left_alone_note(freqs, traces),
        })
    opt.write_family_aliases(out_dir, family_rows, base_xml, phase_plan=phase_plan)
    voicing_variants = []
    if args.voicing_variants == "audition" and best:
        voicing_variants = opt.write_voicing_variants(out_dir, base_xml, best[0][2], phase_plan)
    sub_blend = None
    if args.sub_blend == "recommend":
        sub_blend = opt.same_level_sub_blend_recommendation(
            freqs, traces, target, measurement_session, args.headroom_db
        )

    baseline_pred = opt.predict_traces(freqs, traces, baseline_groups)
    baseline_score = opt.tune_scorecard(freqs, baseline_pred, target)
    baseline_score["components"] = component_score(baseline_groups)
    ns = argparse.Namespace(
        baseline=args.baseline,
        target=args.target,
        validation=validation,
        gate_ms=args.gate_ms,
        sample_rate=args.sample_rate,
        level_calibration=args.level_calibration,
        measurement_session=measurement_session,
        phase_peq_rejections=phase_peq_rejections[:20],
        phase_cache=args.phase_cache,
        phase_diagnostic_cache=phase_diagnostic_cache,
        mode=args.mode,
        phase_session=phase_session,
        freqs=freqs,
        rich_traces=rich_traces,
        voicing_variants=voicing_variants,
        sub_blend_recommendation=sub_blend,
        trials=sum(json.loads((w / "stream_state.json").read_text(encoding="utf-8")).get("completed_trials", 0)
                   for w in worker_dirs if (w / "stream_state.json").exists()),
        # Previously never reached write_report()'s assistant_summary at all
        # (this Namespace is built fresh and did not carry it forward from
        # `args`) - the merged report's problem_census was silently always
        # empty regardless of what workers actually found. Fixed alongside
        # the DEFECT 4b hard gate above; see CHANGELOG.md.
        proposal_audit=args.proposal_audit,
        census_found_nothing_eligible=args.census_found_nothing_eligible,
    )
    opt.write_report(out_dir, rows, baseline_score, interference_notes(freqs, traces), ns,
                     family_rows=family_rows, crossover_rows=crossover_rows, phase_plan=phase_plan)
    print("Merged", len(worker_dirs), "workers into", out_dir)
    print("Total worker candidates:", len(items))
    print("Top objective:", rows[0]["objective"])
    print(opt.format_bands(rows[0]["groups"]))


if __name__ == "__main__":
    main()
