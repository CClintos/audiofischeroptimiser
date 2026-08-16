"""Merge resumable streaming optimizer workers into one ranked AFPX folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _optimizer as opt
import baseline_rehabilitation as rehab
from _optimizer_stream import (
    beam_entry_from_json, build_rows, candidate_plan_from_json, configure_profile, interference_notes,
    stream_input_fingerprint,
)


def load_worker_payload(
    worker_dir: Path, expected_fingerprint: str | None = None,
):
    state = worker_dir / "stream_state.json"
    if not state.exists():
        raise ValueError(f"{worker_dir.name} is missing stream_state.json")
    try:
        payload = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{worker_dir.name} has an unreadable stream_state.json"
        ) from exc
    if expected_fingerprint is not None:
        actual = payload.get("input_fingerprint")
        if not actual:
            raise ValueError(
                f"{worker_dir.name} is missing input fingerprint"
            )
        if actual != expected_fingerprint:
            raise ValueError(
                f"{worker_dir.name} fingerprint does not match current inputs"
            )
    return payload


def load_worker_best(
    worker_dir: Path, expected_fingerprint: str | None = None,
):
    payload = load_worker_payload(worker_dir, expected_fingerprint)
    out = []
    for bucket in ("best", "archive"):
        for item in payload.get(bucket, []):
            entry = beam_entry_from_json(item)
            out.append((
                float(entry.objective),
                entry.signature,
                entry.plan,
                worker_dir.name,
            ))
    return out

def attach_fingerprint_context(args, level_calibration, measurement_noise_guard):
    """Mirror worker-loaded session values before fingerprinting merge inputs."""
    args.loaded_level_calibration = dict(level_calibration or {})
    args.measurement_noise_guard = dict(measurement_noise_guard or {})

def write_merge_progress(path: Path, stage: str, completed: int, total: int):
    payload = {
        "stage": str(stage),
        "completed": max(0, int(completed)),
        "total": max(0, int(total)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def unique_best(items, keep):
    out = []
    seen = set()
    for value, sig, plan, source in sorted(items, key=lambda x: x[0]):
        if sig in seen:
            continue
        seen.add(sig)
        out.append((value, sig, plan, source))
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
    return [
        item for item in items
        if item[3] == "baseline"
        or (
            isinstance(item[2], rehab.CandidatePlan)
            and bool(item[2].slot_edits)
        )
    ]


def main():
    parser = argparse.ArgumentParser(description="Merge streaming optimizer worker outputs.")
    parser.add_argument("root", type=Path, help="Folder containing worker_* folders.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--baseline", type=Path, default=opt.DEFAULT_BASELINE)
    parser.add_argument("--target", type=Path, default=opt.DEFAULT_TARGET)
    parser.add_argument("--filter-cost-scale", type=float, default=0.1)
    parser.add_argument("--worst-weight", type=float, default=0.10)
    parser.add_argument("--min-total-bands", type=int, default=0)
    parser.add_argument("--profile", choices=("safe", "explore"), default="explore")
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
    parser.add_argument("--progress-file", type=Path, default=None)
    args = parser.parse_args()
    progress_path = args.progress_file or (args.root / "merge_progress.json")
    write_merge_progress(progress_path, "loading_inputs", 0, 1)

    measurement_session, level_calibration = opt.prepare_measurement_session(
        args.baseline, args.target, args.level_calibration
    )
    measurement_noise_guard = opt.configure_repeatability_floor(
        args.repeatability_folder, level_calibration
    )
    attach_fingerprint_context(
        args, level_calibration, measurement_noise_guard,
    )
    opt.sync_external_objective(args.baseline, args.target, level_calibration)
    configure_profile(args.profile)
    args.measurement_session = measurement_session
    channel_roles = dict(opt.CH_TRACE)
    channel_roles.update({6: "Left Sub", 7: "Right Sub"})
    rehabilitation_config = opt.rehabilitation_config(
        channel_roles, explore=args.profile == "explore"
    )
    expected_fingerprint = stream_input_fingerprint(
        args, rehabilitation_config
    )
    worker_dirs = sorted(p for p in args.root.glob("worker_*") if p.is_dir())
    try:
        worker_state_payloads = [
            load_worker_payload(worker, expected_fingerprint)
            for worker in worker_dirs
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    rehabilitation_rows = [
        dict(payload.get("rehabilitation", {}))
        for payload in worker_state_payloads
        if payload.get("rehabilitation")
    ]
    args.rehabilitation = (
        rehabilitation_rows[0] if rehabilitation_rows else {}
    )
    items = []
    for worker in worker_dirs:
        items.extend(load_worker_best(worker, expected_fingerprint))
    write_merge_progress(progress_path, "ranking_candidates", 1, 1)
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
    baseline_plan = rehab.CandidatePlan()
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
    try:
        rehabilitation_plan = candidate_plan_from_json(
            dict(args.rehabilitation).get("best_plan", {})
        )
    except (TypeError, ValueError, KeyError):
        rehabilitation_plan = rehab.CandidatePlan()
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
    baseline_components = dict(score_plan(baseline_plan))
    items.append((
        float(baseline_components["objective"]),
        rehab.candidate_plan_signature(baseline_plan),
        baseline_plan,
        "baseline",
    ))
    items = apply_census_gate(
        items, args.census_found_nothing_eligible
    )

    target_family_size = max(args.top * 10, 200)
    ranked_items = unique_best(items, len(items))
    rescored_items = []
    rescored_components = {}
    phase_peq_rejections = []
    write_merge_progress(
        progress_path, "rescoring_finalists", 0,
        min(target_family_size, len(ranked_items)),
    )
    for _stored_value, _sig, plan, source in ranked_items:
        groups = {group: [] for group in opt.GROUPS}
        groups.update(rehab.thaw_groups(plan.groups))
        if phase_valid and phase_plan:
            if not opt.complex_crossover_verification(
                freqs, rich_traces, groups, phase_plan
            )["pass"]:
                continue
        else:
            conflicts = opt.phase_peq_conflicts(
                freqs, groups, phase_plan
            )
            if conflicts:
                phase_peq_rejections.extend(conflicts)
                continue
        components = dict(score_plan(plan))
        if components.get("phase_peq_conflict_count", 0.0) > 0.0:
            phase_peq_rejections.extend(
                opt.candidate_plan_phase_conflicts(
                    freqs, plan, phase_plan
                )
            )
            continue
        signature = rehab.candidate_plan_signature(plan)
        rescored_items.append((
            float(components["objective"]), signature, plan, source,
        ))
        rescored_components[signature] = components
        write_merge_progress(
            progress_path, "rescoring_finalists", len(rescored_items),
            min(target_family_size, len(ranked_items)),
        )
        if len(rescored_items) >= target_family_size:
            break
    if not rescored_items:
        raise SystemExit("Every stored candidate conflicts with an attached crossover phase write")
    family_pool = unique_best(rescored_items, target_family_size)
    best = family_pool[: args.top]
    out_dir = args.out or (args.root / "_merged_top")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("candidate_*.afpx"):
        old.unlink()

    rows = []
    write_merge_progress(progress_path, "writing_candidates", 0, max(1, len(best)))
    for rank, (value, sig, plan, source) in enumerate(best, start=1):
        groups = {group: [] for group in opt.GROUPS}
        groups.update(rehab.thaw_groups(plan.groups))
        pred = opt.predict_candidate_plan(freqs, traces, plan)
        score = opt.tune_scorecard(freqs, pred, target)
        components = dict(rescored_components[sig])
        file_name = (
            f"candidate_{rank:02d}_objective_{value:.4f}_{source}.afpx"
        )
        path = out_dir / file_name
        lint = opt.write_candidate_plan(
            base_xml, path, plan, phase_plan=phase_plan
        )
        rows.append({
            "rank": rank,
            "file": file_name,
            "objective": value,
            "score": score,
            "components": components,
            "groups": groups,
            "plan": plan,
            "signature": sig,
            "lint": lint,
            "headroom": opt.candidate_plan_headroom(freqs, plan),
            "source": source,
            "left_alone": opt.left_alone_note(freqs, traces),
        })
        write_merge_progress(progress_path, "writing_candidates", rank, len(best))
    family_rows = []
    write_merge_progress(progress_path, "building_families", 0, max(1, len(family_pool)))
    for rank, (value, sig, plan, source) in enumerate(
        family_pool, start=1
    ):
        groups = {group: [] for group in opt.GROUPS}
        groups.update(rehab.thaw_groups(plan.groups))
        pred = opt.predict_candidate_plan(freqs, traces, plan)
        score = opt.tune_scorecard(freqs, pred, target)
        components = dict(rescored_components[sig])
        family_rows.append({
            "rank": rank,
            "file": (
                f"candidate_{rank:02d}_objective_"
                f"{value:.4f}_{source}.afpx"
            ),
            "objective": value,
            "score": score,
            "components": components,
            "groups": groups,
            "plan": plan,
            "signature": sig,
            "source": source,
            "left_alone": opt.left_alone_note(freqs, traces),
        })
        write_merge_progress(progress_path, "building_families", rank, len(family_pool))
    rehabilitation_file = ""
    if rows:
        final_plan = rows[0]["plan"]
        empty_signature = rehab.candidate_plan_signature(rehab.CandidatePlan())
        rehabilitation_signature = rehab.candidate_plan_signature(
            rehabilitation_plan
        )
        final_signature = rehab.candidate_plan_signature(final_plan)
        if (
            rehabilitation_signature != empty_signature
            and rehabilitation_signature != final_signature
        ):
            rehabilitation_file = "rehabilitated_baseline.afpx"
            opt.write_candidate_plan(
                base_xml,
                out_dir / rehabilitation_file,
                rehabilitation_plan,
                phase_plan=phase_plan,
            )
    if args.rehabilitation:
        args.rehabilitation["file"] = rehabilitation_file
        args.rehabilitation["objective"] = float(
            rehabilitation_components["objective"]
        )
    opt.write_family_aliases(out_dir, family_rows, base_xml, phase_plan=phase_plan)
    voicing_variants = []
    if args.voicing_variants == "audition" and best:
        best_plan = best[0][2]
        variant_base = rehab.apply_slot_edits(
            base_xml, best_plan.slot_edits
        )
        voicing_variants = opt.write_voicing_variants(
            out_dir,
            variant_base,
            rehab.thaw_groups(best_plan.groups),
            phase_plan,
        )
    sub_blend = None
    if args.sub_blend == "recommend":
        sub_blend = opt.same_level_sub_blend_recommendation(
            freqs, traces, target, measurement_session, args.headroom_db
        )

    baseline_groups = {group: [] for group in opt.GROUPS}
    baseline_pred = opt.predict_traces(freqs, traces, baseline_groups)
    baseline_score = opt.tune_scorecard(freqs, baseline_pred, target)
    baseline_score["components"] = baseline_components
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
        rehabilitation=args.rehabilitation,
    )
    write_merge_progress(progress_path, "writing_summary", 0, 1)
    opt.write_report(out_dir, rows, baseline_score, interference_notes(freqs, traces), ns,
                     family_rows=family_rows, crossover_rows=crossover_rows, phase_plan=phase_plan)
    write_merge_progress(progress_path, "complete", 1, 1)
    print("Merged", len(worker_dirs), "workers into", out_dir)
    print("Total worker candidates:", len(items))
    print("Top objective:", rows[0]["objective"])
    print(opt.format_bands(rows[0]["groups"]))


if __name__ == "__main__":
    main()
