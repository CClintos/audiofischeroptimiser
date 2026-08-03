#!/usr/bin/env python3
"""Build or validate one shared PEQ baseline-rehabilitation cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _optimizer as opt
import _optimizer_stream as stream
import baseline_rehabilitation as rehab


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build one fingerprinted rehabilitation cache for PEQ workers."
    )
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--target", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--seconds", type=int, default=1200)
    result.add_argument("--profile", choices=("safe", "explore"), default="explore")
    result.add_argument("--filter-cost-scale", type=float, default=0.1)
    result.add_argument("--worst-weight", type=float, default=0.10)
    result.add_argument("--min-total-bands", type=int, default=0)
    result.add_argument("--validation-threshold", type=float, default=2.5)
    result.add_argument("--sample-rate", type=float, default=96000.0)
    result.add_argument("--impulse-root", type=Path, default=None)
    result.add_argument("--phase-cache", type=Path, default=None)
    result.add_argument("--level-calibration", type=Path, default=None)
    result.add_argument("--repeatability-folder", type=Path, default=None)
    result.add_argument("--role-map", type=Path, default=None)
    result.add_argument("--phase-writes", choices=("auto", "off"), default="off")
    result.add_argument("--print-mode", choices=("compact", "none"), default="compact")
    return result


def _configure_environment(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.baseline = args.baseline.resolve()
    args.target = args.target.resolve()
    os.environ["AFPX_DATA_ROOT"] = str(args.data_root)
    os.environ["AFPX_BASELINE"] = str(args.baseline)
    os.environ["AFPX_TARGET"] = str(args.target)
    if args.role_map:
        os.environ["AFPX_ROLE_MAP"] = str(args.role_map.resolve())
    else:
        os.environ.pop("AFPX_ROLE_MAP", None)


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.data_root.is_dir():
        raise SystemExit(f"Measurement folder not found: {args.data_root}")
    if not args.baseline.is_file():
        raise SystemExit(f"Baseline AFPX not found: {args.baseline}")
    if not args.target.is_file():
        raise SystemExit(f"Target curve not found: {args.target}")


def build(args: argparse.Namespace) -> dict[str, object]:
    _configure_environment(args)
    _validate_paths(args)
    args.mode = "peq"
    args.measurement_session, level_calibration = opt.prepare_measurement_session(
        args.baseline, args.target, args.level_calibration
    )
    args.measurement_noise_guard = opt.configure_repeatability_floor(
        args.repeatability_folder, level_calibration
    )
    opt.sync_external_objective(args.baseline, args.target, level_calibration)
    stream.configure_profile(args.profile)

    channel_roles = dict(opt.CH_TRACE)
    channel_roles.update({6: "Left Sub", 7: "Right Sub"})
    config = opt.rehabilitation_config(
        channel_roles, explore=args.profile == "explore"
    )
    fingerprint_inputs = stream.stream_input_fingerprint_payload(args, config)
    fingerprint = stream.stream_input_fingerprint(args, config)

    if args.out.exists():
        return stream.load_rehabilitation_cache(args.out, fingerprint)

    freqs, traces, rich_traces = opt.load_measurements(level_calibration)
    raw_target = opt.load_target(args.target, freqs)
    target = raw_target + opt.target_anchor_offset(
        freqs, traces["System Sum"], raw_target
    )
    validation = opt.pair_sum_validation(
        freqs, traces, threshold=args.validation_threshold
    )
    failed = [item for item in validation if item.get("pass") is False]
    if failed:
        details = "; ".join(
            f"{item['pair']} {item['rms_db']} dB > {item['threshold_db']} dB"
            for item in failed
        )
        raise SystemExit("Measurement validation gate failed: " + details)

    phase_session = opt.analyze_phase_session(
        freqs,
        traces,
        rich_traces,
        args.measurement_session,
        args.sample_rate,
        args.impulse_root,
        args.phase_cache,
        writes=args.phase_writes != "off",
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
        phase_session["writes"],
        bool(args.measurement_session["audit"].get("phase_valid")),
    )
    base_xml = opt.decode_afpx(args.baseline)
    refs = rehab.active_peq_slot_refs(base_xml, channel_roles)

    return stream.build_or_load_rehabilitation_cache(
        args.out,
        expected_fingerprint=fingerprint,
        fingerprint_inputs=fingerprint_inputs,
        config=config,
        build_stage=lambda: stream.run_rehabilitation_stage(
            mode="peq",
            refs=refs,
            score_plan=score_plan,
            total_seconds=args.seconds,
            config=config,
        ),
    )


def main() -> int:
    args = parser().parse_args()
    try:
        result = build(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.print_mode == "compact":
        state = result["rehabilitation"]
        print(json.dumps({
            "cache": str(args.out.resolve()),
            "fingerprint": result["fingerprint"],
            "status": state.get("completion_status"),
            "evaluations": state.get("evaluations", 0),
        }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
