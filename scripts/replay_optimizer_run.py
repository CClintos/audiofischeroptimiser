from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _job_config(run_root: Path) -> dict[str, Any]:
    path = run_root / "gui_job.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _candidate_paths(run_root: Path, limit: int) -> list[Path]:
    selected = []
    merged = run_root / "_merged_top"
    if merged.is_dir():
        selected.extend(sorted(merged.glob("*.afpx")))
    selected.extend(sorted(run_root.glob("worker_*/candidate_01_*.afpx")))
    unique = []
    seen = set()
    for path in selected:
        fingerprint = (path.name, path.stat().st_size)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(path)
        if len(unique) >= limit:
            break
    return unique


def replay_run(run_root: Path, data_root: Path | None = None,
               baseline: Path | None = None, target: Path | None = None,
               limit: int = 40, out: Path | None = None) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    job = _job_config(run_root)
    data_root = Path(data_root or job.get("data_root", "")).resolve()
    baseline = Path(baseline or job.get("baseline", "")).resolve()
    target = Path(target or job.get("target", "")).resolve()
    role_map = Path(str(job.get("role_map", ""))) if job.get("role_map") else None
    if not data_root.is_dir() or not baseline.is_file() or not target.is_file():
        raise FileNotFoundError(
            "Replay needs the original measurement folder, baseline AFPX, and target curve"
        )
    os.environ["AFPX_DATA_ROOT"] = str(data_root)
    os.environ["AFPX_BASELINE"] = str(baseline)
    os.environ["AFPX_TARGET"] = str(target)
    if role_map:
        os.environ["AFPX_ROLE_MAP"] = str(role_map)

    from objective_module.session import ScorerSession
    import _optimizer as optimizer
    import _optimizer_stream as stream
    from scripts.verify_written_tune import (
        _added_peq_by_channel, decode_afpx, verify,
    )

    session = ScorerSession(data_root, baseline, target)
    baseline_score = session.score_afpx(baseline)
    rows = []
    for candidate in _candidate_paths(run_root, limit):
        score = session.score_afpx(candidate)
        lint = verify(
            baseline, candidate, False, False, False, True,
            data_root, target, role_map,
        )
        old_xml = decode_afpx(baseline)
        new_xml = decode_afpx(candidate)
        added = _added_peq_by_channel(old_xml, new_xml)
        added_filters = [
            {
                "channel": channel,
                "frequency_hz": float(item.get("F") or 0.0),
                "q": float(item.get("Q") or 0.0),
                "gain_db": float(item.get("G") or 0.0),
            }
            for channel, items in added.items() for item in items
        ]
        rows.append({
            "file": str(candidate),
            "objective": round(float(score["objective"]), 6),
            "delta_vs_baseline": round(
                float(score["objective"]) - float(baseline_score["objective"]), 6
            ),
            "accepted_now": bool(lint["pass"]),
            "rejection_reasons": sorted({
                reason
                for item in lint.get("measurement_guardrail_errors", [])
                for reason in item.get("reasons", [])
            }),
            "added_filters": added_filters,
        })

    freqs, traces, _ = optimizer.load_measurements()
    raw_target = optimizer.load_target(target, freqs)
    anchored_target = raw_target + optimizer.target_anchor_offset(
        freqs, traces["System Sum"], raw_target
    )
    stream.configure_profile("safe")
    pools = stream.find_guided_candidates(freqs, traces, anchored_target, "safe")
    fine_peaks = sorted({
        round(float(item["F"]), 1)
        for items in pools.values() for item in items
        if 2550.0 <= float(item["F"]) <= 2800.0 and float(item["G"]) < 0.0
    })

    def audited_candidates(low_hz: float, high_hz: float) -> list[dict[str, Any]]:
        matches = []
        for row in rows:
            filters = [
                item for item in row["added_filters"]
                if low_hz <= item["frequency_hz"] <= high_hz
            ]
            if filters:
                matches.append({
                    "file": row["file"],
                    "accepted_now": row["accepted_now"],
                    "rejection_reasons": row["rejection_reasons"],
                    "filters": filters,
                })
        return matches

    audited_270 = audited_candidates(240.0, 310.0)
    audited_1128 = audited_candidates(1050.0, 1220.0)
    audited_crossover = audited_candidates(2500.0, 2800.0)
    payload = {
        "schema": "audiofischer-run-replay-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_folder": str(run_root),
        "inputs": {
            "data_root": str(data_root),
            "baseline": str(baseline),
            "target": str(target),
        },
        "baseline_objective": round(float(baseline_score["objective"]), 6),
        "candidate_count": len(rows),
        "accepted_count": sum(row["accepted_now"] for row in rows),
        "rejected_count": sum(not row["accepted_now"] for row in rows),
        "candidates": rows,
        "current_problem_census": stream.LAST_PROPOSAL_AUDIT.get("problem_census", {}),
        "fine_crossover_cut_centres_hz": fine_peaks,
        "fine_2671_peak_preserved": any(abs(value - 2671.0) <= 80.0 for value in fine_peaks),
        "audited_guardrail_checks": {
            "one_sided_270_hz": {
                "candidate_count": len(audited_270),
                "all_rejected": bool(audited_270) and all(
                    not row["accepted_now"] for row in audited_270
                ),
                "candidates": audited_270,
            },
            "one_sided_1128_hz": {
                "candidate_count": len(audited_1128),
                "all_rejected": bool(audited_1128) and all(
                    not row["accepted_now"] for row in audited_1128
                ),
                "candidates": audited_1128,
            },
            "all_front_crossover": {
                "candidate_count": len(audited_crossover),
                "all_rejected": bool(audited_crossover) and all(
                    not row["accepted_now"] for row in audited_crossover
                ),
                "candidates": audited_crossover,
            },
            "fine_2671_peak_preserved": any(
                abs(value - 2671.0) <= 80.0 for value in fine_peaks
            ),
        },
    }
    destination = out or run_root / "replay_current_code.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["file"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay and audit an existing optimizer run.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = replay_run(
        args.run_root, args.data_root, args.baseline, args.target,
        args.limit, args.out,
    )
    print(json.dumps({
        "file": payload["file"],
        "baseline_objective": payload["baseline_objective"],
        "candidate_count": payload["candidate_count"],
        "accepted_count": payload["accepted_count"],
        "rejected_count": payload["rejected_count"],
        "fine_2671_peak_preserved": payload["fine_2671_peak_preserved"],
        "audited_guardrail_checks": {
            name: (
                value if isinstance(value, bool)
                else {
                    "candidate_count": value["candidate_count"],
                    "all_rejected": value["all_rejected"],
                }
            )
            for name, value in payload["audited_guardrail_checks"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
