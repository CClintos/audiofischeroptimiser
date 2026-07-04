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
    args = parser.parse_args()

    worker_dirs = sorted(p for p in args.root.glob("worker_*") if p.is_dir())
    items = []
    for worker in worker_dirs:
        items.extend(load_worker_best(worker))
    if not items:
        raise SystemExit("No stream_state.json best candidates found under " + str(args.root))

    freqs, traces = opt.load_measurements()
    raw_target = opt.load_target(args.target, freqs)
    target = raw_target + opt.target_anchor_offset(freqs, traces["System Sum"], raw_target)
    base_xml = opt.decode_afpx(args.baseline)
    validation = opt.pair_sum_validation(freqs, traces, threshold=args.validation_threshold)
    failed_validation = [item for item in validation if not item["pass"]]
    if failed_validation:
        details = "; ".join(
            f"{item['pair']} {item['rms_db']} dB > {item['threshold_db']} dB"
            for item in failed_validation
        )
        raise SystemExit("Measurement validation gate failed: " + details)
    component_score = opt.make_component_scorer(
        freqs, traces, target, args.filter_cost_scale, args.worst_weight
    )

    rescored_items = []
    for _stored_value, sig, groups, source in items:
        rescored_items.append((component_score(groups)["objective"], sig, groups, source))
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
        lint = opt.write_candidate(base_xml, path, groups)
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
    opt.write_family_aliases(out_dir, family_rows, base_xml)

    baseline_groups = {group: [] for group in opt.GROUPS}
    baseline_pred = opt.predict_traces(freqs, traces, baseline_groups)
    baseline_score = opt.tune_scorecard(freqs, baseline_pred, target)
    baseline_score["components"] = component_score(baseline_groups)
    ns = argparse.Namespace(
        baseline=args.baseline,
        target=args.target,
        validation=validation,
        trials=sum(json.loads((w / "stream_state.json").read_text(encoding="utf-8")).get("completed_trials", 0)
                   for w in worker_dirs if (w / "stream_state.json").exists()),
    )
    opt.write_report(out_dir, rows, baseline_score, interference_notes(freqs, traces), ns, family_rows=family_rows)
    print("Merged", len(worker_dirs), "workers into", out_dir)
    print("Total worker candidates:", len(items))
    print("Top objective:", rows[0]["objective"])
    print(opt.format_bands(rows[0]["groups"]))


if __name__ == "__main__":
    main()
