from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def risk_for(row: dict[str, str]) -> str:
    filters = as_float(row, "filter_count")
    headroom = as_float(row, "positive_gain_penalty_db")
    peak = as_float(row, "peak_penalty_db")
    null_boost = as_float(row, "null_boost_avg")
    if headroom > 4.0 or null_boost > 1.5 or filters > 12:
        return "high"
    if headroom > 2.0 or peak > 3.5 or filters > 8:
        return "medium"
    return "low"


def summarise(run_folder: Path, top: int) -> dict[str, object]:
    native_summary = run_folder / "optimizer_summary.json"
    if native_summary.exists():
        payload = json.loads(native_summary.read_text(encoding="utf-8"))
        payload["top_candidates"] = payload.get("top_candidates", [])[:top]
        return payload

    csv_path = run_folder / "optimizer_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Missing optimizer_results.csv in %s" % run_folder)
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    candidates = []
    for row in rows[:top]:
        candidates.append({
            "rank": int(as_float(row, "rank")),
            "file": row.get("file", ""),
            "objective": as_float(row, "objective"),
            "pareto_rank": row.get("pareto_rank", ""),
            "family_role": row.get("family_role", ""),
            "sum_rms_db": as_float(row, "sum_rms_db"),
            "image_weighted_db": as_float(row, "sum_wrms_img_db"),
            "worst_dev_db": as_float(row, "worst_dev_db"),
            "filters": int(as_float(row, "filter_count")),
            "headroom_penalty_db": as_float(row, "positive_gain_penalty_db"),
            "risk": risk_for(row),
            "left_alone": row.get("left_alone", ""),
        })

    warnings: list[str] = []
    report = run_folder / "optimizer_report.md"
    if report.exists():
        for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
            if "FAIL" in line or "low confidence" in line or "not full confidence" in line:
                warnings.append(line.strip("- "))

    return {
        "run_folder": str(run_folder),
        "candidate_count": len(rows),
        "top_candidates": candidates,
        "family_files": sorted(p.name for p in run_folder.glob("family_*.afpx")),
        "warnings": warnings[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise an optimiser output folder as compact JSON.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("latest_run_summary.json"))
    args = parser.parse_args()

    payload = summarise(args.run_folder.resolve(), args.top)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
