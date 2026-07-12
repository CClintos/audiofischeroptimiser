from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Equal-budget guided/beam/CMA search comparison.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--validation-threshold", type=float, default=2.5)
    parser.add_argument("--out", type=Path, default=Path("search_benchmark"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "AFPX_DATA_ROOT": str(args.data_root.resolve()),
        "AFPX_BASELINE": str(args.baseline.resolve()),
        "AFPX_TARGET": str(args.target.resolve()),
    })
    results = []
    for method in ("guided", "beam", "cmaes"):
        folder = args.out / method
        if folder.exists():
            shutil.rmtree(folder)
        command = [
            sys.executable, str(root / "_optimizer_stream.py"),
            "--baseline", str(args.baseline), "--target", str(args.target),
            "--seconds", str(args.seconds), "--seed", str(args.seed),
            "--proposal", method, "--profile", "explore", "--top", "3",
            "--keep", "40", "--archive-size", "200", "--refine-top", "1",
            "--refine-passes", "1", "--phase-writes", "off",
            "--validation-threshold", str(args.validation_threshold),
            "--print-mode", "none", "--out", str(folder),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        row = {"method": method, "wall_seconds": round(elapsed, 3), "returncode": completed.returncode}
        summary = folder / "assistant_summary.json"
        state = folder / "stream_state.json"
        if completed.returncode == 0 and summary.exists():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            state_payload = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
            row.update({
                "objective": payload.get("best", {}).get("objective"),
                "best_file": payload.get("best", {}).get("file"),
                "trials": state_payload.get("completed_trials"),
            })
        else:
            row["error"] = (completed.stderr or completed.stdout)[-1000:]
        results.append(row)
    output = args.out / "benchmark_results.json"
    output.write_text(json.dumps({"seconds_per_method": args.seconds, "seed": args.seed, "results": results}, indent=2), encoding="utf-8")
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
