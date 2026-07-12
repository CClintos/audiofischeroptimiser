from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one shared crossover diagnostic cache.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--impulse-root", type=Path, default=None)
    parser.add_argument("--level-calibration", type=Path, default=None)
    parser.add_argument("--validation-threshold", type=float, default=2.5)
    parser.add_argument("--print-mode", choices=("path", "json", "none"), default="path")
    args = parser.parse_args()

    os.environ["AFPX_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["AFPX_BASELINE"] = str(args.baseline.resolve())
    os.environ["AFPX_TARGET"] = str(args.target.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _optimizer as opt

    session, calibration = opt.prepare_measurement_session(
        args.baseline, args.target, args.level_calibration
    )
    opt.sync_external_objective(args.baseline, args.target, calibration)
    freqs, traces, rich = opt.load_measurements(calibration)
    validation = opt.pair_sum_validation(freqs, traces, threshold=args.validation_threshold)
    failed = [item for item in validation if not item["pass"]]
    if failed:
        raise SystemExit("Measurement validation gate failed: " + "; ".join(
            f"{item['pair']} {item['rms_db']} dB > {item['threshold_db']} dB" for item in failed
        ))
    rows, cache = opt.cached_crossover_phase_diagnostics(
        args.out, freqs, traces, rich, session, args.impulse_root
    )
    payload = {**cache, "row_count": len(rows), "pair_validation": validation}
    if args.print_mode == "json":
        print(json.dumps(payload, indent=2))
    elif args.print_mode == "path":
        print(str(args.out.resolve()))


if __name__ == "__main__":
    main()
