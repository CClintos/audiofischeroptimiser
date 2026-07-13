from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run authoritative optimizer gates without starting workers.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--level-calibration", type=Path, default=None)
    parser.add_argument("--validation-threshold", type=float, default=2.5)
    args = parser.parse_args()

    os.environ["AFPX_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["AFPX_BASELINE"] = str(args.baseline.resolve())
    os.environ["AFPX_TARGET"] = str(args.target.resolve())

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _optimizer as optimizer

    try:
        session, calibration = optimizer.prepare_measurement_session(
            args.baseline, args.target, args.level_calibration
        )
        optimizer.sync_external_objective(args.baseline, args.target, calibration)
        freqs, traces, _rich = optimizer.load_measurements(calibration)
        pairs = optimizer.pair_sum_validation(freqs, traces, args.validation_threshold)
        failed = [row for row in pairs if not row["pass"]]
        result = {
            "valid": not failed,
            "measurement_session": session.get("audit", {}),
            "pair_validation": pairs,
            "errors": [
                f"{row['pair']} solo/together validation is {row['rms_db']} dB; limit is {row['threshold_db']} dB"
                for row in failed
            ],
        }
    except Exception as exc:
        result = {"valid": False, "measurement_session": {}, "pair_validation": [], "errors": [str(exc)]}
    print(json.dumps(result))
    raise SystemExit(0 if result["valid"] else 2)


if __name__ == "__main__":
    main()
