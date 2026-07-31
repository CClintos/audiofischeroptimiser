from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.verify_achieved_response import verify_run


class AchievedVerificationTests(unittest.TestCase):
    def test_predicted_and_level_shifted_achieved_match_after_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            original = root / "original"
            achieved = root / "achieved"
            run.mkdir()
            original.mkdir()
            achieved.mkdir()
            freqs = np.geomspace(60.0, 16000.0, 256)
            baseline = 70.0 - 3.0 * np.log10(freqs / 100.0)
            change = -1.5 * np.exp(-0.5 * (np.log2(freqs / 2650.0) / 0.2) ** 2)
            candidate = baseline + change

            def write(folder: Path, filename: str, values: np.ndarray) -> None:
                (folder / filename).write_text(
                    "\n".join(f"{f:.8f} {value:.8f}" for f, value in zip(freqs, values)),
                    encoding="utf-8",
                )

            for filename in (
                "System Sum.txt", "Front L High.txt", "Front R High.txt",
                "Front L Low.txt", "Front R Low.txt",
            ):
                write(original, filename, baseline)
                write(achieved, filename, candidate + 5.0)
            summary = {
                "data_root": str(original),
                "response_plot": {
                    "frequency_hz": freqs.tolist(),
                    "baseline_error_db": np.zeros_like(freqs).tolist(),
                    "candidate_error_db": change.tolist(),
                    "drivers": {
                        role: {
                            "frequency_hz": freqs.tolist(),
                            "change_db": change.tolist(),
                        }
                        for role in ("FL High", "FR High", "FL Low", "FR Low")
                    },
                },
                "top_candidates": [{"file": "candidate.afpx"}],
            }
            (run / "optimizer_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            payload = verify_run(run, achieved)
            self.assertEqual(payload["verdict"], "model_matched_measurement")
            self.assertLess(payload["system"]["difference_rms_db"], 0.001)
            self.assertEqual(set(payload["drivers"]), {"FL High", "FR High", "FL Low", "FR Low"})
            self.assertTrue(Path(payload["file"]).is_file())


if __name__ == "__main__":
    unittest.main()
