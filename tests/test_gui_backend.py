from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from optimizer_gui.backend import (
    RunConfig, candidate_files, collect_progress, powershell_command, validate_config,
)


class GuiJobTests(unittest.TestCase):
    def test_job_round_trip_and_worker_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig("data", "base.afpx", "target.txt", tmp, cpu_percent=100)
            config.save()
            loaded = RunConfig.load(Path(tmp))
            self.assertEqual(loaded.data_root, "data")
            self.assertLessEqual(loaded.workers, 12)

    def test_command_passes_explicit_user_choices(self) -> None:
        config = RunConfig(
            "C:\\Measurements", "C:\\Measurements\\base.afpx", "C:\\target.txt",
            "C:\\run", voicing_variants="audition", sub_blend="recommend", headroom_db=3.0,
        )
        program, args = powershell_command(config, executable="C:\\python.exe")
        self.assertEqual(program, "powershell.exe")
        self.assertIn("audition", args)
        self.assertIn("recommend", args)
        self.assertIn("C:\\python.exe", args)

    def test_invalid_inputs_block_before_workers_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(tmp, str(Path(tmp) / "missing.afpx"), "missing.txt", str(Path(tmp) / "run"))
            result = validate_config(config)
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(len(result["errors"]), 1)

    def test_progress_and_candidates_read_compact_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker_01"
            worker.mkdir()
            (worker / "stream_state.json").write_text(json.dumps({
                "completed_trials": 42, "elapsed_seconds": 5,
                "best": [{"objective": 3.25}],
            }), encoding="utf-8")
            progress = collect_progress(root)
            self.assertEqual(progress["trials"], 42)
            self.assertEqual(progress["best_objective"], 3.25)

            merged = root / "_merged_top"
            merged.mkdir()
            (merged / "family_balanced.afpx").write_bytes(b"x")
            summary_path = merged / "assistant_summary.json"
            summary = {"families": {"balanced": {"file": "family_balanced.afpx", "objective": 2.0}}}
            rows = candidate_files(summary, summary_path)
            self.assertEqual(rows[0]["role"], "Balanced")


if __name__ == "__main__":
    unittest.main()
