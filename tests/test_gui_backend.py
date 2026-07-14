from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from optimizer_gui.backend import (
    RunConfig, candidate_files, collect_progress, powershell_command, validate_config,
)
from optimizer_gui.reporting import build_report_html
from optimizer_gui.window import OptimizerWindow


class GuiJobTests(unittest.TestCase):
    def test_start_button_boolean_is_not_treated_as_resume_path(self) -> None:
        calls = []

        class DummyWindow:
            start_run = lambda self, *args: calls.append(args)

        OptimizerWindow._start_clicked(DummyWindow(), False)
        self.assertEqual(calls, [()])

    def test_job_round_trip_and_worker_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig("data", "base.afpx", "target.txt", tmp, cpu_percent=100)
            config.save()
            loaded = RunConfig.load(Path(tmp))
            self.assertEqual(loaded.data_root, "data")
            self.assertLessEqual(loaded.workers, 12)

    def test_phase_mode_is_single_worker_and_explicit(self) -> None:
        config = RunConfig("data", "base.afpx", "target.txt", "run", mode="phase", cpu_percent=80)
        self.assertEqual(config.workers, 1)
        _program, args = powershell_command(config, executable="C:\\python.exe")
        self.assertEqual(args[args.index("-Mode") + 1], "phase")

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

    def test_pdf_report_uses_named_components_and_phase_actions(self) -> None:
        summary = {
            "search": {"mode": "phase"},
            "candidate_count": 1,
            "baseline": {"objective": 5.0, "tonal_error_db": 2.0, "presence_error_db": 2.2},
            "best": {
                "file": "candidate.afpx", "objective": 4.0,
                "components": {"objective": 4.0, "tonal_error_db": 1.5, "presence_error_db": 1.8},
                "left_alone": "450 Hz null: destructive, not EQ-able",
            },
            "phase_actions": [{
                "source": "Left mid to tweeter", "delay_samples": -12,
                "confidence": "warning",
            }],
            "gates": {"measurement_session": {"phase_valid": True}},
        }
        report = build_report_html(summary, {}, Path("assistant_summary.json"))
        self.assertIn("Phase / Timing Diagnostic", report)
        self.assertIn("Vocal / presence error", report)
        self.assertIn("delay -12 samples", report)
        self.assertIn("destructive, not EQ-able", report)

    def test_peq_report_leads_with_plain_language_and_fixed_anchor_graph(self) -> None:
        summary = {
            "search": {"mode": "peq"},
            "candidate_count": 2,
            "baseline": {
                "objective": 7.0, "tonal_error_db": 2.5, "presence_error_db": 2.4,
                "peak_penalty_db": 2.0, "balance_penalty_db": 2.8,
            },
            "best": {
                "file": "candidate.afpx", "objective": 5.0,
                "components": {
                    "objective": 5.0, "tonal_error_db": 1.8, "presence_error_db": 1.9,
                    "peak_penalty_db": 1.3, "balance_penalty_db": 2.4,
                },
                "fixed_anchor_response": {
                    "checkpoints": [
                        {"frequency_hz": 100.0, "baseline_error_db": 3.0,
                         "candidate_error_db": 1.0, "raw_system_delta_db": -2.0},
                        {"frequency_hz": 1000.0, "baseline_error_db": -2.0,
                         "candidate_error_db": -1.0, "raw_system_delta_db": 1.0},
                    ]
                },
            },
            "gates": {"measurement_session": {"phase_valid": False}},
        }
        report = build_report_html(summary, {}, Path("assistant_summary.json"))
        self.assertIn("What You Should Notice", report)
        self.assertIn("target is anchored once", report)
        self.assertIn("data:image/png;base64,", report)
        self.assertIn("Tonal accuracy", report)


if __name__ == "__main__":
    unittest.main()
