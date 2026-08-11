from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from optimizer_gui.backend import (
    RunConfig, RunRootBusyError, active_run_pid, candidate_files, claim_run_root,
    collect_progress, create_measurement_template, default_export_name,
    export_candidate, load_target_curve, measurement_checklist, powershell_command, release_run_claim,
    locate_summary, record_run_decision, runner_completed_successfully, runner_failure_reason, save_role_map,
    start_detached_process, suggest_measurement_role, timestamped_run_root, validate_config,
    _communicate_cancellable,
)
from optimizer_gui import __version__
import optimizer_gui.reporting as reporting
from optimizer_gui.reporting import (
    build_report_html, improvement_verdict, line_chart_data_uri,
    load_response_plot, metric_card_data, response_chart_series,
)
from optimizer_gui.warning_text import warning_info
from optimizer_gui.window import OptimizerWindow
from scripts.make_measurement_manifest import build_manifest


class GuiJobTests(unittest.TestCase):
    def test_validation_worker_drains_large_stdout_without_deadlock(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr, cancelled = _communicate_cancellable(process)
        self.assertFalse(cancelled)
        self.assertEqual(len(stdout), 200000)
        self.assertEqual(stderr, "")
    def test_run_tab_uses_scrollable_high_dpi_safe_layout(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = OptimizerWindow()
        window.resize(920, 650)
        window.tabs.setTabEnabled(window.TAB_RUN, True)
        window.tabs.setCurrentIndex(window.TAB_RUN)
        window.show()
        app.processEvents()
        try:
            self.assertGreater(window.run_scroll.verticalScrollBar().maximum(), 0)
            self.assertGreaterEqual(
                window.workflow_value.height(),
                window.workflow_value.minimumHeight(),
            )
            self.assertGreaterEqual(window.run_log.height(), window.run_log.minimumHeight())
        finally:
            window.close()

    def test_version_has_one_package_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(__version__, "0.9.3")
        self.assertEqual(project["project"]["dynamic"], ["version"])
        self.assertNotIn("version", project["project"])
        self.assertEqual(
            project["tool"]["setuptools"]["dynamic"]["version"]["attr"],
            "optimizer_gui.__version__",
        )

    def test_run_roots_are_unique_and_live_claims_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            first = timestamped_run_root(parent)
            second = timestamped_run_root(parent)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            claim_run_root(first, os.getpid())
            self.assertEqual(active_run_pid(first), os.getpid())
            with self.assertRaises(RunRootBusyError):
                claim_run_root(first, os.getpid())
            release_run_claim(first, os.getpid())
            self.assertEqual(active_run_pid(first), 0)

    def test_only_merged_verified_runner_output_counts_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker_01"
            worker.mkdir()
            (worker / "assistant_summary.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(locate_summary(root))
            self.assertFalse(runner_completed_successfully(root))

            merged = root / "_merged_top"
            merged.mkdir()
            expected = merged / "assistant_summary.json"
            expected.write_text("{}", encoding="utf-8")
            self.assertEqual(locate_summary(root), expected)
            self.assertFalse(runner_completed_successfully(root))
            (root / ".runner_success").touch()
            self.assertTrue(runner_completed_successfully(root))

    def test_runner_failure_reason_translates_checkpoint_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".runner_failed").write_text(
                "PermissionError: stream_state.json was locked", encoding="utf-8",
            )
            reason = runner_failure_reason(root)
        self.assertIn("Windows kept the state file locked", reason)
        self.assertIn("resumable", reason)

    def test_windows_runner_uses_hidden_process_group_without_detached_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "runner.log"
            with patch(
                "optimizer_gui.backend.subprocess.Popen",
                return_value=SimpleNamespace(pid=1234),
            ) as popen:
                pid = start_detached_process("powershell.exe", ["-NoProfile"], Path(tmp), log_path)
        self.assertEqual(pid, 1234)
        flags = popen.call_args.kwargs["creationflags"]
        self.assertTrue(flags & subprocess.CREATE_NEW_PROCESS_GROUP)
        self.assertTrue(flags & subprocess.CREATE_NO_WINDOW)
        self.assertFalse(flags & subprocess.DETACHED_PROCESS)

    def test_start_button_boolean_is_not_treated_as_resume_path(self) -> None:
        calls = []

        class DummyWindow:
            start_run = lambda self, *args: calls.append(args)

        OptimizerWindow._start_clicked(DummyWindow(), False)
        self.assertEqual(calls, [()])

    def test_job_round_trip_and_worker_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                "data", "base.afpx", "target.txt", tmp,
                cpu_percent=100, seed=12345,
            )
            config.save()
            loaded = RunConfig.load(Path(tmp))
            self.assertEqual(loaded.data_root, "data")
            self.assertLessEqual(loaded.workers, 12)
            self.assertEqual(loaded.seed, 12345)

    def test_phase_mode_is_single_worker_and_explicit(self) -> None:
        config = RunConfig("data", "base.afpx", "target.txt", "run", mode="phase", cpu_percent=80)
        self.assertEqual(config.workers, 1)
        _program, args = powershell_command(config, executable="C:\\python.exe")
        self.assertEqual(args[args.index("-Mode") + 1], "phase")

    def test_retarget_workflow_runs_through_peq_engine(self) -> None:
        config = RunConfig(
            "data", "current.afpx", "alternate_target.txt", "run",
            mode="peq", workflow="retarget", phase_writes="off",
        )
        self.assertEqual(config.ui_workflow, "retarget")
        _program, args = powershell_command(config, executable="C:\\python.exe")
        self.assertEqual(args[args.index("-Mode") + 1], "peq")
        self.assertEqual(args[args.index("-Target") + 1], "alternate_target.txt")
        self.assertEqual(args[args.index("-PhaseWrites") + 1], "off")

    def test_old_jobs_fall_back_to_backend_mode_for_workflow(self) -> None:
        config = RunConfig("data", "base.afpx", "target.txt", "run", mode="phase")
        self.assertEqual(config.ui_workflow, "phase")

    def test_command_passes_explicit_user_choices(self) -> None:
        config = RunConfig(
            "C:\\Measurements", "C:\\Measurements\\base.afpx", "C:\\target.txt",
            "C:\\run", voicing_variants="audition", sub_blend="recommend",
            headroom_db=3.0, seed=24680,
        )
        program, args = powershell_command(config, executable="C:\\python.exe")
        self.assertEqual(program, "powershell.exe")
        self.assertIn("audition", args)
        self.assertIn("recommend", args)
        self.assertIn("C:\\python.exe", args)
        self.assertEqual(args[args.index("-StartSeed") + 1], "24680")

    def test_command_passes_role_map_to_runner(self) -> None:
        config = RunConfig(
            "C:\\Measurements", "C:\\base.afpx", "C:\\target.txt", "C:\\run",
            role_map="C:\\run\\role_map.json",
        )
        _program, args = powershell_command(config, executable="C:\\python.exe")
        self.assertEqual(args[args.index("-RoleMap") + 1], "C:\\run\\role_map.json")

    def test_invalid_inputs_block_before_workers_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(tmp, str(Path(tmp) / "missing.afpx"), "missing.txt", str(Path(tmp) / "run"))
            result = validate_config(config)
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(len(result["errors"]), 1)
        self.assertEqual(result["diagnostics"]["job_config"]["baseline"], config.baseline)
        self.assertIn("manifest", result["diagnostics"])

    def test_unparseable_preflight_preserves_copyable_diagnostics(self) -> None:
        class DummyProcess:
            returncode = 7

            @staticmethod
            def poll():
                return 7

            @staticmethod
            def communicate(timeout=None):
                return "not-json", "worker import failed"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.afpx"
            target = root / "target.txt"
            baseline.write_bytes(b"x")
            target.write_text("20 0\n1000 0\n", encoding="utf-8")
            config = RunConfig(
                str(root), str(baseline), str(target), str(root / "run"),
            )
            manifest = {
                "measurements_missing": [],
                "baseline_exists": True,
                "target_exists": True,
            }
            with (
                patch("optimizer_gui.backend.build_manifest", return_value=manifest),
                patch("optimizer_gui.backend.compact_manifest", return_value={"missing": []}),
                patch("optimizer_gui.backend.subprocess.Popen", return_value=DummyProcess()),
            ):
                result = validate_config(config)
        self.assertFalse(result["valid"])
        self.assertEqual(result["diagnostics"]["stderr"], "worker import failed")
        self.assertEqual(result["diagnostics"]["manifest"], manifest)
        self.assertTrue(result["preflight"]["parse_failed"])

    def test_target_curve_preview_is_normalized_at_one_khz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "alternate.txt"
            target.write_text("20 6\n1000 0\n20000 -4\n", encoding="utf-8")
            curve = load_target_curve(target)
        index = curve["frequency_hz"].index(1000.0)
        self.assertEqual(curve["relative_db"][index], 0.0)
        self.assertEqual(curve["file"], "alternate.txt")

    def test_custom_measurement_names_resolve_through_role_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {
                "System Sum": "Cabin Average.txt",
                "Sub": "Sub Sweep.txt",
                "FL High": "FL High Sweep.txt",
                "FR High": "FR High Sweep.txt",
                "Tweeters Together": "High Pair.txt",
                "FL Low": "FL Door.txt",
                "FR Low": "FR Door.txt",
                "Mid Bass Together": "Doors Pair.txt",
            }
            rows = "\n".join(f"{20 + index} {70 + index / 10}" for index in range(20))
            for filename in mapping.values():
                (root / filename).write_text(rows, encoding="utf-8")
            baseline = root / "baseline.afpx"
            target = root / "target.txt"
            baseline.write_bytes(b"test")
            target.write_text("20 6\n1000 0\n20000 -4\n", encoding="utf-8")
            role_map = save_role_map(
                root / "run" / "role_map.json", mapping, "front_2way_plus_sub",
            )
            manifest = build_manifest(root, baseline, target, role_map)
        self.assertEqual(manifest["measurements_missing"], [])
        self.assertEqual(
            Path(manifest["resolved_roles"]["FL High"]).name, "FL High Sweep.txt",
        )
        self.assertEqual(manifest["detected_layout"], "front_2way_plus_sub")

    def test_declared_two_way_layout_wins_when_midbass_files_use_mid_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {
                "System Sum": "System Sum.txt",
                "Sub": "Subwoofer.txt",
                "FL High": "Front L Tweeter.txt",
                "FR High": "Front R Tweeter.txt",
                "FL Low": "Front L Mid.txt",
                "FR Low": "Front R Mid.txt",
            }
            rows = "\n".join(f"{20 + index} {70 + index / 10}" for index in range(20))
            for filename in mapping.values():
                (root / filename).write_text(rows, encoding="utf-8")
            baseline = root / "baseline.afpx"
            target = root / "target.txt"
            baseline.write_bytes(b"test")
            target.write_text("20 6\n1000 0\n20000 -4\n", encoding="utf-8")
            role_map = save_role_map(
                root / "run" / "role_map.json", mapping, "front_2way_plus_sub",
            )

            manifest = build_manifest(root, baseline, target, role_map)

        self.assertEqual(manifest["detected_layout"], "front_2way_plus_sub")
        self.assertIn("FL Low", manifest["resolved_roles"])
        self.assertNotIn("FL Mid", manifest["resolved_roles"])

    def test_together_traces_are_optional_for_peq_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = "\n".join(f"{20 + index} {70 + index / 10}" for index in range(20))
            for filename in (
                "System Sum.txt", "Sub.txt", "Front L High.txt", "Front R High.txt",
                "Front L Low.txt", "Front R Low.txt",
            ):
                (root / filename).write_text(rows, encoding="utf-8")
            baseline = root / "baseline.afpx"
            target = root / "target.txt"
            baseline.write_bytes(b"test")
            target.write_text("20 6\n1000 0\n20000 -4\n", encoding="utf-8")

            manifest = build_manifest(root, baseline, target)
            checklist = measurement_checklist(root)

        self.assertEqual(manifest["measurements_missing"], [])
        self.assertEqual(
            set(manifest["optional_missing_roles"]),
            {"Tweeters Together", "Mid Bass Together", "FL Nearfield", "FR Nearfield"},
        )
        self.assertFalse(manifest["pair_measurements_complete"])
        self.assertIn("optional_pair_measurements_missing:", " ".join(manifest["warnings"]))
        optional_rows = [row for row in checklist["rows"] if not row["required"]]
        self.assertEqual({row["role"] for row in optional_rows}, {
            "Tweeters Together", "Mid Bass Together", "FL Nearfield", "FR Nearfield",
        })

    def test_independent_objective_consumes_role_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measurement = root / "Front Left High Sweep.txt"
            measurement.write_text("20 70\n30 71\n", encoding="utf-8")
            role_map = save_role_map(
                root / "role_map.json",
                {"FL High": measurement.name},
                "front_2way_plus_sub",
            )
            objective_path = (
                Path(__file__).resolve().parents[1]
                / "objective_module" / "afpx_objective.py"
            )
            spec = importlib.util.spec_from_file_location(
                "role_map_objective_test", objective_path,
            )
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, {
                "AFPX_DATA_ROOT": str(root),
                "AFPX_ROLE_MAP": str(role_map),
            }, clear=False):
                spec.loader.exec_module(module)
            resolved = module._resolve_txt(module.SOLO_FILES["FL High"], "FL High")
        self.assertEqual(resolved.name, "Front Left High Sweep.txt")

    def test_fuzzy_role_guess_and_template_are_safe(self) -> None:
        self.assertEqual(
            suggest_measurement_role("Front Left High Sweep.txt"), "FL High",
        )
        self.assertEqual(suggest_measurement_role("FL High.txt"), "FL High")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "template"
            created = create_measurement_template(folder, "front_2way_plus_sub")
            self.assertTrue((folder / "System Sum.txt").exists())
            self.assertEqual((folder / "System Sum.txt").stat().st_size, 0)
            self.assertTrue(any(path.name.startswith("README") for path in created))
            with self.assertRaises(FileExistsError):
                create_measurement_template(folder, "front_2way_plus_sub")

    def test_warning_tokens_include_plain_fix_and_severity(self) -> None:
        warning = warning_info("measurement_source_volume_changed")
        self.assertEqual(warning["severity"], "error")
        self.assertIn("without touching the volume knob", warning["text"])
        dynamic = warning_info("missing_required_measurements:3")
        self.assertIn("Details: 3", dynamic["text"])
        optional = warning_info("optional_pair_measurements_missing:Tweeters Together")
        self.assertEqual(optional["severity"], "warning")
        self.assertIn("You may continue with PEQ", optional["text"])

    def test_results_chart_uses_fixed_anchor_response_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "assistant_summary.json"
            summary_path.write_text(json.dumps({
                "best": {"fixed_anchor_response": {"checkpoints": [
                    {"frequency_hz": 100.0, "baseline_error_db": 3.0,
                     "candidate_error_db": 1.0, "raw_system_delta_db": -2.0},
                    {"frequency_hz": 1000.0, "baseline_error_db": -2.0,
                     "candidate_error_db": -1.0, "raw_system_delta_db": 1.0},
                ]}},
            }), encoding="utf-8")
            plot = load_response_plot(summary_path)
        self.assertEqual(plot["candidate_error_db"], [1.0, -1.0])
        chart = line_chart_data_uri([{
            "label": "Candidate", "x": plot["frequency_hz"],
            "y": plot["candidate_error_db"], "color": "#16805d",
        }])
        self.assertTrue(chart.startswith("data:image/png;base64,"))
        series = response_chart_series({
            **plot,
            "drivers": {
                "FL High": {
                    "frequency_hz": [100.0, 1000.0],
                    "change_db": [-1.0, 0.5],
                },
            },
        })
        self.assertEqual([row["label"] for row in series[:3]], [
            "Before", "Candidate", "Target",
        ])
        self.assertFalse(series[3]["visible"])

    def test_results_metrics_flag_negligible_improvement(self) -> None:
        baseline = {
            "objective": 10.0, "tonal_error_db": 2.0,
            "presence_error_db": 2.0, "narrow_peak_penalty_db": 1.0,
            "balance_penalty_db": 1.0,
        }
        best = dict(baseline)
        best["objective"] = 9.95
        verdict = improvement_verdict(baseline, best)
        self.assertFalse(verdict["meaningful"])
        self.assertIn("Not meaningfully better", verdict["heading"])
        self.assertNotIn("%", verdict["detail"])
        best["tonal_error_db"] = 1.5
        cards = metric_card_data(baseline, best)
        self.assertEqual(cards[0]["state"], "good")

    def test_progress_and_candidates_read_compact_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".phase_preparing").touch()
            self.assertEqual(collect_progress(root)["phase"], "preparing")
            worker = root / "worker_01"
            worker.mkdir()
            (worker / "stream_state.json").write_text(json.dumps({
                "completed_trials": 42, "elapsed_seconds": 5,
                "best": [{"objective": 3.25}],
                "convergence": {
                    "verdict": "still_improving",
                    "stalled_seconds": 1.5,
                    "events": [{"elapsed_seconds": 4.0, "objective": 3.25}],
                },
            }), encoding="utf-8")
            (root / ".phase_searching").touch()
            progress = collect_progress(root)
            self.assertEqual(progress["trials"], 42)
            self.assertEqual(progress["best_objective"], 3.25)
            self.assertEqual(progress["phase"], "searching")
            self.assertEqual(progress["convergence"]["verdict"], "still_improving")
            self.assertEqual(progress["convergence"]["events"][0]["objective"], 3.25)

            (root / ".phase_merging").touch()
            self.assertEqual(collect_progress(root)["phase"], "merging")

            merged = root / "_merged_top"
            merged.mkdir()
            (merged / "family_balanced.afpx").write_bytes(b"x")
            summary_path = merged / "assistant_summary.json"
            summary = {
                "baseline": {"objective": 2.5},
                "inputs": {"baseline": {"file": "baseline.afpx"}},
                "families": {
                    "balanced": {
                        "file": "family_balanced.afpx", "objective": 2.0,
                    },
                },
            }
            rows = candidate_files(summary, summary_path)
            self.assertEqual(rows[0]["role"], "Current tune (baseline)")
            self.assertFalse(rows[0]["exportable"])
            self.assertEqual(rows[1]["role"], "Balanced")

    def test_listening_decision_links_latest_achieved_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verification = root / "verification"
            verification.mkdir()
            (verification / "verification_20260731_010203.json").write_text(
                json.dumps({
                    "verdict": "confirmed",
                    "system": {"difference_rms_db": 0.18},
                }),
                encoding="utf-8",
            )
            path = record_run_decision(
                root, "family_balanced.afpx", "Kept - clearly better",
                "Stable centre image.",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry = payload["entries"][0]
            self.assertEqual(entry["candidate"], "family_balanced.afpx")
            self.assertEqual(entry["verification"]["verdict"], "confirmed")
            self.assertEqual(entry["verification"]["system_difference_rms_db"], 0.18)

    def test_export_uses_run_timestamp_and_never_clobbers_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "family_balanced.afpx"
            source.write_bytes(b"candidate")
            destination = root / "exports"
            name = default_export_name(
                source, "Balanced", "2026-07-30T21:15:30",
            )
            self.assertEqual(name, "20260730_211530_balanced.afpx")
            target = export_candidate(source, destination, filename=name)
            self.assertEqual(target.read_bytes(), b"candidate")
            with self.assertRaises(FileExistsError):
                export_candidate(source, destination, filename=name)
            source.write_bytes(b"replacement")
            export_candidate(source, destination, filename=name, overwrite=True)
            self.assertEqual(target.read_bytes(), b"replacement")

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
            "warnings": ["phase_writes_disabled_timing_reference_missing"],
            "gates": {"measurement_session": {"phase_valid": True}},
        }
        report = build_report_html(summary, {}, Path("assistant_summary.json"))
        self.assertIn("Phase / Timing Diagnostic", report)
        self.assertIn("Vocal / presence error", report)
        self.assertIn("delay -12 samples", report)
        self.assertIn("destructive, not EQ-able", report)
        self.assertIn("Re-measure using REW acoustic timing reference", report)

    def test_peq_report_explains_existing_filter_operations(self) -> None:
        summary = {
            "search": {"mode": "peq"},
            "baseline": {"objective": 10.0},
            "best": {
                "file": "candidate.afpx",
                "objective": 7.0,
                "components": {"objective": 7.0},
            },
            "rehabilitation": {
                "verdict": "meaningful_improvement",
                "evaluation_count": 321,
                "cache_source": "rehabilitation_cache.json",
                "operation_counts": {
                    "modify": 1, "remove": 1, "merge": 0, "append": 0,
                },
                "accepted_operations": [{
                    "operation": "modify",
                    "channel_role": "FL Low",
                    "slot": 7,
                    "old": {
                        "frequency_hz": 97.0, "q": 3.0, "gain_db": -1.5,
                    },
                    "new": {
                        "frequency_hz": 100.0, "q": 1.2, "gain_db": -1.5,
                    },
                    "reason": "Re-centred because the revised band cleared every hard gate.",
                }],
                "component_deltas": {
                    "baseline_to_rehabilitated": {"tonal_error_db": -0.8},
                },
                "headroom": {
                    "supplied": {"2": 2.0},
                    "rehabilitated": {"2": 2.5},
                    "final": {"2": 2.5},
                },
            },
            "gates": {"measurement_session": {}},
        }

        report = build_report_html(summary, {}, Path("assistant_summary.json"))

        self.assertIn("Existing Tune Rehabilitation", report)
        self.assertIn("97.0 Hz", report)
        self.assertIn("100.0 Hz", report)
        self.assertIn("Re-centred because", report)
        self.assertIn("321", report)
    def test_tie_report_uses_repeatability_language_without_percentage(self) -> None:
        summary = {
            "search": {"mode": "peq"},
            "baseline": {
                "objective": 10.0,
                "tonal_error_db": 2.0,
                "presence_error_db": 2.0,
                "narrow_peak_penalty_db": 1.0,
                "balance_penalty_db": 1.0,
            },
            "best": {
                "file": "candidate.afpx",
                "objective": 9.99,
                "components": {
                    "objective": 9.99,
                    "tonal_error_db": 1.99,
                    "presence_error_db": 2.0,
                    "narrow_peak_penalty_db": 1.0,
                    "balance_penalty_db": 1.0,
                },
            },
            "rehabilitation": {
                "verdict": "no_meaningful_improvement",
                "comparison_stages": [{"key": "supplied", "label": "Supplied"}],
            },
        }

        report = build_report_html(summary, {}, Path("assistant_summary.json"))

        self.assertIn("No meaningful improvement", report)
        self.assertNotIn("0% better", report)
        self.assertNotIn("Objective improvement</td><td><b>0.1%", report)
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




class RehabilitationCacheLaunchTests(unittest.TestCase):
    def test_peq_runner_receives_one_shared_rehabilitation_cache_path(self):
        config = RunConfig(
            "C:\\Measurements", "C:\\baseline.afpx", "C:\\target.txt",
            "C:\\run", mode="peq",
        )

        _program, args = powershell_command(config, executable="C:\\python.exe")

        self.assertEqual(args.count("-RehabilitationCache"), 1)
        self.assertEqual(
            args[args.index("-RehabilitationCache") + 1],
            str(Path(config.run_root) / "rehabilitation_cache.json"),
        )

    def test_phase_runner_receives_no_rehabilitation_cache(self):
        config = RunConfig(
            "C:\\Measurements", "C:\\baseline.afpx", "C:\\target.txt",
            "C:\\run", mode="phase",
        )

        _program, args = powershell_command(config, executable="C:\\python.exe")

        self.assertNotIn("-RehabilitationCache", args)

    def test_launcher_passes_shared_stop_file_to_cache_preparation(self):
        script = (
            Path(__file__).resolve().parents[1] / "run_guided_stream_workers.ps1"
        ).read_text(encoding="utf-8-sig")
        preparation = script.split("scripts\\build_rehabilitation_cache.py", 1)[1]
        preparation = preparation.split("& $pythonExe @rehabilitationArgs", 1)[0]

        self.assertIn('"--stop-file", $stopFilePath', preparation)
    def test_windows_paths_with_spaces_remain_single_runner_arguments(self):
        config = RunConfig(
            "C:\\Fresh Measurements\\RTA session",
            "C:\\Fresh Measurements\\baseline tune.afpx",
            "C:\\Target Curves\\target curve.txt",
            "C:\\Optimizer Runs\\run one",
            mode="peq",
        )

        _program, args = powershell_command(
            config, executable="C:\\Python Runtime\\python.exe"
        )

        self.assertEqual(args[args.index("-DataRoot") + 1], config.data_root)
        self.assertEqual(args[args.index("-Baseline") + 1], config.baseline)
        self.assertEqual(args[args.index("-Root") + 1], config.run_root)


class RehabilitationResultsTests(unittest.TestCase):
    def test_candidate_rows_include_distinct_rehabilitated_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rehabilitated_baseline.afpx").write_bytes(b"rehabilitated")
            (root / "candidate.afpx").write_bytes(b"final")
            summary_path = root / "assistant_summary.json"
            summary = {
                "baseline": {"objective": 10.0},
                "inputs": {"baseline": {"file": "baseline.afpx"}},
                "rehabilitation": {
                    "verdict": "meaningful_improvement",
                    "file": "rehabilitated_baseline.afpx",
                    "objective": 8.0,
                    "comparison_stages": [
                        {"key": "supplied", "label": "Supplied"},
                        {"key": "rehabilitated", "label": "Existing tune improved"},
                        {"key": "final", "label": "Final"},
                    ],
                },
                "best": {"file": "candidate.afpx", "objective": 7.0},
            }
            rows = candidate_files(summary, summary_path)

        self.assertEqual(
            [row["role"] for row in rows[:3]],
            ["Supplied", "Existing tune improved", "Final"],
        )
        self.assertEqual(
            [row["comparison_stage"] for row in rows[:3]],
            ["supplied", "rehabilitated", "final"],
        )

    def test_candidate_rows_omit_unchanged_final_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate.afpx").write_bytes(b"unchanged")
            summary_path = root / "assistant_summary.json"
            summary = {
                "baseline": {"objective": 10.0},
                "inputs": {"baseline": {"file": "baseline.afpx"}},
                "rehabilitation": {
                    "verdict": "no_meaningful_improvement",
                    "comparison_stages": [
                        {"key": "supplied", "label": "Supplied"},
                    ],
                },
                "best": {"file": "candidate.afpx", "objective": 9.99},
            }
            rows = candidate_files(summary, summary_path)

        self.assertEqual([row["role"] for row in rows], ["Supplied"])
    def test_results_window_loads_operation_markers_for_staged_run(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate.afpx").write_bytes(b"final")
            (root / "rehabilitated.afpx").write_bytes(b"rehabilitated")
            summary_path = root / "assistant_summary.json"
            summary_path.write_text(json.dumps({
                "baseline": {"objective": 10.0},
                "best": {
                    "file": "candidate.afpx",
                    "objective": 7.0,
                    "components": {"objective": 7.0},
                },
                "rehabilitation": {
                    "verdict": "meaningful_improvement",
                    "file": "rehabilitated.afpx",
                    "objective": 8.0,
                    "comparison_stages": [
                        {"key": "supplied", "label": "Supplied"},
                        {"key": "rehabilitated", "label": "Existing tune improved"},
                        {"key": "final", "label": "Final"},
                    ],
                    "accepted_operations": [{
                        "operation": "append",
                        "group": "low_sym",
                        "old": None,
                        "new": {"frequency_hz": 180.0, "q": 0.8, "gain_db": -1.0},
                        "reason": "Supported residual correction.",
                    }],
                },
                "details": {"optimizer_summary": "optimizer_summary.json"},
                "search": {"mode": "peq"},
            }), encoding="utf-8")
            (root / "optimizer_summary.json").write_text(json.dumps({
                "response_comparisons": {
                    key: {
                        "frequency_hz": [100.0, 1000.0],
                        "baseline_error_db": [3.0, -2.0],
                        "candidate_error_db": [2.0, -1.0],
                        "raw_system_delta_db": [-1.0, 1.0],
                    }
                    for key in ("supplied", "rehabilitated", "final")
                },
            }), encoding="utf-8")
            window = OptimizerWindow()
            window._start_report_generation = lambda _path: None
            try:
                window.load_results(summary_path)
                app.processEvents()
                self.assertEqual(window.result_table.rowCount(), 3)
                window.result_table.selectRow(2)
                app.processEvents()
                self.assertEqual(window.result_filters[0][1], 180.0)
            finally:
                window.close()
    def test_stage_selection_returns_supplied_rehabilitated_and_final_curves(self):
        full = {
            "response_comparisons": {
                "supplied": {
                    "frequency_hz": [100.0, 1000.0],
                    "candidate_error_db": [3.0, -2.0],
                },
                "rehabilitated": {
                    "frequency_hz": [100.0, 1000.0],
                    "candidate_error_db": [1.5, -1.5],
                },
                "final": {
                    "frequency_hz": [100.0, 1000.0],
                    "candidate_error_db": [1.0, -1.0],
                },
            },
        }

        plots = reporting.result_comparison_plots({}, full)

        self.assertEqual(set(plots), {"supplied", "rehabilitated", "final"})
        self.assertEqual(plots["rehabilitated"]["candidate_error_db"], [1.5, -1.5])
        self.assertEqual(
            response_chart_series(plots["final"])[1]["label"], "Final",
        )
if __name__ == "__main__":
    unittest.main()
