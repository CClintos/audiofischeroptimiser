from __future__ import annotations

import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _optimizer as optimizer
import _optimizer_stream as stream
from objective_module import afpx_objective as objective
from scripts.summarise_optimizer_run import summarise


def _manifest(source_volumes, timing_references=None, phase=True):
    timing_references = timing_references or ["Rear R"] * len(source_volumes)
    roles = ["System Sum", "Sub", "FL High", "FR High", "Tweeters Together",
             "FL Low", "FR Low", "Mid Bass Together"]
    metadata = {}
    resolved = {}
    for role, volume, timing in zip(roles, source_volumes, timing_references):
        metadata[role] = {
            "source_volume": volume,
            "sweep_dbfs": -12.0,
            "timing_reference": timing,
        }
        resolved[role] = f"C:/measurements/{role}.txt"
    return {
        "measurement_metadata": metadata,
        "resolved_roles": resolved,
        "measurements_missing": [],
        "measurement_conditions": {
            "timing_references": sorted({item for item in timing_references if item}),
        },
        "phase_available": phase,
        "warnings": [],
    }


class ObjectiveInvariantTests(unittest.TestCase):
    def test_positive_deviation_has_extra_peak_cost(self) -> None:
        freqs = np.geomspace(80.0, 12000.0, 256)
        valid = np.ones_like(freqs, dtype=bool)
        positive = objective.tonal_components(freqs, np.full_like(freqs, 3.0), valid)
        negative = objective.tonal_components(freqs, np.full_like(freqs, -3.0), valid)

        self.assertAlmostEqual(positive["tonal_masked"], negative["tonal_masked"], places=12)
        self.assertGreater(positive["peak_penalty_db"], 0.0)
        self.assertEqual(negative["peak_penalty_db"], 0.0)

    def test_balance_bias_cannot_hide_sign_changing_mismatch(self) -> None:
        freqs = np.geomspace(200.0, 5000.0, 256)
        difference = np.where(np.arange(len(freqs)) % 2, -5.0, 5.0)
        parts = objective.balance_components(freqs, difference, (200.0, 5000.0))

        self.assertLess(abs(parts["bias_db"]), 0.01)
        self.assertAlmostEqual(parts["mismatch_rms_db"], 5.0, places=10)
        self.assertAlmostEqual(parts["mismatch_abs_db"], 5.0, places=10)

    def test_tonal_presence_and_peak_components_are_distinct(self) -> None:
        freqs = np.geomspace(60.0, 16000.0, 512)
        deviation = np.zeros_like(freqs)
        deviation[(freqs >= 300.0) & (freqs <= 2000.0)] = 4.0
        deviation[freqs >= 9000.0] = -6.0
        parts = objective.tonal_components(freqs, deviation, np.ones_like(freqs, dtype=bool))

        self.assertNotEqual(parts["tonal_masked"], parts["sum_tonal_anchor_db"])
        self.assertNotEqual(parts["presence_error_db"], parts["peak_penalty_db"])


class MeasurementSessionGateTests(unittest.TestCase):
    def test_tonal_mode_rejects_uncalibrated_level_change(self) -> None:
        manifest = _manifest([0.90] * 7 + [0.75])
        audit = optimizer.measurement_session_audit(manifest, {})
        self.assertFalse(audit["tonal_valid"])
        self.assertEqual(audit["missing_calibration_roles"], ["Mid Bass Together"])

    def test_explicit_level_calibration_allows_tonal_mode(self) -> None:
        manifest = _manifest([0.90] * 7 + [0.75])
        audit = optimizer.measurement_session_audit(manifest, {"Mid Bass Together": 2.4})
        self.assertTrue(audit["tonal_valid"])
        self.assertTrue(audit["phase_valid"])

    def test_mixed_timing_reference_disables_phase_writes_only(self) -> None:
        refs = ["Rear R"] * 7 + ["Rear L"]
        audit = optimizer.measurement_session_audit(_manifest([0.90] * 8, refs), {})
        self.assertTrue(audit["tonal_valid"])
        self.assertFalse(audit["phase_valid"])

    def test_missing_level_provenance_requires_explicit_calibration(self) -> None:
        manifest = _manifest([0.90] * 8)
        manifest["measurement_metadata"]["Sub"]["source_volume"] = None
        manifest["measurement_metadata"]["Sub"]["sweep_dbfs"] = None
        audit = optimizer.measurement_session_audit(manifest, {})
        self.assertFalse(audit["tonal_valid"])
        self.assertIn("Sub", audit["missing_calibration_roles"])

    def test_phase_requires_one_named_timing_reference(self) -> None:
        manifest = _manifest([0.90] * 8, [""] * 8)
        audit = optimizer.measurement_session_audit(manifest, {})
        self.assertTrue(audit["tonal_valid"])
        self.assertFalse(audit["phase_valid"])


class PhasePeqProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.geomspace(20.0, 20000.0, 2048)
        self.plan = [{
            "source": "Left mid to tweeter",
            "crossover_channels": (0, 2),
            "crossover_band": (1800.0, 4500.0),
        }]

    def test_rejects_peq_that_changes_written_crossover(self) -> None:
        groups = {name: [] for name in optimizer.GROUPS}
        group = next(name for name, spec in optimizer.GROUPS.items() if 0 in spec["channels"])
        groups[group] = [(3000.0, 1.0, -3.0)]
        conflicts = optimizer.phase_peq_conflicts(self.freqs, groups, self.plan)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["source"], "Left mid to tweeter")

    def test_allows_peq_outside_written_crossover(self) -> None:
        groups = {name: [] for name in optimizer.GROUPS}
        group = next(name for name, spec in optimizer.GROUPS.items() if 0 in spec["channels"])
        groups[group] = [(500.0, 1.0, -3.0)]
        self.assertEqual(optimizer.phase_peq_conflicts(self.freqs, groups, self.plan), [])


class RunIntegrityTests(unittest.TestCase):
    def test_v4_resume_state_is_rescored_with_current_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream_state.json"
            groups = {name: [] for name in optimizer.GROUPS}
            payload = {
                "version": 4,
                "completed_trials": 12,
                "elapsed_seconds": 4.0,
                "best": [{"objective": -999.0, "groups": groups}],
                "archive": [],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            rng = np.random.default_rng(1)
            best, _archive, _scores, trials, _elapsed = stream.load_state(
                path, rng, lambda _groups: {"objective": 7.25}, 10
            )
            self.assertEqual(best[0][0], 7.25)
            self.assertEqual(trials, 12)

    def test_summariser_prefers_assistant_decision_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "assistant_summary.json").write_text(
                json.dumps({"schema": "assistant", "best": {"objective": 1.0}}), encoding="utf-8"
            )
            (root / "optimizer_summary.json").write_text(
                json.dumps({"schema": "full", "best": {"objective": 2.0}}), encoding="utf-8"
            )
            self.assertEqual(summarise(root, 5)["schema"], "assistant")


class ModernGoldenBenchmarkTests(unittest.TestCase):
    def test_modern_txt_and_afpx_golden_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freqs = np.geomspace(30.0, 20000.0, 384)
            x = np.log2(freqs / 500.0)
            traces = {
                "FL High": 64.0 + 1.2 * np.sin(x),
                "FR High": 63.3 - 0.8 * np.sin(x * 0.8),
                "FL Low": 67.5 + 1.4 * np.cos(x * 0.7),
                "FR Low": 66.6 - 1.1 * np.cos(x * 0.6),
                "Sub": 70.0 - 8.0 * np.maximum(np.log2(freqs / 90.0), 0.0),
            }

            def power_sum(*values):
                return 10.0 * np.log10(sum(10.0 ** (value / 10.0) for value in values))

            traces["Tweeters Together"] = power_sum(traces["FL High"], traces["FR High"])
            traces["Mid Bass Together"] = power_sum(traces["FL Low"], traces["FR Low"])
            traces["System Sum"] = power_sum(
                traces["Sub"], traces["Tweeters Together"], traces["Mid Bass Together"]
            )
            filenames = {
                "FL High": "Front L High.txt", "FR High": "Front R High.txt",
                "FL Low": "Front L Low.txt", "FR Low": "Front R Low.txt",
                "Sub": "Sub.txt", "System Sum": "System Sum.txt",
                "Tweeters Together": "Tweeters Together.txt",
                "Mid Bass Together": "Mid Bass Together.txt",
            }
            for role, filename in filenames.items():
                rows = ["* volume: 0.90", "* sweeps at -12 dBFS", "* reference played from Rear R"]
                rows.extend(
                    f"{f:.9f} {s:.9f} 0.0 0.99 1" for f, s in zip(freqs, traces[role])
                )
                (root / filename).write_text("\n".join(rows), encoding="utf-8")

            target = root / "target.txt"
            target.write_text("\n".join(
                f"{f:.9f} {75.0 - 4.0 * np.log10(f / 100.0):.9f}" for f in freqs
            ), encoding="utf-8")
            baseline = root / "baseline.afpx"
            xml = "<Root>" + "".join("<OC></OC>" for _ in range(8)) + "</Root>"
            baseline.write_bytes(b"AFPX" + zlib.compress(xml.encode("utf-8")))

            solo_files = {key: (Path(name).stem,) for key, name in filenames.items()}
            pair_specs = {
                "low": ("FL Low", "FR Low", "Mid Bass Together", (80.0, 2600.0), (200.0, 2000.0)),
                "high": ("FL High", "FR High", "Tweeters Together", (2600.0, 16000.0), (2800.0, 16000.0)),
            }
            with patch.multiple(
                objective,
                REW_DIR=root,
                TARGET=target,
                BASELINE_AFPX=baseline,
                LEVEL_CALIBRATION={},
                SOLO_FILES=solo_files,
                PAIR_SPECS=pair_specs,
                CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
                _F=None,
                _T={},
                _TGT=None,
                _NULL_MASK=None,
                _V5=None,
            ):
                result = objective.score_bands([[] for _ in range(8)])

            golden_path = Path(__file__).parent / "fixtures" / "objective_golden.json"
            golden = json.loads(golden_path.read_text(encoding="utf-8"))
            for key, expected in golden.items():
                self.assertAlmostEqual(float(result[key]), float(expected), places=8, msg=key)
            self.assertNotEqual(result["objective"], round(result["objective"], 4))


if __name__ == "__main__":
    unittest.main()
