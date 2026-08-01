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
import _merge_stream_results as merge_results
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
    def test_matched_front_voicing_gets_uniform_protective_trim(self) -> None:
        freqs = np.geomspace(20.0, 20000.0, 512)
        token = (len(freqs), float(freqs[0]), float(freqs[-1]), hash(freqs.tobytes()))
        baseline = [[] for _ in range(8)]
        candidate = [[] for _ in range(8)]
        for index in range(4):
            candidate[index] = [(2600.0, 1.1, 5.5)]
        objective._cached_peaking.cache_clear()
        with patch.multiple(
            objective,
            _F=freqs,
            _GRID_TOKEN=token,
            _V5=baseline,
            _BASE_CASCADES=[np.zeros_like(freqs) for _ in range(8)],
            _BASE_OUTPUT_DB=[-5.0] * 8,
            CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
        ):
            plan = objective.output_trim_plan(candidate)

        self.assertEqual(set(plan), {0, 1, 2, 3})
        self.assertEqual(len(set(plan.values())), 1)
        self.assertLess(plan[0], 0.0)
        self.assertAlmostEqual(plan[0] * 4.0, round(plan[0] * 4.0), places=10)

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

    def test_target_shape_error_is_anchor_independent(self) -> None:
        freqs = np.geomspace(800.0, 6000.0, 512)
        shape = 4.0 * np.exp(-0.5 * (np.log2(freqs / 2600.0) / 0.45) ** 2)
        valid = np.ones_like(freqs, dtype=bool)
        first = objective.tonal_components(freqs, shape, valid)
        shifted = objective.tonal_components(freqs, shape + 12.0, valid)

        self.assertAlmostEqual(
            first["target_shape_error_db"], shifted["target_shape_error_db"], places=10
        )

    def test_fixed_anchor_audit_dilutes_unilateral_eq_in_combined_pair(self) -> None:
        freqs = np.array([300.0, 424.3, 600.0, 848.5, 1200.0])
        left = np.full_like(freqs, 60.0)
        right = np.full_like(freqs, 60.0)
        inactive = np.full_like(freqs, -100.0)
        low_together = optimizer.power_sum_db([left, right])
        high_together = optimizer.power_sum_db([inactive, inactive])
        system = optimizer.power_sum_db([low_together, high_together, inactive])
        traces = {
            "FL High": inactive,
            "FR High": inactive,
            "FL Low": left,
            "FR Low": right,
            "Sub": inactive,
            "Tweeters Together": high_together,
            "Mid Bass Together": low_together,
            "System Sum": system,
        }
        baseline = [[] for _ in range(8)]
        candidate = [[] for _ in range(8)]
        candidate[2] = [(600.0, 1.0, -6.0)]
        with patch.multiple(
            objective,
            _F=freqs,
            _T=traces,
            _TGT=system + 2.0,
            _V5=baseline,
            _BASE_CASCADES=[np.zeros_like(freqs) for _ in range(8)],
            _TOTAL_DB=system,
            _SMOOTHER=None,
        ):
            audit = objective.response_audit(candidate)
            plot = objective.report_plot_data(candidate, max_points=5)

        center = next(row for row in audit["checkpoints"] if row["frequency_hz"] == 600.0)
        expected_pair_delta = 10.0 * np.log10(10.0 ** (-6.0 / 10.0) + 1.0) - 10.0 * np.log10(2.0)
        self.assertAlmostEqual(center["pair_delta_db"]["low"], expected_pair_delta, places=3)
        self.assertNotAlmostEqual(center["pair_delta_db"]["low"], -6.0, places=1)
        self.assertAlmostEqual(
            center["candidate_error_db"] - center["baseline_error_db"],
            center["raw_system_delta_db"],
            places=3,
        )
        self.assertEqual(
            audit["anchor_policy"], "target_anchored_once_from_baseline_system_sum"
        )
        center_index = plot["frequency_hz"].index(600.0)
        self.assertAlmostEqual(
            plot["candidate_error_db"][center_index] - plot["baseline_error_db"][center_index],
            plot["raw_system_delta_db"][center_index],
            places=3,
        )


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

    def test_missing_optional_pair_keeps_tonal_valid_but_disables_phase(self) -> None:
        manifest = _manifest([0.90] * 8)
        manifest["optional_missing_roles"] = ["Tweeters Together"]
        manifest["pair_measurements_complete"] = False
        audit = optimizer.measurement_session_audit(manifest, {})
        self.assertTrue(audit["tonal_valid"])
        self.assertFalse(audit["phase_valid"])
        self.assertIn("phase_writes_disabled_pair_measurements_missing", audit["warnings"])

    def test_missing_pair_is_synthesized_and_validation_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freqs = np.geomspace(20.0, 20000.0, 64)
            paths = {
                "System Sum": root / "System Sum.txt",
                "Sub": root / "Sub.txt",
                "FL High": root / "FL High.txt",
                "FR High": root / "FR High.txt",
                "Tweeters Together": root / "missing high pair.txt",
                "FL Low": root / "FL Low.txt",
                "FR Low": root / "FR Low.txt",
                "Mid Bass Together": root / "Mid Bass Together.txt",
            }
            for role, path in paths.items():
                if role == "Tweeters Together":
                    continue
                levels = np.full_like(freqs, 60.0 if role != "Sub" else 45.0)
                path.write_text(
                    "\n".join(f"{f:.6f} {level:.6f}" for f, level in zip(freqs, levels)),
                    encoding="utf-8",
                )
            pair_defs = {
                "high": {
                    "left": "FL High", "right": "FR High",
                    "together": "Tweeters Together", "branch_band": (2000.0, 16000.0),
                },
                "low": {
                    "left": "FL Low", "right": "FR Low",
                    "together": "Mid Bass Together", "branch_band": (80.0, 2000.0),
                },
            }
            with patch.multiple(
                optimizer,
                MEASUREMENT_FILES=paths,
                OPTIONAL_PAIR_ROLES={"Tweeters Together", "Mid Bass Together"},
                PAIR_DEFS=pair_defs,
            ):
                loaded_freqs, traces, rich = optimizer.load_measurements({})
                validation = optimizer.pair_sum_validation(loaded_freqs, traces)

        expected = optimizer.power_sum_db([traces["FL High"], traces["FR High"]])
        np.testing.assert_allclose(traces["Tweeters Together"], expected)
        self.assertIn("synthetic_pair", rich["Tweeters Together"])
        high_row = next(row for row in validation if row["pair"] == "high")
        self.assertIsNone(high_row["pass"])
        self.assertFalse(high_row["available"])


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
        group = next(
            name for name, spec in optimizer.GROUPS.items()
            if 0 in spec["channels"] and not spec.get("system_transfer")
        )
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


class CensusHardGateTests(unittest.TestCase):
    """DEFECT 4b: the pre-search census was computed and reported ("No
    eligible correction centres were found") but nothing gated the run's
    actual output on it - a candidate could still be selected and written
    anyway. See CHANGELOG.md."""

    def test_empty_worth_fixing_is_detected(self) -> None:
        self.assertFalse(merge_results.census_found_nothing_eligible([]))
        self.assertFalse(merge_results.census_found_nothing_eligible(
            [{"problem_census": {"worth_fixing": [{"group": "sub", "frequency_hz": 33.0}]}}]
        ))
        self.assertTrue(merge_results.census_found_nothing_eligible(
            [{"problem_census": {"worth_fixing": []}}]
        ))
        self.assertTrue(merge_results.census_found_nothing_eligible(
            [{"problem_census": {"worth_fixing": []}}, {"problem_census": {"worth_fixing": []}}]
        ))

    def test_gate_discards_every_non_baseline_candidate(self) -> None:
        items = [
            (7.48, "sig1", {"fl_high": [(2650.0, 2.0, -1.5)]}, "worker_00"),
            (7.50, "sig2", {}, "worker_01"),
            (7.51, "sig3", {}, "baseline"),
        ]
        gated = merge_results.apply_census_gate(items, gate_active=True)
        self.assertEqual(gated, [items[2]])
        self.assertEqual([item[3] for item in gated], ["baseline"])

    def test_gate_is_a_no_op_when_census_found_something(self) -> None:
        items = [
            (7.48, "sig1", {"fl_high": [(2650.0, 2.0, -1.5)]}, "worker_00"),
            (7.51, "sig3", {}, "baseline"),
        ]
        gated = merge_results.apply_census_gate(items, gate_active=False)
        self.assertEqual(gated, items)


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

    def _build_synthetic_measurement_root(
        self, root, with_ear_positions=False, baseline_channel_filters=None,
    ):
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
        if with_ear_positions:
            for prefix, shift in (("Left Ear ", 1.5), ("Right Ear ", -1.5)):
                rows = ["* volume: 0.90", "* sweeps at -12 dBFS", "* reference played from Rear R"]
                rows.extend(
                    f"{f:.9f} {s + shift:.9f} 0.0 0.99 1"
                    for f, s in zip(freqs, traces["System Sum"])
                )
                (root / f"{prefix}System Sum.txt").write_text("\n".join(rows), encoding="utf-8")

        target = root / "target.txt"
        target.write_text("\n".join(
            f"{f:.9f} {75.0 - 4.0 * np.log10(f / 100.0):.9f}" for f in freqs
        ), encoding="utf-8")
        baseline = root / "baseline.afpx"
        baseline_channel_filters = baseline_channel_filters or {}
        channels_xml = []
        for index in range(8):
            fils = "".join(
                f'<Fil F="{f}" Q="{q}" G="{g}" T="17"/>'
                for f, q, g in baseline_channel_filters.get(index, [])
            )
            channels_xml.append(f"<OC>{fils}</OC>")
        xml = "<Root>" + "".join(channels_xml) + "</Root>"
        baseline.write_bytes(b"AFPX" + zlib.compress(xml.encode("utf-8")))
        return target, baseline, filenames

    def _score_synthetic(self, with_ear_positions):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, baseline, filenames = self._build_synthetic_measurement_root(
                root, with_ear_positions,
            )
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
                return objective.score_bands([[] for _ in range(8)])

    def test_worst_position_weight_is_zero_with_one_position(self) -> None:
        """A single-position (centre-only) session must not let
        spatial_worst_db double-count the tonal term it is derived from.
        Regression for the false 0.4% "win" that was entirely this duplicate
        (see CHANGELOG.md)."""
        one_position = self._score_synthetic(with_ear_positions=False)
        self.assertEqual(one_position["spatial_position_count"], 1)
        self.assertEqual(one_position["active_weights"]["worst"], 0.0)

        three_positions = self._score_synthetic(with_ear_positions=True)
        self.assertEqual(three_positions["spatial_position_count"], 3)
        self.assertEqual(
            three_positions["active_weights"]["worst"], objective.W["worst"],
        )

    def test_target_rms_reported_both_with_and_without_masked_nulls(self) -> None:
        """A masked (null-excluded) win must never be reportable as the only
        number: a candidate can improve target_rms_null_excluded_db while its
        filter skirts spill enough unrequested boost outside the mask to
        worsen target_rms_null_included_db. Both must be visible."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, baseline, filenames = self._build_synthetic_measurement_root(root)
            solo_files = {key: (Path(name).stem,) for key, name in filenames.items()}
            pair_specs = {
                "low": ("FL Low", "FR Low", "Mid Bass Together", (80.0, 2600.0), (200.0, 2000.0)),
                "high": ("FL High", "FR High", "Tweeters Together", (2600.0, 16000.0), (2800.0, 16000.0)),
            }
            with patch.multiple(
                objective,
                REW_DIR=root, TARGET=target, BASELINE_AFPX=baseline, LEVEL_CALIBRATION={},
                SOLO_FILES=solo_files, PAIR_SPECS=pair_specs,
                CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
                _F=None, _T={}, _TGT=None, _NULL_MASK=None, _V5=None,
            ):
                no_mask = objective.score_bands([[] for _ in range(8)])
                self.assertIn("target_rms_null_excluded_db", no_mask)
                self.assertIn("target_rms_null_included_db", no_mask)
                # With no destructive-summing evidence in this synthetic fixture,
                # the null mask is empty and both numbers must agree exactly.
                self.assertAlmostEqual(
                    no_mask["target_rms_null_excluded_db"],
                    no_mask["target_rms_null_included_db"],
                    places=10,
                )

                forced_mask = objective._F < 200.0
                with patch.object(objective, "_NULL_MASK", forced_mask):
                    masked = objective.score_bands([[] for _ in range(8)])
                # Once real bins are excluded, the two numbers must be free to
                # diverge - never collapsed into one reported figure.
                self.assertNotAlmostEqual(
                    masked["target_rms_null_excluded_db"],
                    masked["target_rms_null_included_db"],
                    places=6,
                )

    def _score_with_baseline_filter(self, baseline_gain, candidate_channel_0_bands):
        """Baseline has one real PEQ filter at 6000 Hz/Q1 on FL High (channel
        0) with the given gain. A single peaking filter's cascade MAX is 0 dB
        away from its own center regardless of sign, so any all-negative (or
        absent) baseline filter yields peak=0 dB / not clip-risky, while a
        positive baseline gain directly sets that channel's peak (and, above
        0 dB, its clip_risk). Scores `candidate_channel_0_bands` as channel
        0's full band list (baseline bands must be included explicitly, since
        band_sets is always the complete final list, never just additions)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, baseline, filenames = self._build_synthetic_measurement_root(
                root, baseline_channel_filters={0: [(6000.0, 1.0, baseline_gain)]},
            )
            solo_files = {key: (Path(name).stem,) for key, name in filenames.items()}
            pair_specs = {
                "low": ("FL Low", "FR Low", "Mid Bass Together", (80.0, 2600.0), (200.0, 2000.0)),
                "high": ("FL High", "FR High", "Tweeters Together", (2600.0, 16000.0), (2800.0, 16000.0)),
            }
            with patch.multiple(
                objective,
                REW_DIR=root, TARGET=target, BASELINE_AFPX=baseline, LEVEL_CALIBRATION={},
                SOLO_FILES=solo_files, PAIR_SPECS=pair_specs,
                CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
                _F=None, _T={}, _TGT=None, _NULL_MASK=None, _V5=None,
            ):
                band_sets = [[] for _ in range(8)]
                band_sets[0] = candidate_channel_0_bands
                return objective.score_bands(band_sets)

    def test_headroom_is_a_hard_constraint_not_a_tradeable_penalty(self) -> None:
        """Regression for the run that paid +1.598 on the old soft headroom
        penalty to buy a 0.026/7.5 objective "win" on a channel already
        clip-risky (FL midbass, clip_risk already True). That must now be
        infeasible, not merely expensive (see CHANGELOG.md)."""
        # Reproducing a clip-risky baseline exactly (same F/Q/G) must stay
        # selectable - a no-op/baseline-preserving candidate is never a
        # headroom violation, even though this channel is already clip-risky.
        unchanged = self._score_with_baseline_filter(4.0, [(6000.0, 1.0, 4.0)])
        self.assertEqual(unchanged["headroom_violation_count"], 0)
        self.assertLess(unchanged["objective"], 1000.0)

        # Raising gain further on an already clip-risky channel must be
        # rejected outright, regardless of how much it improves everything
        # else - this is the exact shape of the old bug.
        worse = self._score_with_baseline_filter(4.0, [(6000.0, 1.0, 5.0)])
        self.assertGreaterEqual(worse["headroom_violation_count"], 1)
        self.assertGreaterEqual(worse["objective"], objective.HEADROOM_VIOLATION_PENALTY)
        self.assertGreater(worse["objective"], unchanged["objective"])

        # Cutting gain on an already clip-risky channel is a real improvement
        # and must never be rejected.
        improved = self._score_with_baseline_filter(4.0, [(6000.0, 1.0, 1.0)])
        self.assertEqual(improved["headroom_violation_count"], 0)

        # A channel that starts safe (baseline peak at 0 dB, no clip risk)
        # but that the candidate pushes below the 1.5 dB margin floor must
        # also be rejected, even without a pre-existing clip-risk flag.
        safe_baseline = self._score_with_baseline_filter(-2.0, [(6000.0, 1.0, -2.0)])
        self.assertEqual(safe_baseline["headroom_violation_count"], 0)
        pushed_too_far = self._score_with_baseline_filter(-2.0, [(6000.0, 1.0, 2.0)])
        self.assertGreaterEqual(pushed_too_far["headroom_violation_count"], 1)
        self.assertGreaterEqual(
            pushed_too_far["objective"], objective.HEADROOM_VIOLATION_PENALTY,
        )

    def test_nearfield_confirmed_null_forbids_positive_gain_skirt_overlap(self) -> None:
        """DEFECT 4a end-to-end: a narrow, deep System Sum dip with flat
        nearfield captures on both sides must be confirmed room-only and hard
        -reject any positive-gain candidate whose own -3dB skirt still lands
        on it - never merely the soft null_boost discouragement it got
        before. A cut at the same spot, or a boost far away, must both stay
        untouched by this specific guardrail."""
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
            system = power_sum(
                traces["Sub"], traces["Tweeters Together"], traces["Mid Bass Together"]
            )
            dip_center = 150.0
            traces["System Sum"] = system - 16.0 * np.exp(
                -0.5 * (np.log2(freqs / dip_center) / (1 / 40)) ** 2
            )
            filenames = {
                "FL High": "Front L High.txt", "FR High": "Front R High.txt",
                "FL Low": "Front L Low.txt", "FR Low": "Front R Low.txt",
                "Sub": "Sub.txt", "System Sum": "System Sum.txt",
                "Tweeters Together": "Tweeters Together.txt",
                "Mid Bass Together": "Mid Bass Together.txt",
            }
            for role, filename in filenames.items():
                rows = [f"{f:.9f} {s:.9f}" for f, s in zip(freqs, traces[role])]
                (root / filename).write_text("\n".join(rows), encoding="utf-8")
            # Close-mic captures: no room path, so the seat-only dip above
            # must not show up here if it is genuinely room/summation-caused.
            for filename in ("Front L Nearfield.txt", "Front R Nearfield.txt"):
                rows = [f"{f:.9f} {68.0:.9f}" for f in freqs]
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
                REW_DIR=root, TARGET=target, BASELINE_AFPX=baseline, LEVEL_CALIBRATION={},
                SOLO_FILES=solo_files, PAIR_SPECS=pair_specs,
                CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
                _F=None, _T={}, _TGT=None, _NULL_MASK=None, _V5=None,
                _NEARFIELD_GUARD_MASK=None,
            ):
                objective.score_bands([[] for _ in range(8)])
                self.assertEqual(objective._MASK_AUDIT["nearfield"]["state"], "DETECTED")
                self.assertGreater(int(np.sum(objective._NEARFIELD_GUARD_MASK)), 0)

                overlapping = [[] for _ in range(8)]
                overlapping[2] = [(dip_center, 2.0, 4.0)]  # FL Low, positive gain on the null
                overlapping_score = objective.score_bands(overlapping)
                self.assertGreaterEqual(overlapping_score["nearfield_skirt_violation_count"], 1)
                self.assertGreaterEqual(
                    overlapping_score["objective"], objective.NEARFIELD_SKIRT_PENALTY,
                )

                cutting = [[] for _ in range(8)]
                cutting[2] = [(dip_center, 2.0, -4.0)]  # a cut is never penalized here
                cutting_score = objective.score_bands(cutting)
                self.assertEqual(cutting_score["nearfield_skirt_violation_count"], 0)

                elsewhere = [[] for _ in range(8)]
                elsewhere[2] = [(1000.0, 2.0, 4.0)]  # far from the confirmed null
                elsewhere_score = objective.score_bands(elsewhere)
                self.assertEqual(elsewhere_score["nearfield_skirt_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
