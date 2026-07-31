from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _tunefit as public_tunefit
import _optimizer as optimizer
import _optimizer_stream as optimizer_stream
from objective_module import _tunefit as canonical_tunefit
from objective_module import afpx_objective as objective
from objective_module.session import ScorerSession
from scripts.verify_written_tune import measurement_guardrail_errors


class StrictMeasurementTests(unittest.TestCase):
    def test_missing_measurement_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                objective._load_txt_rich(Path(tmp) / "missing.txt")

    def test_truncated_measurement_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.txt"
            path.write_text("\n".join(f"{100 + i} 60" for i in range(8)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncated"):
                objective._load_txt_rich(path)

    def test_invalid_numeric_row_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            rows = [f"{100 + i} 60" for i in range(16)] + ["400"]
            path.write_text("\n".join(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frequency and SPL"):
                objective._load_txt_rich(path)


class PerceptualObjectiveTests(unittest.TestCase):
    def test_vocal_weight_has_no_boxcar_edge(self) -> None:
        freqs = np.geomspace(80.0, 12000.0, 4096)
        weights = objective.perceptual_weights(freqs)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=8)
        self.assertAlmostEqual(float(weights[-1]), 1.0, places=8)
        self.assertLess(float(np.max(np.abs(np.diff(weights)))), 0.01)
        self.assertGreater(float(np.max(weights)), 1.79)

    def test_one_sixth_octave_term_sees_peak_hidden_by_erb_smoothing(self) -> None:
        freqs = np.geomspace(60.0, 16000.0, 2048)
        raw = 7.0 * np.exp(-0.5 * (np.log2(freqs / 2400.0) / 0.025) ** 2)
        broad = canonical_tunefit.erb_smooth(freqs, raw)
        narrow = np.maximum(objective._fractional_octave_smooth(freqs, raw, 6), raw)
        parts = objective.tonal_components(
            freqs, broad, np.ones_like(freqs, dtype=bool), narrow
        )
        self.assertGreater(float(np.max(narrow)), float(np.max(broad)))
        self.assertGreater(parts["narrow_peak_penalty_db"], 0.0)
        self.assertGreater(parts["narrow_peak_max_db"], 2.0)


class BalanceGuardrailTests(unittest.TestCase):
    def test_alternating_lr_comb_is_not_a_correctable_offset(self) -> None:
        freqs = np.array([700.0, 800.0, 900.0, 1100.0, 1128.0, 1200.0, 1400.0])
        difference = np.array([-1.4, -1.3, 2.7, 2.9, 3.6, 2.9, -2.3])

        evidence = canonical_tunefit.signed_offset_evidence(
            freqs, difference, 1128.0, "low"
        )

        self.assertFalse(evidence["eligible"])
        self.assertEqual(evidence["reason"], "alternating_lr_comb")
        self.assertGreater(evidence["sign_changes"], 1)

    def test_broad_one_sign_offset_must_clear_local_floor(self) -> None:
        freqs = np.geomspace(500.0, 2000.0, 64)
        systematic = canonical_tunefit.signed_offset_evidence(
            freqs, np.full_like(freqs, 3.0), 1128.0, "low"
        )
        below_floor = canonical_tunefit.signed_offset_evidence(
            freqs, np.full_like(freqs, 0.7), 1128.0, "low"
        )

        self.assertTrue(systematic["eligible"])
        self.assertEqual(systematic["sign_changes"], 0)
        self.assertFalse(below_floor["eligible"])
        self.assertEqual(below_floor["reason"], "below_measurement_noise_threshold")
        self.assertGreater(below_floor["required_deviation_db"], 0.7)

    def test_imaging_balance_is_heavily_deweighted_below_400_hz(self) -> None:
        low, imaging = canonical_tunefit.imaging_balance_weight([270.0, 1000.0])
        self.assertLess(low, 0.05)
        self.assertAlmostEqual(imaging, 1.0, places=10)

    def test_crossover_groups_offer_tweeters_mids_and_both_scopes(self) -> None:
        groups = optimizer.groups_for_layout("2way")
        self.assertEqual(groups["high_crossover_sym"]["channels"], (0, 1))
        self.assertEqual(groups["low_crossover_sym"]["channels"], (2, 3))
        self.assertEqual(groups["front_voicing"]["channels"], (0, 1, 2, 3))
        for name in ("high_crossover_sym", "low_crossover_sym", "front_voicing"):
            lo, hi = groups[name]["range"]
            self.assertLessEqual(lo, 2650.0)
            self.assertGreaterEqual(hi, 2650.0)
        self.assertGreaterEqual(groups["low_sym"]["range"][1], 2600.0)
        three_way = optimizer.groups_for_layout("3way")
        self.assertGreaterEqual(three_way["mid_sym"]["range"][1], 2600.0)

    def test_tweeter_crossover_scope_maps_only_to_tweeter_outputs(self) -> None:
        band = (2650.0, 2.0, -1.5)
        with patch.multiple(
            optimizer,
            GROUPS=optimizer.groups_for_layout("2way"),
            AFPX_OBJECTIVE=None,
        ), patch.object(optimizer, "baseline_band_sets", return_value=[[] for _ in range(8)]):
            band_sets = optimizer.groups_to_band_sets({"high_crossover_sym": [band]})
        self.assertEqual(band_sets[0], [band])
        self.assertEqual(band_sets[1], [band])
        self.assertTrue(all(not bands for bands in band_sets[2:]))

    def test_symmetric_tweeter_cut_near_crossover_is_not_balance_blocked(self) -> None:
        freqs = np.geomspace(200.0, 16000.0, 512)
        target = np.full_like(freqs, 60.0)
        peak = 2.0 * np.exp(-0.5 * (np.log2(freqs / 2650.0) / 0.20) ** 2)
        system = target + peak
        flat = np.full_like(freqs, 60.0)
        traces = {
            "FL High": flat,
            "FR High": flat,
            "FL Low": flat,
            "FR Low": flat,
            "Tweeters Together": flat,
            "Mid Bass Together": flat,
            "System Sum": system,
            "Sub": np.full_like(freqs, -100.0),
        }
        candidate = [[] for _ in range(8)]
        candidate[0] = [(2650.0, 2.0, -1.5)]
        candidate[1] = [(2650.0, 2.0, -1.5)]
        with patch.multiple(
            objective,
            _F=freqs,
            _T=traces,
            _TGT=target,
            _V5=[[] for _ in range(8)],
            _BASE_CASCADES=[np.zeros_like(freqs) for _ in range(8)],
            _TOTAL_DB=system,
            _SMOOTH_T={},
            _SMOOTHER=None,
            CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
        ):
            guard = objective._guardrail_score(candidate)

        self.assertEqual(guard["balance_guardrail_violation_count"], 0)
        self.assertEqual(guard["filter_noise_floor_violation_count"], 0)
        self.assertEqual(guard["balance_guardrail_penalty"], 0.0)


class MaskIntegrityTests(unittest.TestCase):
    def test_synthetic_pair_is_unknown_and_blocks_one_sided_pool(self) -> None:
        freqs = np.geomspace(100.0, 4000.0, 512)
        target = np.full_like(freqs, 60.0)
        left = target + 3.0
        right = target
        together = 10.0 * np.log10(10.0 ** (left / 10.0) + 10.0 ** (right / 10.0))
        traces = {
            "FL Low": left,
            "FR Low": right,
            "Mid Bass Together": together,
            "System Sum": target + 2.0,
            "Sub": np.full_like(freqs, -100.0),
        }
        groups = {
            "fl_low": {
                "channels": (0,),
                "branch": "low",
                "trace": "FL Low",
                "pair": "low",
                "side": "left",
                "range": (200.0, 2000.0),
                "q_range": (0.5, 6.0),
                "gain_range": (-6.0, 3.0),
                "max_bands": 2,
            }
        }
        pairs = {
            "low": {
                "left": "FL Low",
                "right": "FR Low",
                "together": "Mid Bass Together",
                "branch_band": (100.0, 2600.0),
                "balance_band": (200.0, 2000.0),
            }
        }
        with patch.multiple(
            optimizer,
            GROUPS=groups,
            PAIR_DEFS=pairs,
            SYNTHETIC_PAIR_ROLES={"Mid Bass Together"},
        ):
            masks, states = optimizer_stream.interference_masks(freqs, traces)
            pools = optimizer_stream.find_guided_candidates(freqs, traces, target, "safe")
        self.assertEqual(states["low"]["state"], canonical_tunefit.MASK_UNKNOWN)
        self.assertFalse(np.any(masks["fl_low"]))
        self.assertFalse(any(item["source"] == "balance" for item in pools["fl_low"]))

    def test_measured_destructive_pair_is_detected(self) -> None:
        freqs = np.geomspace(300.0, 3000.0, 512)
        left = np.full_like(freqs, 60.0)
        right = np.full_like(freqs, 60.0)
        together = 10.0 * np.log10(10.0 ** 6 + 10.0 ** 6) * np.ones_like(freqs)
        together -= 8.0 * np.exp(-0.5 * (np.log2(freqs / 1128.0) / 0.05) ** 2)
        evidence = canonical_tunefit.interference_mask_evidence(
            freqs, left, right, together, band=(700.0, 1600.0)
        )
        self.assertEqual(evidence["state"], canonical_tunefit.MASK_DETECTED)
        self.assertTrue(np.any(evidence["mask"]))

    def test_spatially_shifting_lf_dip_is_modal_but_stable_dip_is_not(self) -> None:
        freqs = np.geomspace(20.0, 250.0, 1024)

        def dip(center: float) -> np.ndarray:
            return -12.0 * np.exp(-0.5 * (np.log2(freqs / center) / 0.045) ** 2)

        moving = canonical_tunefit.modal_null_evidence(
            freqs, dip(52.0), {"left": dip(59.0), "right": dip(45.0)}
        )
        stable = canonical_tunefit.modal_null_evidence(
            freqs, dip(52.0), {"left": dip(52.5), "right": dip(51.5)}
        )
        single = canonical_tunefit.modal_null_evidence(freqs, dip(52.0))
        self.assertEqual(moving["state"], canonical_tunefit.MASK_DETECTED)
        self.assertEqual(stable["state"], canonical_tunefit.MASK_CLEAR)
        self.assertEqual(single["state"], canonical_tunefit.MASK_DETECTED)
        self.assertEqual(single["confidence"], "low")

    def _balance_pools(self, freqs: np.ndarray, difference: np.ndarray,
                       system_deviation: float = 1.0) -> dict[str, list[dict[str, object]]]:
        target = np.full_like(freqs, 60.0)
        left = target + difference / 2.0
        right = target - difference / 2.0
        together = optimizer.power_sum_db([left, right])
        traces = {
            "FL Low": left,
            "FR Low": right,
            "Mid Bass Together": together,
            "System Sum": target + system_deviation,
            "Sub": np.full_like(freqs, -100.0),
        }
        groups = {
            side: {
                "channels": (index,),
                "branch": "low",
                "trace": role,
                "pair": "low",
                "side": side_name,
                "range": (500.0, 2000.0),
                "q_range": (0.5, 6.0),
                "gain_range": (-6.0, 3.0),
                "max_bands": 2,
            }
            for side, index, role, side_name in (
                ("fl_low", 0, "FL Low", "left"),
                ("fr_low", 1, "FR Low", "right"),
            )
        }
        pairs = {
            "low": {
                "left": "FL Low",
                "right": "FR Low",
                "together": "Mid Bass Together",
                "branch_band": (500.0, 2200.0),
                "balance_band": (500.0, 2000.0),
            }
        }
        with patch.multiple(
            optimizer,
            GROUPS=groups,
            PAIR_DEFS=pairs,
            SYNTHETIC_PAIR_ROLES=set(),
        ):
            return optimizer_stream.find_guided_candidates(freqs, traces, target, "safe")

    def test_alternating_audit_imbalance_produces_no_balance_candidate(self) -> None:
        audit_f = np.array([700.0, 800.0, 900.0, 1100.0, 1128.0, 1200.0, 1400.0])
        audit_d = np.array([-1.4, -1.3, 2.7, 2.9, 3.6, 2.9, -2.3])
        freqs = np.geomspace(500.0, 2200.0, 768)
        difference = np.interp(np.log10(freqs), np.log10(audit_f), audit_d)
        pools = self._balance_pools(freqs, difference)
        in_audit_band = [
            item for items in pools.values() for item in items
            if item["source"] == "balance" and 700.0 <= item["F"] <= 1400.0
        ]
        self.assertEqual(in_audit_band, [])
        self.assertTrue(any(
            item["reason"] == "alternating_lr_comb"
            for item in optimizer_stream.LAST_PROPOSAL_AUDIT["suppressions"]
        ))

    def test_broad_two_db_offset_still_produces_balance_candidate(self) -> None:
        freqs = np.geomspace(500.0, 2200.0, 768)
        pools = self._balance_pools(freqs, np.full_like(freqs, 2.0))
        self.assertTrue(any(
            item["source"] == "balance"
            for items in pools.values() for item in items
        ))

    def test_balance_cut_cannot_deepen_summed_hole(self) -> None:
        freqs = np.geomspace(500.0, 2200.0, 768)
        pools = self._balance_pools(freqs, np.full_like(freqs, 3.0), -2.2)
        self.assertFalse(any(
            item["source"] == "balance" and item["G"] < 0.0
            for items in pools.values() for item in items
        ))
        self.assertTrue(any(
            item["reason"] == "summed_response_already_below_target"
            for item in optimizer_stream.LAST_PROPOSAL_AUDIT["suppressions"]
        ))

    def test_write_lint_rejects_one_sided_cut_into_summed_hole(self) -> None:
        freqs = np.geomspace(100.0, 3000.0, 768)
        target = np.full_like(freqs, 60.0)
        traces = {
            "FL Low": target + 1.5,
            "FR Low": target - 1.5,
            "Mid Bass Together": target,
            "System Sum": target - 2.2,
        }
        groups = {
            "fl_low": {
                "channels": (0,), "branch": "low", "pair": "low", "side": "left",
            },
            "fr_low": {
                "channels": (1,), "branch": "low", "pair": "low", "side": "right",
            },
        }
        pairs = {
            "low": {
                "left": "FL Low", "right": "FR Low",
                "together": "Mid Bass Together", "branch_band": (80.0, 2600.0),
            }
        }
        errors = measurement_guardrail_errors(
            freqs,
            traces,
            target,
            {0: [{"F": "270", "Q": "2", "G": "-3"}], 1: []},
            groups,
            pairs,
            set(),
        )
        reasons = {reason for item in errors for reason in item["reasons"]}
        self.assertIn("summed_response_already_below_target", reasons)

    def test_write_lint_rejects_sub_floor_filter(self) -> None:
        freqs = np.geomspace(500.0, 2200.0, 768)
        target = np.full_like(freqs, 60.0)
        traces = {
            "FL Low": target + 0.35,
            "FR Low": target - 0.35,
            "Mid Bass Together": target,
            "System Sum": target + 0.7,
        }
        groups = {
            "fl_low": {
                "channels": (0,), "branch": "low", "pair": "low", "side": "left",
            },
            "fr_low": {
                "channels": (1,), "branch": "low", "pair": "low", "side": "right",
            },
        }
        pairs = {
            "low": {
                "left": "FL Low", "right": "FR Low",
                "together": "Mid Bass Together", "branch_band": (500.0, 2200.0),
            }
        }
        errors = measurement_guardrail_errors(
            freqs,
            traces,
            target,
            {0: [{"F": "1128", "Q": "2", "G": "-0.75"}], 1: []},
            groups,
            pairs,
            set(),
        )
        reasons = {reason for item in errors for reason in item["reasons"]}
        self.assertIn("below_measurement_noise_floor", reasons)

    def test_repeatability_folder_builds_and_installs_empirical_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            repeat = root / "repeat"
            primary.mkdir()
            repeat.mkdir()
            freqs = np.geomspace(200.0, 6000.0, 384)
            roles = ("FL Low", "FR Low", "FL High", "FR High")
            primary_files = {}
            repeat_files = {}
            for index, role in enumerate(roles):
                filename = role.replace(" ", "_") + ".txt"
                first = 60.0 + 0.2 * index + np.zeros_like(freqs)
                noise = 0.35 * np.sin(np.log(freqs) * (4.0 + index))
                first_path = primary / filename
                repeat_path = repeat / filename
                first_path.write_text(
                    "\n".join(f"{f:.8f} {value:.8f}" for f, value in zip(freqs, first)),
                    encoding="utf-8",
                )
                repeat_path.write_text(
                    "\n".join(
                        f"{f:.8f} {value:.8f}"
                        for f, value in zip(freqs, first + 1.0 + noise)
                    ),
                    encoding="utf-8",
                )
                primary_files[role] = first_path
                repeat_files[role] = repeat_path
            (repeat / "known_eq_delta.json").write_text(
                json.dumps({role: 1.0 for role in roles}), encoding="utf-8"
            )
            with patch.multiple(
                optimizer,
                MEASUREMENT_FILES=primary_files,
                OPTIONAL_PAIR_ROLES=set(),
            ), patch.object(optimizer, "resolve_measurement_files", return_value=repeat_files):
                model = optimizer.empirical_repeatability_model(repeat)
            canonical_tunefit.configure_measurement_noise_model(model)
            try:
                self.assertEqual(model["id"], "empirical_same_day_repeatability_v1")
                self.assertEqual(model["known_eq_delta_status"], "subtracted")
                self.assertEqual(set(model["roles_used"]), set(roles))
                self.assertTrue(model["branches"]["low"])
                self.assertTrue(model["branches"]["high"])
                self.assertGreater(
                    float(canonical_tunefit.measurement_noise_floor_db([1128.0], "low")[0]),
                    0.05,
                )
            finally:
                canonical_tunefit.configure_measurement_noise_model(None)


class ComplexPredictionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.geomspace(100.0, 10000.0, 512)
        x = np.log2(self.freqs / 1000.0)
        self.left = {
            "spl": np.full_like(self.freqs, 60.0),
            "phase": 25.0 * np.sin(x),
        }
        self.right = {
            "spl": np.full_like(self.freqs, 59.0),
            "phase": 25.0 * np.sin(x) + 55.0,
        }
        summed = objective._trace_complex(self.left) + objective._trace_complex(self.right)
        self.together = {
            "spl": 20.0 * np.log10(np.abs(summed)),
            "phase": np.rad2deg(np.unwrap(np.angle(summed))),
        }

    def test_valid_complex_model_reproduces_measured_baseline(self) -> None:
        meta = {"L": self.left, "R": self.right, "Together": self.together}
        with patch.object(objective, "_F", self.freqs):
            model, reason = objective._make_complex_sum_model(
                meta, ("L", "R"), "Together", (150.0, 8000.0)
            )
        self.assertEqual(reason, "pass")
        self.assertIsNotNone(model)
        self.assertLess(model["validation_rms_db"], 1e-10)

        with patch.multiple(
            objective,
            _F=self.freqs,
            _V5=[[], []],
            CH_KEYS=["L", "R"],
        ):
            baseline = objective._predict_complex_model(model, [[], []], {})
            candidate = objective._predict_complex_model(
                model, [[(1800.0, 1.2, 3.0)], []], {}
            )
        np.testing.assert_allclose(baseline, self.together["spl"], atol=1e-10)
        self.assertGreater(float(np.max(np.abs(candidate - baseline))), 0.1)

    def test_constant_placeholder_phase_is_rejected(self) -> None:
        meta = {
            "L": {"spl": self.left["spl"], "phase": np.zeros_like(self.freqs)},
            "R": {"spl": self.right["spl"], "phase": np.zeros_like(self.freqs)},
            "Together": {"spl": self.together["spl"], "phase": np.zeros_like(self.freqs)},
        }
        with patch.object(objective, "_F", self.freqs):
            model, reason = objective._make_complex_sum_model(
                meta, ("L", "R"), "Together", (150.0, 8000.0)
            )
        self.assertIsNone(model)
        self.assertIn("placeholder", reason)

    def test_inconsistent_together_trace_falls_back(self) -> None:
        inconsistent = dict(self.together)
        inconsistent["spl"] = self.together["spl"] + 5.0 * np.sin(
            8.0 * np.log2(self.freqs / 1000.0)
        )
        meta = {"L": self.left, "R": self.right, "Together": inconsistent}
        with patch.object(objective, "_F", self.freqs):
            model, reason = objective._make_complex_sum_model(
                meta, ("L", "R"), "Together", (150.0, 8000.0)
            )
        self.assertIsNone(model)
        self.assertIn("exceeds", reason)


class SyntheticPairFallbackTests(unittest.TestCase):
    def test_synthetic_pairs_preserve_measured_baseline_and_cut_cannot_raise_sum(self) -> None:
        freqs = np.geomspace(500.0, 5000.0, 128)
        high = np.full_like(freqs, 70.0)
        low = np.full_like(freqs, 68.0)
        sub = np.full_like(freqs, 55.0)
        high_pair = 10.0 * np.log10(2.0 * 10.0 ** (high / 10.0))
        low_pair = 10.0 * np.log10(2.0 * 10.0 ** (low / 10.0))
        measured_system = np.full_like(freqs, 66.0)
        traces = {
            "FL High": high, "FR High": high,
            "FL Low": low, "FR Low": low,
            "Tweeters Together": high_pair,
            "Mid Bass Together": low_pair,
            "Sub": sub,
            "System Sum": measured_system,
        }
        baseline = [[] for _ in range(8)]
        candidate = [[] for _ in range(8)]
        candidate[0] = [(2500.0, 2.0, -2.0)]
        candidate[1] = [(2500.0, 2.0, -2.0)]
        with patch.multiple(
            objective,
            _F=freqs,
            _T=traces,
            _V5=baseline,
            _BASE_CASCADES=[np.zeros_like(freqs) for _ in range(8)],
            _BASE_OUTPUT_DB=[0.0] * 8,
            _COMPLEX_MODELS={},
            _SYNTHETIC_PAIRS={"Tweeters Together", "Mid Bass Together"},
            CH_KEYS=["FL High", "FR High", "FL Low", "FR Low"],
        ):
            predicted_baseline = objective._predict(baseline, {})
            predicted_candidate = objective._predict(candidate, {})

        np.testing.assert_allclose(predicted_baseline["System Sum"], measured_system)
        self.assertTrue(np.all(predicted_candidate["System Sum"] <= measured_system + 1e-12))
        self.assertLess(float(np.min(predicted_candidate["System Sum"] - measured_system)), -1.0)

class ScorerSessionTests(unittest.TestCase):
    def test_sessions_keep_independent_roots_and_modules(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            one = ScorerSession(first_root, first_root / "one.afpx", first_root / "one.txt")
            two = ScorerSession(second_root, second_root / "two.afpx", second_root / "two.txt")
        self.assertIsNot(one._module, two._module)
        self.assertEqual(one._module.REW_DIR, first_root.resolve())
        self.assertEqual(two._module.REW_DIR, second_root.resolve())

class CanonicalDspTests(unittest.TestCase):
    def test_public_module_uses_canonical_implementation(self) -> None:
        self.assertIs(public_tunefit.allpass_fil_str, canonical_tunefit.allpass_fil_str)
        xml = public_tunefit.allpass_fil_str(174.0, 8.0, FN="20")
        self.assertIn('Q="8.0"', xml)


if __name__ == "__main__":
    unittest.main()
