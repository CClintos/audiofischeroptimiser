from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _tunefit as public_tunefit
from objective_module import _tunefit as canonical_tunefit
from objective_module import afpx_objective as objective
from objective_module.session import ScorerSession


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