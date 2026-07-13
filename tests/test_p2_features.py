from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _optimizer as optimizer
from objective_module._tunefit import cascade_complex, peaking_complex


class ComplexPeqTests(unittest.TestCase):
    def test_complex_transfer_magnitude_matches_db_model(self) -> None:
        freqs = np.geomspace(20.0, 20000.0, 1024)
        bands = [(900.0, 1.4, -3.0), (3200.0, 0.8, 1.5)]
        complex_db = 20.0 * np.log10(np.abs(cascade_complex(freqs, bands)))
        expected = sum(optimizer.peaking_db(freqs, *band) for band in bands)
        np.testing.assert_allclose(complex_db, expected, atol=1e-10)
        self.assertGreater(np.max(np.abs(np.angle(peaking_complex(freqs, *bands[0])))), 0.0)

    def test_phase_valid_scorer_uses_complex_gate_not_coarse_veto(self) -> None:
        freqs = np.geomspace(1000.0, 6000.0, 256)
        phase = np.zeros_like(freqs)
        rich = {
            "FL Low": {"spl": np.full_like(freqs, 60.0), "phase": phase},
            "FL High": {"spl": np.full_like(freqs, 60.0), "phase": phase},
        }
        together = 20.0 * np.log10(2.0 * 10.0 ** (60.0 / 20.0)) * np.ones_like(freqs)
        specs = [{"name": "left_mid_to_tweeter", "a": "FL Low", "b": "FL High",
                  "together": {"spl": together}, "band": (1800.0, 4500.0)}]
        with patch.object(optimizer, "crossover_specs", return_value=specs):
            scorer = optimizer.complex_phase_component_scorer(
                lambda groups: {"objective": 1.0}, freqs, rich,
                [{"crossover_band": (1800.0, 4500.0), "crossover_channels": (0, 2)}], True,
            )
            result = scorer({name: [] for name in optimizer.GROUPS})
        self.assertEqual(result["complex_crossover_pass"], 1.0)
        self.assertEqual(result["objective"], 1.0)


class ExplicitChoiceTests(unittest.TestCase):
    def test_sub_blend_requires_headroom(self) -> None:
        freqs = np.geomspace(20.0, 20000.0, 256)
        traces = {"System Sum": np.zeros_like(freqs)}
        session = {"audit": {"tonal_valid": True, "missing_calibration_roles": []}}
        result = optimizer.same_level_sub_blend_recommendation(
            freqs, traces, np.ones_like(freqs), session, None
        )
        self.assertEqual(result["status"], "blocked")

    def test_sub_blend_is_trim_recommendation_not_peq(self) -> None:
        freqs = np.geomspace(20.0, 20000.0, 256)
        traces = {"System Sum": np.zeros_like(freqs)}
        session = {"audit": {"tonal_valid": True, "missing_calibration_roles": []}}
        result = optimizer.same_level_sub_blend_recommendation(
            freqs, traces, np.full_like(freqs, 2.0), session, 1.25
        )
        self.assertEqual(result["status"], "recommendation_only")
        self.assertLessEqual(result["recommended_sub_output_trim_db"], 1.25)
        self.assertIn("output level", result["warning"])

    def test_canonical_phase_schema(self) -> None:
        with patch.object(optimizer, "cached_crossover_phase_diagnostics", return_value=([], {"source": "cache"})):
            result = optimizer.analyze_phase_session(
                np.array([100.0]), {}, {}, {"audit": {"phase_valid": False}}, writes=False
            )
        self.assertEqual(result["schema"], "audiofischer-phase-session-v1")
        self.assertEqual(result["writes"], [])
        self.assertIn("experimental_engines", result)


if __name__ == "__main__":
    unittest.main()
