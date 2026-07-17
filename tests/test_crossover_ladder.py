from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import _optimizer as optimizer
from _make_v3 import afpx_roundtrip_lint, apply_output_trim


class CrossoverLadderTests(unittest.TestCase):
    def test_lint_allows_only_declared_protective_output_trim(self) -> None:
        block = '<OC><Vol i="0" L="1" T="15"/></OC>'
        old = '<Root>' + block * 4 + '</Root>'
        new_blocks = [apply_output_trim(block, -1.0) for _ in range(4)]
        new = '<Root>' + ''.join(new_blocks) + '</Root>'

        rejected = afpx_roundtrip_lint(old, new, allowed_added_types=())
        accepted = afpx_roundtrip_lint(
            old, new, allowed_added_types=(),
            allowed_volume_trims={index: -1.0 for index in range(4)},
        )

        self.assertFalse(rejected["pass"])
        self.assertTrue(accepted["pass"], accepted)
        self.assertEqual(set(accepted["output_volume_changes_db"]), {0, 1, 2, 3})

    def test_band_limited_impulse_finds_arrival_and_polarity(self) -> None:
        sample_rate = 48000.0
        a = np.zeros(4096)
        b = np.zeros(4096)
        a[800] = 1.0
        b[824] = -1.0

        result = optimizer._impulse_pair_result(
            {"samples": a, "sample_rate": sample_rate, "path": "a.wav"},
            {"samples": b, "sample_rate": sample_rate, "path": "b.wav"},
            (1800.0, 4500.0),
        )

        self.assertTrue(result["usable"])
        self.assertEqual(result["polarity"], "inverted")
        self.assertAlmostEqual(float(result["arrival_delay_ms_B"]), 0.5, places=3)
        self.assertAlmostEqual(float(result["correction_delay_ms_B"]), -0.5, places=3)

    def test_writer_scopes_pm_polarity_and_delay_changes(self) -> None:
        old = (
            '<Root><OC CINV="0" CN="1"></OC><OC CINV="0" CN="2"></OC>'
            '<T P="0" PM="1" T="100"/><T P="0" PM="4" T="200"/></Root>'
        )
        plan = [{
            "polarity_channels": (0,),
            "channels": (1,),
            "delay_samples": 7,
            "apf": False,
            "apf_channels": (),
        }]

        new = optimizer.apply_phase_writes(old, plan)
        lint = afpx_roundtrip_lint(
            old,
            new,
            allow_delay_changes=True,
            allow_polarity_changes=True,
            allowed_added_types=(),
        )

        self.assertTrue(lint["pass"], lint)
        self.assertTrue(lint["polarity_changed"])
        self.assertTrue(lint["delay_changed"])
        self.assertFalse(lint["channel_attributes_changed"])
        self.assertFalse(lint["delay_attributes_changed"])
        self.assertIn('PM="4" T="100"', new)
        self.assertIn('PM="4" T="207"', new)

    def test_lint_rejects_unapproved_polarity(self) -> None:
        old = '<Root><OC CINV="0"></OC><T P="0" PM="1" T="100"/></Root>'
        new = '<Root><OC CINV="0"></OC><T P="0" PM="4" T="100"/></Root>'
        lint = afpx_roundtrip_lint(old, new, allowed_added_types=())
        self.assertFalse(lint["pass"])
        self.assertIn("output polarity changed", lint["errors"])

    def test_impulse_fallback_replaces_invalid_complex_reference(self) -> None:
        freqs = np.geomspace(50.0, 120.0, 192)
        flat = np.zeros_like(freqs)
        rich = {
            "Sub": {"freq": freqs, "spl": flat, "phase": flat},
            "Mid Bass Together": {"freq": freqs, "spl": flat, "phase": flat},
        }
        traces = {"Sub": flat, "Mid Bass Together": flat}
        measured_together = {"freq": freqs, "spl": np.full_like(freqs, -12.0)}
        a = np.zeros(4096)
        b = np.zeros(4096)
        a[800] = 1.0
        b[824] = -1.0
        impulses = {
            "Sub": {"samples": a, "sample_rate": 48000.0, "path": "sub.wav"},
            "Mid Bass Together": {"samples": b, "sample_rate": 48000.0, "path": "mids.wav"},
        }
        specs = [{
            "name": "sub_to_front",
            "label": "Sub to front midbass",
            "a": "Sub",
            "b": "Mid Bass Together",
            "together": measured_together,
            "band": (50.0, 120.0),
        }]

        with patch.object(optimizer, "load_optional_impulses", return_value=impulses), patch.object(
            optimizer, "crossover_specs", return_value=specs
        ):
            row = optimizer.crossover_phase_diagnostics(freqs, traces, rich)[0]

        ladder = row["crossover_ladder"]
        self.assertEqual(row["predicted_sum_match"], "low")
        self.assertEqual(ladder["source"], "band_limited_impulse_fallback")
        self.assertTrue(ladder["write_eligible"])
        self.assertTrue(ladder["polarity_flip_B"])
        self.assertAlmostEqual(float(ladder["correction_delay_ms_B"]), -0.5, places=3)


if __name__ == "__main__":
    unittest.main()
