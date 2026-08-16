from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _optimizer as optimizer
from _make_v3 import afpx_roundtrip_lint, apply_output_trim
import scripts.verify_written_tune as written_verifier
from scripts.verify_written_tune import unapproved_removed_filters


class CrossoverLadderTests(unittest.TestCase):
    def test_verifier_cli_exits_nonzero_when_candidate_fails(self):
        with patch.object(
            written_verifier, "verify", return_value={"pass": False},
        ), patch.object(
            written_verifier.sys,
            "argv",
            ["verify_written_tune.py", "baseline.afpx", "candidate.afpx"],
        ), patch.object(Path, "write_text"):
            with self.assertRaises(SystemExit) as raised:
                written_verifier.main()

        self.assertEqual(raised.exception.code, 1)
    def test_verifier_can_allow_peq_edits_without_allowing_crossover_edits(self):
        removed = (
            tuple(sorted({"T": "17", "F": "100"}.items())),
            tuple(sorted({"T": "15", "F": "80"}.items())),
        )

        self.assertEqual(
            unapproved_removed_filters(removed, allow_peq_edits=True),
            [removed[1]],
        )
        self.assertEqual(
            unapproved_removed_filters(removed, allow_peq_edits=False),
            list(removed),
        )

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
    def test_phase_search_scores_worst_case_across_snapshots_and_drift(self) -> None:
        freqs = np.geomspace(1800.0, 4500.0, 160)
        a = np.ones_like(freqs, dtype=complex)
        b_center = -np.exp(1j * 2.0 * np.pi * freqs * 0.00012)
        b_left = -np.exp(1j * 2.0 * np.pi * freqs * 0.00016)

        result = optimizer.polarity_delay_search(
            freqs,
            a,
            b_center,
            (1800.0, 4500.0),
            max_delay_ms=0.4,
            steps=81,
            snapshots=[(a, b_center), (a, b_left)],
        )

        self.assertEqual(result["phase_snapshot_count"], 2)
        self.assertEqual(result["perturbation_count"], 7)
        self.assertEqual(len(result["robust_snapshot_scores_after"]), 2)
        self.assertAlmostEqual(
            result["score_after"],
            max(result["robust_snapshot_scores_after"]),
            places=3,
        )
        self.assertGreater(result["improvement_pct"], 10.0)

    def test_allpass_search_reports_robust_snapshot_envelope(self) -> None:
        freqs = np.geomspace(50.0, 120.0, 128)
        a = np.ones_like(freqs, dtype=complex)
        b_center = np.exp(1j * np.deg2rad(130.0)) * np.ones_like(freqs, dtype=complex)
        b_right = np.exp(1j * np.deg2rad(145.0)) * np.ones_like(freqs, dtype=complex)

        result = optimizer.optimize_allpass(
            freqs,
            a,
            b_center,
            (50.0, 120.0),
            f_steps=10,
            q_steps=5,
            snapshots=[(a, b_center), (a, b_right)],
        )

        self.assertEqual(result["phase_snapshot_count"], 2)
        self.assertEqual(result["perturbation_count"], 7)
        self.assertEqual(len(result["robust_snapshot_scores"]), 2)
        self.assertGreaterEqual(result["robust_score_after"], 0.0)

    def test_spatial_phase_snapshot_requires_reference_and_pair_validation(self) -> None:
        freqs = np.geomspace(1800.0, 4500.0, 96)
        phase_b = np.full_like(freqs, -35.0)
        pair_spl = 60.0 + 20.0 * np.log10(
            np.abs(1.0 + np.exp(1j * np.deg2rad(phase_b)))
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "FL Low": root / "Left Ear Front L Mid.txt",
                "FL High": root / "Left Ear Front L High.txt",
                "pair": root / "Left Ear Front L Mid + Tweeter.txt",
            }

            def write_export(path: Path, spl, phase) -> None:
                rows = ["* reference played from Rear Right with loopback"]
                rows.extend(
                    f"{f:.6f} {level:.6f} {angle:.6f}"
                    for f, level, angle in zip(freqs, spl, phase)
                )
                path.write_text("\n".join(rows), encoding="utf-8")

            write_export(paths["FL Low"], np.full_like(freqs, 60.0), np.zeros_like(freqs))
            write_export(paths["FL High"], np.full_like(freqs, 60.0), phase_b)
            write_export(paths["pair"], pair_spl, np.zeros_like(freqs))
            spec = {
                "a": "FL Low",
                "b": "FL High",
                "band": (1800.0, 4500.0),
                "together_aliases": ("Front L Mid + Tweeter.txt",),
            }
            session = {
                "manifest": {
                    "spatial_bundles": {
                        "left": {
                            "FL Low": str(paths["FL Low"]),
                            "FL High": str(paths["FL High"]),
                        }
                    }
                },
                "audit": {"timing_references": ["Rear Right"]},
            }

            with patch.object(optimizer, "DATA_ROOT", root):
                snapshots, audit = optimizer._spatial_phase_snapshots(
                    spec, freqs, session
                )

        self.assertEqual(len(snapshots), 1)
        self.assertTrue(audit[0]["usable"])
        self.assertLessEqual(audit[0]["predicted_sum_rms_db"], 0.01)



class HardwareRealPhaseTests(unittest.TestCase):
    def test_delay_search_returns_an_exact_hardware_sample(self) -> None:
        freqs = np.geomspace(1800.0, 4500.0, 160)
        sample_rate = 48000.0
        a = np.ones_like(freqs, dtype=complex)
        b = -np.exp(1j * 2.0 * np.pi * freqs * 0.000137)

        result = optimizer.polarity_delay_search(
            freqs, a, b, (1800.0, 4500.0), max_delay_ms=0.4,
            steps=121, sample_rate_hz=sample_rate,
        )

        self.assertEqual(
            result["delay_samples_B"],
            round(result["delay_ms_B"] * sample_rate / 1000.0),
        )
        self.assertAlmostEqual(
            result["delay_ms_B"] * sample_rate / 1000.0,
            result["delay_samples_B"],
            places=10,
        )
        self.assertAlmostEqual(result["delay_step_ms"], 1000.0 / sample_rate, places=6)

    def test_group_delay_cost_is_frequency_relative(self) -> None:
        freqs = np.geomspace(50.0, 5000.0, 512)
        delay = np.full_like(freqs, 1.0)
        low = optimizer.temporal_group_delay_cost(freqs, delay, (80.0, 120.0))
        high = optimizer.temporal_group_delay_cost(freqs, delay, (2500.0, 3500.0))
        self.assertGreater(high, low + 1.0)

    def test_temporal_cost_avoids_a_razor_upper_mid_apf(self) -> None:
        freqs = np.geomspace(1000.0, 7000.0, 300)
        a = np.ones_like(freqs, dtype=complex)
        measured = np.exp(
            -1j * np.angle(optimizer.allpass_H(freqs, 2200.0, 0.8)) * 0.7
        )

        acoustic_only = optimizer.optimize_allpass(
            freqs, a, measured, (1800.0, 5000.0), apply_to="B",
            f_steps=14, q_steps=6, gd_penalty=0.0, robust=False,
        )
        temporal_aware = optimizer.optimize_allpass(
            freqs, a, measured, (1800.0, 5000.0), apply_to="B",
            f_steps=14, q_steps=6, gd_penalty=0.5, robust=False,
        )

        self.assertEqual(acoustic_only["Q"], 2.0)
        self.assertLessEqual(temporal_aware["Q"], 0.8)
        self.assertLess(
            temporal_aware["temporal_gd_cost"],
            acoustic_only["temporal_gd_cost"],
        )

    def test_phase_write_uses_the_scored_sample_count(self) -> None:
        rows = [{
            "name": "left_mid_to_tweeter",
            "label": "Left mid to tweeter",
            "a": "FL Low",
            "b": "FL High",
            "band": "1800-4500 Hz",
            "crossover_ladder": {
                "write_eligible": True,
                "source": "complex_phase",
                "polarity_flip_B": False,
                "correction_delay_ms_B": 0.062,
                "correction_delay_samples_B": 6,
            },
        }]

        plan = optimizer.phase_write_plan(rows, 96000.0)

        self.assertEqual(plan[0]["delay_samples"], 6)
        self.assertEqual(plan[0]["delay_ms"], 0.0625)

if __name__ == "__main__":
    unittest.main()
