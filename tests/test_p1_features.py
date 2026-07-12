from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _optimizer as optimizer
import _optimizer_stream as stream
from objective_module import afpx_objective as objective
from scripts.make_measurement_manifest import first_position_existing


class SpatialObjectiveTests(unittest.TestCase):
    def test_three_position_score_uses_centre_and_both_ears(self) -> None:
        freqs = np.geomspace(60.0, 16000.0, 384)
        shape = 2.0 * np.sin(np.log(freqs))
        keep = np.ones_like(freqs, dtype=bool)
        positions = {
            "left": {"system": shape, "target": np.zeros_like(freqs)},
            "right": {"system": -0.7 * shape, "target": np.zeros_like(freqs)},
        }
        with patch.multiple(
            objective,
            _F=freqs,
            _T={"System Sum": np.zeros_like(freqs)},
            _TGT=np.zeros_like(freqs),
            _SMOOTHER=None,
            _POSITION_TRACES=positions,
            _POSITION_BASELINE={"left": 0.0, "right": 0.0},
        ), patch.object(objective, "_has_fragile_filters", return_value=False):
            parts = objective._spatial_components(
                {"System Sum": np.zeros_like(freqs)}, [[] for _ in range(8)], keep
            )
        self.assertEqual(parts["spatial_position_count"], 3)
        self.assertEqual(parts["spatial_model"], "system_delta")
        self.assertGreater(parts["spatial_tonal_db"], parts["tonal_masked"])

    def test_fragile_correction_must_hold_at_ear_positions(self) -> None:
        freqs = np.geomspace(60.0, 16000.0, 384)
        shape = 2.0 * np.sin(np.log(freqs))
        keep = np.ones_like(freqs, dtype=bool)
        with patch.multiple(
            objective,
            _F=freqs,
            _T={"System Sum": np.zeros_like(freqs)},
            _TGT=np.zeros_like(freqs),
            _SMOOTHER=None,
            _POSITION_TRACES={"left": {"system": shape, "target": np.zeros_like(freqs)}},
            _POSITION_BASELINE={"left": 0.0},
        ), patch.object(objective, "_has_fragile_filters", return_value=True):
            parts = objective._spatial_components(
                {"System Sum": np.zeros_like(freqs)}, [[] for _ in range(8)], keep
            )
        self.assertGreater(parts["spatial_fragility_penalty"], 1.0)


class CacheTests(unittest.TestCase):
    def test_peaking_cache_is_numerically_identical(self) -> None:
        freqs = np.geomspace(20.0, 20000.0, 512)
        token = (len(freqs), float(freqs[0]), float(freqs[-1]), hash(freqs.tobytes()))
        objective._cached_peaking.cache_clear()
        with patch.multiple(objective, _F=freqs, _GRID_TOKEN=token):
            first = objective._casc([(1000.0, 1.2, -2.5)])
            second = objective._casc([(1000.0, 1.2, -2.5)])
            info = objective._cached_peaking.cache_info()
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(info.hits, 1)

    def test_phase_diagnostics_are_reused_by_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "phase.json"
            session = {"manifest": {"resolved_roles": {}, "impulse_files": {}, "detected_layout": "2way"}, "audit": {}}
            with patch.object(optimizer, "crossover_phase_diagnostics", return_value=[{"name": "x"}]) as compute:
                first, first_meta = optimizer.cached_crossover_phase_diagnostics(
                    cache, np.array([1.0]), {}, {}, session
                )
                second, second_meta = optimizer.cached_crossover_phase_diagnostics(
                    cache, np.array([1.0]), {}, {}, session
                )
            self.assertEqual(first, second)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(first_meta["source"], "computed")
            self.assertEqual(second_meta["source"], "cache")


class BeamSearchTests(unittest.TestCase):
    def test_beam_is_deterministic_and_keeps_best_partial_combination(self) -> None:
        first_group = next(iter(optimizer.GROUPS))
        pools = {name: [] for name in optimizer.GROUPS}
        pools[first_group] = [
            {"F": 500.0, "Q": 1.0, "G": -2.0, "strength": 3.0},
            {"F": 800.0, "Q": 1.2, "G": -1.5, "strength": 2.0},
        ]

        def score(groups):
            gain = sum(abs(band[2]) for bands in groups.values() for band in bands)
            return {"objective": -gain}

        a, eval_a = stream.deterministic_beam_combinations(pools, score, beam_width=6, pool_limit=2)
        b, eval_b = stream.deterministic_beam_combinations(pools, score, beam_width=6, pool_limit=2)
        self.assertEqual([(v, s) for v, s, _g in a], [(v, s) for v, s, _g in b])
        self.assertEqual(eval_a, eval_b)
        self.assertLess(a[0][0], 0.0)


class PositionDiscoveryTests(unittest.TestCase):
    def test_discovers_prefixed_or_subfolder_position_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "Left Ear System Sum.txt"
            path.write_text("20 70", encoding="utf-8")
            self.assertEqual(
                first_position_existing(root, ("Left Ear ",), ("System Sum.txt",)), path
            )


if __name__ == "__main__":
    unittest.main()
