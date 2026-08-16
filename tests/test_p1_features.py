from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    def test_checkpoint_replace_retries_a_transient_windows_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "stream_state.json"
            args = SimpleNamespace(
                seed=1,
                profile="explore",
                proposal="beam",
                mode="peq",
                filter_cost_scale=0.1,
                worst_weight=0.1,
                min_total_bands=0,
                archive_size=10,
            )
            rng = np.random.default_rng(1)
            groups = {name: [] for name in optimizer.GROUPS}
            best = [(1.0, optimizer.bands_signature(groups), groups)]
            real_replace = Path.replace
            calls = 0

            def transient_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError("transient reader lock")
                return real_replace(source, destination)

            with (
                patch.object(Path, "replace", transient_replace),
                patch.object(stream.time, "sleep"),
            ):
                stream.save_state(state, best, rng, 5, 1.0, args)

            self.assertEqual(calls, 2)
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8"))["completed_trials"], 5,
            )

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
    def test_peq_beam_continues_with_guided_variations_until_its_deadline(self) -> None:
        self.assertTrue(stream.beam_uses_timed_guided_continuation("beam", "peq"))
        self.assertFalse(stream.beam_uses_timed_guided_continuation("beam", "phase"))
        self.assertFalse(stream.beam_uses_timed_guided_continuation("guided", "peq"))

    def test_guided_pool_can_express_broad_matched_front_target_shape(self) -> None:
        freqs = np.geomspace(80.0, 16000.0, 768)
        flat = np.full_like(freqs, 60.0)
        inactive = np.full_like(freqs, -100.0)
        pair = optimizer.power_sum_db([flat, flat])
        system = optimizer.power_sum_db([pair, pair, inactive])
        requested = 5.0 * np.exp(-0.5 * (np.log2(freqs / 2700.0) / 0.45) ** 2)
        traces = {
            "FL High": flat, "FR High": flat,
            "FL Low": flat, "FR Low": flat,
            "Tweeters Together": pair, "Mid Bass Together": pair,
            "Sub": inactive, "System Sum": system,
        }
        pools = stream.find_guided_candidates(freqs, traces, system + requested, "safe")
        candidates = pools["front_voicing"]

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["source"], "target_shape")
        self.assertGreater(candidates[0]["G"], 0.0)
        self.assertLessEqual(candidates[0]["Q"], 1.8)

    def test_crossover_peak_is_offered_to_all_three_channel_scopes(self) -> None:
        freqs = np.geomspace(200.0, 8000.0, 768)
        flat = np.full_like(freqs, 60.0)
        inactive = np.full_like(freqs, -100.0)
        pair = optimizer.power_sum_db([flat, flat])
        base_system = optimizer.power_sum_db([pair, pair, inactive])
        peak = 6.0 * np.exp(-0.5 * (np.log2(freqs / 2650.0) / 0.16) ** 2)
        traces = {
            "FL High": flat, "FR High": flat,
            "FL Low": flat, "FR Low": flat,
            "Tweeters Together": pair, "Mid Bass Together": pair,
            "Sub": inactive, "System Sum": base_system + peak,
        }

        pools = stream.find_guided_candidates(freqs, traces, base_system, "safe")

        for group in ("high_crossover_sym", "low_crossover_sym", "front_voicing"):
            cuts = [
                candidate for candidate in pools[group]
                if 2400.0 <= candidate["F"] <= 2900.0 and candidate["G"] < 0.0
            ]
            self.assertTrue(cuts, group)

    def test_guided_pool_proposes_editing_an_existing_band_not_only_appending(self) -> None:
        """DEFECT 1: real fix. With a baseline sub cut already at 33 Hz that
        is too shallow for the measured deviation there, the pool must offer
        an edit of THAT band (deepening it), not only free-slot append
        candidates elsewhere. See CHANGELOG.md."""
        freqs = np.geomspace(20.0, 16000.0, 1024)
        inactive = np.full_like(freqs, -100.0)
        front_pair = optimizer.power_sum_db([inactive, inactive])
        sub_bump = 8.0 * np.exp(-0.5 * (np.log2(freqs / 33.0) / 0.30) ** 2)
        sub_flat = np.where(freqs <= 200.0, 60.0, -100.0)
        sub_measured = np.where(freqs <= 200.0, 60.0 + sub_bump, -100.0)
        system = optimizer.power_sum_db([front_pair, front_pair, sub_measured])
        target = optimizer.power_sum_db([front_pair, front_pair, sub_flat])
        traces = {
            "FL High": inactive, "FR High": inactive,
            "FL Low": inactive, "FR Low": inactive,
            "Tweeters Together": front_pair, "Mid Bass Together": front_pair,
            "Sub": sub_measured, "System Sum": system,
        }
        baseline = [[] for _ in range(8)]
        baseline[6] = [(33.0, 2.0, -2.0)]
        baseline[7] = [(33.0, 2.0, -2.0)]

        with patch.object(optimizer, "baseline_band_sets", return_value=baseline):
            pools = stream.find_guided_candidates(freqs, traces, target, "safe")

        edits = [
            candidate for candidate in pools["sub"]
            if candidate.get("source") == "tonal_edit"
            and abs(candidate["F"] - 33.0) < 2.0
        ]
        self.assertTrue(edits, pools["sub"])
        self.assertEqual(edits[0]["edit_target"], (33.0, 2.0, -2.0))
        self.assertLess(edits[0]["G"], -2.0)  # deeper than the existing -2.0 dB cut

    def test_candidate_peaks_proposes_removal_when_existing_band_is_no_longer_justified(
        self,
    ) -> None:
        """DEFECT 1, removal side: when the data-supported setting for an
        existing band rounds to nothing (too small to be a real filter),
        candidate_peaks must propose retiring it via the G=0.0 sentinel, not
        silently drop the finding the way a plain append candidate would be
        dropped. See CHANGELOG.md."""
        freqs = np.geomspace(20.0, 200.0, 512)
        target_index = int(np.argmin(np.abs(np.log10(freqs) - np.log10(33.0))))
        # A narrow single-bin spike is diluted to near-nothing by ERB
        # smoothing inside candidate_peaks, so use a bump wide enough to
        # survive it - this test is about the removal branch, not smoothing.
        strength = 3.0 * np.exp(-0.5 * (np.log2(freqs / 33.0) / 0.25) ** 2)
        desired_gain = np.full_like(freqs, -0.3)  # clears the 0.25 dB gate, rounds to <0.5 dB

        candidates = stream.candidate_peaks(
            freqs, strength, desired_gain, 30.0, 90.0, (0.5, 5.0), (-6.0, 0.0),
            "tonal", "safe", forced_targets={target_index: (33.0, 2.0, -2.0)},
        )

        removals = [c for c in candidates if c["source"] == "tonal_remove"]
        self.assertTrue(removals, candidates)
        self.assertEqual(removals[0]["F"], 33.0)
        self.assertEqual(removals[0]["G"], 0.0)
        self.assertEqual(removals[0]["edit_target"], (33.0, 2.0, -2.0))

    def _sub_bump_scenario(self):
        freqs = np.geomspace(20.0, 16000.0, 1024)
        inactive = np.full_like(freqs, -100.0)
        front_pair = optimizer.power_sum_db([inactive, inactive])
        sub_bump = 8.0 * np.exp(-0.5 * (np.log2(freqs / 33.0) / 0.30) ** 2)
        sub_flat = np.where(freqs <= 200.0, 60.0, -100.0)
        sub_measured = np.where(freqs <= 200.0, 60.0 + sub_bump, -100.0)
        system = optimizer.power_sum_db([front_pair, front_pair, sub_measured])
        target = optimizer.power_sum_db([front_pair, front_pair, sub_flat])
        traces = {
            "FL High": inactive, "FR High": inactive,
            "FL Low": inactive, "FR Low": inactive,
            "Tweeters Together": front_pair, "Mid Bass Together": front_pair,
            "Sub": sub_measured, "System Sum": system,
        }
        return freqs, traces, target, system, sub_bump

    def test_persistence_gate_suppresses_a_deviation_only_the_primary_session_shows(
        self,
    ) -> None:
        """DEFECT 6: a single MMM session can't tell a real deviation from
        run-to-run capture noise. Extra sessions whose System Sum matches the
        target (no bump - exactly what noise on the primary session would
        look like if it weren't real) must suppress the candidate, even
        though the primary session alone clearly supports it."""
        freqs, traces, target, system, sub_bump = self._sub_bump_scenario()

        def near_bump(candidates):
            return any(abs(np.log2(c["F"] / 33.0)) < 1.0 for c in candidates)

        without_sessions = stream.find_guided_candidates(freqs, traces, target, "safe")
        self.assertTrue(near_bump(without_sessions["sub"]))

        quiet_session = {"system_sum": system - sub_bump}
        gated = stream.find_guided_candidates(
            freqs, traces, target, "safe",
            persistence_sessions=[quiet_session, quiet_session],
        )
        self.assertFalse(near_bump(gated["sub"]))

    def test_persistence_gate_allows_a_deviation_confirmed_by_every_session(self) -> None:
        """DEFECT 6, positive path: when every supplied session's System Sum
        shows the same bump, the candidate must survive and carry the
        session count it was confirmed against - the evidence success
        criterion #15 needs in the final report."""
        freqs, traces, target, system, _sub_bump = self._sub_bump_scenario()
        matching_session = {"system_sum": system}
        gated = stream.find_guided_candidates(
            freqs, traces, target, "safe",
            persistence_sessions=[matching_session, matching_session],
        )
        matches = [c for c in gated["sub"] if abs(np.log2(c["F"] / 33.0)) < 1.0]
        self.assertTrue(matches, gated["sub"])
        self.assertEqual(matches[0]["persistence_session_count"], 3)

    def test_persistence_gate_anchors_each_session_before_comparing(self) -> None:
        """DEFECT 6 methodology gap found while reviewing the real v9
        re-run: two REW sessions rarely share the exact same absolute source
        volume or mic gain. Without per-session level anchoring, a session
        that is simply captured several dB louder/quieter overall - but
        otherwise flat, i.e. matches target's SHAPE exactly - would look
        like it "confirms" a positive deviation at every frequency purely
        from its own broadband level offset. Anchoring each session to
        `target` first (same convention `target` itself was anchored to the
        primary session with) must recognize this session has no real
        spectral deviation and refuse to let it confirm anything."""
        freqs, traces, target, system, _sub_bump = self._sub_bump_scenario()
        flat_but_louder_session = {"system_sum": target + 5.0}
        gated = stream.find_guided_candidates(
            freqs, traces, target, "safe",
            persistence_sessions=[flat_but_louder_session, flat_but_louder_session],
        )
        self.assertFalse(any(abs(np.log2(c["F"] / 33.0)) < 1.0 for c in gated["sub"]))

    def test_beam_selects_tweeter_only_crossover_scope_when_rms_is_lower(self) -> None:
        band = {
            "F": 2642.7, "Q": 2.0, "G": -1.5, "strength": 2.0,
            "width_oct": 0.25, "source": "crossover_scope",
        }
        pools = {name: [] for name in optimizer.GROUPS}
        for group in ("high_crossover_sym", "low_crossover_sym", "front_voicing"):
            pools[group] = [dict(band)]

        def score(groups):
            active = [
                name for name in ("high_crossover_sym", "low_crossover_sym", "front_voicing")
                if groups.get(name)
            ]
            if active == ["high_crossover_sym"]:
                value = 1.031
            elif active == ["front_voicing"]:
                value = 1.053
            elif active == ["low_crossover_sym"]:
                value = 1.071
            elif active:
                value = 1.090
            else:
                value = 1.100
            return {"objective": value}

        ranked, _ = stream.deterministic_beam_combinations(
            pools, score, beam_width=12, pool_limit=1
        )
        self.assertAlmostEqual(ranked[0][0], 1.031, places=6)
        self.assertEqual(ranked[0][2]["high_crossover_sym"], [(2642.7, 2.0, -1.5)])
        self.assertFalse(ranked[0][2]["front_voicing"])
        self.assertFalse(ranked[0][2]["low_crossover_sym"])

    def test_adaptive_peak_spacing_keeps_strong_2671_hz_neighbour(self) -> None:
        freqs = np.geomspace(1800.0, 3600.0, 2048)
        strength = (
            7.0 * np.exp(-0.5 * (np.log2(freqs / 2400.0) / 0.025) ** 2)
            + 8.0 * np.exp(-0.5 * (np.log2(freqs / 2671.0) / 0.025) ** 2)
        )
        candidates = stream.candidate_peaks(
            freqs,
            strength,
            -0.25 * strength,
            1800.0,
            3600.0,
            (0.5, 6.0),
            (-6.0, 0.0),
            "tonal",
            "safe",
        )
        centres = [item["F"] for item in candidates]
        self.assertTrue(any(abs(center - 2400.0) < 80.0 for center in centres))
        self.assertTrue(any(abs(center - 2671.0) < 80.0 for center in centres))

    def test_recoverable_error_allocates_more_search_budget(self) -> None:
        pools = {name: [] for name in optimizer.GROUPS}
        pools["high_crossover_sym"] = [
            {"strength": 8.0}, {"strength": 6.0}, {"strength": 4.0}
        ]
        pools["sub"] = [{"strength": 0.5}]
        budgets = stream.search_budgets(pools, pool_limit=6, beam_width=24)
        self.assertGreater(
            budgets["high_crossover_sym"]["pool_limit"],
            budgets["sub"]["pool_limit"],
        )
        self.assertGreater(
            budgets["high_crossover_sym"]["beam_width"],
            budgets["sub"]["beam_width"],
        )

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


class FamilySelectionTests(unittest.TestCase):
    def test_single_safe_candidate_only_writes_balanced_family(self) -> None:
        row = {"components": {"objective": 1.0}, "name": "baseline"}
        picks = optimizer.select_family_rows([row])
        self.assertEqual(list(picks), ["balanced"])
        self.assertIs(picks["balanced"], row)
    def test_equal_family_scores_do_not_compare_row_dicts(self) -> None:
        metrics = {
            "objective": 1.0,
            "pareto_tonal_db": 1.0,
            "tonal_error_db": 1.0,
            "sum_tonal_anchor_db": 1.0,
            "presence_error_db": 1.0,
            "peak_penalty_db": 1.0,
            "balance_penalty_db": 1.0,
            "positive_gain_penalty_db": 1.0,
            "filter_count": 1.0,
        }
        rows = [{"components": dict(metrics), "name": name} for name in ("a", "b")]
        ranked = optimizer.family_pick_scores(rows, "balanced")
        self.assertEqual([row["name"] for _score, row in ranked], ["a", "b"])


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
