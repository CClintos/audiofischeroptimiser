from __future__ import annotations

import re
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import baseline_rehabilitation as rehab
import _optimizer_stream as stream
import _merge_stream_results as merge_stream
import _optimizer as optimizer
from _make_v3 import decode_afpx


def fixture_afpx_xml(active_by_channel):
    channels = []
    for channel in range(8):
        active = {
            slot: (frequency, q, gain)
            for slot, frequency, q, gain in active_by_channel.get(channel, ())
        }
        filters = []
        for slot in range(12):
            if slot in active:
                frequency, q, gain = active[slot]
                filters.append(
                    '<Fil T="17" F="%.2f" Q="%s" G="%s" dF="%d" FN="%d"/>'
                    % (frequency, q, gain, slot + 25, slot)
                )
            else:
                filters.append(
                    '<Fil T="1" F="%.2f" Q="4.3" G="0" dF="%d" FN="%d"/>'
                    % (25.0 + slot, slot + 25, slot)
                )
        channels.append('<OC Name="channel-%d">%s</OC>' % (channel, "".join(filters)))
    return "<Root>%s</Root>" % "".join(channels)


def filter_slots(xml, channel):
    outputs = re.findall(r"<OC\b.*?</OC>", xml, re.S)
    return [
        dict(re.findall(r'([A-Za-z]+)="([^"]*)"', tag))
        for tag in re.findall(r"<Fil\b[^>]*/?>", outputs[channel])
    ]


def bands_from_afpx(path):
    xml = decode_afpx(path)
    band_sets = []
    for channel in range(8):
        active = [
            (float(slot["F"]), float(slot["Q"]), float(slot["G"]))
            for slot in filter_slots(xml, channel)
            if slot["T"] == "17" and float(slot["G"]) != 0.0
        ]
        band_sets.append(tuple(sorted(active)))
    return tuple(band_sets)

class SlotIdentityTests(unittest.TestCase):
    def test_recentre_beyond_old_frequency_tolerance_keeps_same_slot(self):
        xml = fixture_afpx_xml({2: [(7, 97.0, 3.0, -1.5)]})
        refs = rehab.active_peq_slot_refs(xml, {2: "FL Low"})
        edit = rehab.SlotEdit.modify(refs[0], (100.0, 1.2, -1.5))

        written = rehab.apply_slot_edits(xml, (edit,))

        before_slots = filter_slots(xml, channel=2)
        after_slots = filter_slots(written, channel=2)
        self.assertEqual(after_slots[7]["F"], "100.00")
        self.assertEqual(after_slots[7]["Q"], "1.2")
        self.assertEqual(before_slots[:7] + before_slots[8:], after_slots[:7] + after_slots[8:])

    def test_remove_frees_exact_duplicate_slot_without_frequency_guessing(self):
        xml = fixture_afpx_xml({2: [(4, 100.0, 1.0, -2.0), (9, 100.0, 1.0, -2.0)]})
        refs = rehab.active_peq_slot_refs(xml, {2: "FL Low"})

        written = rehab.apply_slot_edits(xml, (rehab.SlotEdit.remove(refs[1]),))

        slots = filter_slots(written, channel=2)
        self.assertEqual(slots[4]["T"], "17")
        self.assertEqual(slots[9]["T"], "1")

    def test_write_rejects_slot_changed_since_census(self):
        xml = fixture_afpx_xml({2: [(7, 97.0, 3.0, -1.5)]})
        ref = rehab.active_peq_slot_refs(xml, {2: "FL Low"})[0]
        changed = xml.replace('F="97.00"', 'F="98.00"', 1)

        with self.assertRaisesRegex(
            ValueError, "AFPX slot changed since rehabilitation census"
        ):
            rehab.apply_slot_edits(changed, (rehab.SlotEdit.remove(ref),))


class PlanResolverTests(unittest.TestCase):
    def setUp(self):
        self.xml = fixture_afpx_xml({2: [(7, 97.0, 3.0, -1.5)]})
        self.ref_97 = rehab.active_peq_slot_refs(self.xml, {2: "FL Low"})[0]
        self.baseline = ((), (), ((97.0, 3.0, -1.5),), (), (), (), (), ())
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.out = Path(self.temp_dir.name) / "candidate.afpx"

    def test_plan_resolution_is_identical_for_score_report_and_write(self):
        plan = rehab.CandidatePlan(
            slot_edits=(
                rehab.SlotEdit.modify(
                    self.ref_97, (100.1234, 1.2345, -1.5678)
                ),
            ),
            groups=rehab.freeze_groups({
                "low_sym": [(184.126, 0.63456, -3.98765)]
            }),
        )

        class FakeObjective:
            def __init__(self):
                self.scored = []

            def score_bands(self, band_sets):
                self.scored.append(band_sets)
                return {"objective": 1.2345678901234567, "tonal_masked": 2.0}

        objective = FakeObjective()
        with patch.object(
            optimizer, "baseline_band_sets", return_value=self.baseline
        ), patch.object(optimizer, "phase_peq_conflicts", return_value=[]), patch.object(
            optimizer, "AFPX_OBJECTIVE", objective
        ):
            resolved = optimizer.resolve_candidate_plan(plan)
            score = optimizer.make_band_set_component_scorer(
                np.array([100.0]), {}, np.array([0.0])
            )(resolved.band_sets)
            written = optimizer.write_candidate_plan(self.xml, self.out, plan)

        parsed = bands_from_afpx(self.out)
        self.assertEqual(objective.scored, [resolved.band_sets])
        self.assertEqual(score["filter_count"], sum(map(len, parsed)))
        self.assertIn((100.12, 1.2345, -1.5678), resolved.band_sets[2])
        self.assertIn((184.13, 0.63456, -3.98765), resolved.band_sets[2])
        canonical_plan = rehab.CandidatePlan(
            slot_edits=(
                rehab.SlotEdit.modify(
                    self.ref_97, (100.12, 1.2345, -1.5678)
                ),
            ),
            groups=rehab.freeze_groups({
                "low_sym": [(184.13, 0.63456, -3.98765)]
            }),
        )
        self.assertEqual(
            resolved.signature,
            optimizer.candidate_plan_signature(canonical_plan),
        )
        self.assertEqual(parsed, resolved.band_sets)
        self.assertEqual(written["operation_signature"], resolved.signature)
        self.assertEqual(
            [
                (row["channel_index"], row["slot_index"])
                for row in written["permitted_filter_slot_changes"]
            ],
            [(2, 7)],
        )

    def test_unchanged_plan_resolves_to_exact_baseline(self):
        baseline = (((200.0, 1.0, -2.0), (100.0, 1.2, -1.0)),) + ((),) * 7
        with patch.object(optimizer, "baseline_band_sets", return_value=baseline):
            resolved = optimizer.resolve_candidate_plan(rehab.CandidatePlan())

        self.assertEqual(resolved.band_sets, baseline)

    def test_frozen_groups_preserve_resolution_order(self):
        baseline = (
            ((1000.0, 1.5, -2.0),),
            ((1000.0, 1.5, -2.0),),
        ) + ((),) * 6
        plan = rehab.CandidatePlan(
            groups=rehab.freeze_groups({
                "high_sym": [(1010.0, 1.4, -2.5)],
                "fl_high": [(1020.0, 1.3, -3.0)],
            })
        )

        with patch.object(optimizer, "baseline_band_sets", return_value=baseline):
            resolved = optimizer.resolve_candidate_plan(plan)

        self.assertEqual(resolved.band_sets[0], ((1020.0, 1.3, -3.0),))

    def test_duplicate_slot_edits_are_rejected(self):
        plan = rehab.CandidatePlan(
            slot_edits=(
                rehab.SlotEdit.remove(self.ref_97),
                rehab.SlotEdit.modify(self.ref_97, (100.0, 1.2, -1.5)),
            )
        )
        with patch.object(optimizer, "baseline_band_sets", return_value=self.baseline):
            with self.assertRaisesRegex(ValueError, "duplicate rehabilitation slot edit"):
                optimizer.resolve_candidate_plan(plan)

    def test_group_cannot_reedit_slot_removed_by_rehabilitation(self):
        plan = rehab.CandidatePlan(
            slot_edits=(rehab.SlotEdit.remove(self.ref_97),),
            groups=rehab.freeze_groups({"low_sym": [(97.0, 1.2, -2.0)]}),
        )
        with patch.object(optimizer, "baseline_band_sets", return_value=self.baseline):
            with self.assertRaisesRegex(ValueError, "group action targets removed"):
                optimizer.resolve_candidate_plan(plan)

    def test_group_cannot_reuse_slot_removed_by_an_earlier_group(self):
        xml = fixture_afpx_xml({
            2: [(4, 100.0, 1.0, -2.0), (9, 102.0, 1.0, -2.0)]
        })
        ref_100 = rehab.active_peq_slot_refs(xml, {2: "FL Low"})[0]
        baseline = (
            (),
            (),
            ((100.0, 1.0, -2.0), (102.0, 1.0, -2.0)),
            ((102.0, 1.0, -2.0),),
        ) + ((),) * 4
        plan = rehab.CandidatePlan(
            slot_edits=(rehab.SlotEdit.remove(ref_100),),
            groups=rehab.freeze_groups({
                "low_sym": [(102.0, 1.0, 0.0)],
                "fl_low": [(102.5, 1.0, -2.0)],
            }),
        )

        with patch.object(optimizer, "baseline_band_sets", return_value=baseline):
            with self.assertRaisesRegex(
                ValueError, "group action targets removed or consumed slot"
            ):
                optimizer.resolve_candidate_plan(plan)

class FilterSearchTests(unittest.TestCase):
    def setUp(self):
        self.ref_97_q3 = rehab.FilterRef(
            channel=2, slot=7, role="FL Low", filter_type="17",
            original=(97.0, 3.0, -1.5), pair_key="low-97",
        )
        self.ref_97_q3_right = rehab.FilterRef(
            channel=3, slot=7, role="FR Low", filter_type="17",
            original=(97.0, 3.0, -1.5), pair_key="low-97",
        )
        self.ref_sub_33 = rehab.FilterRef(
            channel=6, slot=4, role="Sub", filter_type="17",
            original=(33.0, 3.0, -2.0),
        )
        self.refs = (self.ref_97_q3, self.ref_97_q3_right, self.ref_sub_33)
        self.config = rehab.RehabilitationConfig(
            retained_per_slot=8,
            refinement_passes=4,
            max_evaluations_per_slot=5000,
            role_limits=(
                ("FL Low", 80.0, 2000.0, 0.5, 2.5, -6.0, 3.0),
                ("FR Low", 80.0, 2000.0, 0.5, 2.5, -6.0, 3.0),
                ("Sub", 30.0, 90.0, 0.5, 5.0, -6.0, 0.0),
            ),
            paired_role_limits=(
                ("FL Low", "FR Low", 80.0, 2600.0, 0.5, 5.0, -6.0, 0.0),
            ),
        )

    def score_plan(self, plan):
        objective = 20.0
        balance = 2.0
        headroom = 1.0
        for edit in plan.slot_edits:
            if edit.replacement is None:
                objective += 3.0
                continue
            frequency, q, gain = edit.replacement
            if edit.ref.role == "Sub":
                objective += (
                    abs(frequency - 33.0) * 0.2
                    + abs(q - 2.2) * 2.0
                    + abs(gain + 3.5)
                )
                headroom += max(gain, 0.0)
            else:
                objective += (
                    abs(frequency - 100.0) * 0.1
                    + abs(q - 1.2) * 2.0
                    + abs(gain + 1.5)
                )
                if len(plan.slot_edits) == 1:
                    balance += 1.0
        return {
            "objective": objective,
            "balance_penalty_db": balance,
            "positive_gain_penalty_db": headroom,
            "filter_count": 3.0 - sum(
                edit.replacement is None for edit in plan.slot_edits
            ),
        }

    def test_paired_search_rejects_missing_symmetric_limits(self):
        config = rehab.RehabilitationConfig(
            role_limits=self.config.role_limits,
            retained_per_slot=1,
            max_evaluations_per_slot=2,
        )

        with self.assertRaisesRegex(ValueError, "symmetric limits"):
            rehab.search_filter_operations(
                self.ref_97_q3,
                self.score_plan,
                config,
                paired_ref=self.ref_97_q3_right,
            )

    def test_paired_search_uses_symmetric_limits_but_asymmetric_uses_side_limits(
        self
    ):
        config = optimizer.rehabilitation_config(
            {2: "FL Low", 3: "FR Low"},
            retained_per_slot=12,
            max_evaluations_per_slot=5000,
        )

        def favour_side_only_values(plan):
            if not plan.slot_edits:
                return {
                    "objective": 20.0,
                    "balance_penalty_db": 2.0,
                    "positive_gain_penalty_db": 1.0,
                }
            replacement = plan.slot_edits[0].replacement
            if replacement is None:
                return {
                    "objective": 30.0,
                    "balance_penalty_db": 2.0,
                    "positive_gain_penalty_db": 1.0,
                }
            _frequency, q, gain = replacement
            return {
                "objective": abs(q - 5.5) + abs(gain - 1.0),
                "balance_penalty_db": 2.0,
                "positive_gain_penalty_db": max(gain, 0.0),
            }

        candidates = rehab.search_filter_operations(
            self.ref_97_q3,
            favour_side_only_values,
            config,
            paired_ref=self.ref_97_q3_right,
            asymmetry_eligible=lambda ref, replacement: replacement is not None,
        )
        paired = [item for item in candidates if len(item.edits) == 2]
        one_sided = [item for item in candidates if len(item.edits) == 1]

        self.assertEqual(
            config.limits_for_refs((self.ref_97_q3, self.ref_97_q3_right)),
            (80.0, 2600.0, 0.5, 5.0, -6.0, 0.0),
        )
        self.assertTrue(paired)
        self.assertTrue(all(
            item.edit.replacement[1] <= 5.0
            and item.edit.replacement[2] <= 0.0
            for item in paired if item.edit.replacement
        ))
        self.assertTrue(any(
            item.edit.replacement[1] > 5.0
            or item.edit.replacement[2] > 0.0
            for item in one_sided if item.edit.replacement
        ))

    def test_census_attaches_distinct_owner_rankings_to_retained_regions(self):
        low_left = rehab.FilterRef(
            2, 1, "FL Low", "17", (80.0, 1.0, -2.0), "low-80"
        )
        low_right = rehab.FilterRef(
            3, 1, "FR Low", "17", (80.0, 1.0, -2.0), "low-80"
        )
        sub = rehab.FilterRef(6, 1, "Sub", "17", (60.0, 1.0, -2.0))
        config = rehab.RehabilitationConfig(
            retained_per_slot=6,
            refinement_passes=2,
            max_evaluations_per_slot=2500,
            role_limits=(
                ("FL Low", 50.0, 500.0, 0.5, 6.0, -6.0, 3.0),
                ("FR Low", 50.0, 500.0, 0.5, 6.0, -6.0, 3.0),
                ("Sub", 30.0, 90.0, 0.5, 5.0, -6.0, 0.0),
            ),
            paired_role_limits=(
                ("FL Low", "FR Low", 50.0, 500.0, 0.5, 5.0, -6.0, 0.0),
            ),
        )

        def score_plan(plan):
            if not plan.slot_edits:
                return {
                    "objective": 100.0,
                    "sum_rms_db": 100.0,
                    "balance_penalty_db": 2.0,
                    "positive_gain_penalty_db": 1.0,
                }
            replacement = plan.slot_edits[0].replacement
            if replacement is None:
                value = 120.0
            else:
                frequency, q, gain = replacement
                channels = {edit.ref.channel for edit in plan.slot_edits}
                target = 60.0 if channels == {6} else 80.0
                value = (
                    abs(frequency - target)
                    + abs(q - 1.0)
                    + abs(gain + 2.25)
                )
            return {
                "objective": value,
                "sum_rms_db": value,
                "balance_penalty_db": 2.0,
                "positive_gain_penalty_db": 1.0,
            }

        rows = rehab.build_filter_census(
            (low_left, low_right, sub),
            score_plan,
            config,
        )
        retained = [
            candidate
            for row in rows
            for candidate in row.candidates
            if candidate.region is not None
        ]
        low_region = next(
            item for item in retained
            if abs(item.region.frequency - 80.0) <= 0.1
            and abs(item.region.q - 1.0) <= 0.01
        )
        sub_region = next(
            item for item in retained
            if abs(item.region.frequency - 60.0) <= 0.1
            and abs(item.region.q - 1.0) <= 0.01
        )

        expected_owners = {frozenset({2, 3}), frozenset({6})}
        for candidate in (low_region, sub_region):
            self.assertEqual(
                {
                    frozenset(ref.channel for ref in item.owner_refs)
                    for item in candidate.owner_attributions
                },
                expected_owners,
            )
        self.assertEqual(
            {ref.channel for ref in low_region.owner_attributions[0].owner_refs},
            {2, 3},
        )
        self.assertEqual(
            {ref.channel for ref in sub_region.owner_attributions[0].owner_refs},
            {6},
        )
        self.assertFalse(hasattr(rows[0], "driver_rank"))

    def test_region_attribution_ranks_same_probe_across_low_pair_and_sub(self):
        low_left = rehab.FilterRef(
            2, 1, "FL Low", "17", (80.0, 1.0, -2.0), "low-80"
        )
        low_right = rehab.FilterRef(
            3, 1, "FR Low", "17", (80.0, 1.0, -2.0), "low-80"
        )
        sub = rehab.FilterRef(6, 1, "Sub", "17", (80.0, 1.0, -2.0))
        config = rehab.RehabilitationConfig(
            role_limits=(
                ("FL Low", 50.0, 500.0, 0.5, 6.0, -6.0, 3.0),
                ("FR Low", 50.0, 500.0, 0.5, 6.0, -6.0, 3.0),
                ("Sub", 30.0, 90.0, 0.5, 5.0, -6.0, 0.0),
            ),
            paired_role_limits=(
                ("FL Low", "FR Low", 50.0, 500.0, 0.5, 5.0, -6.0, 0.0),
            ),
        )

        def score_owner(plan):
            if not plan.slot_edits:
                return {
                    "objective": 10.0,
                    "sum_rms_db": 5.0,
                    "balance_penalty_db": 2.0,
                    "positive_gain_penalty_db": 1.0,
                }
            channels = {edit.ref.channel for edit in plan.slot_edits}
            if channels == {6}:
                values = (6.0, 3.0, 1.5, 1.0)
            else:
                values = (8.0, 4.0, 1.0, 1.2)
            return {
                "objective": values[0],
                "sum_rms_db": values[1],
                "balance_penalty_db": values[2],
                "positive_gain_penalty_db": values[3],
            }

        ranked = rehab.attribute_correction_region(
            rehab.CorrectionRegion(80.0, 1.0, -0.25),
            ((low_left, low_right), (sub,)),
            score_owner,
            config,
        )

        self.assertEqual(ranked[0].owner_refs, (sub,))
        self.assertEqual([item.rank for item in ranked], [1, 2])
        self.assertEqual(
            {item.probe_band[:2] for item in ranked},
            {(80.0, 1.0)},
        )
        self.assertLess(ranked[0].system_delta, ranked[1].system_delta)
        self.assertNotEqual(ranked[0].balance_delta, ranked[1].balance_delta)
        self.assertNotEqual(ranked[0].headroom_delta, ranked[1].headroom_delta)

    def test_census_probe_is_bounded_to_configured_q_cap(self):
        captured = []

        def capture_score(plan):
            for edit in plan.slot_edits:
                if edit.replacement is not None:
                    captured.append(edit.replacement)
            return {
                "objective": 10.0,
                "balance_penalty_db": 2.0,
                "positive_gain_penalty_db": 1.0,
            }

        row = rehab.build_filter_census(
            (self.ref_97_q3,),
            capture_score,
            self.config,
        )[0]

        self.assertIsNone(row.probe_skip_reason)
        self.assertTrue(captured)
        self.assertLessEqual(row.probe_band[1], 2.5)
        self.assertTrue(all(80.0 <= band[0] <= 2000.0 for band in captured))
        self.assertTrue(all(0.5 <= band[1] <= 2.5 for band in captured))

    def test_census_probe_records_reason_when_no_valid_perturbation_exists(self):
        fixed = rehab.FilterRef(2, 2, "Fixed", "17", (100.0, 1.0, -2.0))
        config = rehab.RehabilitationConfig(
            role_limits=(
                ("Fixed", 100.0, 100.0, 1.0, 1.0, -2.0, -2.0),
            ),
            retained_per_slot=1,
            max_evaluations_per_slot=1,
        )

        row = rehab.build_filter_census(
            (fixed,), self.score_plan, config
        )[0]

        self.assertIsNone(row.probe_components)
        self.assertIsNone(row.probe_band)
        self.assertEqual(row.probe_skip_reason, "no valid bounded perturbation")

    def test_optimizer_plan_scorer_resolves_before_direct_scoring(self):
        captured = []
        baseline = ((), (), ((97.0, 3.0, -1.5),),) + ((),) * 5
        plan = rehab.CandidatePlan(slot_edits=(
            rehab.SlotEdit.modify(self.ref_97_q3, (100.0, 1.2, -1.5)),
        ))

        with patch.object(optimizer, "baseline_band_sets", return_value=baseline):
            score = optimizer.make_candidate_plan_scorer(
                lambda band_sets: captured.append(band_sets)
                or {"objective": 1.0}
            )(plan)

        self.assertEqual(score["objective"], 1.0)
        self.assertEqual(
            captured,
            [((), (), ((100.0, 1.2, -1.5),),) + ((),) * 5],
        )

    def test_optimizer_config_uses_detected_role_passbands(self):
        config = optimizer.rehabilitation_config({
            2: "FL Low",
            3: "FR Low",
            6: "Sub",
        })

        self.assertEqual(config.limits_for(self.ref_97_q3)[:2], (80.0, 2000.0))
        self.assertEqual(config.limits_for(self.ref_sub_33)[:2], (30.0, 90.0))

    def test_every_eligible_existing_filter_gets_removal_trial(self):
        result = rehab.build_filter_census(self.refs, self.score_plan, self.config)
        self.assertEqual({row.ref for row in result}, set(self.refs))
        self.assertTrue(all(row.removal_components is not None for row in result))

    def test_coarse_search_reaches_known_recentre_and_q_change(self):
        candidates = rehab.search_filter_operations(
            self.ref_97_q3, self.score_plan, self.config
        )
        settings = {
            item.edit.replacement for item in candidates if item.edit.replacement
        }
        self.assertIn((100.0, 1.2, -1.5), settings)

    def test_sub_search_reaches_gain_and_q_change(self):
        candidates = rehab.search_filter_operations(
            self.ref_sub_33, self.score_plan, self.config
        )
        self.assertTrue(any(
            abs(item.edit.replacement[0] - 33.0) <= 0.1
            and abs(item.edit.replacement[1] - 2.2) <= 0.01
            and item.edit.replacement[2] <= -3.25
            for item in candidates if item.edit.replacement
        ))

    def test_matched_filters_are_searched_as_one_symmetric_operation(self):
        rows = rehab.build_filter_census(
            (self.ref_97_q3, self.ref_97_q3_right),
            self.score_plan,
            self.config,
        )
        paired = [
            candidate
            for row in rows
            for candidate in row.candidates
            if len(candidate.edits) == 2
        ]
        self.assertTrue(paired)
        self.assertTrue(all(
            {edit.ref.channel for edit in candidate.edits} == {2, 3}
            for candidate in paired
        ))
        self.assertFalse(any(
            len(candidate.edits) == 1
            for row in rows
            for candidate in row.candidates
        ))

    def test_asymmetric_variant_requires_evidence_callback(self):
        rows = rehab.build_filter_census(
            (self.ref_97_q3, self.ref_97_q3_right),
            self.score_plan,
            self.config,
            asymmetry_eligible=lambda ref, replacement: ref.channel == 2,
        )
        one_sided = [
            candidate
            for row in rows
            for candidate in row.candidates
            if len(candidate.edits) == 1
        ]
        self.assertTrue(one_sided)
        self.assertTrue(all(item.edit.ref.channel == 2 for item in one_sided))


    def test_rejected_asymmetric_variants_never_reach_scorer(self):
        def paired_only_scorer(plan):
            if len(plan.slot_edits) == 1:
                raise AssertionError("one-sided plan reached scorer without evidence")
            return self.score_plan(plan)

        rows = rehab.build_filter_census(
            (self.ref_97_q3, self.ref_97_q3_right),
            paired_only_scorer,
            self.config,
            asymmetry_eligible=lambda ref, replacement: False,
        )

        self.assertTrue(rows)
        self.assertTrue(all(
            len(candidate.edits) == 2
            for row in rows
            for candidate in row.candidates
        ))

    def test_census_records_probe_driver_deltas(self):
        row = rehab.build_filter_census(
            (self.ref_sub_33,), self.score_plan, self.config
        )[0]
        self.assertIsNotNone(row.probe_components)
        self.assertAlmostEqual(
            row.system_delta,
            row.probe_components["objective"]
            - row.baseline_components["objective"],
        )
        self.assertIn("balance_penalty_db", row.probe_components)
        self.assertIn("positive_gain_penalty_db", row.probe_components)

class InteractionBeamTests(unittest.TestCase):
    def setUp(self):
        self.left = rehab.FilterRef(2, 1, "FL Low", "17", (500.0, 1.0, -2.0))
        self.second = rehab.FilterRef(2, 2, "FL Low", "17", (900.0, 1.0, -2.0))
        self.edit_left = rehab.SlotEdit.modify(self.left, (520.0, 1.2, -2.5))
        self.edit_second = rehab.SlotEdit.modify(self.second, (880.0, 1.1, -2.5))
        self.baseline = self.components(10.0, 3)
        self.operations = (
            rehab.OperationCandidate(self.edit_left, rehab.CandidatePlan((self.edit_left,)), self.components(11.0, 3), self.baseline),
            rehab.OperationCandidate(self.edit_second, rehab.CandidatePlan((self.edit_second,)), self.components(11.0, 3), self.baseline),
        )

    @staticmethod
    def components(objective, filter_count, tonal=2.0, headroom=4.0):
        return {
            "objective": float(objective), "tonal_masked": float(tonal),
            "presence_error_db": 1.0, "balance_penalty_db": 0.5,
            "positive_gain_penalty_db": 0.0, "asymmetric_eq_penalty": 0.0,
            "filter_count": float(filter_count),
            "headroom_margin_db": {"FL Low": float(headroom)},
            "headroom_violation_count": 0, "nearfield_skirt_violation_count": 0,
            "balance_guardrail_violation_count": 0,
            "filter_noise_floor_violation_count": 0,
        }

    def score_plan(self, plan):
        keys = {(edit.ref.channel, edit.ref.slot) for edit in plan.slot_edits}
        if keys == {(2, 1), (2, 2)}:
            return self.components(5.0, 3, tonal=1.0)
        if keys:
            return self.components(11.0, 3, tonal=2.1)
        return self.baseline

    def test_interaction_beam_keeps_two_edits_that_win_only_together(self):
        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(), self.operations, self.score_plan,
            beam_width=16, max_depth=4,
        )
        self.assertEqual(set(result.best.slot_edits), {self.edit_left, self.edit_second})

    def test_unchanged_baseline_survives_every_generation(self):
        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(), self.operations,
            lambda plan: self.components(10.0 + len(plan.slot_edits), 3, tonal=2.0 + len(plan.slot_edits)),
            beam_width=2, max_depth=2,
        )
        baseline_signature = rehab.candidate_plan_signature(rehab.CandidatePlan())
        self.assertEqual(result.best.slot_edits, ())
        self.assertTrue(all(
            baseline_signature in {rehab.candidate_plan_signature(candidate.plan) for candidate in generation}
            for generation in result.generations
        ))

    def test_supplied_repeatability_keeps_partial_for_one_generation(self):
        refs = tuple(
            rehab.FilterRef(2, slot, "FL Low", "17", (500.0, 1.0, -2.0))
            for slot in (3, 4, 5)
        )
        edits = tuple(
            rehab.SlotEdit.modify(ref, (510.0 + index, 1.0, -2.5))
            for index, ref in enumerate(refs)
        )
        operations = tuple(
            rehab.OperationCandidate(
                edit,
                rehab.CandidatePlan((edit,)),
                self.components(20.0, 3, tonal=2.15),
                self.baseline,
            )
            for edit in edits
        )

        def score(plan):
            if not plan.slot_edits:
                return self.baseline
            slot = plan.slot_edits[0].ref.slot
            if slot == 3:
                return self.components(20.0, 3, tonal=2.15)
            return self.components(float(slot + 5), 3, tonal=3.0)

        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(), operations, score,
            beam_width=3, max_depth=1, repeatability_db=0.2,
        )
        retained = {
            rehab.candidate_plan_signature(candidate.plan)
            for candidate in result.generations[1]
        }
        self.assertIn(
            rehab.candidate_plan_signature(rehab.CandidatePlan((edits[0],))),
            retained,
        )

    def test_global_tie_selection_cannot_chain_past_acoustic_reference(self):
        candidates = tuple(
            rehab.ScoredCandidate(
                rehab.CandidatePlan((
                    rehab.SlotEdit.modify(
                        rehab.FilterRef(
                            2, slot, "FL Low", "17", (500.0, 1.0, -2.0)
                        ),
                        (500.0, 1.0, -2.5),
                    ),
                )),
                self.components(value, filter_count, tonal=value),
            )
            for slot, value, filter_count in (
                (6, 0.00, 3),
                (7, 0.04, 2),
                (8, 0.08, 1),
            )
        )

        winner = rehab.select_best_candidate(
            candidates, repeatability_db=0.05
        )

        self.assertIs(winner, candidates[1])
        self.assertLessEqual(
            abs(winner.components["tonal_masked"] - 0.00), 0.05
        )

    def test_beam_final_selection_uses_global_acoustic_reference(self):
        refs = tuple(
            rehab.FilterRef(2, slot, "FL Low", "17", (500.0, 1.0, -2.0))
            for slot in (6, 7, 8)
        )
        edits = tuple(
            rehab.SlotEdit.modify(ref, (500.0, 1.0, -2.5))
            for ref in refs
        )
        values = {6: (0.00, 3), 7: (0.04, 2), 8: (0.08, 1)}
        operations = tuple(
            rehab.OperationCandidate(
                edit,
                rehab.CandidatePlan((edit,)),
                self.components(*values[edit.ref.slot], tonal=values[edit.ref.slot][0]),
                self.baseline,
            )
            for edit in edits
        )

        def score(plan):
            if not plan.slot_edits:
                return self.components(10.0, 4, tonal=10.0)
            objective, filter_count = values[plan.slot_edits[0].ref.slot]
            return self.components(
                objective, filter_count, tonal=objective
            )

        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(), operations, score,
            beam_width=4, max_depth=1, repeatability_db=0.05,
        )

        self.assertEqual(result.best.slot_edits, (edits[1],))
    def test_survival_only_bridge_cannot_be_exported(self):
        refs = tuple(
            rehab.FilterRef(2, slot, "FL Low", "17", (500.0, 1.0, -2.0))
            for slot in (6, 7)
        )
        edits = tuple(
            rehab.SlotEdit.modify(ref, (510.0 + index, 1.0, -2.5))
            for index, ref in enumerate(refs)
        )
        operations = tuple(
            rehab.OperationCandidate(
                edit,
                rehab.CandidatePlan((edit,)),
                self.components(
                    20.0 + 10.0 * index,
                    2 if index == 0 else 1,
                    tonal=0.04 + index,
                ),
                self.baseline,
            )
            for index, edit in enumerate(edits)
        )

        def score(plan):
            if not plan.slot_edits:
                return self.components(10.0, 3, tonal=0.0)
            slot = plan.slot_edits[0].ref.slot
            if slot == 6:
                return self.components(20.0, 2, tonal=0.04)
            return self.components(30.0, 1, tonal=1.0)

        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(), operations, score,
            beam_width=3, max_depth=1, repeatability_db=0.05,
        )

        bridge_signature = rehab.candidate_plan_signature(
            rehab.CandidatePlan((edits[0],))
        )
        self.assertIn(
            bridge_signature,
            {
                rehab.candidate_plan_signature(candidate.plan)
                for candidate in result.generations[1]
            },
        )
        self.assertEqual(result.best.slot_edits, ())

    def test_bridge_can_seed_meaningful_completed_interaction(self):
        refs = tuple(
            rehab.FilterRef(2, slot, "FL Low", "17", (500.0, 1.0, -2.0))
            for slot in (6, 7, 8)
        )
        edits = tuple(
            rehab.SlotEdit.modify(ref, (510.0 + index, 1.0, -2.5))
            for index, ref in enumerate(refs)
        )
        operations = tuple(
            rehab.OperationCandidate(
                edit,
                rehab.CandidatePlan((edit,)),
                self.components(20.0, 3, tonal=0.04),
                self.baseline,
            )
            for edit in edits
        )

        def score(plan):
            slots = {edit.ref.slot for edit in plan.slot_edits}
            if not slots:
                return self.components(10.0, 3, tonal=0.0)
            if slots == {6}:
                return self.components(20.0, 3, tonal=0.04)
            if slots == {6, 7}:
                return self.components(5.0, 3, tonal=-1.0)
            if slots == {8}:
                return self.components(11.0, 3, tonal=1.0)
            return self.components(40.0, 3, tonal=1.0)

        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(), operations, score,
            beam_width=3, max_depth=2, repeatability_db=0.05,
        )

        bridge = next(
            candidate
            for candidate in result.generations[1]
            if {edit.ref.slot for edit in candidate.slot_edits} == {6}
        )
        self.assertTrue(bridge.bridge_only)
        self.assertFalse(bridge.export_eligible)
        self.assertEqual(
            {edit.ref.slot for edit in result.best.slot_edits}, {6, 7}
        )

    def test_acoustic_tie_prefers_removal(self):
        keep = rehab.ScoredCandidate(rehab.CandidatePlan((self.edit_left,)), self.components(8.0, 79))
        remove_edit = rehab.SlotEdit.remove(self.left)
        remove = rehab.ScoredCandidate(rehab.CandidatePlan((remove_edit,)), self.components(8.3, 78))
        self.assertIs(rehab.compare_candidates(keep, remove, repeatability_db=0.05), remove)

    def test_default_repeatability_is_frequency_aware(self):
        low_edit = rehab.SlotEdit.modify(self.left, (400.0, 1.0, -2.5))
        high_ref = rehab.FilterRef(
            0, 1, "FL High", "17", (10000.0, 1.0, -2.0)
        )
        high_edit = rehab.SlotEdit.modify(high_ref, (10000.0, 1.0, -2.5))
        baseline = rehab.ScoredCandidate(
            rehab.CandidatePlan(), self.components(10.0, 3, tonal=2.0)
        )
        low = rehab.ScoredCandidate(
            rehab.CandidatePlan((low_edit,)),
            self.components(9.0, 3, tonal=2.3),
        )
        high = rehab.ScoredCandidate(
            rehab.CandidatePlan((high_edit,)),
            self.components(9.0, 3, tonal=2.3),
        )
        self.assertIs(rehab.compare_candidates(baseline, low), low)
        self.assertIs(rehab.compare_candidates(baseline, high), baseline)

    def test_merge_rejects_any_channel_headroom_regression(self):
        first = rehab.SlotEdit.modify(self.left, (1000.0, 1.0, -3.0))
        second = rehab.SlotEdit.modify(self.second, (1000.0, 1.0, -0.02))
        plan = rehab.CandidatePlan((first, second))

        def score(candidate_plan):
            count = sum(
                edit.replacement is not None
                for edit in candidate_plan.slot_edits
            )
            components = self.components(5.0, count, tonal=1.0)
            components["headroom_margin_db"] = (
                {"FL Low": 2.0, "FR Low": 5.0}
                if count == 2
                else {"FL Low": 3.0, "FR Low": 4.5}
            )
            return components

        result = rehab.consolidate_candidate(plan, score)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "headroom regressed")

    def test_merge_rejects_boolean_pass_to_fail_gate_transition(self):
        first = rehab.SlotEdit.modify(self.left, (1000.0, 1.0, -3.0))
        second = rehab.SlotEdit.modify(self.second, (1000.0, 1.0, -0.02))
        plan = rehab.CandidatePlan((first, second))

        def score(candidate_plan):
            count = sum(
                edit.replacement is not None
                for edit in candidate_plan.slot_edits
            )
            components = self.components(5.0, count, tonal=1.0, headroom=5.0)
            components["spatial_hold_pass"] = count == 2
            return components

        result = rehab.consolidate_candidate(plan, score)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "hard gate regressed")

    def test_merge_rejects_new_hard_gate_violation(self):
        first = rehab.SlotEdit.modify(self.left, (1000.0, 1.0, -3.0))
        second = rehab.SlotEdit.modify(self.second, (1000.0, 1.0, -0.02))
        plan = rehab.CandidatePlan((first, second))

        def score(candidate_plan):
            count = sum(
                edit.replacement is not None
                for edit in candidate_plan.slot_edits
            )
            components = self.components(5.0, count, tonal=1.0, headroom=5.0)
            if count == 2:
                components.pop("nearfield_skirt_violation_count")
            else:
                components["nearfield_skirt_violation_count"] = 1
            return components

        result = rehab.consolidate_candidate(plan, score)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "hard gate regressed")

    def test_merge_rejected_above_point_one_db(self):
        first = rehab.SlotEdit.modify(self.left, (1000.0, 0.5, 6.0))
        second = rehab.SlotEdit.modify(self.second, (1000.0, 10.0, -6.0))
        result = rehab.consolidate_candidate(rehab.CandidatePlan((first, second)), self.score_plan)
        self.assertFalse(result.accepted)
        self.assertGreater(result.max_cascade_error_db, 0.1)

    def test_merge_requires_fewer_filters_without_regression(self):
        first = rehab.SlotEdit.modify(self.left, (1000.0, 1.0, -3.0))
        second = rehab.SlotEdit.modify(self.second, (1000.0, 1.0, -0.02))
        plan = rehab.CandidatePlan((first, second))

        def score(candidate_plan):
            count = sum(edit.replacement is not None for edit in candidate_plan.slot_edits)
            return self.components(5.0, count, tonal=1.0, headroom=5.0)

        result = rehab.consolidate_candidate(plan, score)
        self.assertTrue(result.accepted)
        self.assertLessEqual(result.max_cascade_error_db, 0.1)
        self.assertEqual(result.candidate.components["filter_count"], 1.0)


class StreamIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.ref = rehab.FilterRef(
            2, 7, "FL Low", "17", (97.0, 3.0, -1.5)
        )
        self.edit = rehab.SlotEdit.modify(
            self.ref, (100.0, 1.2, -1.5)
        )
        self.plan = rehab.CandidatePlan(slot_edits=(self.edit,))

    def test_resume_round_trips_candidate_plan(self):
        payload = stream.beam_entry_to_json(stream.BeamEntry(
            objective=1.25,
            signature=rehab.candidate_plan_signature(self.plan),
            plan=self.plan,
            components={"objective": 1.25, "tonal_masked": 0.8},
        ))

        loaded = stream.beam_entry_from_json(payload)

        self.assertEqual(loaded.plan, self.plan)
        self.assertEqual(loaded.components["tonal_masked"], 0.8)

    def test_v1_group_only_entry_loads_as_candidate_plan(self):
        payload = {
            "objective": 2.0,
            "groups": {"sub": [[50.0, 1.0, -2.0]]},
        }

        loaded = stream.beam_entry_from_json(payload)

        self.assertEqual(loaded.plan.slot_edits, ())
        self.assertEqual(
            dict(loaded.plan.groups)["sub"], ((50.0, 1.0, -2.0),)
        )

    def test_guided_beam_keeps_rehabilitation_slot_edits(self):
        pools = {group: [] for group in stream.opt.GROUPS}
        group = next(iter(stream.opt.GROUPS))
        pools[group] = [{
            "F": 500.0, "Q": 1.0, "G": -2.0, "strength": 4.0,
        }]
        seen = []

        def score_plan(plan):
            seen.append(plan)
            return {
                "objective": 10.0 - len(plan.slot_edits) - sum(
                    len(bands) for _name, bands in plan.groups
                )
            }

        entries, _evaluations = stream.deterministic_beam_combinations(
            pools,
            score_plan=score_plan,
            seed_plans=(rehab.CandidatePlan(), self.plan),
            beam_width=8,
            pool_limit=2,
        )

        self.assertTrue(any(
            entry.plan.slot_edits == (self.edit,) for entry in entries
        ))
        self.assertTrue(any(
            entry.plan.slot_edits == (self.edit,)
            and any(bands for _name, bands in entry.plan.groups)
            for entry in entries
        ))
        self.assertTrue(all(
            isinstance(plan, rehab.CandidatePlan) for plan in seen
        ))

    def test_peq_runs_rehabilitation_before_guided_beam(self):
        config = rehab.RehabilitationConfig(
            frequency_octaves=(0.0, 1 / 24),
            q_multipliers=(1.0,),
            gain_offsets_db=(-0.5, 0.0),
            retained_per_slot=2,
            refinement_passes=0,
            max_evaluations_per_slot=8,
            role_limits=((
                "FL Low", 80.0, 200.0, 0.5, 4.0, -6.0, 0.0
            ),),
        )

        def score(plan):
            edit = plan.slot_edits[0] if plan.slot_edits else None
            replacement = None if edit is None else edit.replacement
            distance = (
                4.0
                if replacement is None
                else abs(replacement[0] - 100.0) / 10.0
            )
            return {
                "objective": distance,
                "tonal_masked": distance,
                "filter_count": 1.0,
                "positive_gain_penalty_db": 0.0,
            }

        result = stream.run_rehabilitation_stage(
            mode="peq",
            refs=(self.ref,),
            score_plan=score,
            total_seconds=2.0,
            config=config,
        )

        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["evaluations"], 0)
        self.assertIsInstance(result["best_plan"], rehab.CandidatePlan)
    def test_phase_mode_does_not_run_peq_rehabilitation(self):
        result = stream.run_rehabilitation_stage(
            mode="phase",
            refs=(self.ref,),
            score_plan=lambda _plan: {"objective": 1.0},
            total_seconds=20.0,
        )

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["evaluations"], 0)

    def test_v2_checkpoint_round_trips_candidate_plan_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream_state.json"
            args = stream.argparse.Namespace(
                seed=9,
                profile="safe",
                proposal="beam",
                mode="peq",
                filter_cost_scale=0.1,
                worst_weight=0.1,
                min_total_bands=0,
                archive_size=4,
                rehabilitation={"status": "complete", "evaluations": 3},
            )
            rng = np.random.default_rng(9)
            entry = stream.BeamEntry(
                1.25,
                rehab.candidate_plan_signature(self.plan),
                self.plan,
                {"objective": 1.25, "tonal_masked": 0.8},
            )

            stream.save_state(path, [entry], rng, 3, 0.5, args)
            payload = stream.json.loads(path.read_text(encoding="utf-8"))
            loaded, _archive, _scores, _trials, _elapsed = stream.load_state(
                path,
                rng,
                archive_size=4,
                score_plan=lambda plan: {
                    "objective": 1.25 if plan == self.plan else 10.0
                },
            )

            self.assertEqual(
                payload["schema"], "audiofischer-stream-state-v2"
            )
            self.assertEqual(loaded[0].plan, self.plan)
    def test_checkpoint_rejects_changed_input_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream_state.json"
            args = stream.argparse.Namespace(
                seed=9,
                profile="safe",
                proposal="beam",
                mode="peq",
                filter_cost_scale=0.1,
                worst_weight=0.1,
                min_total_bands=0,
                archive_size=4,
                rehabilitation={},
                input_fingerprint="session-a",
            )
            rng = np.random.default_rng(9)
            entry = stream.BeamEntry(
                1.25,
                rehab.candidate_plan_signature(self.plan),
                self.plan,
                {"objective": 1.25},
            )
            stream.save_state(path, [entry], rng, 3, 0.5, args)

            loaded = stream.load_state(
                path,
                rng,
                archive_size=4,
                score_plan=lambda _plan: {"objective": 1.25},
                expected_fingerprint="session-b",
            )

            self.assertEqual(loaded, ([], [], {}, 0, 0.0))
    def test_merge_loader_preserves_candidate_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = Path(tmp) / "worker_01"
            worker.mkdir()
            args = stream.argparse.Namespace(
                seed=9,
                profile="safe",
                proposal="beam",
                mode="peq",
                filter_cost_scale=0.1,
                worst_weight=0.1,
                min_total_bands=0,
                archive_size=4,
                rehabilitation={},
            )
            entry = stream.BeamEntry(
                1.25,
                rehab.candidate_plan_signature(self.plan),
                self.plan,
                {"objective": 1.25},
            )
            stream.save_state(
                worker / "stream_state.json",
                [entry],
                np.random.default_rng(9),
                3,
                0.5,
                args,
            )

            loaded = merge_stream.load_worker_best(worker)

            self.assertEqual(loaded[0][2], self.plan)
    def test_candidate_plan_slot_edit_keeps_phase_conflict_veto(self):
        freqs = np.geomspace(1000.0, 4000.0, 256)
        phase_plan = [{
            "source": "Mid to tweeter",
            "crossover_band": (1800.0, 3000.0),
            "crossover_channels": (0, 2),
        }]
        scorer = optimizer.make_candidate_plan_component_scorer(
            lambda _band_sets: {"objective": 1.0},
            freqs,
            {},
            phase_plan,
            False,
        )
        plan = rehab.CandidatePlan(slot_edits=(
            rehab.SlotEdit.modify(
                self.ref, (2200.0, 1.0, -4.0)
            ),
        ))

        band_sets = [[] for _ in range(8)]
        band_sets[2] = [self.ref.original]
        with patch.object(
            optimizer, "baseline_band_sets", return_value=band_sets
        ):
            components = scorer(plan)

        self.assertGreater(components["phase_peq_conflict_count"], 0)
        self.assertGreater(components["objective"], 1e6)
    def test_short_run_reserves_half_for_guided_beam(self):
        budget = stream.rehabilitation_budget(12.0)
        self.assertLessEqual(budget.seconds, 6.0)
        self.assertGreater(budget.max_evaluations, 0)


class Task5ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        xml = fixture_afpx_xml({2: [(7, 100.0, 1.0, -2.0)]})
        self.ref = rehab.active_peq_slot_refs(xml, {2: "FL Low"})[0]
        self.edit = rehab.SlotEdit.modify(
            self.ref, (105.0, 1.1, -2.5)
        )

    def test_coordinate_refinement_scores_complete_candidate_plan(self):
        group = "fl_low"
        plan = rehab.CandidatePlan(
            slot_edits=(self.edit,),
            groups=rehab.freeze_groups({group: [(500.0, 1.0, -2.0)]}),
        )
        entry = stream.BeamEntry(
            10.0, rehab.candidate_plan_signature(plan), plan,
            {"objective": 10.0},
        )
        seen = []

        def score_plan(candidate):
            seen.append(candidate)
            self.assertEqual(candidate.slot_edits, (self.edit,))
            return {"objective": 9.0}

        stream.refine_entries(
            [entry], score_plan=score_plan, top=1, passes=1
        )

        self.assertTrue(seen)
        self.assertTrue(all(item.slot_edits == (self.edit,) for item in seen))

    def test_rehabilitation_beam_stops_at_deadline(self):
        operation = rehab.OperationCandidate(
            edit=self.edit,
            plan=rehab.CandidatePlan(slot_edits=(self.edit,)),
            components={"objective": 0.5},
            baseline_components={"objective": 1.0},
        )
        calls = []

        result = rehab.rehabilitation_beam(
            rehab.CandidatePlan(),
            (operation,),
            lambda plan: calls.append(plan) or {"objective": 1.0},
            deadline=0.0,
        )

        self.assertEqual(result.score_count, 1)
        self.assertEqual(calls, [rehab.CandidatePlan()])

    def test_merge_rejects_missing_or_mismatched_worker_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = Path(tmp) / "worker_01"
            worker.mkdir()
            state = worker / "stream_state.json"
            state.write_text(json.dumps({"version": 7, "best": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing input fingerprint"):
                merge_stream.load_worker_best(worker, expected_fingerprint="current")

            state.write_text(json.dumps({
                "version": 7,
                "input_fingerprint": "stale",
                "best": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match current inputs"):
                merge_stream.load_worker_best(worker, expected_fingerprint="current")

    def test_full_plan_diagnostics_resolve_slot_edits(self):
        baseline = [[] for _ in range(8)]
        baseline[2] = [self.ref.original]
        plan = rehab.CandidatePlan(slot_edits=(self.edit,))

        class Objective:
            def __init__(self):
                self.audit_sets = None
                self.plot_sets = None

            def response_audit(self, band_sets):
                self.audit_sets = band_sets
                return {"ok": True}

            def report_plot_data(self, band_sets):
                self.plot_sets = band_sets
                return {"ok": True}

        objective = Objective()
        with patch.object(optimizer, "AFPX_OBJECTIVE", objective), patch.object(
            optimizer, "baseline_band_sets", return_value=baseline
        ):
            optimizer.fixed_anchor_response_audit(plan)
            optimizer.report_plot_data(plan)
            headroom = optimizer.candidate_plan_headroom(
                np.geomspace(20.0, 20000.0, 64), plan
            )

        self.assertIn(self.edit.replacement, objective.audit_sets[2])
        self.assertIn(self.edit.replacement, objective.plot_sets[2])
        self.assertIn("FL Low", headroom)

    def test_candidate_plan_prediction_uses_slot_edit_delta(self):
        freqs = np.geomspace(50.0, 5000.0, 96)
        traces = {
            name: np.zeros_like(freqs)
            for name in optimizer.CH_TRACE.values()
        }
        traces["Sub"] = np.zeros_like(freqs)
        for pair in optimizer.PAIR_DEFS.values():
            traces[pair["together"]] = optimizer.power_sum_db([
                traces[pair["left"]], traces[pair["right"]]
            ])
        traces["System Sum"] = optimizer.power_sum_db([
            traces[pair["together"]]
            for pair in optimizer.PAIR_DEFS.values()
        ] + [traces["Sub"]])
        baseline = [[] for _ in range(8)]
        baseline[2] = [self.ref.original]
        plan = rehab.CandidatePlan(slot_edits=(self.edit,))

        with patch.object(
            optimizer, "baseline_band_sets", return_value=baseline
        ), patch.object(
            optimizer, "output_trim_for_band_sets", return_value={}
        ):
            predicted = optimizer.predict_candidate_plan(
                freqs, traces, plan
            )

        expected = (
            optimizer.cascade_db(freqs, [self.edit.replacement])
            - optimizer.cascade_db(freqs, [self.ref.original])
        )
        np.testing.assert_allclose(predicted["FL Low"], expected)
        np.testing.assert_allclose(predicted["FR Low"], 0.0)
    def test_build_rows_predicts_and_scores_the_full_plan(self):
        plan = rehab.CandidatePlan(slot_edits=(self.edit,))
        entry = stream.BeamEntry(
            1.0, rehab.candidate_plan_signature(plan), plan,
            {"objective": 1.0},
        )
        traces = {"System Sum": np.zeros(4)}
        predicted = {"System Sum": np.ones(4)}
        with patch.object(
            stream.opt, "predict_candidate_plan", return_value=predicted,
            create=True,
        ) as predict, patch.object(
            stream.opt, "tune_scorecard", return_value={"sum_rms_db": 1.0}
        ), patch.object(
            stream.opt, "candidate_plan_headroom", return_value={"FL Low": {}},
            create=True,
        ), patch.object(
            stream.opt, "left_alone_note", return_value="left alone"
        ):
            rows = stream.build_rows(
                np.arange(4.0), traces, np.zeros(4), [entry],
                score_plan=lambda _plan: {"objective": 1.0},
            )

        predict.assert_called_once()
        self.assertIs(predict.call_args.args[2], plan)
        self.assertEqual(rows[0]["headroom"], {"FL Low": {}})

    def test_guided_continuation_keeps_baseline_and_rehabilitated_lineages(self):
        rehabilitated = rehab.CandidatePlan(slot_edits=(self.edit,))
        groups = {group: [] for group in stream.opt.GROUPS}
        groups["fl_low"] = [(500.0, 1.0, -2.0)]

        plans = stream.guided_continuation_plans(
            groups, (rehab.CandidatePlan(), rehabilitated)
        )

        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].slot_edits, ())
        self.assertEqual(plans[1].slot_edits, (self.edit,))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.config = rehab.RehabilitationConfig()
        self.fingerprint_inputs = {
            "baseline": {"sha256": "baseline"},
            "target": {"sha256": "target"},
        }

    def _stage(self):
        return {
            "status": "no_eligible_filters",
            "evaluations": 1,
            "best_plan": rehab.CandidatePlan(),
            "result": None,
            "census": (),
        }

    def test_shared_cache_reused_without_rescoring(self):
        calls = []

        def build_stage():
            calls.append("built")
            return self._stage()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            first = stream.build_or_load_rehabilitation_cache(
                path,
                expected_fingerprint="same-session",
                fingerprint_inputs=self.fingerprint_inputs,
                config=self.config,
                build_stage=build_stage,
            )
            second = stream.build_or_load_rehabilitation_cache(
                path,
                expected_fingerprint="same-session",
                fingerprint_inputs=self.fingerprint_inputs,
                config=self.config,
                build_stage=lambda: self.fail("cache reuse rescored rehabilitation"),
            )

        self.assertEqual(calls, ["built"])
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["best_plan"], rehab.CandidatePlan())

    def test_cache_rejects_changed_target_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            stream.build_or_load_rehabilitation_cache(
                path,
                expected_fingerprint="target-a",
                fingerprint_inputs=self.fingerprint_inputs,
                config=self.config,
                build_stage=self._stage,
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                stream.build_or_load_rehabilitation_cache(
                    path,
                    expected_fingerprint="target-b",
                    fingerprint_inputs={
                        **self.fingerprint_inputs,
                        "target": {"sha256": "different"},
                    },
                    config=self.config,
                    build_stage=lambda: self.fail("stale cache was silently rebuilt"),
                )

    def test_worker_keeps_compact_cache_state_after_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            loaded = stream.build_or_load_rehabilitation_cache(
                path,
                expected_fingerprint="same-session",
                fingerprint_inputs=self.fingerprint_inputs,
                config=self.config,
                build_stage=self._stage,
            )

            state = stream.compact_rehabilitation_cache_state(loaded, path)

        self.assertNotIn("census_detail", state)
        self.assertNotIn("scored_candidates", state)
        self.assertEqual(state["cache_fingerprint"], "same-session")
        self.assertEqual(state["cache_path"], str(path.resolve()))

    def test_cache_fingerprint_includes_phase_scoring_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.afpx"
            target = root / "target.txt"
            baseline.write_bytes(b"baseline")
            target.write_text("20 0\n", encoding="utf-8")
            common = dict(
                baseline=baseline,
                target=target,
                measurement_session={"manifest": {}, "audit": {}},
                filter_cost_scale=0.1,
                worst_weight=0.1,
                min_total_bands=0,
                mode="peq",
                profile="explore",
                phase_cache=None,
                sample_rate=96000.0,
                level_calibration=None,
                repeatability_folder=None,
            )
            off = stream.stream_input_fingerprint(
                SimpleNamespace(**common, phase_writes="off"), self.config
            )
            automatic = stream.stream_input_fingerprint(
                SimpleNamespace(**common, phase_writes="auto"), self.config
            )

        self.assertNotEqual(off, automatic)
    def _fingerprint_args(self, root, **overrides):
        baseline = root / "baseline.afpx"
        target = root / "target.txt"
        baseline.write_bytes(b"baseline")
        target.write_text("20 0\n", encoding="utf-8")
        values = dict(
            baseline=baseline,
            target=target,
            measurement_session={"manifest": {}, "audit": {}},
            measurement_noise_guard={"source": "default", "repeatability_db": 0.25},
            loaded_level_calibration={},
            filter_cost_scale=0.1,
            worst_weight=0.1,
            min_total_bands=0,
            mode="peq",
            profile="explore",
            phase_cache=None,
            phase_writes="off",
            sample_rate=96000.0,
            level_calibration=None,
            repeatability_folder=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_cache_fingerprint_changes_with_calibration_content_and_loaded_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "level calibration.json"
            calibration.write_text('{"FL Low": 1.0}', encoding="utf-8")
            args = self._fingerprint_args(
                root,
                level_calibration=calibration,
                loaded_level_calibration={"FL Low": 1.0},
            )
            first = stream.stream_input_fingerprint(args, self.config)
            calibration.write_text('{"FL Low": 2.0}', encoding="utf-8")
            second = stream.stream_input_fingerprint(args, self.config)
            args.loaded_level_calibration = {"FL Low": 2.0}
            third = stream.stream_input_fingerprint(args, self.config)

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_cache_fingerprint_changes_with_repeatability_content_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repeatability = root / "repeatability folder"
            repeatability.mkdir()
            capture = repeatability / "System Sum.txt"
            capture.write_text("20 70\n", encoding="utf-8")
            args = self._fingerprint_args(root, repeatability_folder=repeatability)
            first = stream.stream_input_fingerprint(args, self.config)
            capture.write_text("20 71\n", encoding="utf-8")
            second = stream.stream_input_fingerprint(args, self.config)
            args.measurement_noise_guard = {
                "source": "empirical", "repeatability_db": 0.7
            }
            third = stream.stream_input_fingerprint(args, self.config)

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_cache_preparation_cancels_without_publishing_or_leaving_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            stopped = threading.Event()

            def build_stage():
                stopped.set()
                return self._stage()

            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                stream.build_or_load_rehabilitation_cache(
                    path,
                    expected_fingerprint="cancelled-session",
                    fingerprint_inputs=self.fingerprint_inputs,
                    config=self.config,
                    build_stage=build_stage,
                    stop_requested=stopped.is_set,
                )

            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob("rehabilitation_cache.json.*")), [])

    def test_rehabilitation_scoring_stops_before_the_first_evaluation(self):
        ref = rehab.FilterRef(2, 0, "FL Low", "PEQ", (500.0, 1.0, -2.0))
        calls = []

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            stream.run_rehabilitation_stage(
                mode="peq",
                refs=(ref,),
                score_plan=lambda _plan: calls.append("scored") or {"objective": 1.0},
                total_seconds=60,
                config=self.config,
                stop_requested=lambda: True,
            )

        self.assertEqual(calls, [])
    def test_abandoned_lock_and_temporary_file_are_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            path.with_name(f"{path.name}.lock").write_text(
                "2147483647", encoding="ascii"
            )
            path.with_name(f"{path.name}.orphan.tmp").write_text(
                "partial", encoding="utf-8"
            )

            loaded = stream.build_or_load_rehabilitation_cache(
                path,
                expected_fingerprint="recovered-session",
                fingerprint_inputs=self.fingerprint_inputs,
                config=self.config,
                build_stage=self._stage,
                lock_timeout_seconds=0.2,
            )

            self.assertEqual(loaded["fingerprint"], "recovered-session")
            self.assertEqual(list(path.parent.glob("rehabilitation_cache.json.*")), [])
    def test_concurrent_cache_preparation_builds_and_publishes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            started = threading.Event()
            release = threading.Event()
            calls = []
            results = []
            errors = []

            def build_stage():
                calls.append("built")
                started.set()
                release.wait(2)
                return self._stage()

            def invoke():
                try:
                    results.append(stream.build_or_load_rehabilitation_cache(
                        path,
                        expected_fingerprint="concurrent-session",
                        fingerprint_inputs=self.fingerprint_inputs,
                        config=self.config,
                        build_stage=build_stage,
                    ))
                except BaseException as exc:
                    errors.append(exc)

            first = threading.Thread(target=invoke)
            second = threading.Thread(target=invoke)
            first.start()
            self.assertTrue(started.wait(1))
            second.start()
            time.sleep(0.05)
            release.set()
            first.join(2)
            second.join(2)

            self.assertEqual(errors, [])
            self.assertEqual(calls, ["built"])
            self.assertEqual(len(results), 2)
            self.assertEqual(json.loads(path.read_text())["fingerprint"], "concurrent-session")
            self.assertEqual(list(path.parent.glob("rehabilitation_cache.json.*")), [])
    def test_malformed_cache_fails_instead_of_rescoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehabilitation_cache.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                stream.load_rehabilitation_cache(path, "same-session")

if __name__ == "__main__":
    unittest.main()
