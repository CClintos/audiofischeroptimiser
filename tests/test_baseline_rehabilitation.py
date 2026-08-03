from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import baseline_rehabilitation as rehab
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


if __name__ == "__main__":
    unittest.main()
