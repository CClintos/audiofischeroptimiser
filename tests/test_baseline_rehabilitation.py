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


if __name__ == "__main__":
    unittest.main()
