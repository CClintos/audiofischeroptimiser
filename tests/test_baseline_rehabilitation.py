from __future__ import annotations

import re
import unittest

import baseline_rehabilitation as rehab


def fixture_afpx_xml(active_by_channel):
    channels = []
    for channel in range(4):
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


if __name__ == "__main__":
    unittest.main()
