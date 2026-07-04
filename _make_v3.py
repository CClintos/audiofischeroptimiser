# _make_v3.py — hardened .afpx writer TEMPLATE (roundtrip lint + FN-ignoring semantic diff,
# PEQ-limit validation, reserved special slots, delay+crossover preservation).
# Upgraded 2026-07-03 from the reviewed ChatGPT 'improved copy' (kept the writer safety;
# rejected the over-engineered scorers — see MDAT_AFPX_INSTRUCTIONS.md §6).
import re
import struct
import zlib
from pathlib import Path


SRC = Path(r'C:\Users\Adroit\Desktop\New Tune.afpx')
DST = Path(__file__).resolve().with_name('New Tune_v3.afpx')

# Free slots at the extremes are useful for shelves/all-pass/tone-control quirks.
# Use ordinary middle slots for normal PEQ unless the channel is genuinely full.
RESERVED_FREE_DF = {'25', '32', '20000'}


def decode_afpx(path):
    raw = Path(path).read_bytes()
    if len(raw) < 5:
        raise ValueError('AFPX file is too short: %s' % path)
    declared_len = struct.unpack('>I', raw[:4])[0]
    xml = zlib.decompress(raw[4:]).decode('utf-8', 'replace')
    actual_len = len(xml.encode('utf-8'))
    if declared_len != actual_len:
        print('warning: AFPX header length %d != decoded XML length %d' %
              (declared_len, actual_len))
    return xml


def encode_afpx(xml, path):
    payload = xml.encode('utf-8')
    Path(path).write_bytes(struct.pack('>I', len(payload)) + zlib.compress(payload, 9))


def at(text, key):
    m = re.search(r'(?<![A-Za-z])' + re.escape(key) + r'="([^"]*)"', text)
    return m.group(1) if m else None


def set_attr(text, key, value):
    value = str(value)
    pat = r'(?<![A-Za-z])' + re.escape(key) + r'="[^"]*"'
    if re.search(pat, text):
        return re.sub(pat, '%s="%s"' % (key, value), text, count=1)
    return text[:-2] + ' %s="%s"/>' % (key, value) if text.endswith('/>') else text


def active_filter_tags(xml):
    return re.findall(r'<Fil\b[^>]*/?>', xml)


def semantic_tag_key(tag_text):
    """Order-independent identity for a self-closing tag. PC-Tool round-trips
    REORDER attributes inside a tag on save (verified 2026-07-03: a <T PM= T=
    P=/> delay tag came back as <T T= P= PM=/>, same values) -- so raw string
    comparison of a PC-Tool-saved tag is a false-positive trap. Sort attrs."""
    return tuple(sorted(re.findall(r'([A-Za-z]+)="([^"]*)"', tag_text)))


def delay_tags(xml):
    return [semantic_tag_key(t) for t in re.findall(r'<T [^>]*>', xml)]


def semantic_filter_key(fil):
    """Filter identity for round-trip diffing.
    PC-Tool may renumber FN, so ignore FN and compare the acoustic meaning."""
    keep = []
    for key in ('T', 'F', 'Q', 'G', 'dF', 'I'):
        keep.append((key, at(fil, key)))
    return tuple(keep)


def crossover_tags(xml):
    return [semantic_filter_key(f) for f in active_filter_tags(xml) if at(f, 'T') in ('15', '16')]


def channel_filter_inventory(xml):
    inv = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S):
        inv.append([semantic_filter_key(f) for f in re.findall(r'<Fil\b[^>]*/?>', oc)])
    return inv


def multiset_delta(old_items, new_items):
    old_counts, new_counts = {}, {}
    for item in old_items:
        old_counts[item] = old_counts.get(item, 0) + 1
    for item in new_items:
        new_counts[item] = new_counts.get(item, 0) + 1
    added, removed = [], []
    for item, count in new_counts.items():
        added.extend([item] * max(0, count - old_counts.get(item, 0)))
    for item, count in old_counts.items():
        removed.extend([item] * max(0, count - new_counts.get(item, 0)))
    return added, removed


def afpx_roundtrip_lint(old_xml, new_xml, allow_delay_changes=False,
                        allow_crossover_changes=False, allowed_added_types=('17', '20')):
    old_inv = channel_filter_inventory(old_xml)
    new_inv = channel_filter_inventory(new_xml)
    channel_diffs = []
    for i in range(max(len(old_inv), len(new_inv))):
        old_items = old_inv[i] if i < len(old_inv) else []
        new_items = new_inv[i] if i < len(new_inv) else []
        added, removed = multiset_delta(old_items, new_items)
        if added or removed:
            channel_diffs.append({'channel_index': i, 'added': added, 'removed': removed})

    forbidden = []
    old_all = [semantic_filter_key(f) for f in active_filter_tags(old_xml)]
    new_all = [semantic_filter_key(f) for f in active_filter_tags(new_xml)]
    added_all, _ = multiset_delta(old_all, new_all)
    for item in added_all:
        t = dict(item).get('T')
        if t not in allowed_added_types:
            forbidden.append(item)

    delay_changed = delay_tags(old_xml) != delay_tags(new_xml)
    crossover_changed = crossover_tags(old_xml) != crossover_tags(new_xml)
    errors = []
    if delay_changed and not allow_delay_changes:
        errors.append('delay tags changed')
    if crossover_changed and not allow_crossover_changes:
        errors.append('crossover filters changed')
    if forbidden:
        errors.append('forbidden filter type added')

    return {'pass': not errors,
            'errors': errors,
            'delay_changed': delay_changed,
            'crossover_changed': crossover_changed,
            'forbidden_added_filters': forbidden,
            'channel_diffs': channel_diffs}


def validate_peq_band(F, Q, G):
    F, Q, G = float(F), float(Q), float(G)
    if not (20.0 <= F <= 20000.0):
        raise ValueError('PEQ frequency out of range: %.2f' % F)
    if not (0.5 <= Q <= 15.0):
        raise ValueError('PEQ Q out of range: %.3f' % Q)
    if not (-15.0 <= G <= 6.0):
        raise ValueError('PEQ gain out of Helix range: %.2f' % G)
    if G > 3.0:
        print('warning: boost %.2f dB is inside hardware range but above the app safety cap' % G)


def choose_free_slots(oc, needed):
    fils = re.findall(r'<Fil\b[^>]*/?>', oc)
    free = [f for f in fils if at(f, 'T') == '1']
    preferred = [f for f in free if at(f, 'dF') not in RESERVED_FREE_DF]
    slots = preferred if len(preferred) >= needed else free
    if len(slots) < needed:
        raise ValueError('not enough free PEQ slots: need %d, have %d' % (needed, len(slots)))
    return slots[:needed]


def edit_tweeter(oc):
    # Change the 13000 Hz boost from +2.5 to -1.5, but fail if the expected
    # source band is missing or duplicated.
    before = oc.count('F="13000.00" G="2.5"')
    if before != 1:
        raise ValueError('expected exactly one 13000 Hz +2.5 tweeter band, found %d' % before)
    return oc.replace('F="13000.00" G="2.5"', 'F="13000.00" G="-1.5"', 1)


def add_bands(oc, bands):
    # bands: list of (F, Q, G). Convert safe free slots (T="1") into active PEQs.
    slots = choose_free_slots(oc, len(bands))
    for (F, Q, G), slot in zip(bands, slots):
        validate_peq_band(F, Q, G)
        new = slot
        new = set_attr(new, 'T', '17')
        new = set_attr(new, 'Q', Q)
        new = set_attr(new, 'F', '%.2f' % float(F))
        new = set_attr(new, 'G', G)
        oc = oc.replace(slot, new, 1)
    return oc


def main():
    xml = decode_afpx(SRC)
    ocs = list(re.finditer(r'<OC\b.*?</OC>', xml, re.S))
    if len(ocs) < 4:
        raise ValueError('expected at least 4 output-channel blocks, found %d' % len(ocs))
    blocks = [m.group() for m in ocs]

    new_blocks = list(blocks)
    new_blocks[0] = edit_tweeter(blocks[0])   # FL High
    new_blocks[1] = edit_tweeter(blocks[1])   # FR High
    new_blocks[2] = add_bands(blocks[2], [(110.0, 1.3, -4.0), (2500.0, 2.5, -4.0)])  # FL Low
    new_blocks[3] = add_bands(blocks[3], [(110.0, 1.3, -4.0), (2500.0, 2.5, -4.0)])  # FR Low

    new_xml = xml
    for old, new in zip(blocks, new_blocks):
        if old != new:
            new_xml = new_xml.replace(old, new, 1)

    encode_afpx(new_xml, DST)

    written = decode_afpx(DST)
    lint = afpx_roundtrip_lint(xml, written, allowed_added_types=('17',))
    if not lint['pass']:
        raise AssertionError('AFPX lint failed: ' + '; '.join(lint['errors']))

    names = ['FL High', 'FR High', 'FL Low', 'FR Low']
    for i, oc in enumerate(re.findall(r'<OC\b.*?</OC>', written, re.S)[:4]):
        act = sorted((float(at(f, 'F')), at(f, 'Q'), at(f, 'G'))
                     for f in re.findall(r'<Fil\b[^>]*/?>', oc)
                     if at(f, 'T') == '17')
        print('ch%d %-8s' % (i, names[i]))
        for F, Q, G in act:
            mark = '   <== NEW/CHANGED' if (round(F) in (110, 2500, 13000)) else ''
            print('     F=%-8.1f Q=%-5s G=%s%s' % (F, Q, G, mark))
    print('delays preserved: OK')
    print('crossovers preserved: OK')
    print('semantic channel diffs:', len(lint['channel_diffs']))
    print('written:', DST)


if __name__ == '__main__':
    main()
