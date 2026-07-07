import argparse
import json
import re
import struct
import sys
import zlib


def decode(path):
    raw = open(path, 'rb').read()
    if len(raw) < 5:
        raise ValueError('file too short to be a valid .afpx: %s' % path)
    declared = struct.unpack('>I', raw[:4])[0]
    xml = zlib.decompress(raw[4:]).decode('utf-8', 'replace')
    if declared != len(xml.encode('utf-8')):
        print('warning: header length %d != decoded length %d' % (declared, len(xml.encode('utf-8'))),
              file=sys.stderr)
    return xml


def encode(xml, path):
    payload = xml.encode('utf-8')
    open(path, 'wb').write(struct.pack('>I', len(payload)) + zlib.compress(payload, 9))


def attrs(tag):
    return dict(re.findall(r'([A-Za-z]+)="([^"]*)"', tag))


def channel_blocks(xml):
    return re.findall(r'<OC\b.*?</OC>', xml, re.S)


def filters(block):
    return re.findall(r'<Fil\b[^>]*/?>', block)


TYPE = {'1': 'free', '17': 'PEQ', '15': 'LP', '16': 'HP',
        '3': 'low_shelf', '4': 'high_shelf', '19': 'allpass1', '20': 'allpass2', '9': 'LP'}


def infer_role(hp_hz, lp_hz):
    if lp_hz is not None and lp_hz <= 120:
        return 'sub'
    if hp_hz is not None and hp_hz >= 1500:
        return 'tweeter'
    if hp_hz is not None and hp_hz <= 250 and lp_hz is not None and lp_hz <= 4000:
        return 'midbass/mid'
    if hp_hz is not None and 250 < hp_hz < 1500:
        return 'midrange'
    if (hp_hz is None or hp_hz <= 60) and (lp_hz is None or lp_hz >= 6000):
        return 'wide/full-range?'
    return 'unknown (confirm manually)'


def channel_summary(block):
    fils = [attrs(f) for f in filters(block)]
    active = [a for a in fils if a.get('T') != '1']
    hp = next((a for a in fils if a.get('T') == '16'), None)
    lp = next((a for a in fils if a.get('T') in ('15', '9')), None)
    hp_engaged = hp is not None and float(hp.get('G', 0)) != 0 and hp.get('FilBy') != '1'
    lp_engaged = lp is not None and float(lp.get('G', 0)) != 0 and lp.get('FilBy') != '1'
    hp_f = float(hp['F']) if hp_engaged else None
    lp_f = float(lp['F']) if lp_engaged else None
    role = infer_role(hp_f, lp_f)
    peqs = [(float(a['F']), float(a['Q']), float(a['G']))
            for a in fils if a.get('T') == '17' and float(a.get('G', 0)) != 0]
    apfs = [(TYPE[a['T']], float(a['F']), None if a['T'] == '19' else float(a.get('Q', 0)))
            for a in fils if a.get('T') in ('19', '20')]
    shelves = [(TYPE[a['T']], float(a['F']), float(a['Q']), float(a['G']))
               for a in fils if a.get('T') in ('3', '4') and float(a.get('G', 0)) != 0]
    free_mid = sum(1 for a in fils if a.get('T') == '1' and a.get('dF') not in ('25', '32', '20000'))
    oc = attrs(re.match(r'<OC\b[^>]*>', block).group(0))
    polarity = None
    if 'CINV' in oc:
        polarity = 'inverted' if oc.get('CINV') == '1' else 'normal'
    return {
        'hp_hz': hp_f, 'lp_hz': lp_f, 'inferred_role': role,
        'active_filter_count': len(active),
        'peqs': peqs, 'all_passes': apfs, 'shelves': shelves,
        'free_middle_slots': free_mid,
        'low_shelf_slot_free': any(a.get('T') == '1' and a.get('dF') == '25' for a in fils),
        'high_shelf_slot_free': any(a.get('T') == '1' and a.get('dF') == '20000' for a in fils),
        'polarity': polarity, 'cinv_raw': oc.get('CINV'),
        'delay_samples': oc.get('__delay__'),
    }


def delay_tags(xml):
    return re.findall(r'<T [^>]*/?>', xml)


def channels(xml):
    blocks = channel_blocks(xml)
    delays = [attrs(t) for t in delay_tags(xml)]
    out = []
    for i, b in enumerate(blocks):
        s = channel_summary(b)
        if i < len(delays):
            s['delay_samples'] = delays[i].get('T')
            s['polarity_delay_tag_raw'] = {'PM': delays[i].get('PM'), 'P': delays[i].get('P')}
        s['index'] = i
        out.append(s)
    for i in range(0, len(out) - 1, 2):
        if out[i]['inferred_role'] == out[i + 1]['inferred_role']:
            out[i]['pair_guess'] = out[i + 1]['pair_guess'] = (i, i + 1)
    return out


def semantic_delay_key(xml):
    return [tuple(sorted(attrs(t).items())) for t in delay_tags(xml)]


def semantic_xover_key(xml):
    keep = ('T', 'F', 'Q', 'G', 'dF', 'FilBy')
    return sorted(tuple((k, attrs(f).get(k)) for k in keep)
                  for f in filters(xml) if attrs(f).get('T') in ('15', '16', '9'))


def roundtrip_lint(old_xml, new_xml, expect_changed=None, allow_delay=False, allow_xover=False):
    errors = []
    if not allow_delay and semantic_delay_key(old_xml) != semantic_delay_key(new_xml):
        errors.append('delay tags changed')
    if not allow_xover and semantic_xover_key(old_xml) != semantic_xover_key(new_xml):
        errors.append('crossover filters changed')

    def sig(xml):
        out = []
        for b in channel_blocks(xml):
            out.append([tuple((k, attrs(f).get(k)) for k in ('T', 'F', 'Q', 'G', 'dF', 'I'))
                        for f in filters(b)])
        return out
    so, sn = sig(old_xml), sig(new_xml)
    changed = sum(1 for co, cn in zip(so, sn) for a, b in zip(co, cn) if a != b)
    if expect_changed is not None and changed != expect_changed:
        errors.append('changed %d slots, expected %d' % (changed, expect_changed))
    return {'pass': not errors, 'errors': errors, 'slots_changed': changed}


def _fmt_ch(c):
    xo = '%s-%s Hz' % ((('%.0f' % c['hp_hz']) if c['hp_hz'] else 'DC'),
                       (('%.0f' % c['lp_hz']) if c['lp_hz'] else 'open'))
    line = 'ch%d  %-16s  band %-14s  %d active' % (
        c['index'], c['inferred_role'], xo, c['active_filter_count'])
    extra = []
    if c.get('all_passes'):
        extra.append('APF ' + ','.join('%s@%.0f' % (t, f) for t, f, q in c['all_passes']))
    if c.get('shelves'):
        extra.append('shelf ' + ','.join('%s@%.0f' % (t, f) for t, f, q, g in c['shelves']))
    extra.append('%d free mid slots' % c['free_middle_slots'])
    return line + '   [' + ' | '.join(extra) + ']'


def main():
    ap = argparse.ArgumentParser(description='Inspect / analyze a Helix .afpx file.')
    ap.add_argument('cmd', choices=['inspect', 'channels'])
    ap.add_argument('file')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    xml = decode(a.file)
    chans = channels(xml)
    if a.json:
        print(json.dumps(chans, indent=2))
        return
    print('%d output channels. Inferred driver roles (CONFIRM these with the user):\n' % len(chans))
    for c in chans:
        print(_fmt_ch(c))
    print('\nDelays present:', len(delay_tags(xml)),
          '| Reminder: roles are inferred from crossover corners -- verify against the actual install.')


if __name__ == '__main__':
    main()
