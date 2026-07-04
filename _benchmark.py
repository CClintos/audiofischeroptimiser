# _benchmark.py -- score N .afpx tunes against the ResoNix target on equal footing.
# Adopted 2026-07-03 from the R&D brief (its "benchmark harness" idea, cut down to
# what our data actually supports): all tunes are applied as EQ-deltas-from-the-
# BASELINE tune to the measured traces (which were captured at baseline state),
# then scored with the one canonical tune_scorecard. This replaces the ad-hoc
# comparison scripts that were re-derived three times during the v7 session.
#
# Usage:  python _benchmark.py <measurements.mdat> <baseline.afpx> <tune1.afpx> [tune2.afpx ...]
# Assumes the 8-trace RTA capture convention (names: Front L/R High, Front L/R Low,
# Sub, System Sum, Tweeters Together, Mid Bass Together) and magnitude-only RTA data.
import os
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _tunefit import peaking_db, tune_scorecard

TARGET = os.environ.get("AFPX_TARGET", str(ROOT / "ResoNix Target Curve 2026.txt"))
NAMES = ['FL High', 'FR High', 'FL Low', 'FR Low', 'Sub', 'System Sum',
         'Tweeters Together', 'Mid Bass Together']
CH = ['FL High', 'FR High', 'FL Low', 'FR Low']


def load_mdat_rta(path):
    """8-trace RTA .mdat -> (freqs, {name: spl}). Axis: 96/oct anchored 24 kHz.
    VALIDATES against crossover physics before returning (rule: never trust an
    axis you haven't validated)."""
    data = open(path, 'rb').read()
    arrs, i = [], 0
    while True:
        j = data.find(b'\x75\x71\x00\x7e', i)
        if j < 0:
            break
        p = j + 6
        n = struct.unpack('>I', data[p:p + 4])[0]
        body = p + 4
        if 0 < n < 5000000 and body + 4 * n <= len(data):
            a = np.frombuffer(data[body:body + 4 * n], dtype='>f4')
            with np.errstate(invalid='ignore'):
                if n >= 256 and np.isfinite(a).all():
                    arrs.append((j, n, a.astype(float)))
        i = j + 2
    spl = [(j, a) for j, n, a in arrs if n == 1229]
    traces = [spl[k][1] for k in range(0, len(spl), 2)]
    if len(traces) != len(NAMES):
        raise ValueError('expected %d traces, found %d — capture convention mismatch'
                         % (len(NAMES), len(traces)))
    n = 1229
    # AXIS CORRECTED 2026-07-03: REW's own .txt export shows the true grid is
    # 3.2958984 Hz .. 23369.487 Hz (96 ppo) -- NOT anchored at 24 kHz. The old
    # 24k anchor was 0.038 octave (2.7%) high; the 415 Hz cabin null proved it.
    freqs = 3.2958984 * ((2 ** (1 / 96.0)) ** np.arange(n))
    T = {nm: traces[k] for k, nm in enumerate(NAMES)}

    def at(f):
        return int(np.argmin(np.abs(freqs - f)))
    ok_sub = T['Sub'][at(60)] - T['Sub'][at(160)] > 10       # LP80 rolloff
    ok_tw = T['FL High'][at(2600)] - T['FL High'][at(800)] > 10  # HP2600 rise
    if not (ok_sub and ok_tw):
        raise ValueError('axis validation FAILED (sub/tweeter crossover corners wrong)')
    return freqs, T


def load_target(freqs, sum_trace):
    tf, ts = [], []
    for line in open(TARGET, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line or line[0].isalpha() or line.startswith('*'):
            continue
        p = line.replace(',', ' ').split()
        try:
            tf.append(float(p[0])); ts.append(float(p[1]))
        except Exception:
            continue
    tgt = np.interp(np.log10(freqs), np.log10(np.array(tf)), np.array(ts))
    band = (freqs >= 300) & (freqs <= 3000)
    return tgt + float(np.median(sum_trace[band] - tgt[band]))


def peqset(path):
    xml = zlib.decompress(open(path, 'rb').read()[4:]).decode('utf-8', 'replace')
    out = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S)[:8]:
        bands = []
        for t in re.findall(r'<Fil\b[^>]*/>', oc):
            a = dict(re.findall(r'([A-Za-z]+)="([^"]*)"', t))
            if a.get('T') == '17' and float(a.get('G', 0)) != 0:
                bands.append((float(a['F']), float(a['Q']), float(a['G'])))
        out.append(bands)
    return out


def predict(freqs, T, baseline, tune):
    def casc(bands):
        d = np.zeros_like(freqs)
        for f, q, g in bands:
            d += peaking_db(freqs, f, q, g)
        return d

    pred = {}
    for ci, nm in enumerate(CH):
        pred[nm] = T[nm] + (casc(tune[ci]) - casc(baseline[ci]))

    def ps(a, b):
        return 10 * np.log10(10 ** (a / 10) + 10 ** (b / 10))
    pred['Tweeters Together'] = ps(pred['FL High'], pred['FR High']) + (
        T['Tweeters Together'] - ps(T['FL High'], T['FR High']))
    pred['Mid Bass Together'] = ps(pred['FL Low'], pred['FR Low']) + (
        T['Mid Bass Together'] - ps(T['FL Low'], T['FR Low']))
    rest = np.maximum(10 ** (T['System Sum'] / 10)
                      - 10 ** (ps(T['Tweeters Together'], T['Mid Bass Together']) / 10), 1e-9)
    pred['System Sum'] = 10 * np.log10(rest + 10 ** (pred['Tweeters Together'] / 10)
                                       + 10 ** (pred['Mid Bass Together'] / 10))
    return pred


def main():
    if len(sys.argv) < 4:
        print(__doc__ or 'usage: python _benchmark.py <mdat> <baseline.afpx> <tune.afpx> [...]')
        sys.exit(1)
    mdat, base_path, tune_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    freqs, T = load_mdat_rta(mdat)
    tgt = load_target(freqs, T['System Sum'])
    baseline = peqset(base_path)
    rows = []
    for label, path in [('BASELINE', base_path)] + [(None, p) for p in tune_paths]:
        import ntpath
        name = label or ntpath.basename(path)
        sc = tune_scorecard(freqs, predict(freqs, T, baseline, peqset(path)), tgt)
        rows.append((name, sc))
    hdr = ['tune', 'sum_rms', 'wrms_img', 'worst', 'mid_bal', 'tw_bal']
    print('%-32s %8s %9s %6s %8s %7s' % tuple(hdr))
    for name, sc in rows:
        print('%-32s %8.2f %9.2f %6.1f %+8.2f %+7.2f' % (
            name, sc['sum_rms_db'], sc['sum_wrms_img_db'], sc['worst_dev_db'],
            sc.get('mid_balance_db', float('nan')), sc.get('tweeter_balance_db', float('nan'))))
    print('\nNOTE: predictions from baseline-state magnitude RTA; phase effects (APFs,')
    print('delay changes) are NOT modeled here — verify those with a live re-measure.')


if __name__ == '__main__':
    main()
