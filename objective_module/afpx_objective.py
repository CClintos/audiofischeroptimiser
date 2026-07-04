# afpx_objective.py -- the SINGLE objective function for the tuning optimizer.
# Hand this to the optimizer so its objective == the independent check. It bakes
# in every guardrail that was previously applied by hand: null-masking, headroom
# penalty (hard into nulls), L/R balance from solos, vocal-band weighting.
#
# The optimizer minimizes objective()['objective'] (a scalar). It also gets the
# named components so a human can see WHY one candidate beat another.
#
# Two entry points:
#   score_bands(band_sets)  -> in-loop scoring; band_sets = list of 8 lists of
#                              (F, Q, G) tuples (one list per output channel).
#   score_afpx(path)        -> parse an .afpx file and score it.
# CLI: python afpx_objective.py candidate.afpx [candidate2.afpx ...]
#
# REQUIRES (same folder / same env): _tunefit.py, the 8 REW .txt solo/together
# exports, the target curve, and the baseline .afpx that matches the measurements.
#
# MAGNITUDE-ONLY: this scores EQ/gain. It does NOT model all-pass/delay (phase).
# Keep phase edits (APF) out of the optimizer's search until phase-valid sweeps
# exist; grafting a verified APF onto the EQ winner is a separate, additive step.
import re
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('AFPX_DATA_ROOT', str(ROOT.parent)))
if not (DATA_ROOT / 'System Sum.txt').exists() and (DATA_ROOT.parent / 'System Sum.txt').exists():
    DATA_ROOT = DATA_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA_ROOT))
from _tunefit import peaking_db, erb_smooth, interference_audit, headroom_report

# ---- config ---------------------------------------------------------------
REW_DIR = DATA_ROOT
SOLO_FILES = {  # REW .txt export names, with aliases for different session labels
    'FL High': ('Front L High', 'Front L Tweeter'),
    'FR High': ('Front R High', 'Front R Tweeter'),
    'FL Low': ('Front L Low', 'Front L Mid', 'Front L MID'),
    'FR Low': ('Front R Low', 'Front R Mid', 'Front R MID'),
    'Sub': ('Sub', 'SUB'),
    'System Sum': ('System Sum', 'SYSTEM SUM'),
    'Tweeters Together': ('Tweeters Together', 'Both Tweeters'),
    'Mid Bass Together': ('Mid Bass Together', 'Both Mids'),
}
TARGET = Path(os.environ.get('AFPX_TARGET', str(DATA_ROOT / 'ResoNix Target Curve 2026.txt')))
BASELINE_AFPX = Path(os.environ.get('AFPX_BASELINE', str(DATA_ROOT / 'baseline.afpx')))
CH_KEYS = ['FL High', 'FR High', 'FL Low', 'FR Low']  # the 4 front channels EQ acts on
ANCHOR_BAND = (300.0, 3000.0)

# ---- objective weights (tunable; defaults encode the reviewed priorities) --
W = {
    'tonal': 1.0,        # null-masked, vocal-weighted sum RMS  (primary)
    'mid_balance': 0.6,  # |FL Low - FR Low| median, image band  (imaging)
    'tw_balance': 0.2,   # |FL High - FR High| median
    'worst': 0.15,       # masked worst-case deviation
    'headroom': 0.4,     # per dB of cascade boost above SOFT_CAP
    'null_boost': 0.8,   # per dB of EQ BOOST landing in a masked null bin (the exploit)
    'parsimony': 0.02,   # per active band
}
SOFT_CAP_DB = 3.0        # cascade boost above this starts costing
VOCAL_BAND = (200.0, 6000.0)
VOCAL_WEIGHT = 1.8
INBAND = (60.0, 16000.0)


# ---- load measured data + target (once) -----------------------------------
def _load_txt(path):
    f, s = [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line or line.startswith('*'):
            continue
        p = line.split()
        try:
            f.append(float(p[0])); s.append(float(p[1]))
        except Exception:
            continue
    return np.array(f), np.array(s)


def _resolve_txt(names):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        path = REW_DIR / (name + '.txt')
        if path.exists():
            return path
    return REW_DIR / (names[0] + '.txt')


_F = None
_T = {}
_TGT = None
_NULL_MASK = None
_V5 = None


def _attrs(t):
    return dict(re.findall(r'([A-Za-z]+)="([^"]*)"', t))


def _peqset(xml):
    out = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S)[:8]:
        out.append([(float(a['F']), float(a['Q']), float(a['G']))
                    for a in (_attrs(t) for t in re.findall(r'<Fil\b[^>]*/>', oc))
                    if a['T'] == '17' and float(a['G']) != 0])
    return out


def _init():
    global _F, _T, _TGT, _NULL_MASK, _V5
    if _F is not None:
        return
    F = None
    for key, nm in SOLO_FILES.items():
        f, s = _load_txt(_resolve_txt(nm))
        if F is None:
            F = f
        _T[key] = s
    _F = F
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
    tgt = np.interp(np.log10(F), np.log10(np.array(tf)), np.array(ts))
    band = (F >= ANCHOR_BAND[0]) & (F <= ANCHOR_BAND[1])
    _TGT = tgt + float(np.median(_T['System Sum'][band] - tgt[band]))
    # null mask: destructive-interference bins in either front pair (from MEASURED
    # data -- a property of acoustic summation, ~stable under EQ). Filling these
    # earns no reward; boosting into them is penalized. Restrict each pair to its
    # own passband AND to bins where the pair is actually playing (within 20 dB of
    # its in-band max) -- otherwise the audit flags rolled-off noise-floor regions
    # outside the passband and masks most of the axis.
    def _pair_null(a_key, b_key, tog_key, lo, hi):
        _, _, _, flagged = interference_audit(F, _T[a_key], _T[b_key], _T[tog_key])
        band = (F >= lo) & (F <= hi)
        tog = _T[tog_key]
        alive = tog > (np.max(tog[band]) - 20.0)  # pair is meaningfully present
        return flagged & band & alive
    m1 = _pair_null('FL Low', 'FR Low', 'Mid Bass Together', 80.0, 2600.0)
    m2 = _pair_null('FL High', 'FR High', 'Tweeters Together', 2600.0, 16000.0)
    _NULL_MASK = m1 | m2
    _V5 = _peqset(zlib.decompress(open(BASELINE_AFPX, 'rb').read()[4:]).decode('utf-8', 'replace'))


def baseline_band_sets():
    """Return the baseline PEQ bands as 8 channel lists."""
    _init()
    return [list(bands) for bands in _V5]


def _casc(bands):
    d = np.zeros_like(_F)
    for f, q, g in bands:
        d += peaking_db(_F, f, q, g)
    return d


def _predict(band_sets):
    """band_sets: 8 lists of (F,Q,G). Returns predicted magnitude traces."""
    pr = {}
    for i, k in enumerate(CH_KEYS):
        pr[k] = _T[k] + (_casc(band_sets[i]) - _casc(_V5[i]))
    if len(band_sets) > 6:
        pr['Sub'] = _T['Sub'] + (_casc(band_sets[6]) - _casc(_V5[6]))
    else:
        pr['Sub'] = _T['Sub'].copy()

    def ps(a, b):
        return 10 * np.log10(10 ** (a / 10) + 10 ** (b / 10))
    pr['Tweeters Together'] = ps(pr['FL High'], pr['FR High']) + (
        _T['Tweeters Together'] - ps(_T['FL High'], _T['FR High']))
    pr['Mid Bass Together'] = ps(pr['FL Low'], pr['FR Low']) + (
        _T['Mid Bass Together'] - ps(_T['FL Low'], _T['FR Low']))
    old = ps(ps(_T['Tweeters Together'], _T['Mid Bass Together']), _T['Sub'])
    rest = np.maximum(10 ** (_T['System Sum'] / 10) - 10 ** (old / 10), 1e-9)
    new = ps(ps(pr['Tweeters Together'], pr['Mid Bass Together']), pr['Sub'])
    pr['System Sum'] = 10 * np.log10(rest + 10 ** (new / 10))
    return pr


def objective(band_sets):
    """The single scalar the optimizer minimizes, plus named components."""
    _init()
    pr = _predict(band_sets)
    dev = erb_smooth(_F, pr['System Sum'] - _TGT)
    inb = (_F >= INBAND[0]) & (_F <= INBAND[1])
    keep = inb & ~_NULL_MASK  # nulls MASKED OUT of tonal error + worst-case

    w = np.ones_like(_F)
    w[(_F >= VOCAL_BAND[0]) & (_F <= VOCAL_BAND[1])] = VOCAL_WEIGHT
    tonal = float(np.sqrt(np.sum((dev[keep] * w[keep]) ** 2) / np.sum(w[keep] ** 2)))

    worst = float(np.max(np.abs(dev[keep & (_F >= 100) & (_F <= 8000)])))

    mb = erb_smooth(_F, pr['FL Low'] - pr['FR Low'])
    mid_bal = float(np.median(mb[(_F >= 200) & (_F <= 2000)]))
    tb = erb_smooth(_F, pr['FL High'] - pr['FR High'])
    tw_bal = float(np.median(tb[(_F >= 2800) & (_F <= 16000)]))

    # headroom: worst front-channel cascade peak, + boost landing in null bins
    head_peak = 0.0
    null_boost = 0.0
    for i in range(4):
        r = headroom_report(_F, band_sets[i])
        head_peak = max(head_peak, r['peak_cascade_gain_db'])
        b = _casc(band_sets[i])  # this channel's EQ curve; penalize boost in null bins
        null_boost += float(np.sum(np.maximum(b[_NULL_MASK], 0.0))) / max(np.sum(_NULL_MASK), 1)

    n_bands = sum(len(bs) for bs in band_sets[:4])

    comp = {
        'tonal_masked': round(tonal, 3),
        'worst_masked': round(worst, 2),
        'mid_balance': round(mid_bal, 2),
        'tweeter_balance': round(tw_bal, 2),
        'headroom_peak': round(head_peak, 2),
        'null_boost_avg': round(null_boost, 2),
        'n_front_bands': n_bands,
    }
    scalar = (W['tonal'] * tonal
              + W['mid_balance'] * abs(mid_bal)
              + W['tw_balance'] * abs(tw_bal)
              + W['worst'] * worst
              + W['headroom'] * max(0.0, head_peak - SOFT_CAP_DB)
              + W['null_boost'] * null_boost
              + W['parsimony'] * n_bands)
    comp['objective'] = round(scalar, 3)
    return comp


def score_bands(band_sets):
    return objective(band_sets)


def score_afpx(path):
    xml = zlib.decompress(open(path, 'rb').read()[4:]).decode('utf-8', 'replace')
    return objective(_peqset(xml))


if __name__ == '__main__':
    _init()
    print('null-masked bins: %d of %d (%.0f-%.0f Hz zones excluded from tonal error)'
          % (int(np.sum(_NULL_MASK)), len(_F), _F[_NULL_MASK].min() if _NULL_MASK.any() else 0,
             _F[_NULL_MASK].max() if _NULL_MASK.any() else 0))
    for p in sys.argv[1:]:
        import ntpath
        c = score_afpx(p)
        print('\n%s' % ntpath.basename(p))
        print('  OBJECTIVE = %.3f   (lower = better)' % c['objective'])
        print('  tonal_masked=%.3f worst_masked=%.2f mid_bal=%+.2f tw_bal=%+.2f headroom=%.2f null_boost=%.2f bands=%d'
              % (c['tonal_masked'], c['worst_masked'], c['mid_balance'], c['tweeter_balance'],
                 c['headroom_peak'], c['null_boost_avg'], c['n_front_bands']))
