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
from collections import Counter
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
def _has_any(names):
    return any((REW_DIR / (name + '.txt')).exists() for name in names)


THREE_WAY = _has_any(('Front L Mid', 'Front L MID', 'Front L Midrange', 'Front Left Mid')) and _has_any(('Front R Mid', 'Front R MID', 'Front R Midrange', 'Front Right Mid')) and _has_any(('Both Mids', 'Mids Together', 'Midrange Together')) and _has_any(('Front L Low', 'Front L Midbass', 'Front L Mid Bass', 'Front Left Low')) and _has_any(('Front R Low', 'Front R Midbass', 'Front R Mid Bass', 'Front Right Low')) and _has_any(('Mid Bass Together', 'Both Midbass', 'Both Midbasses', 'Both Mid Bass'))

if THREE_WAY:
    SOLO_FILES = {
        'FL High': ('Front L High', 'Front L Tweeter', 'Front Left High', 'Front Left Tweeter'),
        'FR High': ('Front R High', 'Front R Tweeter', 'Front Right High', 'Front Right Tweeter'),
        'FL Mid': ('Front L Mid', 'Front L MID', 'Front L Midrange', 'Front Left Mid'),
        'FR Mid': ('Front R Mid', 'Front R MID', 'Front R Midrange', 'Front Right Mid'),
        'FL Low': ('Front L Low', 'Front L Midbass', 'Front L Mid Bass', 'Front Left Low'),
        'FR Low': ('Front R Low', 'Front R Midbass', 'Front R Mid Bass', 'Front Right Low'),
        'Sub': ('Sub', 'SUB', 'Subwoofer'),
        'System Sum': ('System Sum', 'SYSTEM SUM'),
        'Tweeters Together': ('Tweeters Together', 'Both Tweeters'),
        'Mids Together': ('Both Mids', 'Mids Together', 'Midrange Together'),
        'Mid Bass Together': ('Mid Bass Together', 'Both Midbass', 'Both Midbasses', 'Both Mid Bass'),
    }
    CH_KEYS = ['FL High', 'FR High', 'FL Mid', 'FR Mid', 'FL Low', 'FR Low']
    PAIR_SPECS = {
        'low': ('FL Low', 'FR Low', 'Mid Bass Together', (50.0, 700.0), (80.0, 500.0)),
        'mid': ('FL Mid', 'FR Mid', 'Mids Together', (250.0, 4500.0), (300.0, 3500.0)),
        'high': ('FL High', 'FR High', 'Tweeters Together', (1800.0, 16000.0), (2500.0, 12000.0)),
    }
else:
    SOLO_FILES = {
        'FL High': ('Front L High', 'Front L Tweeter', 'Front Left High', 'Front Left Tweeter'),
        'FR High': ('Front R High', 'Front R Tweeter', 'Front Right High', 'Front Right Tweeter'),
        'FL Low': ('Front L Low', 'Front L Mid', 'Front L MID', 'Front Left Mid'),
        'FR Low': ('Front R Low', 'Front R Mid', 'Front R MID', 'Front Right Mid'),
        'Sub': ('Sub', 'SUB', 'Subwoofer'),
        'System Sum': ('System Sum', 'SYSTEM SUM'),
        'Tweeters Together': ('Tweeters Together', 'Both Tweeters'),
        'Mid Bass Together': ('Mid Bass Together', 'Both Mids'),
    }
    CH_KEYS = ['FL High', 'FR High', 'FL Low', 'FR Low']
    PAIR_SPECS = {
        'low': ('FL Low', 'FR Low', 'Mid Bass Together', (80.0, 2600.0), (200.0, 2000.0)),
        'high': ('FL High', 'FR High', 'Tweeters Together', (2600.0, 16000.0), (2800.0, 16000.0)),
    }
TARGET = Path(os.environ.get('AFPX_TARGET', str(DATA_ROOT / 'ResoNix Target Curve 2026.txt')))
BASELINE_AFPX = Path(os.environ.get('AFPX_BASELINE', str(DATA_ROOT / 'baseline.afpx')))
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
    'added_band': 0.05,   # a new filter must beat the one-seat noise floor
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


def _optimization_grid(freqs, points_per_octave=96):
    freqs = np.asarray(freqs, dtype=float)
    if len(freqs) < 3 or np.any(freqs <= 0.0):
        return freqs
    log_f = np.log2(freqs)
    steps = np.diff(log_f)
    expected = 1.0 / float(points_per_octave)
    already_log = (
        abs(float(np.median(steps)) - expected) <= expected * 0.02
        and float(np.percentile(np.abs(steps - np.median(steps)), 95)) <= expected * 0.02
    )
    if already_log:
        return freqs
    first = int(np.ceil(log_f[0] * points_per_octave))
    last = int(np.floor(log_f[-1] * points_per_octave))
    if last <= first:
        return freqs
    return 2.0 ** (np.arange(first, last + 1, dtype=float) / float(points_per_octave))


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
    raw = {}
    F = None
    for key, nm in SOLO_FILES.items():
        f, s = _load_txt(_resolve_txt(nm))
        if F is None:
            F = f
        raw[key] = (f, s)
    F = _optimization_grid(F)
    log_f = np.log10(F)
    for key, (source_f, source_s) in raw.items():
        _T[key] = np.interp(log_f, np.log10(source_f), source_s)
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
    _NULL_MASK = np.zeros_like(F, dtype=bool)
    for _name, (left, right, together, band_range, _balance) in PAIR_SPECS.items():
        _NULL_MASK |= _pair_null(left, right, together, band_range[0], band_range[1])
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


def _band_key(band):
    f, q, g = band
    return (round(float(f), 1), round(float(q), 2), round(float(g) * 4.0) / 4.0)


def _added_bands_by_channel(band_sets):
    """Return only filters added on top of the matching baseline tune."""
    added = {}
    for i, _key in enumerate(CH_KEYS):
        candidate = list(band_sets[i]) if i < len(band_sets) else []
        baseline = list(_V5[i]) if i < len(_V5) else []
        remaining = Counter(_band_key(b) for b in baseline)
        new_bands = []
        for band in candidate:
            key = _band_key(band)
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                new_bands.append((float(band[0]), float(band[1]), float(band[2])))
        added[i] = new_bands
    return added


def _interp_at(values, f):
    return float(np.interp(np.log10(float(f)), np.log10(_F), values))


def _system_branch_total_db():
    total = 10 ** (_T['Sub'] / 10)
    for _name, (_left, _right, together, _band_range, _balance) in PAIR_SPECS.items():
        total += 10 ** (_T[together] / 10)
    return 10 * np.log10(np.maximum(total, 1e-30))


def _driver_share_db(channel_key, f, total_db=None):
    # Side solos are naturally about 3 dB below their pair when L/R are equal,
    # so add that back before judging whether the driver is meaningfully active.
    if total_db is None:
        total_db = _system_branch_total_db()
    share = _interp_at(_T[channel_key] - total_db, f)
    if channel_key.startswith('F'):
        share += 3.0
    return share


def _solo_peak_support(channel_key, f):
    """True when a narrow/deep cut is backed by a real local solo peak."""
    sm = erb_smooth(_F, _T[channel_key])
    oct_dist = np.abs(np.log2(_F / float(f)))
    window = oct_dist <= (1 / 3)
    center = oct_dist <= (1 / 12)
    if not np.any(window) or not np.any(center):
        return False
    side = window & ~center
    side_ref = sm[side] if np.any(side) else sm[window]
    center_peak = float(np.max(sm[center]))
    local_peak = float(np.max(sm[window]))
    prominence = center_peak - float(np.median(side_ref))
    return center_peak >= local_peak - 0.4 and prominence >= 1.25


def _delta_channel(i, band_sets):
    candidate = list(band_sets[i]) if i < len(band_sets) else []
    baseline = list(_V5[i]) if i < len(_V5) else []
    return _casc(candidate) - _casc(baseline)


def _asymmetry_penalty(band_sets, total=None):
    if total is None:
        total = _system_branch_total_db()
    penalty = 0.0
    for _name, (left, right, together, _band_range, balance_band) in PAIR_SPECS.items():
        if left not in CH_KEYS or right not in CH_KEYS:
            continue
        li = CH_KEYS.index(left)
        ri = CH_KEYS.index(right)
        eq_diff = erb_smooth(_F, _delta_channel(li, band_sets) - _delta_channel(ri, band_sets))
        solo_diff = np.abs(erb_smooth(_F, _T[left] - _T[right]))
        allowed = 0.75 + 0.55 * solo_diff
        active = (_T[together] - total) >= -10.0
        sel = (_F >= balance_band[0]) & (_F <= balance_band[1]) & active
        excess = np.maximum(np.abs(eq_diff) - allowed, 0.0)
        if np.any(sel):
            penalty += 0.35 * float(np.sqrt(np.mean(excess[sel] ** 2)))
    return penalty


def _guardrail_score(band_sets):
    added = _added_bands_by_channel(band_sets)
    total_db = _system_branch_total_db()
    shape = 0.0
    unsupported = 0.0
    wasted = 0.0
    boost_q = 0.0
    n_added = 0
    worst_share = None
    for i, bands in added.items():
        channel_key = CH_KEYS[i]
        for f, q, g in bands:
            n_added += 1
            shape += 0.012 * abs(g) * q
            if g > 0.0 and q > 1.8:
                boost_q += 0.08 * g * q * (1.0 + max(0.0, q - 2.0))
            needs_solo_proof = g < -4.0 or q > 2.5
            if needs_solo_proof and not (g < 0.0 and _solo_peak_support(channel_key, f)):
                unsupported += 0.75
                unsupported += 0.85 * max(0.0, -g - 4.0)
                unsupported += 0.65 * max(0.0, q - 2.5)
            share = _driver_share_db(channel_key, f, total_db)
            worst_share = share if worst_share is None else min(worst_share, share)
            if share < -6.0:
                wasted += 0.18 * (-6.0 - share) * (0.5 + abs(g) / 4.0)
    asym = _asymmetry_penalty(band_sets, total_db)
    parsimony = W['added_band'] * n_added
    total = shape + unsupported + wasted + boost_q + asym + parsimony
    return {
        'guardrail_penalty': round(total, 3),
        'shape_penalty': round(shape, 3),
        'unsupported_filter_penalty': round(unsupported, 3),
        'wasted_band_penalty': round(wasted, 3),
        'asymmetric_eq_penalty': round(asym, 3),
        'high_q_boost_penalty': round(boost_q, 3),
        'added_band_penalty': round(parsimony, 3),
        'n_added_front_bands': n_added,
        'worst_driver_share_db': round(worst_share if worst_share is not None else 0.0, 2),
    }


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

    branch_outputs = []
    for _name, (left, right, together, _band_range, _balance) in PAIR_SPECS.items():
        pr[together] = ps(pr[left], pr[right]) + (_T[together] - ps(_T[left], _T[right]))
        branch_outputs.append(pr[together])

    old = _T['Sub'].copy()
    for _name, (_left, _right, together, _band_range, _balance) in PAIR_SPECS.items():
        old = 10 * np.log10(10 ** (old / 10) + 10 ** (_T[together] / 10))
    rest = np.maximum(10 ** (_T['System Sum'] / 10) - 10 ** (old / 10), 1e-9)
    new = pr['Sub'].copy()
    for branch in branch_outputs:
        new = 10 * np.log10(10 ** (new / 10) + 10 ** (branch / 10))
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

    balances = {}
    for name, (left, right, _together, _band_range, balance_band) in PAIR_SPECS.items():
        diff = erb_smooth(_F, pr[left] - pr[right])
        sel = (_F >= balance_band[0]) & (_F <= balance_band[1])
        balances[name] = float(np.median(diff[sel])) if np.any(sel) else 0.0

    # headroom: worst front-channel cascade peak, + boost landing in null bins
    head_peak = 0.0
    null_boost = 0.0
    for i in range(len(CH_KEYS)):
        r = headroom_report(_F, band_sets[i])
        head_peak = max(head_peak, r['peak_cascade_gain_db'])
        b = _casc(band_sets[i])  # this channel's EQ curve; penalize boost in null bins
        null_boost += float(np.sum(np.maximum(b[_NULL_MASK], 0.0))) / max(np.sum(_NULL_MASK), 1)

    n_bands = sum(len(bs) for bs in band_sets[:len(CH_KEYS)])
    guard = _guardrail_score(band_sets)

    comp = {
        'tonal_masked': round(tonal, 3),
        'worst_masked': round(worst, 2),
        'headroom_peak': round(head_peak, 2),
        'null_boost_avg': round(null_boost, 2),
        'n_front_bands': n_bands,
    }
    if 'low' in balances:
        comp['low_balance'] = round(balances['low'], 2)
    if 'mid' in balances:
        comp['mid_balance'] = round(balances['mid'], 2)
    if 'high' in balances:
        comp['tweeter_balance'] = round(balances['high'], 2)
    balance_term = (
        W['mid_balance'] * abs(balances.get('mid', balances.get('low', 0.0)))
        + W['tw_balance'] * abs(balances.get('high', 0.0))
        + (0.25 * abs(balances.get('low', 0.0)) if 'mid' in balances else 0.0)
    )
    scalar = (W['tonal'] * tonal
              + balance_term
              + W['worst'] * worst
              + W['headroom'] * max(0.0, head_peak - SOFT_CAP_DB)
              + W['null_boost'] * null_boost
              + W['parsimony'] * n_bands
              + guard['guardrail_penalty'])
    comp.update(guard)
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
        balance_mid = c.get('mid_balance', c.get('low_balance', 0.0))
        print('\n%s' % ntpath.basename(p))
        print('  OBJECTIVE = %.3f   (lower = better)' % c['objective'])
        print('  tonal_masked=%.3f worst_masked=%.2f mid_bal=%+.2f tw_bal=%+.2f headroom=%.2f null_boost=%.2f bands=%d'
              % (c['tonal_masked'], c['worst_masked'], balance_mid, c.get('tweeter_balance', 0.0),
                 c['headroom_peak'], c['null_boost_avg'], c['n_front_bands']))
        print('  guardrail=%.3f shape=%.3f unsupported=%.3f wasted=%.3f asym=%.3f added_bands=%d'
              % (c.get('guardrail_penalty', 0.0), c.get('shape_penalty', 0.0),
                 c.get('unsupported_filter_penalty', 0.0), c.get('wasted_band_penalty', 0.0),
                 c.get('asymmetric_eq_penalty', 0.0), c.get('n_added_front_bands', 0)))
