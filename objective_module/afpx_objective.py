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
# PEQ magnitude is always scored. When phase-valid solos reproduce the measured
# together trace, candidate biquads are also complex-summed; otherwise the scorer
# automatically retains the conservative measured-residual magnitude model.
import re
import os
import sys
import zlib
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('AFPX_DATA_ROOT', str(ROOT.parent)))
if not (DATA_ROOT / 'System Sum.txt').exists() and (DATA_ROOT.parent / 'System Sum.txt').exists():
    DATA_ROOT = DATA_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA_ROOT))
from _tunefit import (
    LOGSTEP,
    cascade_complex,
    erb_hz,
    erb_smooth,
    interference_audit,
    peaking_db,
)

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
LEVEL_CALIBRATION = {}
ANCHOR_BAND = (300.0, 3000.0)

# ---- objective weights (tunable; defaults encode the reviewed priorities) --
W = {
    'tonal': 1.0,        # null-masked, vocal-weighted sum RMS  (primary)
    'target_shape': 0.35, # anchor-independent requested contour through presence
    'peak': 0.35,        # positive deviations are more audible than equal dips
    'narrow_peak': 0.18, # light raw/1/6-oct check catches peaks hidden by ERB smoothing
    'mid_balance': 0.6,  # weighted RMS FL/FR mismatch in the image band
    'tw_balance': 0.2,   # weighted RMS tweeter mismatch
    'balance_bias': 0.12, # broad signed image pull, separate from mismatch RMS
    'worst': 0.15,       # masked worst-case deviation
    'headroom': 0.4,     # per dB of cascade boost above SOFT_CAP
    'output_gain': 1.0,  # never reward candidate-level output gain
    'null_boost': 0.8,   # per dB of EQ BOOST landing in a masked null bin (the exploit)
    'parsimony': 0.02,   # per active band
    'added_band': 0.05,   # a new filter must beat the one-seat noise floor
    'spatial_fragility': 1.0,
}
BALANCE_RMS_SHARE = 0.65
BALANCE_ABS_SHARE = 0.35
SOFT_CAP_DB = 3.0        # cascade boost above this starts costing
VOCAL_BAND = (200.0, 6000.0)
VOCAL_WEIGHT = 1.8
COMPLEX_VALIDATION_RMS_DB = 2.5
TARGET_SHAPE_BAND = (1300.0, 5000.0)
TARGET_SHAPE_REFERENCE = (1000.0, 1400.0)
INBAND = (60.0, 16000.0)


# ---- load measured data + target (once) -----------------------------------
def _load_txt_rich(path, min_points=16):
    """Load a REW-style trace and fail loudly on missing or truncated inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required measurement is missing: %s' % path)
    columns = [[], [], [], [], []]
    numeric_rows = 0
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith(('*', '#', ';')):
                continue
            parts = text.replace(',', ' ').split()
            try:
                values = [float(value) for value in parts[:5]]
            except (ValueError, TypeError):
                numeric_start = False
                try:
                    float(parts[0])
                    numeric_start = True
                except (ValueError, TypeError, IndexError):
                    pass
                if not numeric_start and any(char.isalpha() for char in text):
                    continue
                raise ValueError('Malformed numeric row in %s at line %d' % (path, line_number))
            if len(values) < 2:
                raise ValueError('Measurement row needs frequency and SPL in %s at line %d'
                                 % (path, line_number))
            numeric_rows += 1
            for index, value in enumerate(values):
                columns[index].append(value)
    if numeric_rows < int(min_points):
        raise ValueError('Measurement %s is truncated: %d points, need at least %d'
                         % (path, numeric_rows, min_points))
    freqs = np.asarray(columns[0], dtype=float)
    spl = np.asarray(columns[1], dtype=float)
    if np.any(~np.isfinite(freqs)) or np.any(~np.isfinite(spl)):
        raise ValueError('Measurement contains non-finite frequency or SPL values: %s' % path)
    if np.any(freqs <= 0.0) or np.any(np.diff(freqs) <= 0.0):
        raise ValueError('Measurement frequencies must be positive and strictly increasing: %s' % path)
    result = {'freq': freqs, 'spl': spl, 'path': str(path)}
    if len(columns[2]) == numeric_rows:
        phase = np.asarray(columns[2], dtype=float)
        if np.all(np.isfinite(phase)):
            result['phase'] = phase
    if len(columns[3]) == numeric_rows:
        coherence = np.asarray(columns[3], dtype=float)
        if np.all(np.isfinite(coherence)):
            result['coherence'] = coherence
    if len(columns[4]) == numeric_rows:
        result['position_id'] = np.asarray(columns[4], dtype=float)
    return result


def _load_txt(path):
    trace = _load_txt_rich(path)
    return trace['freq'], trace['spl']

def _resolve_txt(names):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        path = REW_DIR / (name + '.txt')
        if path.exists():
            return path
    expected = ', '.join(str(REW_DIR / (name + '.txt')) for name in names)
    raise FileNotFoundError('Missing required measurement; expected one of: ' + expected)


def _calibration_offset(role, path):
    for key in (role, path.name, path.stem, str(path)):
        if key in LEVEL_CALIBRATION:
            return float(LEVEL_CALIBRATION[key])
    return 0.0


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


def _weighted_rms(values, weights, mask):
    selected = np.asarray(mask, dtype=bool) & np.isfinite(values) & np.isfinite(weights)
    if not np.any(selected):
        return float('inf')
    weighted = np.asarray(values, dtype=float)[selected] * np.asarray(weights, dtype=float)[selected]
    den = float(np.sum(np.asarray(weights, dtype=float)[selected] ** 2))
    return float(np.sqrt(np.sum(weighted ** 2) / max(den, 1e-30)))


def perceptual_weights(freqs):
    """Smooth vocal/presence emphasis without hard 200 Hz or 6 kHz edges."""
    freqs = np.asarray(freqs, dtype=float)
    weights = np.ones_like(freqs)

    def raised_log_ramp(lo, hi, rising):
        selected = (freqs > lo) & (freqs < hi)
        x = np.clip(np.log2(freqs[selected] / lo) / np.log2(hi / lo), 0.0, 1.0)
        curve = 0.5 - 0.5 * np.cos(np.pi * x)
        if not rising:
            curve = 1.0 - curve
        weights[selected] += (VOCAL_WEIGHT - 1.0) * curve

    weights[(freqs >= 300.0) & (freqs <= 4000.0)] = VOCAL_WEIGHT
    raised_log_ramp(120.0, 300.0, True)
    raised_log_ramp(4000.0, 8000.0, False)
    return weights


def _fractional_octave_smooth(freqs, values, fraction=6):
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    log_f = np.log2(freqs)
    half_width = 1.0 / (2.0 * float(fraction))
    starts = np.searchsorted(log_f, log_f - half_width, side='left')
    ends = np.searchsorted(log_f, log_f + half_width, side='right')
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[ends] - cumulative[starts]) / np.maximum(ends - starts, 1)


def tonal_components(freqs, deviation_db, valid_mask, narrow_deviation_db=None):
    """Return distinct broad tonal, presence, broad-peak and narrow-peak metrics."""
    freqs = np.asarray(freqs, dtype=float)
    dev = np.asarray(deviation_db, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    vocal = (freqs >= VOCAL_BAND[0]) & (freqs <= VOCAL_BAND[1])
    weights = perceptual_weights(freqs)
    narrow = (_fractional_octave_smooth(freqs, dev, 6)
              if narrow_deviation_db is None else np.asarray(narrow_deviation_db, dtype=float))
    tonal = _weighted_rms(dev, weights, valid)
    anchor = _weighted_rms(dev, np.ones_like(freqs), valid)
    presence = _weighted_rms(dev, np.ones_like(freqs), valid & vocal)
    peak = _weighted_rms(np.maximum(dev, 0.0), weights, valid)
    narrow_peak = _weighted_rms(np.maximum(narrow, 0.0), weights, valid)
    shape_reference = valid & (freqs >= TARGET_SHAPE_REFERENCE[0]) & (freqs <= TARGET_SHAPE_REFERENCE[1])
    shape_band = valid & (freqs >= TARGET_SHAPE_BAND[0]) & (freqs <= TARGET_SHAPE_BAND[1])
    reference_db = float(np.median(dev[shape_reference])) if np.any(shape_reference) else 0.0
    target_shape = _weighted_rms(dev - reference_db, np.ones_like(freqs), shape_band)
    return {
        'tonal_masked': tonal,
        'sum_tonal_anchor_db': anchor,
        'presence_error_db': presence,
        'peak_penalty_db': peak,
        'narrow_peak_penalty_db': narrow_peak,
        'narrow_peak_max_db': float(np.max(np.maximum(narrow[valid], 0.0)))
        if np.any(valid) else 0.0,
        'target_shape_error_db': target_shape,
        'target_shape_reference_db': reference_db,
    }

def balance_components(freqs, difference_db, band):
    """Return broad signed bias and non-cancelling weighted L/R mismatch."""
    freqs = np.asarray(freqs, dtype=float)
    diff = np.asarray(difference_db, dtype=float)
    selected = (freqs >= band[0]) & (freqs <= band[1]) & np.isfinite(diff)
    if not np.any(selected):
        return {'bias_db': 0.0, 'mismatch_rms_db': 0.0, 'mismatch_abs_db': 0.0}
    weights = perceptual_weights(freqs)
    w = weights[selected]
    d = diff[selected]
    return {
        'bias_db': float(np.median(d)),
        'mismatch_rms_db': float(np.sqrt(np.sum((d * w) ** 2) / max(np.sum(w ** 2), 1e-30))),
        'mismatch_abs_db': float(np.sum(np.abs(d) * w) / max(np.sum(w), 1e-30)),
    }


def _balance_mismatch(parts):
    return (
        BALANCE_RMS_SHARE * parts.get('mismatch_rms_db', 0.0)
        + BALANCE_ABS_SHARE * parts.get('mismatch_abs_db', 0.0)
    )


_F = None
_T = {}
_TGT = None
_NULL_MASK = None
_V5 = None
_GRID_TOKEN = None
_BASE_CASCADES = []
_TOTAL_DB = None
_SMOOTH_T = {}
_POSITION_TRACES = {}
_POSITION_BASELINE = {}
_SMOOTHER = None
_BASE_OUTPUT_DB = []
_TRACE_META = {}
_COMPLEX_MODELS = {}
_POSITION_COMPLEX_MODELS = {}
_PREDICTION_AUDIT = {}


def _attrs(t):
    return dict(re.findall(r'([A-Za-z]+)="([^"]*)"', t))


def _peqset(xml):
    out = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S)[:8]:
        out.append([(float(a['F']), float(a['Q']), float(a['G']))
                    for a in (_attrs(t) for t in re.findall(r'<Fil\b[^>]*/>', oc))
                    if a['T'] == '17' and float(a['G']) != 0])
    return out


def _output_levels_db(xml):
    levels = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S)[:8]:
        tag = re.search(r'<Vol\b[^>]*/?>', oc)
        attrs = _attrs(tag.group()) if tag else {}
        linear = float(attrs.get('L', 1.0))
        levels.append(20.0 * np.log10(max(linear, 1e-30)))
    while len(levels) < 8:
        levels.append(0.0)
    return levels


def _position_path(prefixes, aliases):
    for prefix in prefixes:
        for alias in aliases:
            candidates = (
                REW_DIR / (prefix + alias + '.txt'),
                REW_DIR / prefix.strip() / (alias + '.txt'),
            )
            for path in candidates:
                if path.exists():
                    return path
    return None


def _build_smoother(freqs):
    dlog = np.log(LOGSTEP)
    starts = []
    ends = []
    for i, f in enumerate(freqs):
        hb = max(1, int(round(np.log(1 + 0.5 * erb_hz(float(f)) / float(f)) / dlog)))
        starts.append(max(0, i - hb))
        ends.append(min(len(freqs), i + hb + 1))
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    widths = (ends - starts).astype(float)

    def smooth(values):
        cumulative = np.empty(len(values) + 1, dtype=float)
        cumulative[0] = 0.0
        np.cumsum(values, out=cumulative[1:])
        return (cumulative[ends] - cumulative[starts]) / widths
    return smooth


def _smooth(values):
    return _SMOOTHER(values) if _SMOOTHER is not None else erb_smooth(_F, values)


def _trace_complex(meta):
    phase = np.deg2rad(meta['phase'])
    return 10.0 ** (np.asarray(meta['spl'], dtype=float) / 20.0) * np.exp(1j * phase)


def _coherence_mask(meta):
    if 'coherence' not in meta:
        return np.ones_like(_F, dtype=bool)
    coherence = np.asarray(meta['coherence'], dtype=float)
    if np.nanmax(coherence) > 1.5:
        coherence = coherence / 100.0
    return coherence >= 0.60


def _complex_agreement(measured_meta, predicted_complex, mask):
    predicted_db = 20.0 * np.log10(np.maximum(np.abs(predicted_complex), 1e-30))
    selected = np.asarray(mask, dtype=bool) & np.isfinite(predicted_db)
    if np.sum(selected) < 12:
        return float('inf'), 0.0, np.zeros_like(predicted_db)
    offset = float(np.median(np.asarray(measured_meta['spl'])[selected] - predicted_db[selected]))
    residual = np.asarray(measured_meta['spl']) - (predicted_db + offset)
    rms = float(np.sqrt(np.mean(residual[selected] ** 2)))
    return rms, offset, residual


def _align_trace(path, calibration_role):
    trace = _load_txt_rich(path)
    log_f = np.log10(_F)
    aligned = {
        'spl': np.interp(log_f, np.log10(trace['freq']),
                         trace['spl'] + _calibration_offset(calibration_role, Path(path))),
        'path': str(path),
    }
    if 'phase' in trace:
        unwrapped = np.unwrap(np.deg2rad(trace['phase']))
        aligned['phase'] = np.rad2deg(
            np.interp(log_f, np.log10(trace['freq']), unwrapped)
        )
    if 'coherence' in trace:
        aligned['coherence'] = np.interp(
            log_f, np.log10(trace['freq']), trace['coherence']
        )
    return aligned


def _make_complex_sum_model(trace_meta, roles, measured_role, band):
    required = list(roles) + [measured_role]
    if any(role not in trace_meta or 'phase' not in trace_meta[role] for role in required):
        return None, 'phase column missing'
    if any(float(np.ptp(np.asarray(trace_meta[role]['phase'], dtype=float))) < 1e-3
           for role in required):
        return None, 'phase column is constant or placeholder data'
    baseline_sum = np.zeros_like(_F, dtype=complex)
    mask = (_F >= band[0]) & (_F <= band[1])
    for role in roles:
        baseline_sum += _trace_complex(trace_meta[role])
        mask &= _coherence_mask(trace_meta[role])
    measured = trace_meta[measured_role]
    mask &= _coherence_mask(measured)
    alive_band = (_F >= band[0]) & (_F <= band[1])
    alive_floor = float(np.max(measured['spl'][alive_band])) - 25.0
    mask &= measured['spl'] >= alive_floor
    rms, offset, residual = _complex_agreement(measured, baseline_sum, mask)
    if not np.isfinite(rms) or rms > COMPLEX_VALIDATION_RMS_DB:
        return None, 'solo/together agreement %.3f dB exceeds %.1f dB' % (
            rms, COMPLEX_VALIDATION_RMS_DB
        )
    return {
        'roles': tuple(roles),
        'measured_role': measured_role,
        'trace_meta': trace_meta,
        'baseline_sum': baseline_sum,
        'offset_db': offset,
        'residual_db': residual,
        'validation_rms_db': rms,
        'validation_points': int(np.sum(mask)),
    }, 'pass'


def _build_complex_models(position_specs):
    mode = os.environ.get('AFPX_COMPLEX_TONAL', 'auto').strip().lower()
    audit = {'mode': mode, 'pairs': {}, 'positions': {}}
    if mode in ('0', 'off', 'false', 'disabled'):
        audit['system'] = {'active': False, 'reason': 'disabled by AFPX_COMPLEX_TONAL'}
        return {}, {}, audit

    models = {'pairs': {}}
    for name, (left, right, together, band, _balance) in PAIR_SPECS.items():
        model, reason = _make_complex_sum_model(_TRACE_META, (left, right), together, band)
        audit['pairs'][name] = {
            'active': model is not None,
            'reason': reason,
            'validation_rms_db': model['validation_rms_db'] if model else None,
        }
        if model is not None:
            models['pairs'][name] = model

    system_roles = tuple(CH_KEYS) + ('Sub',)
    system_model, reason = _make_complex_sum_model(
        _TRACE_META, system_roles, 'System Sum', INBAND
    )
    models['system'] = system_model
    audit['system'] = {
        'active': system_model is not None,
        'reason': reason,
        'validation_rms_db': system_model['validation_rms_db'] if system_model else None,
    }

    position_models = {}
    for position, prefixes in position_specs.items():
        if position not in _POSITION_TRACES:
            continue
        meta = {}
        missing = []
        for role in system_roles + ('System Sum',):
            path = _position_path(prefixes, SOLO_FILES[role])
            if path is None:
                missing.append(role)
                continue
            meta[role] = _align_trace(path, position + ':' + role)
        if missing:
            audit['positions'][position] = {
                'active': False,
                'reason': 'optional per-position solos missing: ' + ', '.join(missing),
            }
            continue
        model, reason = _make_complex_sum_model(meta, system_roles, 'System Sum', INBAND)
        audit['positions'][position] = {
            'active': model is not None,
            'reason': reason,
            'validation_rms_db': model['validation_rms_db'] if model else None,
        }
        if model is not None:
            position_models[position] = model
    return models, position_models, audit

def _init():
    global _F, _T, _TGT, _NULL_MASK, _V5, _GRID_TOKEN
    global _BASE_CASCADES, _TOTAL_DB, _SMOOTH_T, _POSITION_TRACES, _POSITION_BASELINE, _SMOOTHER
    global _BASE_OUTPUT_DB, _TRACE_META, _COMPLEX_MODELS, _POSITION_COMPLEX_MODELS, _PREDICTION_AUDIT
    if _F is not None:
        return
    raw = {}
    F = None
    for key, nm in SOLO_FILES.items():
        path = _resolve_txt(nm)
        trace = _load_txt_rich(path)
        trace['spl'] = trace['spl'] + _calibration_offset(key, path)
        if F is None:
            F = trace['freq']
        raw[key] = trace
    F = _optimization_grid(F)
    log_f = np.log10(F)
    _TRACE_META = {}
    for key, trace in raw.items():
        source_f = trace['freq']
        source_s = trace['spl']
        _T[key] = np.interp(log_f, np.log10(source_f), source_s)
        aligned = {'spl': _T[key], 'path': trace['path']}
        if 'phase' in trace:
            unwrapped = np.unwrap(np.deg2rad(trace['phase']))
            aligned['phase'] = np.rad2deg(np.interp(log_f, np.log10(source_f), unwrapped))
        if 'coherence' in trace:
            aligned['coherence'] = np.interp(log_f, np.log10(source_f), trace['coherence'])
        _TRACE_META[key] = aligned
    _F = F
    _GRID_TOKEN = (len(F), float(F[0]), float(F[-1]), hash(F.tobytes()))
    _SMOOTHER = _build_smoother(F)
    target_trace = _load_txt_rich(TARGET)
    tf, ts = target_trace['freq'], target_trace['spl']
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
    with open(BASELINE_AFPX, 'rb') as handle:
        baseline_xml = zlib.decompress(handle.read()[4:]).decode('utf-8', 'replace')
    _V5 = _peqset(baseline_xml)
    _BASE_OUTPUT_DB = _output_levels_db(baseline_xml)
    _BASE_CASCADES = [_casc_uncached(bands) for bands in _V5]
    _TOTAL_DB = _system_branch_total_uncached()
    _SMOOTH_T = {key: _smooth(values) for key, values in _T.items()}

    _POSITION_TRACES = {}
    position_specs = {
        'left': ('Left Ear ', 'Left '),
        'right': ('Right Ear ', 'Right '),
    }
    for position, prefixes in position_specs.items():
        path = _position_path(prefixes, SOLO_FILES['System Sum'])
        if path is None:
            continue
        position_trace = _load_txt_rich(path)
        pf = position_trace['freq']
        ps = position_trace['spl'] + _calibration_offset(position + ':System Sum', path)
        measured = np.interp(np.log10(_F), np.log10(pf), ps)
        target = tgt + float(np.median(measured[band] - tgt[band]))
        _POSITION_TRACES[position] = {'system': measured, 'target': target, 'file': str(path)}
    keep = (_F >= INBAND[0]) & (_F <= INBAND[1]) & ~_NULL_MASK
    _POSITION_BASELINE = {
        name: tonal_components(_F, _smooth(data['system'] - data['target']), keep)['tonal_masked']
        for name, data in _POSITION_TRACES.items()
    }
    _COMPLEX_MODELS, _POSITION_COMPLEX_MODELS, _PREDICTION_AUDIT = _build_complex_models(position_specs)


def baseline_band_sets():
    """Return the baseline PEQ bands as 8 channel lists."""
    _init()
    return [list(bands) for bands in _V5]


def _casc_uncached(bands):
    d = np.zeros_like(_F)
    for f, q, g in bands:
        d += peaking_db(_F, f, q, g)
    return d


@lru_cache(maxsize=8192)
def _cached_peaking(grid_token, f, q, g):
    return peaking_db(_F, f, q, g)


def _casc(bands):
    d = np.zeros_like(_F)
    for f, q, g in bands:
        d += _cached_peaking(_GRID_TOKEN, float(f), float(q), float(g))
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


def _matched_front_keys(added):
    if not CH_KEYS:
        return set()
    common = Counter(_band_key(band) for band in added.get(0, []))
    for index in range(1, len(CH_KEYS)):
        common &= Counter(_band_key(band) for band in added.get(index, []))
    return {key for key, count in common.items() if count > 0}


def _interp_at(values, f):
    return float(np.interp(np.log10(float(f)), np.log10(_F), values))


def _system_branch_total_uncached():
    total = 10 ** (_T['Sub'] / 10)
    for _name, (_left, _right, together, _band_range, _balance) in PAIR_SPECS.items():
        total += 10 ** (_T[together] / 10)
    return 10 * np.log10(np.maximum(total, 1e-30))


def _system_branch_total_db():
    return _TOTAL_DB if _TOTAL_DB is not None else _system_branch_total_uncached()


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
    sm = _SMOOTH_T.get(channel_key)
    if sm is None:
        sm = _smooth(_T[channel_key])
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
    baseline_cascade = _BASE_CASCADES[i] if i < len(_BASE_CASCADES) else _casc(baseline)
    return _casc(candidate) - baseline_cascade


def _asymmetry_penalty(band_sets, total=None):
    if total is None:
        total = _system_branch_total_db()
    penalty = 0.0
    for _name, (left, right, together, _band_range, balance_band) in PAIR_SPECS.items():
        if left not in CH_KEYS or right not in CH_KEYS:
            continue
        li = CH_KEYS.index(left)
        ri = CH_KEYS.index(right)
        eq_diff = _smooth(_delta_channel(li, band_sets) - _delta_channel(ri, band_sets))
        solo_diff = np.abs(_smooth(_T[left] - _T[right]))
        allowed = 0.75 + 0.55 * solo_diff
        active = (_T[together] - total) >= -10.0
        sel = (_F >= balance_band[0]) & (_F <= balance_band[1]) & active
        excess = np.maximum(np.abs(eq_diff) - allowed, 0.0)
        if np.any(sel):
            penalty += 0.35 * float(np.sqrt(np.mean(excess[sel] ** 2)))
    return penalty


def _guardrail_score(band_sets):
    added = _added_bands_by_channel(band_sets)
    matched_front = _matched_front_keys(added)
    matched_seen = set()
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
            key = _band_key((f, q, g))
            matched = key in matched_front
            if matched and key in matched_seen:
                continue
            matched_seen.add(key)
            n_added += 1
            shape += 0.012 * abs(g) * q
            if g > 0.0 and q > 1.8:
                boost_q += 0.08 * g * q * (1.0 + max(0.0, q - 2.0))
            needs_solo_proof = g < -4.0 or q > 2.5
            if needs_solo_proof and not (g < 0.0 and _solo_peak_support(channel_key, f)):
                unsupported += 0.75
                unsupported += 0.85 * max(0.0, -g - 4.0)
                unsupported += 0.65 * max(0.0, q - 2.5)
            share = 0.0 if matched else _driver_share_db(channel_key, f, total_db)
            worst_share = share if worst_share is None else min(worst_share, share)
            if share < -6.0:
                wasted += 0.18 * (-6.0 - share) * (0.5 + abs(g) / 4.0)
    asym = _asymmetry_penalty(band_sets, total_db)
    parsimony = W['added_band'] * n_added
    total = shape + unsupported + wasted + boost_q + asym + parsimony
    return {
        'guardrail_penalty': float(total),
        'shape_penalty': float(shape),
        'unsupported_filter_penalty': float(unsupported),
        'wasted_band_penalty': float(wasted),
        'asymmetric_eq_penalty': float(asym),
        'high_q_boost_penalty': float(boost_q),
        'added_band_penalty': float(parsimony),
        'n_added_front_bands': n_added,
        'n_matched_front_voicing_bands': len(matched_front),
        'worst_driver_share_db': float(worst_share if worst_share is not None else 0.0),
    }


def output_trim_plan(band_sets):
    """Return uniform front attenuation when matched voicing raises peak gain."""
    _init()
    added = _added_bands_by_channel(band_sets)
    matched = _matched_front_keys(added)
    has_positive_voicing = any(
        1300.0 <= f <= 6000.0 and q <= 2.0 and g > 0.0
        for f, q, g in matched
    )
    if not has_positive_voicing:
        return {}
    base_peak = max(
        float(np.max(_BASE_CASCADES[i])) + float(_BASE_OUTPUT_DB[i])
        for i in range(len(CH_KEYS))
    )
    candidate_peak = max(
        float(np.max(_casc(band_sets[i]))) + float(_BASE_OUTPUT_DB[i])
        for i in range(len(CH_KEYS))
    )
    needed = max(0.0, candidate_peak - base_peak)
    trim = -min(6.0, math.ceil((needed - 1e-9) * 4.0) / 4.0)
    if trim >= -0.01:
        return {}
    return {index: trim for index in range(len(CH_KEYS))}


def _candidate_transfer(role, band_sets, trim_plan):
    if role == 'Sub':
        index = 6
    elif role in CH_KEYS:
        index = CH_KEYS.index(role)
    else:
        return np.ones_like(_F, dtype=complex)
    candidate = list(band_sets[index]) if index < len(band_sets) else []
    baseline = list(_V5[index]) if index < len(_V5) else []
    denominator = cascade_complex(_F, baseline)
    transfer = cascade_complex(_F, candidate) / np.where(
        np.abs(denominator) > 1e-30, denominator, 1.0
    )
    return transfer * (10.0 ** (float(trim_plan.get(index, 0.0)) / 20.0))


def _model_compatible(model):
    return (
        model is not None
        and len(model.get('baseline_sum', ())) == len(_F)
        and all(len(meta.get('spl', ())) == len(_F) for meta in model.get('trace_meta', {}).values())
    )


def _predict_complex_model(model, band_sets, trim_plan):
    candidate_sum = np.zeros_like(_F, dtype=complex)
    for role in model['roles']:
        candidate_sum += (
            _trace_complex(model['trace_meta'][role])
            * _candidate_transfer(role, band_sets, trim_plan)
        )
    predicted_db = 20.0 * np.log10(np.maximum(np.abs(candidate_sum), 1e-30))
    return predicted_db + model['offset_db'] + model['residual_db']


def _predict_position_system(position, band_sets, trim_plan, center_system_delta):
    model = _POSITION_COMPLEX_MODELS.get(position)
    if _model_compatible(model):
        return _predict_complex_model(model, band_sets, trim_plan)
    return _POSITION_TRACES[position]['system'] + center_system_delta


def _predict(band_sets, output_trim_override=None):
    """Predict magnitude, using complex sums only after measured validation passes."""
    trim_plan = output_trim_plan(band_sets) if output_trim_override is None else dict(output_trim_override)
    pr = {}
    for i, key in enumerate(CH_KEYS):
        candidate = list(band_sets[i]) if i < len(band_sets) else []
        pr[key] = _T[key] + (_casc(candidate) - _BASE_CASCADES[i]) + float(trim_plan.get(i, 0.0))
    if len(band_sets) > 6:
        baseline = _BASE_CASCADES[6] if len(_BASE_CASCADES) > 6 else _casc(_V5[6])
        pr['Sub'] = _T['Sub'] + (_casc(band_sets[6]) - baseline) + float(trim_plan.get(6, 0.0))
    else:
        pr['Sub'] = _T['Sub'].copy()

    def power_sum(a, b):
        return 10.0 * np.log10(10.0 ** (a / 10.0) + 10.0 ** (b / 10.0))

    branch_outputs = []
    pair_models = _COMPLEX_MODELS.get('pairs', {})
    for name, (left, right, together, _band_range, _balance) in PAIR_SPECS.items():
        if name in pair_models and _model_compatible(pair_models[name]):
            pr[together] = _predict_complex_model(pair_models[name], band_sets, trim_plan)
        else:
            incoherent = power_sum(pr[left], pr[right])
            baseline_incoherent = power_sum(_T[left], _T[right])
            pr[together] = incoherent + (_T[together] - baseline_incoherent)
        branch_outputs.append(pr[together])

    system_model = _COMPLEX_MODELS.get('system')
    if _model_compatible(system_model):
        pr['System Sum'] = _predict_complex_model(system_model, band_sets, trim_plan)
        pr['_prediction_model'] = 'validated_complex_sum'
        return pr

    old = _T['Sub'].copy()
    for _name, (_left, _right, together, _band_range, _balance) in PAIR_SPECS.items():
        old = power_sum(old, _T[together])
    rest = np.maximum(10.0 ** (_T['System Sum'] / 10.0) - 10.0 ** (old / 10.0), 1e-9)
    new = pr['Sub'].copy()
    for branch in branch_outputs:
        new = power_sum(new, branch)
    pr['System Sum'] = 10.0 * np.log10(rest + 10.0 ** (new / 10.0))
    pr['_prediction_model'] = 'magnitude_residual_fallback'
    return pr

def _changed_band_centers(band_sets):
    """Return frequencies whose hardware-rounded PEQ differs from baseline."""
    centers = set()
    count = max(len(_V5), len(band_sets))
    for index in range(count):
        baseline = Counter(_band_key(b) for b in (_V5[index] if index < len(_V5) else []))
        candidate = Counter(_band_key(b) for b in (band_sets[index] if index < len(band_sets) else []))
        for key in baseline.keys() | candidate.keys():
            if baseline[key] != candidate[key]:
                centers.add(float(key[0]))
    return sorted(centers)


def response_audit(band_sets):
    """Report raw candidate deltas against one baseline-derived target anchor."""
    _init()
    baseline = _predict(_V5)
    candidate = _predict(band_sets)
    system_delta = candidate['System Sum'] - baseline['System Sum']
    inband = (_F >= INBAND[0]) & (_F <= INBAND[1])

    checkpoints = set()
    for center in _changed_band_centers(band_sets):
        for ratio in (2 ** -0.5, 1.0, 2 ** 0.5):
            frequency = center * ratio
            if _F[0] <= frequency <= _F[-1]:
                checkpoints.add(round(float(frequency), 1))

    rows = []
    baseline_error = baseline['System Sum'] - _TGT
    candidate_error = candidate['System Sum'] - _TGT
    for frequency in sorted(checkpoints):
        pair_delta = {}
        balance_delta = {}
        for name, (left, right, together, _band_range, _balance) in PAIR_SPECS.items():
            pair_change = _interp_at(candidate[together] - baseline[together], frequency)
            lr_change = _interp_at(
                (candidate[left] - candidate[right]) - (baseline[left] - baseline[right]),
                frequency,
            )
            if abs(pair_change) >= 0.0005:
                pair_delta[name] = round(pair_change, 4)
            if abs(lr_change) >= 0.0005:
                balance_delta[name] = round(lr_change, 4)
        rows.append({
            'frequency_hz': frequency,
            'baseline_error_db': round(_interp_at(baseline_error, frequency), 4),
            'candidate_error_db': round(_interp_at(candidate_error, frequency), 4),
            'raw_system_delta_db': round(_interp_at(system_delta, frequency), 4),
            'pair_delta_db': pair_delta,
            'lr_balance_delta_db': balance_delta,
        })

    return {
        'anchor_policy': 'target_anchored_once_from_baseline_system_sum',
        'delta_policy': 'candidate_prediction_minus_baseline_prediction_no_reanchoring',
        'pair_model': baseline.get('_prediction_model', 'magnitude_residual_fallback'),
        'complex_validation': _PREDICTION_AUDIT,
        'system_delta_rms_db': round(
            float(np.sqrt(np.mean(system_delta[inband] ** 2))) if np.any(inband) else 0.0,
            4,
        ),
        'system_delta_max_abs_db': round(
            float(np.max(np.abs(system_delta[inband]))) if np.any(inband) else 0.0,
            4,
        ),
        'checkpoints': rows,
    }


def report_plot_data(band_sets, max_points=220):
    """Return compact, fixed-anchor curves for local visual reports."""
    _init()
    baseline = _predict(_V5)
    candidate = _predict(band_sets)
    eligible = np.flatnonzero((_F >= INBAND[0]) & (_F <= INBAND[1]))
    count = min(max(int(max_points), 32), len(eligible))
    selected = np.unique(np.linspace(0, len(eligible) - 1, count).round().astype(int))
    indices = eligible[selected]

    def values(curve):
        return [round(float(value), 3) for value in np.asarray(curve)[indices]]

    payload = {
        'schema': 'audiofischer-response-plot-v1',
        'anchor_policy': 'target_anchored_once_from_baseline_system_sum',
        'frequency_hz': values(_F),
        'baseline_error_db': values(baseline['System Sum'] - _TGT),
        'candidate_error_db': values(candidate['System Sum'] - _TGT),
        'raw_system_delta_db': values(candidate['System Sum'] - baseline['System Sum']),
        'pairs': {},
    }
    for name, (left, right, _together, _band_range, balance_band) in PAIR_SPECS.items():
        selected = indices[(_F[indices] >= balance_band[0]) & (_F[indices] <= balance_band[1])]
        payload['pairs'][name] = {
            'frequency_hz': [round(float(value), 3) for value in _F[selected]],
            'baseline_lr_db': [
                round(float(value), 3)
                for value in (baseline[left] - baseline[right])[selected]
            ],
            'candidate_lr_db': [
                round(float(value), 3)
                for value in (candidate[left] - candidate[right])[selected]
            ],
        }
    return payload


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, quantile * cumulative[-1], side='left')])


def _has_fragile_filters(band_sets):
    added = _added_bands_by_channel(band_sets)
    if any(q > 2.0 for bands in added.values() for _f, q, _g in bands):
        return True
    for _name, (left, right, _together, _band, _balance) in PAIR_SPECS.items():
        if left not in CH_KEYS or right not in CH_KEYS:
            continue
        if np.max(np.abs(_delta_channel(CH_KEYS.index(left), band_sets)
                         - _delta_channel(CH_KEYS.index(right), band_sets))) > 0.5:
            return True
    return False


def _spatial_components(pr, band_sets, keep, trim_plan=None):
    trim_plan = {} if trim_plan is None else trim_plan
    center_raw = pr['System Sum'] - _TGT
    center_dev = _smooth(center_raw)
    center_narrow = np.maximum(_fractional_octave_smooth(_F, center_raw, 6), center_raw)
    center = tonal_components(_F, center_dev, keep, center_narrow)
    tonal_values = [center['tonal_masked']]
    peak_values = [center['peak_penalty_db']]
    narrow_peak_values = [center['narrow_peak_penalty_db']]
    shape_values = [center['target_shape_error_db']]
    worst_values = [float(np.max(np.abs(center_dev[keep & (_F >= 100) & (_F <= 8000)])))]
    position_tonal = {'center': center['tonal_masked']}
    system_delta = pr['System Sum'] - _T['System Sum']
    for name, data in _POSITION_TRACES.items():
        position_system = _predict_position_system(name, band_sets, trim_plan, system_delta)
        raw = position_system - data['target']
        dev = _smooth(raw)
        narrow = np.maximum(_fractional_octave_smooth(_F, raw, 6), raw)
        parts = tonal_components(_F, dev, keep, narrow)
        tonal_values.append(parts['tonal_masked'])
        peak_values.append(parts['peak_penalty_db'])
        narrow_peak_values.append(parts['narrow_peak_penalty_db'])
        shape_values.append(parts['target_shape_error_db'])
        worst_values.append(float(np.max(np.abs(dev[keep & (_F >= 100) & (_F <= 8000)]))))
        position_tonal[name] = parts['tonal_masked']
    weights = [2.0] + [1.0] * (len(tonal_values) - 1)
    spatial_tonal = (
        0.55 * _weighted_quantile(tonal_values, weights, 0.5)
        + 0.30 * float(np.percentile(tonal_values, 80))
        + 0.15 * max(tonal_values)
    )
    spatial_peak = (
        0.65 * _weighted_quantile(peak_values, weights, 0.5)
        + 0.35 * max(peak_values)
    )
    spatial_narrow_peak = (
        0.65 * _weighted_quantile(narrow_peak_values, weights, 0.5)
        + 0.35 * max(narrow_peak_values)
    )
    spatial_shape = (
        0.65 * _weighted_quantile(shape_values, weights, 0.5)
        + 0.35 * max(shape_values)
    )
    spatial_worst = 0.70 * float(np.percentile(worst_values, 80)) + 0.30 * max(worst_values)
    fragility = 0.0
    hold_pass = True
    if _POSITION_TRACES and _has_fragile_filters(band_sets):
        worsenings = [
            position_tonal[name] - _POSITION_BASELINE[name]
            for name in _POSITION_TRACES
        ]
        fragility = sum(max(0.0, value - 0.05) for value in worsenings) * 2.0
        if worsenings and max(worsenings) > 0.10:
            fragility += 5.0
            hold_pass = False
    if not _POSITION_TRACES:
        spatial_model = 'centre_only'
    elif len(_POSITION_COMPLEX_MODELS) == len(_POSITION_TRACES):
        spatial_model = 'validated_complex_per_position'
    elif _POSITION_COMPLEX_MODELS:
        spatial_model = 'mixed_complex_and_center_delta'
    else:
        spatial_model = 'system_delta'
    return {
        **center,
        'spatial_tonal_db': float(spatial_tonal),
        'spatial_peak_db': float(spatial_peak),
        'spatial_narrow_peak_db': float(spatial_narrow_peak),
        'target_shape_error_db': float(spatial_shape),
        'spatial_worst_db': float(spatial_worst),
        'spatial_position_count': len(_POSITION_TRACES) + 1,
        'spatial_model': spatial_model,
        'spatial_position_tonal_db': position_tonal,
        'spatial_fragility_penalty': float(fragility),
        'spatial_hold_pass': hold_pass,
    }

def objective(band_sets, output_trim_override=None):
    """The single scalar the optimizer minimizes, plus named components."""
    _init()
    trim_plan = output_trim_plan(band_sets) if output_trim_override is None else dict(output_trim_override)
    pr = _predict(band_sets, trim_plan)
    inb = (_F >= INBAND[0]) & (_F <= INBAND[1])
    keep = inb & ~_NULL_MASK  # nulls MASKED OUT of tonal error + worst-case

    tonal_parts = _spatial_components(pr, band_sets, keep, trim_plan)
    tonal = tonal_parts['spatial_tonal_db']
    peak = tonal_parts['spatial_peak_db']
    narrow_peak = tonal_parts['spatial_narrow_peak_db']
    target_shape = tonal_parts['target_shape_error_db']
    worst = tonal_parts['spatial_worst_db']

    balances = {}
    for name, (left, right, _together, _band_range, balance_band) in PAIR_SPECS.items():
        diff = _smooth(pr[left] - pr[right])
        balances[name] = balance_components(_F, diff, balance_band)

    # headroom: worst front-channel cascade peak, + boost landing in null bins
    head_peak = 0.0
    null_boost = 0.0
    for i in range(len(CH_KEYS)):
        b = _casc(band_sets[i]) + float(trim_plan.get(i, 0.0))
        head_peak = max(head_peak, float(np.max(b)))
        null_boost += float(np.sum(np.maximum(b[_NULL_MASK], 0.0))) / max(np.sum(_NULL_MASK), 1)

    n_bands = sum(len(bs) for bs in band_sets[:len(CH_KEYS)])
    guard = _guardrail_score(band_sets)

    comp = {
        **tonal_parts,
        'worst_masked': worst,
        'headroom_peak': head_peak,
        'null_boost_avg': null_boost,
        'n_front_bands': n_bands,
        'protective_output_trim_db': max(0.0, -min(trim_plan.values())) if trim_plan else 0.0,
        'output_level_gain_db': max(0.0, max(trim_plan.values())) if trim_plan else 0.0,
        'complex_prediction_active': 1.0 if pr.get('_prediction_model') == 'validated_complex_sum' else 0.0,
        'complex_pair_count': float(sum(
            _model_compatible(model) for model in _COMPLEX_MODELS.get('pairs', {}).values()
        )),
        'complex_system_validation_rms_db': float(_COMPLEX_MODELS['system']['validation_rms_db'])
        if _model_compatible(_COMPLEX_MODELS.get('system')) else 0.0,
        'complex_position_count': float(len(_POSITION_COMPLEX_MODELS)),
    }
    if 'low' in balances:
        comp['low_balance'] = balances['low']['bias_db']
        comp['low_balance_rms_db'] = balances['low']['mismatch_rms_db']
        comp['low_balance_abs_db'] = balances['low']['mismatch_abs_db']
    if 'mid' in balances:
        comp['mid_balance'] = balances['mid']['bias_db']
        comp['mid_balance_rms_db'] = balances['mid']['mismatch_rms_db']
        comp['mid_balance_abs_db'] = balances['mid']['mismatch_abs_db']
    if 'high' in balances:
        comp['tweeter_balance'] = balances['high']['bias_db']
        comp['tweeter_balance_rms_db'] = balances['high']['mismatch_rms_db']
        comp['tweeter_balance_abs_db'] = balances['high']['mismatch_abs_db']
    primary = balances.get('mid', balances.get('low', {}))
    high = balances.get('high', {})
    balance_term = (
        W['mid_balance'] * _balance_mismatch(primary)
        + W['tw_balance'] * _balance_mismatch(high)
        + W['balance_bias'] * abs(primary.get('bias_db', 0.0))
        + (0.25 * _balance_mismatch(balances['low']) if 'mid' in balances else 0.0)
    )
    scalar = (W['tonal'] * tonal
              + W['target_shape'] * target_shape
              + W['peak'] * peak
              + W['narrow_peak'] * narrow_peak
              + balance_term
              + W['worst'] * worst
              + W['headroom'] * max(0.0, head_peak - SOFT_CAP_DB)
              + W['output_gain'] * comp['output_level_gain_db']
              + W['null_boost'] * null_boost
              + W['parsimony'] * n_bands
              + W['spatial_fragility'] * tonal_parts['spatial_fragility_penalty']
              + guard['guardrail_penalty'])
    comp.update(guard)
    comp['balance_penalty_db'] = float(
        np.sqrt(np.mean([_balance_mismatch(item) ** 2 for item in balances.values()]))
        if balances else 0.0
    )
    comp['objective'] = float(scalar)
    return comp


def score_bands(band_sets):
    return objective(band_sets)


def prediction_audit():
    _init()
    return dict(_PREDICTION_AUDIT)


def cache_stats():
    info = _cached_peaking.cache_info()
    return {
        'peaking_hits': info.hits,
        'peaking_misses': info.misses,
        'peaking_entries': info.currsize,
        'spatial_positions': sorted(_POSITION_TRACES),
        'complex_pairs': sorted(_COMPLEX_MODELS.get('pairs', {})),
        'complex_system': _model_compatible(_COMPLEX_MODELS.get('system')),
        'complex_positions': sorted(_POSITION_COMPLEX_MODELS),
    }


def score_afpx(path):
    _init()
    xml = zlib.decompress(open(path, 'rb').read()[4:]).decode('utf-8', 'replace')
    candidate_levels = _output_levels_db(xml)
    output_delta = {
        index: float(candidate_levels[index] - _BASE_OUTPUT_DB[index])
        for index in range(min(len(candidate_levels), len(_BASE_OUTPUT_DB)))
        if abs(float(candidate_levels[index] - _BASE_OUTPUT_DB[index])) >= 0.001
    }
    return objective(_peqset(xml), output_delta)


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
