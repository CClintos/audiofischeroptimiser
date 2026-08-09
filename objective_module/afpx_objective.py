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
# REQUIRES (same folder / same env): _tunefit.py, individual-driver and system-sum
# REW exports, the target curve, and the baseline .afpx that matches them.
# Measured left+right pair exports are recommended but optional for PEQ scoring.
#
# PEQ magnitude is always scored. When phase-valid solos reproduce the measured
# together trace, candidate biquads are also complex-summed; otherwise the scorer
# automatically retains the conservative measured-residual magnitude model.
import re
import os
import sys
import json
import zlib
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('AFPX_DATA_ROOT', str(ROOT.parent)))
ROLE_MAP_PATH = os.environ.get('AFPX_ROLE_MAP', '')
try:
    _role_payload = json.loads(Path(ROLE_MAP_PATH).read_text(encoding='utf-8-sig')) if ROLE_MAP_PATH else {}
    ROLE_MAP = dict(_role_payload.get('roles', _role_payload))
except (OSError, ValueError, TypeError):
    ROLE_MAP = {}
_mapped_system = DATA_ROOT / str(ROLE_MAP.get('System Sum', ''))
if (
    not _mapped_system.is_file()
    and not (DATA_ROOT / 'System Sum.txt').exists()
    and (DATA_ROOT.parent / 'System Sum.txt').exists()
):
    DATA_ROOT = DATA_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA_ROOT))
from _tunefit import (
    LOGSTEP,
    cascade_complex,
    erb_hz,
    erb_smooth,
    imaging_balance_weight,
    interference_mask_evidence,
    MASK_UNKNOWN,
    MEASUREMENT_NOISE_MULTIPLIER,
    high_shelf_db,
    low_shelf_db,
    measurement_noise_floor_db,
    modal_null_evidence,
    nearfield_null_evidence,
    peaking_db,
    signed_offset_evidence,
    target_anchor_offset,
)

# ---- config ---------------------------------------------------------------
REW_DIR = DATA_ROOT
def _has_any(names):
    return any((REW_DIR / (name + '.txt')).exists() for name in names)


def _has_role(role, names):
    mapped = ROLE_MAP.get(role)
    return bool(mapped and (REW_DIR / str(mapped)).is_file()) or _has_any(names)


THREE_WAY = _has_role('FL Mid', ('Front L Mid', 'Front L MID', 'Front L Midrange', 'Front Left Mid')) and _has_role('FR Mid', ('Front R Mid', 'Front R MID', 'Front R Midrange', 'Front Right Mid')) and _has_role('FL Low', ('Front L Low', 'Front L Midbass', 'Front L Mid Bass', 'Front Left Low')) and _has_role('FR Low', ('Front R Low', 'Front R Midbass', 'Front R Mid Bass', 'Front Right Low'))

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
# Physical channels covered by cross-cutting safety guardrails (headroom,
# null-boost exposure, total filter-count parsimony): every front role in
# CH_KEYS, plus both subwoofer outputs. GROUPS['sub']['channels'] is always
# (6, 7) in _optimizer.py regardless of front layout, so the same physical
# indices are used here. Previously these guardrail loops iterated only
# range(len(CH_KEYS)), so a filter placed on the sub was scored acoustically
# but bypassed headroom, null-boost, and parsimony penalties entirely - a
# +6 dB/Q6 sub boost that would be hard-rejected on a front channel scored
# BETTER than the untouched baseline when placed on the sub instead. See
# CHANGELOG.md. _guardrail_score's own front-only pair/imaging/measurement-
# noise checks are deliberately NOT extended here - sub has no L/R pair or
# calibrated noise floor of its own yet, and folding it into that per-band
# logic is a separate, larger design task.
GUARDRAIL_CHANNEL_INDICES = tuple(range(len(CH_KEYS))) + (6, 7)


def _channel_label(index):
    if index < len(CH_KEYS):
        return CH_KEYS[index]
    return {6: 'Sub A', 7: 'Sub B'}.get(index, 'Channel %d' % index)


TARGET = Path(os.environ.get('AFPX_TARGET', str(DATA_ROOT / 'ResoNix Target Curve 2026.txt')))
BASELINE_AFPX = Path(os.environ.get('AFPX_BASELINE', str(DATA_ROOT / 'baseline.afpx')))
LEVEL_CALIBRATION = {}
ANCHOR_BAND = (300.0, 3000.0)
# Repo-review finding: a secondary listening position (left/right ear) used
# to get a fully independent target re-anchor, which can silently hide a
# real broad level difference between positions by always re-centering to
# its own local median. Now it gets only a small bounded "nuisance" offset
# around the ONE global (System Sum) anchor - representing plausible
# mic-placement variance, not a license to erase a genuine asymmetry. If a
# position's own raw anchor differs from the global one by more than this,
# the excess stays visible as real deviation instead of vanishing.
POSITION_ANCHOR_NUISANCE_BOUND_DB = 1.5
# Close-mic front-stage captures, driver-only with negligible room path.
# Optional: when both sides are present they let an already-flagged null be
# confirmed (or left unconfirmed) against how it looks right at the driver,
# not only at the seat. See DEFECT 4a in CHANGELOG.md.
NEARFIELD_L_NAMES = ('Front L Nearfield', 'Front Left Nearfield')
NEARFIELD_R_NAMES = ('Front R Nearfield', 'Front Right Nearfield')

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
SOFT_CAP_DB = 3.0        # cascade boost above this starts costing (soft, tiebreaker only)
HEADROOM_REQUIRED_MARGIN_DB = 1.5  # hard floor: dB below 0 dBFS a channel's real peak must clear
HEADROOM_VIOLATION_PENALTY = 1000.0  # same scale as the other hard guardrails below
NEARFIELD_SKIRT_PENALTY = 1000.0  # positive gain whose -3dB skirt overlaps a confirmed null
VOCAL_BAND = (200.0, 6000.0)
VOCAL_WEIGHT = 1.8
COMPLEX_VALIDATION_RMS_DB = 2.5
TARGET_SHAPE_BAND = (1300.0, 5000.0)
TARGET_SHAPE_REFERENCE = (1000.0, 1400.0)
INBAND = (60.0, 16000.0)


# ---- load measured data + target (once) -----------------------------------
def _detect_delimiter_and_decimal(sample_lines):
    """Repo-review finding: REW supports comma, tab, space, and semicolon-
    delimited exports with a selectable decimal convention, but this used
    to blindly `.replace(',', ' ')` on every line - fine for the
    space-delimited, period-decimal exports this project has actually been
    tested against, but a European-locale decimal comma (e.g. "3,295898")
    would silently become "3 295898", split into two garbage tokens
    instead of one number.

    Inspects a handful of real data lines (comments already stripped) and
    decides delimiter and decimal convention TOGETHER. Returns
    (delimiter, decimal_is_comma): delimiter is a literal split() separator,
    or None for "any whitespace".
    """
    for text in sample_lines:
        if '\t' in text:
            return '\t', False
        if ';' in text:
            return ';', True  # semicolon-delimited REW exports are always comma-decimal
    for text in sample_lines:
        if ',' not in text:
            continue
        # Whitespace already splits this line into 2+ tokens, so it's
        # already doing the field-separating job - a comma living INSIDE
        # one of those tokens is a decimal point (e.g. "100,5 60,0" or
        # "3,295898 40,533"), not a second, redundant delimiter.
        if len(text.split()) >= 2:
            return None, True
        # No whitespace at all: the comma(s) must be the field separator
        # (e.g. "20.5,70.3").
        return ',', False
    return None, False


def _load_txt_rich(path, min_points=16):
    """Load a REW-style trace and fail loudly on missing or truncated inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required measurement is missing: %s' % path)
    with open(path, encoding='utf-8', errors='replace') as handle:
        raw_lines = handle.readlines()
    data_preview = [
        stripped for stripped in (line.strip() for line in raw_lines)
        if stripped and not stripped.startswith(('*', '#', ';'))
    ][:20]
    delimiter, decimal_is_comma = _detect_delimiter_and_decimal(data_preview)
    columns = [[], [], [], [], []]
    numeric_rows = 0
    for line_number, line in enumerate(raw_lines, 1):
        text = line.strip()
        if not text or text.startswith(('*', '#', ';')):
            continue
        normalized = text.replace(',', '.') if decimal_is_comma else text
        parts = normalized.split(delimiter) if delimiter is not None else normalized.split()
        parts = [p for p in (p.strip() for p in parts) if p]
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
    if np.any(freqs <= 0.0):
        raise ValueError('Measurement frequencies must be positive: %s' % path)
    # Resolve exact-duplicate frequency rows explicitly (keep the first,
    # drop the rest) rather than letting them fail the ordering check below
    # with a generic "not strictly increasing" error that doesn't say why.
    _, first_indices = np.unique(freqs, return_index=True)
    duplicates_dropped = int(len(freqs) - len(first_indices))
    if duplicates_dropped:
        keep = np.sort(first_indices)
        columns = [[column[i] for i in keep] if column else column for column in columns]
        freqs = np.asarray(columns[0], dtype=float)
        spl = np.asarray(columns[1], dtype=float)
        numeric_rows = len(keep)
    if np.any(np.diff(freqs) <= 0.0):
        raise ValueError('Measurement frequencies must be strictly increasing: %s' % path)
    result = {
        'freq': freqs, 'spl': spl, 'path': str(path),
        'format': {
            'delimiter': delimiter if delimiter is not None else 'whitespace',
            'decimal_is_comma': decimal_is_comma,
            'duplicate_frequency_rows_dropped': duplicates_dropped,
        },
    }
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

def _resolve_txt(names, role=None, required=True):
    mapped = ROLE_MAP.get(role) if role else None
    if mapped:
        path = (REW_DIR / str(mapped)).resolve()
        try:
            path.relative_to(REW_DIR.resolve())
        except ValueError:
            path = None
        if path is not None and path.is_file() and path.suffix.lower() == '.txt':
            return path
    if isinstance(names, str):
        names = (names,)
    for name in names:
        path = REW_DIR / (name + '.txt')
        if path.exists():
            return path
    expected = ', '.join(str(REW_DIR / (name + '.txt')) for name in names)
    if required:
        raise FileNotFoundError('Missing required measurement; expected one of: ' + expected)
    return None


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
    weights = perceptual_weights(freqs) * imaging_balance_weight(freqs)
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
_BASE_SHELF_DB = []
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
_SYNTHETIC_PAIRS = set()
_MASK_AUDIT = {}
_NEARFIELD_GUARD_MASK = None


def _attrs(t):
    return dict(re.findall(r'([A-Za-z]+)="([^"]*)"', t))


def _peqset(xml):
    out = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S)[:8]:
        out.append([(float(a['F']), float(a['Q']), float(a['G']))
                    for a in (_attrs(t) for t in re.findall(r'<Fil\b[^>]*/>', oc))
                    if a['T'] == '17' and float(a['G']) != 0])
    return out


def _shelf_bands(xml):
    """Active low/high-shelf filters (T=3/T=4 - see afpx_format.md) per
    channel: (kind, F, Q, G). The search never proposes or edits a shelf -
    only T=17 PEQ - so this is purely a fixed, baseline-only contribution to
    the real per-channel gain chain, never a candidate variable."""
    out = []
    for oc in re.findall(r'<OC\b.*?</OC>', xml, re.S)[:8]:
        bands = []
        for a in (_attrs(t) for t in re.findall(r'<Fil\b[^>]*/>', oc)):
            if a['T'] not in ('3', '4') or float(a['G']) == 0:
                continue
            bands.append(('low' if a['T'] == '3' else 'high', float(a['F']), float(a['Q']), float(a['G'])))
        out.append(bands)
    return out


def _shelf_chain_db(freqs, bands):
    total = np.zeros_like(freqs)
    for kind, f, q, g in bands:
        total += (low_shelf_db if kind == 'low' else high_shelf_db)(freqs, f, q, g)
    return total


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
    global _BASE_CASCADES, _BASE_SHELF_DB, _TOTAL_DB, _SMOOTH_T, _POSITION_TRACES, _POSITION_BASELINE, _SMOOTHER
    global _BASE_OUTPUT_DB, _TRACE_META, _COMPLEX_MODELS, _POSITION_COMPLEX_MODELS, _PREDICTION_AUDIT
    global _SYNTHETIC_PAIRS, _MASK_AUDIT, _NEARFIELD_GUARD_MASK
    if _F is not None:
        return
    raw = {}
    F = None
    pair_roles = {spec[2] for spec in PAIR_SPECS.values()}
    _SYNTHETIC_PAIRS = set()
    for key, nm in SOLO_FILES.items():
        path = _resolve_txt(nm, key, required=key not in pair_roles)
        if path is None:
            continue
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
    for _name, (left, right, together, _band_range, _balance) in PAIR_SPECS.items():
        if together in _T:
            continue
        a = 10.0 ** (_T[left] / 10.0)
        b = 10.0 ** (_T[right] / 10.0)
        _T[together] = 10.0 * np.log10(np.maximum(a + b, 1e-30))
        _TRACE_META[together] = {
            'spl': _T[together],
            'path': '',
            'synthetic_pair': True,
        }
        _SYNTHETIC_PAIRS.add(together)
    _F = F
    _GRID_TOKEN = (len(F), float(F[0]), float(F[-1]), hash(F.tobytes()))
    _SMOOTHER = _build_smoother(F)
    target_trace = _load_txt_rich(TARGET)
    tf, ts = target_trace['freq'], target_trace['spl']
    tgt = np.interp(np.log10(F), np.log10(np.array(tf)), np.array(ts))
    band = (F >= ANCHOR_BAND[0]) & (F <= ANCHOR_BAND[1])
    # Repo-review finding: this used to be a plain single-band median even
    # though the confidence-weighted, multi-band-fallback target_anchor_offset()
    # already existed and wasn't used anywhere. Robust to a thin/noisy
    # 300-3000 Hz window (falls back to 120-1000 Hz, then 1000-6000 Hz, then
    # anything finite) rather than silently degrading with fewer valid bins.
    global_anchor_offset = target_anchor_offset(F, _T['System Sum'], tgt)
    _TGT = tgt + global_anchor_offset
    # null mask: destructive-interference bins in either front pair (from MEASURED
    with open(BASELINE_AFPX, 'rb') as handle:
        # Strict, not 'replace' - see _make_v3.decode_afpx for why.
        baseline_xml = zlib.decompress(handle.read()[4:]).decode('utf-8', 'strict')
    _V5 = _peqset(baseline_xml)
    _BASE_OUTPUT_DB = _output_levels_db(baseline_xml)
    _BASE_CASCADES = [_casc_uncached(bands) for bands in _V5]
    # Full-chain headroom (repo-review finding): the search never touches
    # shelves (only T=17 PEQ), so this is a fixed, baseline-only addition to
    # each channel's real gain chain - included at headroom-check time only,
    # never in _BASE_CASCADES itself (that stays PEQ-only, matching what
    # _delta_channel and everything else that measures a PEQ CHANGE expects).
    _BASE_SHELF_DB = [_shelf_chain_db(F, bands) for bands in _shelf_bands(baseline_xml)]
    _TOTAL_DB = _system_branch_total_uncached()
    _SMOOTH_T = {key: _smooth(values) for key, values in _T.items()}

    _POSITION_TRACES = {}
    position_specs = {
        'left': ('Left Ear ', 'Left '),
        'right': ('Right Ear ', 'Right '),
    }
    target_anchor_audit = {
        'global_offset_db': round(float(global_anchor_offset), 3),
        'positions': {},
    }
    for position, prefixes in position_specs.items():
        path = _position_path(prefixes, SOLO_FILES['System Sum'])
        if path is None:
            continue
        position_trace = _load_txt_rich(path)
        pf = position_trace['freq']
        ps = position_trace['spl'] + _calibration_offset(position + ':System Sum', path)
        measured = np.interp(np.log10(_F), np.log10(pf), ps)
        # Repo-review finding: each position previously got a FULLY
        # INDEPENDENT re-anchor here, which can silently hide a genuine
        # broad level difference between positions (a real acoustic
        # asymmetry, not mic-placement noise) by always re-centering to
        # match its own target-region median. Now: one global anchor
        # (above) stays fixed for the whole search, and each position gets
        # only a small BOUNDED nuisance offset around it - representing
        # plausible mic-placement variance, not a license to erase a real
        # difference. If the position's own raw anchor would have differed
        # from the global one by more than the bound, the excess stays
        # visible as real deviation instead of being absorbed away.
        position_raw_offset = target_anchor_offset(F, measured, tgt)
        nuisance = float(np.clip(
            position_raw_offset - global_anchor_offset,
            -POSITION_ANCHOR_NUISANCE_BOUND_DB, POSITION_ANCHOR_NUISANCE_BOUND_DB,
        ))
        target = tgt + global_anchor_offset + nuisance
        _POSITION_TRACES[position] = {'system': measured, 'target': target, 'file': str(path)}
        target_anchor_audit['positions'][position] = {
            'raw_offset_db': round(float(position_raw_offset), 3),
            'nuisance_offset_db': round(nuisance, 3),
            'clamped': abs(float(position_raw_offset) - global_anchor_offset) > POSITION_ANCHOR_NUISANCE_BOUND_DB + 1e-9,
        }
    _NULL_MASK = np.zeros_like(F, dtype=bool)
    pair_audit = {}
    for name, (left, right, together, band_range, _balance) in PAIR_SPECS.items():
        evidence = interference_mask_evidence(
            F,
            _T[left],
            _T[right],
            _T.get(together),
            synthetic=together in _SYNTHETIC_PAIRS,
            band=band_range,
        )
        _NULL_MASK |= evidence['mask']
        pair_audit[name] = {
            'state': evidence['state'],
            'reason': evidence['reason'],
            'together': together,
            'band_hz': list(band_range),
        }
    modal = modal_null_evidence(
        F,
        _T['System Sum'],
        {name: data['system'] for name, data in _POSITION_TRACES.items()},
        band=(max(20.0, float(F[0])), 250.0),
    )
    _NULL_MASK |= modal['mask']

    # DEFECT 4a: confirm already-flagged nulls with close-mic nearfield
    # captures, when both sides were measured. Optional - graceful no-op
    # when the files aren't present, same as the position traces above.
    nl_path = _resolve_txt(NEARFIELD_L_NAMES, 'FL Nearfield', required=False)
    nr_path = _resolve_txt(NEARFIELD_R_NAMES, 'FR Nearfield', required=False)
    if nl_path is not None and nr_path is not None:
        nl_trace = _load_txt_rich(nl_path)
        nr_trace = _load_txt_rich(nr_path)
        nl_spl = np.interp(log_f, np.log10(nl_trace['freq']), nl_trace['spl'])
        nr_spl = np.interp(log_f, np.log10(nr_trace['freq']), nr_trace['spl'])
        a = 10.0 ** (nl_spl / 10.0)
        b = 10.0 ** (nr_spl / 10.0)
        nearfield_sum = 10.0 * np.log10(np.maximum(a + b, 1e-30))
        nearfield = nearfield_null_evidence(
            F, _T['System Sum'], nearfield_sum, _NULL_MASK, band=INBAND,
        )
        _NULL_MASK |= nearfield['confirmed_mask']
        _NEARFIELD_GUARD_MASK = nearfield['guard_mask']
        nearfield_audit = {
            'state': nearfield['state'],
            'regions': nearfield['regions'],
            'files': [str(nl_path), str(nr_path)],
        }
    else:
        _NEARFIELD_GUARD_MASK = np.zeros_like(F, dtype=bool)
        nearfield_audit = {'state': 'unavailable', 'reason': 'nearfield_captures_missing'}

    _MASK_AUDIT = {
        'pairs': pair_audit,
        'modal': {
            'state': modal['state'],
            'confidence': modal['confidence'],
            'regions': modal['regions'],
        },
        'nearfield': nearfield_audit,
        'target_anchor': target_anchor_audit,
        'blocking_pairs': [
            name for name, item in pair_audit.items() if item['state'] == MASK_UNKNOWN
        ],
    }
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


# Repo-review finding: post-quantization filter reduction. Deliberately a
# WRITE-TIME-ONLY refinement on the small number of already-selected
# finalists, never wired into the hot search-loop resolution
# (_resolve_group_bands/groups_to_band_sets run for every candidate the
# beam search evaluates - thousands per run - and this does several extra
# _casc() calls per channel, which would meaningfully slow the search for
# no benefit there). It only ever considers bands the search purely
# APPENDED (never an edited or pre-existing baseline band - editing/
# removing an existing filter is a deliberate, already-justified action,
# not redundant filter-count bloat), so it can only ever reduce filter
# count, never change what an edit or removal already decided.
@dataclass(frozen=True)
class ConsolidatedPeqFit:
    replacement: tuple[float, float, float]
    max_cascade_error_db: float
    passband_mask: np.ndarray
    overlap: bool


def _peq_passband_mask(freqs, band):
    magnitude = np.abs(peaking_db(freqs, *band))
    threshold = max(0.01, min(3.0, abs(float(band[2])) * 0.5))
    return magnitude >= threshold


def fit_consolidated_peq_pair(first, second, *, freqs=None):
    """Fit one canonical PEQ to an overlapping two-PEQ complex cascade."""
    from scipy.optimize import least_squares

    first = tuple(float(value) for value in first)
    second = tuple(float(value) for value in second)
    if freqs is None:
        low = max(20.0, min(first[0], second[0]) / 4.0)
        high = min(20000.0, max(first[0], second[0]) * 4.0)
        freqs = np.geomspace(low, high, 2048)
    else:
        freqs = np.asarray(freqs, dtype=float)
    first_mask = _peq_passband_mask(freqs, first)
    second_mask = _peq_passband_mask(freqs, second)
    passband = first_mask | second_mask
    overlap = bool(np.any(first_mask & second_mask))
    target = cascade_complex(freqs, (first, second))
    selected = passband if np.any(passband) else np.ones_like(freqs, dtype=bool)

    weights = np.asarray([abs(first[2]), abs(second[2])], dtype=float)
    if float(np.sum(weights)) <= 1e-12:
        weights[:] = 1.0
    initial_frequency = float(np.exp(np.average(
        np.log([first[0], second[0]]), weights=weights
    )))
    initial_q = float(np.average([first[1], second[1]], weights=weights))
    initial_gain = float(np.clip(first[2] + second[2], -15.0, 6.0))

    def residual(values):
        frequency = math.exp(float(values[0]))
        predicted = cascade_complex(
            freqs, ((frequency, float(values[1]), float(values[2])),)
        )
        ratio = target / np.where(np.abs(predicted) > 1e-15, predicted, 1.0)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(ratio), 1e-15))
        phase_db = 8.685889638 * np.unwrap(np.angle(ratio))
        return np.concatenate((magnitude_db[selected], 0.2 * phase_db[selected]))

    fitted = least_squares(
        residual,
        (math.log(initial_frequency), np.clip(initial_q, 0.5, 15.0), initial_gain),
        bounds=(
            (math.log(float(freqs[0])), 0.5, -15.0),
            (math.log(float(freqs[-1])), 15.0, 6.0),
        ),
        max_nfev=400,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    replacement = (
        float(f"{math.exp(float(fitted.x[0])):.2f}"),
        float(fitted.x[1]),
        float(fitted.x[2]),
    )
    fitted_response = cascade_complex(freqs, (replacement,))
    mismatch = 20.0 * np.log10(np.maximum(
        np.abs(target / np.where(np.abs(fitted_response) > 1e-15, fitted_response, 1.0)),
        1e-15,
    ))
    return ConsolidatedPeqFit(
        replacement=replacement,
        max_cascade_error_db=float(np.max(np.abs(mismatch[selected]))),
        passband_mask=selected,
        overlap=overlap,
    )


FILTER_SIMPLIFICATION_TOLERANCE_DB = 0.1


def simplify_removable_bands(channel_bands, removable_bands, tolerance_db=FILTER_SIMPLIFICATION_TOLERANCE_DB):
    """Drop any of `removable_bands` (a subset of `channel_bands`) whose
    removal changes that channel's own cascade by less than `tolerance_db`
    everywhere - it isn't pulling its weight, so it isn't worth the filter
    slot. Then checks near-cancelling PAIRS among whatever survives alone
    (each individually meaningful, but redundant together). Returns
    (kept_bands, dropped_bands); `kept_bands` always contains every band
    NOT in `removable_bands`, unconditionally."""
    _init()
    removable_set = set(removable_bands)
    protected = [b for b in channel_bands if b not in removable_set]
    working = list(removable_bands)
    dropped = []
    index = 0
    while index < len(working):
        trial = working[:index] + working[index + 1:]
        before = _casc(protected + working)
        after = _casc(protected + trial)
        if float(np.max(np.abs(before - after))) < tolerance_db:
            dropped.append(working[index])
            working = trial
        else:
            index += 1
    changed = True
    while changed and len(working) >= 2:
        changed = False
        for a in range(len(working)):
            for b in range(a + 1, len(working)):
                trial = [x for j, x in enumerate(working) if j not in (a, b)]
                before = _casc(protected + working)
                after = _casc(protected + trial)
                if float(np.max(np.abs(before - after))) < tolerance_db:
                    dropped.extend([working[a], working[b]])
                    working = trial
                    changed = True
                    break
            if changed:
                break
    return protected + working, dropped


def _band_key(band):
    f, q, g = band
    return (round(float(f), 1), round(float(q), 2), round(float(g) * 4.0) / 4.0)


def _added_bands_by_channel(band_sets, channels=None):
    """Return only filters added on top of the matching baseline tune.

    Defaults to front channels only (CH_KEYS), matching every existing
    caller's front-specific logic (L/R pair symmetry, imaging, per-band
    measurement-noise justification - none of which have a sub analog
    defined yet). Pass channels=GUARDRAIL_CHANNEL_INDICES explicitly at a
    call site that should also see sub-channel additions.
    """
    if channels is None:
        channels = range(len(CH_KEYS))
    added = {}
    for i in channels:
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


def _guardrail_score(band_sets, predicted=None):
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
    balance_guard = 0.0
    measurement_guard = 0.0
    balance_violations = 0
    alternating_violations = 0
    sum_hole_violations = 0
    noise_floor_violations = 0
    low_frequency_violations = 0
    filter_noise_violations = 0
    pair_by_channel = {}
    for pair_name, (left, right, _together, _pair_range, _balance_band) in PAIR_SPECS.items():
        if left in CH_KEYS and right in CH_KEYS:
            pair_by_channel[CH_KEYS.index(left)] = (pair_name, left, right, CH_KEYS.index(right), "left")
            pair_by_channel[CH_KEYS.index(right)] = (pair_name, left, right, CH_KEYS.index(left), "right")
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
            noise_branch = "high" if channel_key.endswith("High") else (
                "mid" if channel_key.endswith("Mid") else "low"
            )
            local_floor = float(measurement_noise_floor_db([f], noise_branch)[0])
            required_deviation = MEASUREMENT_NOISE_MULTIPLIER * local_floor
            system_deviation_at_filter = abs(_interp_at(_T['System Sum'] - _TGT, f))
            predicted_effect = (
                abs(_interp_at(predicted['System Sum'] - _T['System Sum'], f))
                if predicted is not None else system_deviation_at_filter
            )
            if min(system_deviation_at_filter, predicted_effect) < required_deviation:
                filter_noise_violations += 1
                measurement_guard += 1000.0
            peer = pair_by_channel.get(i)
            peer_added = added.get(peer[3], []) if peer else []
            pair_symmetric = any(_band_key(other) == key for other in peer_added)
            if g < 0.0 and peer and not pair_symmetric:
                pair_name, left, right, _peer_index, side = peer
                branch = "high" if pair_name == "high" else pair_name
                diff = _smooth(_T[left] - _T[right])
                evidence = signed_offset_evidence(_F, diff, f, branch)
                system_deviation = _interp_at(_T['System Sum'] - _TGT, f)
                imaging_weight = float(imaging_balance_weight([f])[0])
                hot_side_matches = (
                    (side == "left" and evidence["offset_db"] > 0.0)
                    or (side == "right" and evidence["offset_db"] < 0.0)
                )
                reasons = []
                if not evidence["eligible"]:
                    reasons.append(evidence["reason"])
                if evidence["reason"] == "alternating_lr_comb":
                    alternating_violations += 1
                if evidence["reason"] == "below_measurement_noise_threshold":
                    noise_floor_violations += 1
                if system_deviation < 0.0:
                    reasons.append("summed_response_below_target")
                    sum_hole_violations += 1
                if imaging_weight < 0.5:
                    reasons.append("imaging_frequency_too_low")
                    low_frequency_violations += 1
                if not hot_side_matches:
                    reasons.append("cut_not_on_systematically_hotter_side")
                if reasons:
                    balance_violations += 1
                    balance_guard += 1000.0
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
    # Evidence admissibility is enforced before search and independently at
    # write verification.  Keep the counts for diagnosis, but do not create a
    # second hard objective inside the authoritative scalar.
    total = shape + unsupported + wasted + boost_q + asym + parsimony
    return {
        'guardrail_penalty': float(total),
        'shape_penalty': float(shape),
        'unsupported_filter_penalty': float(unsupported),
        'wasted_band_penalty': float(wasted),
        'asymmetric_eq_penalty': float(asym),
        'high_q_boost_penalty': float(boost_q),
        'added_band_penalty': float(parsimony),
        'balance_guardrail_penalty': 0.0,
        'measurement_noise_guardrail_penalty': 0.0,
        'balance_guardrail_violation_count': int(balance_violations),
        'alternating_lr_comb_violation_count': int(alternating_violations),
        'summed_hole_violation_count': int(sum_hole_violations),
        'noise_floor_violation_count': int(noise_floor_violations),
        'filter_noise_floor_violation_count': int(filter_noise_violations),
        'low_frequency_imaging_violation_count': int(low_frequency_violations),
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
    new = pr['Sub'].copy()
    for branch in branch_outputs:
        new = power_sum(new, branch)
    if _SYNTHETIC_PAIRS:
        # Optional Together traces are synthesized from the solo drivers.  Their
        # absolute capture level need not match the separately measured System
        # Sum, so an inferred positive-power residual may not exist.  Apply only
        # the modelled branch delta to the authoritative measured System Sum;
        # this keeps the untouched baseline exact and prevents a magnitude-only
        # cut from appearing to raise total output.
        pr['System Sum'] = _T['System Sum'] + (new - old)
    else:
        rest = np.maximum(
            10.0 ** (_T['System Sum'] / 10.0) - 10.0 ** (old / 10.0),
            1e-9,
        )
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
        'drivers': {},
    }
    for role in (*CH_KEYS, 'Sub'):
        if role not in baseline or role not in candidate:
            continue
        payload['drivers'][role] = {
            'frequency_hz': values(_F),
            'change_db': values(candidate[role] - baseline[role]),
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

    # Plain (unweighted) target-relative RMS reported both ways so a masked
    # win can never hide an unmasked loss behind one number - a candidate can
    # improve the null-excluded score while its filter skirts spill enough
    # unrequested boost outside the mask to worsen the null-included one.
    system_error = pr['System Sum'] - _TGT
    target_rms_null_excluded_db = (
        float(np.sqrt(np.mean(system_error[keep] ** 2))) if np.any(keep) else 0.0
    )
    target_rms_null_included_db = (
        float(np.sqrt(np.mean(system_error[inb] ** 2))) if np.any(inb) else 0.0
    )

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

    # Headroom plus any newly-added correction landing in a masked null.
    #
    # head_peak/SOFT_CAP_DB below is a SOFT tiebreaker only - it must never be
    # the only thing standing between a candidate and a real clipping risk.
    # headroom_margin_db is the hard feasibility gate: real per-channel peak
    # output (cascade + existing baseline trim + any protective trim), signed
    # so positive = dB of headroom remaining below 0 dBFS, negative = already
    # over. A channel is only flagged when the CANDIDATE ITSELF raises that
    # channel's real peak above the baseline tune's own peak (never for a
    # pre-existing tight baseline the candidate leaves untouched or improves -
    # a no-op/baseline-preserving candidate must always stay selectable), and
    # even then only when the resulting margin is below
    # HEADROOM_REQUIRED_MARGIN_DB, or the baseline channel was already
    # clip-risky (any further increase there is disallowed regardless of the
    # 1.5 dB number). Flagged candidates are rejected via the same
    # huge-penalty guardrail convention as the other hard rules below - never
    # a tradeable cost. (bug: a run once paid +1.598 on the old SOFT term to
    # buy a 0.026/7.5 objective "win" on a channel already clip-risky - see
    # CHANGELOG.md.)
    head_peak = 0.0
    null_boost = 0.0
    headroom_violations = 0
    headroom_margin_db = {}
    for i in GUARDRAIL_CHANNEL_INDICES:
        trim_i = float(trim_plan.get(i, 0.0))
        # Full-chain headroom: shelves (T=3/4) carry real gain and are part
        # of the actual per-channel signal chain, but the search never
        # touches them - fold their fixed baseline contribution into the
        # peak everywhere PEQ cascade + output trim used to stand in for
        # the "whole chain" alone. Repo-review finding: a baseline with an
        # active shelf could have its true peak silently underestimated.
        shelf_i = _BASE_SHELF_DB[i] if i < len(_BASE_SHELF_DB) else np.zeros_like(_F)
        candidate_cascade = _casc(band_sets[i]) + shelf_i
        b = candidate_cascade + trim_i
        head_peak = max(head_peak, float(np.max(b)))
        baseline_peak_db = float(np.max(_BASE_CASCADES[i] + shelf_i)) + float(_BASE_OUTPUT_DB[i])
        candidate_peak_db = float(np.max(candidate_cascade)) + float(_BASE_OUTPUT_DB[i]) + trim_i
        margin_db = -candidate_peak_db
        headroom_margin_db[_channel_label(i)] = round(margin_db, 3)
        baseline_clip_risk = baseline_peak_db > 0.0
        candidate_makes_it_worse = candidate_peak_db > baseline_peak_db + 1e-9
        channel_unsafe = candidate_makes_it_worse and (
            margin_db < HEADROOM_REQUIRED_MARGIN_DB or baseline_clip_risk
        )
        if channel_unsafe:
            headroom_violations += 1
        delta = _delta_channel(i, band_sets)
        null_boost += float(np.sum(np.abs(delta[_NULL_MASK]))) / max(np.sum(_NULL_MASK), 1)
    headroom_guardrail_penalty = HEADROOM_VIOLATION_PENALTY * headroom_violations

    # DEFECT 4a: a positive-gain band is rejected outright - not merely
    # discouraged via null_boost above - when its own -3dB-down skirt still
    # reaches into a null the nearfield captures confirmed is a room/summation
    # artifact (see nearfield_null_evidence in _tunefit.py). Only newly added
    # or edited bands are checked; an untouched baseline band already in that
    # spot is left alone. Same huge-penalty guardrail convention as headroom.
    nearfield_skirt_violations = 0
    if _NEARFIELD_GUARD_MASK is not None and np.any(_NEARFIELD_GUARD_MASK):
        for i, bands in _added_bands_by_channel(band_sets, channels=GUARDRAIL_CHANNEL_INDICES).items():
            for f, q, g in bands:
                if g <= 0.0:
                    continue
                skirt = peaking_db(_F, f, q, g) >= (g - 3.0)
                if np.any(skirt & _NEARFIELD_GUARD_MASK):
                    nearfield_skirt_violations += 1
    nearfield_skirt_penalty = NEARFIELD_SKIRT_PENALTY * nearfield_skirt_violations

    # Parsimony must count every filter that costs a slot, not just front
    # ones - a sub-only candidate previously looked "free" here too.
    n_bands = sum(
        len(band_sets[i]) for i in GUARDRAIL_CHANNEL_INDICES if i < len(band_sets)
    )
    guard = _guardrail_score(band_sets, pr)

    # spatial_worst_db only carries independent evidence when there is more
    # than one measured position: at 1 position it is a max-abs-deviation
    # statistic of the SAME trace the tonal term already scores, so applying
    # W['worst'] there double-counts one improvement as two score components.
    # (bug: a candidate with a 0.026/7.5 objective "win" that was entirely
    # this duplicate shipped as a false 0.4% improvement - see CHANGELOG.md.)
    position_count = tonal_parts['spatial_position_count']
    worst_weight = W['worst'] if position_count >= 2 else 0.0

    comp = {
        **tonal_parts,
        'worst_masked': worst,
        'active_weights': {**W, 'worst': worst_weight},
        'target_rms_null_excluded_db': target_rms_null_excluded_db,
        'target_rms_null_included_db': target_rms_null_included_db,
        'headroom_peak': head_peak,
        'headroom_margin_db': headroom_margin_db,
        'headroom_violation_count': headroom_violations,
        'headroom_guardrail_penalty': float(headroom_guardrail_penalty),
        'nearfield_skirt_violation_count': nearfield_skirt_violations,
        'nearfield_skirt_guardrail_penalty': float(nearfield_skirt_penalty),
        'null_boost_avg': null_boost,
        # Key name kept for compatibility with existing reporting/CSV
        # consumers (_optimizer.py's "filter_count"); the VALUE now counts
        # sub-channel filters too - see GUARDRAIL_CHANNEL_INDICES above.
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
              + worst_weight * worst
              + W['headroom'] * max(0.0, head_peak - SOFT_CAP_DB)
              + W['output_gain'] * comp['output_level_gain_db']
              + W['null_boost'] * null_boost
              + W['parsimony'] * n_bands
              + W['spatial_fragility'] * tonal_parts['spatial_fragility_penalty']
              + guard['guardrail_penalty']
              + headroom_guardrail_penalty
              + nearfield_skirt_penalty)
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
        'mask_audit': dict(_MASK_AUDIT),
    }


def score_afpx(path):
    _init()
    # Strict, not 'replace' - see _make_v3.decode_afpx for why.
    xml = zlib.decompress(open(path, 'rb').read()[4:]).decode('utf-8', 'strict')
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
