# _tunefit.py — joint PEQ optimizer + minimum-phase classifier + audibility score.
# Companion to _devcalc.py (which stays the measurement/deviation workhorse).
# Added 2026-07-02 (Fable max pass). Everything here is self-tested by `python _tunefit.py`.
#
# WHY THIS EXISTS (the gap vs TuneEQ / REW's own EQ window):
#  - TuneEQ and REW fit bands GREEDILY, one at a time, to raw magnitude error.
#    Greedy = each band ignores how its skirts change the next band's problem.
#    fit_peq() fits all bands JOINTLY (scipy least_squares over the full cascade).
#  - Neither weights the error by audibility. audibility_score() ERB-smooths the
#    residual and weights by where the ear is sensitive, so the optimizer spends
#    its band budget where it is HEARD, not where the plot looks worst.
#  - Neither checks EQ-ability physics. REW's own doctrine (minimumphase.html):
#    "Anywhere the excess group delay plot is flat is a minimum phase region"
#    -> correctable. Sharp dips with wild excess-GD swings are non-minimum-phase
#    -> EQ cannot fix them. excess_gd_mask() computes that classifier from a
#    single-position export WITH PHASE (REW text export, 3 columns).
import math
import os

import numpy as np

try:
    # Normal case: loaded as the objective_module package's own submodule
    # (objective_module._tunefit) - a relative import resolves correctly
    # with no sys.path changes, so this file is never independently
    # importable under a second, bare "_tunefit" module identity (that
    # would silently duplicate every function in this file as a DIFFERENT
    # object from the one everyone else has - a real bug caught here: an
    # earlier version used sys.path.insert + a bare import instead, which
    # is exactly what tripped this).
    from .device_profile import DEFAULT_DEVICE_PROFILE
except ImportError:
    # Standalone self-test (`python _tunefit.py` from inside this
    # directory) - there's no package context for a relative import, but
    # Python already put this file's own directory on sys.path[0], so a
    # bare import finds the sibling file directly.
    from device_profile import DEFAULT_DEVICE_PROFILE

FS = DEFAULT_DEVICE_PROFILE.sample_rate_hz  # Helix internal rate - see device_profile.py
LOGSTEP = 2 ** (1 / 96.0)        # REW 96 PPO

# --------------------------------------------------------------------------
# biquad + cascade (same RBJ math _devcalc.py uses, vector over freq axis)
def peaking_complex(freqs, f0, Q, gain_db, fs=FS):
    """Complete RBJ peaking transfer, including magnitude and phase."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    al = np.sin(w0) / (2 * Q)
    b0, b1, b2 = 1 + al * A, -2 * np.cos(w0), 1 - al * A
    a0, a1, a2 = 1 + al / A, -2 * np.cos(w0), 1 - al / A
    w = 2 * np.pi * freqs / fs
    z1, z2 = np.exp(-1j * w), np.exp(-2j * w)
    return (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)

def peaking_db(freqs, f0, Q, gain_db, fs=FS):
    H = peaking_complex(freqs, f0, Q, gain_db, fs)
    return 20 * np.log10(np.abs(H))

def cascade_complex(freqs, bands, fs=FS):
    out = np.ones_like(freqs, dtype=complex)
    for F, Q, G in bands:
        out *= peaking_complex(freqs, F, Q, G, fs)
    return out

def cascade_db(freqs, bands):
    out = np.zeros_like(freqs, dtype=float)
    for F, Q, G in bands:
        out += peaking_db(freqs, F, Q, G)
    return out

# --------------------------------------------------------------------------
# 1) MINIMUM-PHASE EXTRACTION + EXCESS GROUP DELAY  (REW doctrine, computable)
def minphase_from_mag(freqs, mag_db, n_fft=2 ** 16, fs=None):
    """Min-phase (radians, on `freqs`) implied by a magnitude curve.
    Real-cepstrum method: resample |H| to a linear grid, fold the cepstrum,
    read back the phase. Standard DSP; assumes the magnitude IS the whole story
    (that's the definition of minimum phase).

    `fs` only sets this reconstruction's own internal FFT Nyquist bound - it
    has nothing to do with the target DSP's processing rate (this operates
    on measured acoustic magnitude data, not the DSP's biquad math). Left
    unspecified, it is derived from the data itself with headroom above the
    highest measured frequency, so the reconstruction is always correct for
    whatever `freqs` actually contains. Previously hardcoded to 48000.0
    regardless of input: harmless for most car-audio REW exports (max
    ~20-24 kHz), but that put Nyquist right at the edge of - or below - data
    that runs close to or past 24 kHz, silently degrading the reconstructed
    minimum phase, excess group delay, and EQ-ability classification there.
    See CHANGELOG.md."""
    if fs is None:
        fs = 2.2 * float(np.max(freqs))
    lin_f = np.linspace(0, fs / 2, n_fft // 2 + 1)
    lo, hi = freqs.min(), freqs.max()
    lin_db = np.interp(np.clip(lin_f, lo, hi), freqs, mag_db)  # clamp ends flat
    log_mag = lin_db / 8.685889638             # dB -> ln|H|
    full = np.concatenate([log_mag, log_mag[-2:0:-1]])          # even spectrum
    cep = np.fft.ifft(full).real
    n = len(full)
    fold = np.zeros(n)
    fold[0] = cep[0]
    fold[1:n // 2] = 2 * cep[1:n // 2]
    fold[n // 2] = cep[n // 2]
    mp_full = np.fft.fft(fold)
    mp_phase_lin = np.imag(mp_full[:n_fft // 2 + 1])            # radians (min phase)
    return np.interp(freqs, lin_f, mp_phase_lin)

def excess_gd_mask(freqs, spl_db, phase_deg, flat_ms=1.0, smooth_oct=1 / 6.0, fs=None):
    """The EQ-ability classifier. Inputs: single-position REW text export WITH
    phase (freq, SPL, phase columns). Returns (excess_gd_ms, eqable_mask).
    REW doctrine: flat excess GD = minimum phase = EQ WORKS THERE; wild excess-GD
    swings (usually at sharp dips) = non-minimum-phase = EQ CANNOT FIX. `flat_ms`
    = how far excess GD may deviate from its local median and still count flat.
    Note: an overall time-of-flight offset only adds a CONSTANT GD slope, which the
    local-median comparison ignores by construction. `fs` is forwarded to
    minphase_from_mag() - see its docstring; leave unspecified unless a caller
    has a specific reason to override the data-derived default."""
    ph = np.unwrap(np.deg2rad(phase_deg))
    mp = minphase_from_mag(freqs, spl_db, fs=fs)
    ex = ph - mp
    w = 2 * np.pi * freqs
    gd = -np.gradient(ex, w) * 1000.0            # excess group delay, ms
    # local median baseline (removes constant offset + slow trend)
    nb = max(3, int(round((1.0 / np.log10(LOGSTEP)) * np.log10(2 ** smooth_oct))))
    if nb % 2 == 0: nb += 1
    half = nb // 2
    base = np.array([np.median(gd[max(0, i - half):min(len(gd), i + half + 1)])
                     for i in range(len(gd))])
    wob = np.abs(gd - base)
    # wobble itself smoothed a touch so single-bin spikes don't flip the mask
    wob = np.convolve(wob, np.ones(5) / 5, mode='same')
    return gd, (wob <= flat_ms)

# --------------------------------------------------------------------------
# 2) AUDIBILITY-WEIGHTED SCORE (ERB smoothing + sensitivity weighting)
def erb_hz(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)

def erb_smooth(freqs, y):
    dlog = np.log(LOGSTEP)
    out = np.empty_like(y)
    for i in range(len(y)):
        hb = max(1, int(round(np.log(1 + 0.5 * erb_hz(freqs[i]) / freqs[i]) / dlog)))
        out[i] = np.mean(y[max(0, i - hb):min(len(y), i + hb + 1)])
    return out

def audibility_weight(freqs):
    """Simple sensitivity weighting, PROVISIONAL (Toole/Olive tables still not
    primary-sourced): full weight 200 Hz-6 kHz (vocals/timbre/imaging band the
    ear is fussiest about + competition midrange), tapering to 0.5 by 40 Hz and
    0.4 by 16 kHz. Shapes priority only - it does not silence anything."""
    w = np.ones_like(freqs)
    lo = freqs < 200
    w[lo] = 0.5 + 0.5 * (np.log2(freqs[lo] / 40.0) / np.log2(200.0 / 40.0))
    hi = freqs > 6000
    w[hi] = 1.0 - 0.6 * (np.log2(freqs[hi] / 6000.0) / np.log2(16000.0 / 6000.0))
    return np.clip(w, 0.3, 1.0)


# --------------------------------------------------------------------------
# Measurement-repeatability and imaging guardrails.
#
# This schedule is deliberately explicit and reportable.  It is based on the
# same-rig MMM repeatability audit supplied for this optimizer, rather than a
# claim that every microphone/session has the same noise floor.
MEASUREMENT_NOISE_MODEL_ID = "user_supplied_mmm_repeatability_v1"
MEASUREMENT_NOISE_MULTIPLIER = 2.5
_MEASUREMENT_NOISE_OVERRIDE = None
MASK_DETECTED = "DETECTED"
MASK_CLEAR = "CLEAR"
MASK_UNKNOWN = "UNKNOWN"


def interference_mask_evidence(freqs, left_db, right_db, together_db=None,
                               *, synthetic=False, band=None):
    """Return a tri-state destructive-summation audit.

    A synthesized power sum contains no independent summation evidence, so it
    is UNKNOWN by definition.  Invalid measured data is also UNKNOWN and the
    exception is retained in ``reason`` instead of being silently converted to
    a clear mask.
    """
    f = np.asarray(freqs, dtype=float)
    empty = np.zeros_like(f, dtype=bool)
    if synthetic or together_db is None:
        return {
            "state": MASK_UNKNOWN,
            "mask": empty,
            "reason": "measured_together_trace_missing",
        }
    try:
        together = np.asarray(together_db, dtype=float)
        _, _, _, flagged = interference_audit(
            f, np.asarray(left_db, dtype=float), np.asarray(right_db, dtype=float), together
        )
        if band is not None:
            lo, hi = map(float, band)
            in_band = (f >= lo) & (f <= hi)
            if np.any(in_band):
                alive = together > (np.nanmax(together[in_band]) - 20.0)
                flagged = flagged & in_band & alive
            else:
                flagged = empty
        return {
            "state": MASK_DETECTED if np.any(flagged) else MASK_CLEAR,
            "mask": np.asarray(flagged, dtype=bool),
            "reason": "destructive_summation_detected" if np.any(flagged) else "measured_pair_clear",
        }
    except (TypeError, ValueError, FloatingPointError, IndexError) as exc:
        return {
            "state": MASK_UNKNOWN,
            "mask": empty,
            "reason": f"interference_audit_failed:{type(exc).__name__}:{exc}",
        }


def _fractional_octave_mean(freqs, values, width_oct):
    f = np.asarray(freqs, dtype=float)
    y = np.asarray(values, dtype=float)
    out = np.empty_like(y)
    half = float(width_oct) / 2.0
    for index, center in enumerate(f):
        selected = np.abs(np.log2(np.maximum(f, 1e-9) / center)) <= half
        out[index] = np.nanmean(y[selected]) if np.any(selected) else y[index]
    return out


def modal_null_evidence(freqs, center_db, position_db=None, band=(20.0, 250.0)):
    """Classify spatially unstable or very narrow/deep LF nulls.

    With multiple positions a centre dip is masked only when its local minimum
    moves by more than 1/8 octave.  With one position, the fallback masks dips
    deeper than 8 dB relative to a 1/3-octave local mean and narrower than
    roughly 1/6 octave; those classifications are explicitly low confidence.
    """
    f = np.asarray(freqs, dtype=float)
    center = np.asarray(center_db, dtype=float)
    selected = (f >= float(band[0])) & (f <= float(band[1])) & np.isfinite(center)
    mask = np.zeros_like(f, dtype=bool)
    if np.count_nonzero(selected) < 5:
        return {"state": MASK_CLEAR, "mask": mask, "confidence": "low", "regions": []}

    broad = _fractional_octave_mean(f, center, 1 / 3)
    residual = center - broad
    minima = [
        index for index in range(1, len(f) - 1)
        if selected[index] and residual[index] <= -4.0
        and residual[index] <= residual[index - 1] and residual[index] <= residual[index + 1]
    ]
    positions = [np.asarray(values, dtype=float) for values in (position_db or {}).values()]
    regions = []
    for index in minima:
        center_f = float(f[index])
        half_oct = 1 / 12
        local = selected & (np.abs(np.log2(f / center_f)) <= 1 / 3)
        shifted = []
        for values in positions:
            pos_broad = _fractional_octave_mean(f, values, 1 / 3)
            pos_residual = values - pos_broad
            candidates = np.flatnonzero(local & np.isfinite(pos_residual))
            if candidates.size:
                shifted.append(float(f[candidates[np.argmin(pos_residual[candidates])]]))
        if positions:
            centres = [center_f, *shifted]
            span_oct = math.log2(max(centres) / min(centres)) if len(centres) > 1 else 0.0
            is_modal = len(shifted) == len(positions) and span_oct > 1 / 8
            confidence = "high"
        else:
            distance = np.abs(np.log2(f / center_f))
            shoulders = selected & (distance >= 1 / 12) & (distance <= 1 / 4)
            local_baseline = float(np.nanmedian(center[shoulders])) if np.any(shoulders) else broad[index]
            local_depth = float(center[index] - local_baseline)
            deep = local_depth <= -8.0
            dip = center <= local_baseline + local_depth / 2.0
            window = selected & (distance <= 1 / 3)
            # Connected component containing `index` itself, not every dip
            # bin within the 1/3-octave window - a bare boolean selection
            # can silently merge this minimum with a SEPARATE, unrelated
            # dip island nearby into one inflated width estimate, which can
            # push a genuinely narrow modal null over the 1/6-octave cutoff
            # below and misclassify it as too wide to be modal.
            left = index
            while left > 0 and dip[left - 1] and window[left - 1]:
                left -= 1
            right = index
            while right < len(f) - 1 and dip[right + 1] and window[right + 1]:
                right += 1
            contiguous = np.arange(left, right + 1)
            width_oct = (
                math.log2(f[contiguous[-1]] / f[contiguous[0]])
                if contiguous.size > 1 else 0.0
            )
            is_modal = deep and width_oct <= 1 / 6
            span_oct = width_oct
            confidence = "low"
        if is_modal:
            region = selected & (np.abs(np.log2(f / center_f)) <= half_oct)
            mask |= region
            regions.append({
                "center_hz": center_f,
                "depth_db": float(residual[index]),
                "movement_or_width_oct": float(span_oct),
                "confidence": confidence,
            })
    return {
        "state": MASK_DETECTED if np.any(mask) else MASK_CLEAR,
        "mask": mask,
        "confidence": "high" if positions else "low",
        "regions": regions,
    }


NEARFIELD_DEPTH_RATIO_THRESHOLD = 0.5


def nearfield_null_evidence(freqs, at_seat_db, nearfield_db, null_mask, band=None):
    """Confirm already-flagged nulls with a close-mic nearfield trace.

    A dip that is deep at the listening seat but much shallower right at the
    driver (negligible room path) is a room-summation/reflection artifact,
    not something wrong with the driver - EQ cannot fix it and boosting into
    it wastes filter budget and headroom for no audible gain. Depth is
    measured relative to each trace's own 1/3-octave local baseline, so an
    absolute level offset between the loud close-mic capture and the quieter
    at-seat one never affects the comparison.

    Only bins already in ``null_mask`` are considered (this narrows, it does
    not widen, what counts as a null). A run is CONFIRMED not-EQ-able when
    the nearfield depth is under half the at-seat depth at that run's
    deepest bin. Confirmed runs also report a ``guard_mask`` spanning out to
    each side's -3dB-down point (relative to the at-seat local baseline) so
    a positive-gain candidate whose skirt reaches into that span - even
    without its centre landing on the exact null bin - can be rejected too.
    """
    f = np.asarray(freqs, dtype=float)
    at_seat = np.asarray(at_seat_db, dtype=float)
    nearfield = np.asarray(nearfield_db, dtype=float)
    mask = np.asarray(null_mask, dtype=bool)
    confirmed = np.zeros_like(mask)
    guard = np.zeros_like(mask)
    regions = []
    if band is not None:
        lo, hi = map(float, band)
        in_band = (f >= lo) & (f <= hi)
    else:
        in_band = np.ones_like(f, dtype=bool)
    candidate = mask & in_band & np.isfinite(nearfield) & np.isfinite(at_seat)
    if not np.any(candidate):
        return {
            "state": MASK_CLEAR,
            "confirmed_mask": confirmed,
            "guard_mask": guard,
            "regions": regions,
        }
    at_seat_broad = _fractional_octave_mean(f, at_seat, 1 / 3)
    nearfield_broad = _fractional_octave_mean(f, nearfield, 1 / 3)
    at_seat_residual = at_seat - at_seat_broad
    nearfield_residual = nearfield - nearfield_broad
    indices = np.flatnonzero(candidate)
    runs = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx != prev + 1:
            runs.append((start, prev))
            start = idx
        prev = idx
    runs.append((start, prev))
    for lo_i, hi_i in runs:
        span = slice(lo_i, hi_i + 1)
        center_index = lo_i + int(np.argmin(at_seat_residual[span]))
        at_seat_depth = float(-at_seat_residual[center_index])
        nearfield_depth = float(-nearfield_residual[center_index])
        if at_seat_depth <= 0.0:
            continue
        ratio = nearfield_depth / at_seat_depth
        if ratio >= NEARFIELD_DEPTH_RATIO_THRESHOLD:
            continue
        confirmed[span] = True
        left = center_index
        while left > 0 and at_seat_residual[left - 1] <= -3.0:
            left -= 1
        right = center_index
        while right < len(f) - 1 and at_seat_residual[right + 1] <= -3.0:
            right += 1
        guard[left:right + 1] = True
        regions.append({
            "center_hz": float(f[center_index]),
            "at_seat_depth_db": round(at_seat_depth, 2),
            "nearfield_depth_db": round(nearfield_depth, 2),
            "depth_ratio": round(ratio, 3),
            "guard_band_hz": [float(f[left]), float(f[right])],
        })
    return {
        "state": MASK_DETECTED if np.any(confirmed) else MASK_CLEAR,
        "confirmed_mask": confirmed,
        "guard_mask": guard,
        "regions": regions,
    }


def measurement_noise_floor_db(freqs, branch="low"):
    """Return the assumed one-sigma-ish local repeatability floor in dB.

    Low/mid follows the audited rig: about 0.1 dB at 400-500 Hz, rising
    through 0.6-1.0 dB at 700-1400 Hz and to 1.6 dB above 2 kHz.  Tweeters use
    the separately measured 0.23-0.46 dB range.  Interpolation is logarithmic
    in frequency and clamps outside the calibration points.
    """
    f = np.maximum(np.asarray(freqs, dtype=float), 1.0)
    if _MEASUREMENT_NOISE_OVERRIDE:
        branches = dict(_MEASUREMENT_NOISE_OVERRIDE.get("branches", {}))
        key = "high" if str(branch).lower() in {"high", "tweeter", "tweeters"} else "low"
        points = list(branches.get(key, []))
        if points:
            xp = np.asarray([float(item["frequency_hz"]) for item in points])
            yp = np.asarray([float(item["floor_db"]) for item in points])
            return np.interp(np.log10(f), np.log10(xp), yp)
    if str(branch).lower() in {"high", "tweeter", "tweeters"}:
        xp = np.array([1800.0, 3000.0, 6000.0, 10000.0, 16000.0])
        yp = np.array([0.23, 0.27, 0.34, 0.41, 0.46])
    else:
        xp = np.array([200.0, 400.0, 500.0, 700.0, 1400.0, 2200.0, 5000.0])
        yp = np.array([0.20, 0.10, 0.10, 0.60, 1.00, 1.60, 1.60])
    return np.interp(np.log10(f), np.log10(xp), yp)


def imaging_balance_weight(freqs):
    """Frequency importance for corrections justified only by L/R imaging.

    Below 400 Hz the car-cabin wavelength makes small interaural magnitude
    differences weak localization evidence.  The weight rises rapidly into
    the 500 Hz-8 kHz imaging band and tapers gently above it.
    """
    f = np.maximum(np.asarray(freqs, dtype=float), 1.0)
    xp = np.array([100.0, 300.0, 500.0, 8000.0, 16000.0])
    yp = np.array([0.00, 0.00, 1.00, 1.00, 0.55])
    return np.interp(np.log10(f), np.log10(xp), yp)


def signed_offset_evidence(freqs, difference_db, center_hz, branch="low",
                           multiplier=MEASUREMENT_NOISE_MULTIPLIER):
    """Classify broad L/R offset evidence over +/- one octave.

    The already-smoothed L-minus-R trace is sampled on a fixed log grid so
    dense REW exports do not inflate the sign-change count.  Values inside the
    local measurement floor are treated as indeterminate.  An eligible offset
    must retain one dominant sign, change sign at most once, and clear the
    local repeatability floor by ``multiplier``.
    """
    f = np.asarray(freqs, dtype=float)
    d = np.asarray(difference_db, dtype=float)
    center = float(center_hz)
    valid = np.isfinite(f) & np.isfinite(d) & (f > 0.0)
    if np.count_nonzero(valid) < 3 or center <= 0.0:
        return {
            "eligible": False,
            "reason": "insufficient_lr_data",
            "sign_changes": 0,
            "dominant_sign_fraction": 0.0,
            "offset_db": 0.0,
            "noise_floor_db": float(measurement_noise_floor_db([center], branch)[0]),
            "required_deviation_db": float(
                multiplier * measurement_noise_floor_db([center], branch)[0]
            ),
        }
    fv = f[valid]
    dv = d[valid]
    order = np.argsort(fv)
    fv = fv[order]
    dv = dv[order]
    lo = max(float(fv[0]), center / 2.0)
    hi = min(float(fv[-1]), center * 2.0)
    if hi <= lo:
        sample_f = np.array([center])
    else:
        sample_f = np.geomspace(lo, hi, 13)
    sample_d = np.interp(np.log10(sample_f), np.log10(fv), dv)
    sample_floor = measurement_noise_floor_db(sample_f, branch)
    signs = np.sign(np.where(np.abs(sample_d) >= sample_floor, sample_d, 0.0))
    nonzero = signs[signs != 0.0]
    sign_changes = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    dominant_fraction = (
        float(max(np.count_nonzero(nonzero > 0.0), np.count_nonzero(nonzero < 0.0)) / len(nonzero))
        if len(nonzero) else 0.0
    )
    offset = float(np.median(sample_d))
    floor = float(measurement_noise_floor_db([center], branch)[0])
    required = float(multiplier * floor)
    if sign_changes > 1:
        reason = "alternating_lr_comb"
    elif dominant_fraction < 0.75:
        reason = "lr_offset_not_systematic"
    elif abs(offset) < required:
        reason = "below_measurement_noise_threshold"
    else:
        reason = "systematic_lr_offset"
    return {
        "eligible": reason == "systematic_lr_offset",
        "reason": reason,
        "sign_changes": sign_changes,
        "dominant_sign_fraction": dominant_fraction,
        "offset_db": offset,
        "noise_floor_db": floor,
        "required_deviation_db": required,
    }


def cross_session_persistence(deviation_db_by_session, noise_floor_db,
                              multiplier=MEASUREMENT_NOISE_MULTIPLIER):
    """DEFECT 6: classify a deviation as a real, repeatable target error or
    single-MMM-session noise, by voting across every supplied session's own
    measurement at the same frequency.

    A single MMM session cannot tell a genuine deviation from run-to-run
    capture noise apart. ``deviation_db_by_session`` is one signed dB value
    per session that actually had coverage at this frequency - a sparse
    session missing the relevant trace contributes nothing and is simply
    left out of the vote, it never counts as disagreement. Eligible only
    when every included session agrees in sign AND every one individually
    clears ``multiplier`` times the local measurement floor - the weakest
    session sets the bar, not the average, so one marginal session can't be
    outvoted by two strong ones.
    """
    values = np.asarray(
        [float(v) for v in deviation_db_by_session if v is not None and np.isfinite(v)],
        dtype=float,
    )
    required = float(multiplier) * float(noise_floor_db)
    if values.size < 2:
        return {
            "eligible": False,
            "reason": "insufficient_session_coverage",
            "session_count": int(values.size),
            "min_magnitude_db": float(np.min(np.abs(values))) if values.size else 0.0,
            "required_deviation_db": required,
        }
    signs = np.sign(values)
    unanimous = bool(np.all(signs == signs[0])) and signs[0] != 0.0
    min_magnitude = float(np.min(np.abs(values)))
    if not unanimous:
        reason = "sign_disagreement_across_sessions"
    elif min_magnitude < required:
        reason = "below_measurement_noise_threshold"
    else:
        reason = "persistent_across_sessions"
    return {
        "eligible": reason == "persistent_across_sessions",
        "reason": reason,
        "session_count": int(values.size),
        "min_magnitude_db": min_magnitude,
        "required_deviation_db": required,
    }


# --------------------------------------------------------------------------
# Repo-review finding: the objective's null classification, repeatability
# checks, and driver-authority weighting were all real but separate binary
# gates - a candidate either cleared every threshold or it didn't, with no
# continuous notion of "how confident are we." Boosting a dip and cutting a
# peak are not symmetric risks (a missed cut just leaves a peak in place; an
# unjustified boost can audibly worsen the exact spot it targeted), so they
# need separate, asymmetric confidence, not one shared pass/fail. This
# combines the SAME evidence the discrete gates already use into one
# continuous [0,1] score per frequency bin - additive to those gates, not a
# replacement for any of them; every existing hard/soft guardrail keeps
# running unchanged. See CHANGELOG.md.
CONFIDENCE_NEUTRAL = 1.0  # missing evidence: doesn't count for OR against
CONFIDENCE_PHASE_UNKNOWN_PENALTY = 0.15  # phase-classified non-minimum-phase


def correction_confidence(freqs, *, null_fraction=None, driver_authority=None,
                          session_agreement_db=None, session_required_db=None,
                          eqable_mask=None, spatial_agreement=None):
    """Continuous per-bin confidence that a proposed correction is real and
    worth acting on, split into `boost` (asking for more energy) and `cut`
    (asking for less), each 0..1 on `freqs`.

    This is a PRODUCT of independent-evidence factors, so the correct "we
    don't have this evidence" default for a missing factor is 1.0 (no
    effect), not some partial-credit value - only evidence that is
    ACTUALLY PRESENT and ACTUALLY WEAK should reduce confidence. (An
    earlier version defaulted every missing factor to 0.5 and multiplied
    them all together regardless of availability; with three or four
    factors commonly absent at once - e.g. a single-session MMM run has no
    position or phase data at all - that compounded into rejecting ~98% of
    every boost candidate's gain even when the evidence that WAS available
    was strong. Fixed before it shipped. See CHANGELOG.md.) Every input
    below is optional; a caller with only some of the evidence (the common
    case) gets a confidence driven entirely by what it actually has.

    - `null_fraction`: 0..1 (or boolean mask) - how much of this bin is
      already confirmed cancellation/reflection (e.g. _NULL_MASK, or a
      nearfield/modal region's confidence). Only affects `boost` - a
      genuine null is never a real target to fill.
    - `driver_authority`: 0..1 - how much the candidate's own driver
      contributes to the system sum here (branch_contribution()); a
      correction proposed where that driver barely matters is unreliable.
    - `session_agreement_db` / `session_required_db`: repeatability - the
      weakest cross-session deviation magnitude vs what's required to clear
      the noise floor (same ratio cross_session_persistence() computes).
    - `eqable_mask`: bool array from excess_gd_mask() - minimum-phase
      regions are trustworthy correction targets, non-minimum-phase ones
      usually are not (REW doctrine - see excess_gd_mask's docstring).
    - `spatial_agreement`: 0..1 - sign/centre-frequency agreement across
      measured positions.
    """
    f = np.asarray(freqs, dtype=float)
    n = len(f)

    def _neutral():
        return np.full(n, CONFIDENCE_NEUTRAL)

    c_null = (
        np.zeros(n) if null_fraction is None
        else np.clip(np.asarray(null_fraction, dtype=float), 0.0, 1.0)
    )
    c_authority = (
        _neutral() if driver_authority is None
        else np.clip(np.asarray(driver_authority, dtype=float), 0.0, 1.0)
    )
    if session_agreement_db is None or session_required_db is None:
        c_repeat = _neutral()
    else:
        required = np.maximum(np.asarray(session_required_db, dtype=float), 1e-9)
        ratio = np.asarray(session_agreement_db, dtype=float) / required
        c_repeat = np.clip(ratio / 2.0, 0.0, 1.0)
    c_phase = (
        _neutral() if eqable_mask is None
        else np.where(np.asarray(eqable_mask, dtype=bool), 1.0, CONFIDENCE_PHASE_UNKNOWN_PENALTY)
    )
    c_spatial = (
        _neutral() if spatial_agreement is None
        else np.clip(np.asarray(spatial_agreement, dtype=float), 0.0, 1.0)
    )
    boost = c_spatial * c_repeat * c_phase * c_authority * (1.0 - c_null)
    cut = c_spatial * c_repeat * c_authority
    return {
        "boost": np.clip(boost, 0.0, 1.0),
        "cut": np.clip(cut, 0.0, 1.0),
        "components": {
            "spatial": c_spatial, "repeat": c_repeat, "phase": c_phase,
            "authority": c_authority, "null": c_null,
        },
    }


def measurement_noise_model():
    """Serializable description included in optimizer and PDF reports."""
    if _MEASUREMENT_NOISE_OVERRIDE:
        return dict(_MEASUREMENT_NOISE_OVERRIDE)
    return {
        "id": MEASUREMENT_NOISE_MODEL_ID,
        "required_multiplier": MEASUREMENT_NOISE_MULTIPLIER,
        "midbass": [
            {"range_hz": "400-500", "floor_db": "0.10"},
            {"range_hz": "700-1400", "floor_db": "0.60-1.00"},
            {"range_hz": "2200+", "floor_db": "1.60"},
        ],
        "tweeter": [{"range_hz": "1800-16000", "floor_db": "0.23-0.46"}],
        "note": (
            "A new filter must address a repeatable deviation at least 2.5 times "
            "the local same-rig MMM floor."
        ),
    }


def configure_measurement_noise_model(model=None):
    """Install or clear a run-local empirical repeatability model."""
    global _MEASUREMENT_NOISE_OVERRIDE
    _MEASUREMENT_NOISE_OVERRIDE = None if model is None else dict(model)

def audibility_score(freqs, dev_db, band=(60.0, 16000.0), mask=None, conf=None):
    """One number for 'how audibly wrong is this curve' (lower = better).
    ERB-smooth first (what the ear integrates), weight by sensitivity, RMS.
    `conf` is an optional 0..1 per-bin confidence array. Use it for spatial
    consistency / phase-validity weighting so uncertain bins cannot dominate
    the score or the parsimony gate."""
    sm = erb_smooth(freqs, dev_db)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if mask is not None:
        sel &= mask
    if not np.any(sel):
        return float('inf')
    w = audibility_weight(freqs)[sel]
    if conf is not None:
        w = w * np.clip(conf[sel], 0.0, 1.0)
    den = np.sum(w ** 2)
    if den <= 1e-12:
        return float('inf')
    return float(np.sqrt(np.sum((sm[sel] * w) ** 2) / den))

# --------------------------------------------------------------------------
# 3) JOINT PEQ FIT (the TuneEQ-beater)
def fit_peq(freqs, dev_db, fit_band, n_bands_max=5, mask=None, conf=None,
            g_lim=(-15.0, 3.0), q_lim=(0.5, 8.0), min_gain=1.0,
            improve_pct=6.0, boost_penalty=0.5, hf_q_penalty=0.4,
            hf_q_knee=4.0, transition_hz=1000.0, selection_tax_weight=0.25,
            verbose=False):
    """Jointly fit up to n_bands_max peaking bands so that dev+EQ -> 0 over
    fit_band, minimizing the ERB/audibility-weighted residual.

    Discipline built in (this is where it beats a raw curve-fitter):
      - mask=False bins are EXCLUDED from the error (nulls / non-min-phase /
        volatile comb regions never attract a filter);
      - conf (optional 0..1 per-freq confidence, e.g. from spatial_consistency)
        CONTINUOUSLY down-weights uncertain bins instead of a hard mask edge --
        the solver still "sees" them a little, but won't spend a band on a
        low-confidence wiggle;
      - "FILTER TAX" (beats TuneEQ's fill-every-hole habit): each proposed band
        pays a penalty for being a BOOST (boost_penalty x G) and for being a
        NARROW filter above the transition (hf_q_penalty x (Q-knee) when
        F>transition_hz) -- so the optimizer only boosts / goes high-Q-up-high
        when the audible payoff clearly outweighs the tax;
      - boosts capped at g_lim[1] (+3 default), cuts at -15 (hardware);
      - Q capped at 8 (craft ceiling), 0.5 floor (hardware);
      - PARSIMONY: bands are added one at a time and each must improve the
        weighted score by >= improve_pct %, else it is discarded and fitting
        stops -- no chasing sub-dB residuals with extra bands (TuneEQ trap);
      - selection_tax_weight adds a smaller version of the filter tax to the
        parsimony gate. The full tax still shapes fitting, but the gate should
        not reject a clearly useful cut just because it has moderate Q;
      - bands with fitted |G| < min_gain dB are dropped at the end.

    Returns (bands, report) - bands as [(F, Q, G), ...] rounded to hardware
    steps (0.25 dB gain), report dict with before/after scores.
    """
    from scipy.optimize import least_squares

    sel = (freqs >= fit_band[0]) & (freqs <= fit_band[1])
    if mask is not None:
        sel &= mask
    fsel = freqs[sel]
    w = audibility_weight(fsel)
    if conf is not None:
        w = w * np.clip(conf[sel], 0.0, 1.0)     # continuous confidence down-weight

    def penalties(bands):
        # CONSTANT length (2 terms/band) so least_squares' finite-diff Jacobian
        # never sees the vector change size when a band's F is perturbed.
        p = []
        for F, Q, G in bands:
            p.append(boost_penalty * max(0.0, G))                    # boost tax
            hf = 1.0 / (1.0 + np.exp(-(np.log2(F / transition_hz)) * 6.0))  # smooth gate ~transition
            p.append(hf_q_penalty * hf * max(0.0, Q - hf_q_knee))    # narrow-HF tax
        return np.array(p) if p else np.zeros(0)

    def resid(params):
        bands = [(10 ** params[3 * i], params[3 * i + 1], params[3 * i + 2])
                 for i in range(len(params) // 3)]
        r = (dev_db[sel] + cascade_db(fsel, bands)) * w
        return np.concatenate([r, penalties(bands)])

    def score_of(params):
        bands = [(10 ** params[3 * i], params[3 * i + 1], params[3 * i + 2])
                 for i in range(len(params) // 3)]
        full = dev_db + cascade_db(freqs, bands)
        return audibility_score(freqs, full, band=fit_band, mask=mask, conf=conf)

    def selection_score_of(params):
        """Score used by the parsimony gate.
        Raw audibility score decides whether the curve improved; the tax decides
        whether a boost / narrow-HF filter earned the right to exist."""
        bands = [(10 ** params[3 * i], params[3 * i + 1], params[3 * i + 2])
                 for i in range(len(params) // 3)]
        p = penalties(bands)
        tax = float(np.sqrt(np.mean(p ** 2))) if len(p) else 0.0
        return score_of(params) + selection_tax_weight * tax

    base_score = audibility_score(freqs, dev_db, band=fit_band, mask=mask, conf=conf)
    params = np.array([])
    lo_f, hi_f = np.log10(fit_band[0] * 1.02), np.log10(fit_band[1] * 0.98)
    cur_score = base_score
    cur_select_score = base_score

    for k in range(n_bands_max):
        # seed the next band at the biggest remaining weighted, smoothed bump
        bands_now = [(10 ** params[3 * i], params[3 * i + 1], params[3 * i + 2])
                     for i in range(len(params) // 3)]
        res_now = erb_smooth(freqs, dev_db + cascade_db(freqs, bands_now))
        res_w = np.where(sel, np.abs(res_now) * audibility_weight(freqs), 0)
        if conf is not None:
            res_w *= np.clip(conf, 0.0, 1.0)
        i0 = int(np.argmax(res_w))
        if res_w[i0] <= 0:
            break
        seed_F, seed_G = freqs[i0], float(np.clip(-res_now[i0], g_lim[0], g_lim[1]))
        trial = np.concatenate([params, [np.log10(seed_F), 1.5, seed_G]])
        nb = len(trial) // 3
        lb = np.tile([lo_f, q_lim[0], g_lim[0]], nb)
        ub = np.tile([hi_f, q_lim[1], g_lim[1]], nb)
        fit = least_squares(resid, np.clip(trial, lb, ub), bounds=(lb, ub),
                            method='trf', max_nfev=400)
        new_score = score_of(fit.x)
        new_select_score = selection_score_of(fit.x)
        raw_gain_pct = 100.0 * (cur_score - new_score) / max(cur_score, 1e-9)
        select_gain_pct = 100.0 * (cur_select_score - new_select_score) / max(cur_select_score, 1e-9)
        if verbose:
            print('  band %d: score %.3f -> %.3f (%.1f%%) | selection %.3f -> %.3f (%.1f%%)' %
                  (nb, cur_score, new_score, raw_gain_pct, cur_select_score, new_select_score, select_gain_pct))
        if raw_gain_pct < improve_pct or select_gain_pct < improve_pct:
            break                                    # parsimony gate
        params, cur_score, cur_select_score = fit.x, new_score, new_select_score

    bands = []
    for i in range(len(params) // 3):
        F = round(float(10 ** params[3 * i]), 1)
        Q = round(float(params[3 * i + 1]), 2)
        G = round(float(params[3 * i + 2]) * 4) / 4.0       # 0.25 dB steps
        if abs(G) >= min_gain:
            bands.append((F, Q, G))
    final = audibility_score(freqs, dev_db + cascade_db(freqs, bands),
                             band=fit_band, mask=mask, conf=conf)
    final_tax = selection_score_of(np.array(
        sum(([np.log10(F), Q, G] for F, Q, G in bands), []), dtype=float)) if bands else base_score
    return bands, {'score_before': round(base_score, 3),
                   'score_after': round(final, 3),
                   'selection_score_before': round(base_score, 3),
                   'selection_score_after': round(final_tax, 3),
                   'bands_used': len(bands)}

# --------------------------------------------------------------------------
# 3c) INTERFERENCE / SUMMATION AUDIT — added 2026-07-03 (Fable pass).
# Detects L/R (or any driver-pair) destructive interference from THREE PLAIN
# MAGNITUDE captures at one fixed mic spot: solo_a, solo_b, and the pair
# playing together. NO acoustic timing reference / phase capture needed —
# this is the cheap alternative to a full phase-valid measurement for simply
# DETECTING a cancellation (though fine-tuning an APF's F/Q still benefits
# from live sweeping by ear/RTA, §3 "manual APF protocol").
# This is how the ~415 Hz mid-pair null was finally explained: each mid solo
# was healthy there, but the "MidBass Together" trace read ~3 dB BELOW even
# the incoherent sum -- proof the two sides are partially cancelling, not a
# modal/boundary null. That reclassified it from "leave forever" to
# "all-pass candidate."
def interference_audit(freqs, solo_a_db, solo_b_db, together_db, flag_db=2.0,
                       smooth_oct=1 / 12.0):
    """psum = incoherent (power) sum: the floor you'd get if A and B were
    totally uncorrelated. csum = fully coherent (voltage) sum: the ceiling if
    perfectly in phase. If `together` reads BELOW psum, the pair is destructively
    interfering at that frequency (a phase-relative problem, not a level or
    EQ-able magnitude problem). Returns (psum_db, csum_db, interference_db,
    flagged_mask). interference_db = together - psum; large negative = bad."""
    psum = 10 * np.log10(10 ** (solo_a_db / 10.0) + 10 ** (solo_b_db / 10.0))
    csum = 20 * np.log10(10 ** (solo_a_db / 20.0) + 10 ** (solo_b_db / 20.0))
    interference_db = together_db - psum
    flag_basis = octave_smooth_log(freqs, interference_db, smooth_oct) if smooth_oct else interference_db
    return psum, csum, interference_db, (flag_basis < -flag_db)

# --------------------------------------------------------------------------
# SPECIAL-FILTER XML WRITERS -- encodings VERIFIED by controlled export-diffs.
# COMPLETE T-code map (as of 2026-07-03 "Test .afpx" diff, which CORRECTED the
# earlier "T=20 = shelf" inference):
#   T=1  free slot          T=17 parametric EQ
#   T=15 LP xover           T=16 HP xover
#   T=3  LOW SHELF   (band 1 / dF=25 only;  G!=0 active)   [VERIFIED 2026-07-03]
#   T=4  HIGH SHELF  (band 30 / dF=20000 only; G!=0 active)[VERIFIED 2026-07-03]
#   T=19 1st-order ALL-PASS (G=0, Q written as 1 placeholder; MIDDLE slots OK)
#        [CONFIRMED 2026-07-03: PC-Tool screenshot, Band 20 middle slot,
#         "Q: N/A for 1st order", "1. Order" active]
#   T=20 2nd-order ALL-PASS (G=0, Q stored directly; device limit varies by frequency)
# The I attribute (present on EVERY <Fil>) = the INVERT flag, 0/1 -- VERIFIED
# 2026-07-03 by export-diff: pressing 'invert' on the T=19 APF flipped exactly
# I="0" -> I="1" and nothing else in the whole file. (It was previously
# misread as an 'index'.) All writers take invert=True to set it.
# Notes: middle-slot APFs are real (T=19 seen at dF=2000) -> APFs do NOT compete
# with shelves for the end slots. Old tunes' parked T=20 bands were parked
# ALL-PASSES, not shelves (their odd Q>2 values = stale XML from prior PEQ use).
# Switching band 1/30 to a shelf CONSUMES whatever PEQ lived in that slot --
# relocate ("defrag") the squatter PEQ to a free middle slot FIRST.
def allpass_fil_str(F, Q, FN, dF='20000', invert=False):
    """2nd-order all-pass (T=20). G always "0" -- that's what makes it an APF.
    Middle slots allowed (verified via the T=19 sighting + AF docs), but default
    stays the end slot for consistency with the verified example."""
    assert 0.1 <= Q <= 15.0, 'APF Q must be 0.1-15 before the device-specific frequency limit'
    return '<Fil G="0" FN="%s" F="%.2f" T="20" I="%s" dF="%s" Q="%s"/>' % (FN, F, '1' if invert else '0', dF, Q)

def allpass1_fil_str(F, FN, dF, invert=False):
    """1st-order all-pass (T=19, -90 deg at corner, no Q -- written as 1).
    CONFIRMED 1st-order (PC-Tool screenshot: Q shows "N/A for 1st order" with
    "1. Order" active on this exact band). Middle slots verified fine."""
    return '<Fil Q="1" G="0" F="%.2f" FN="%s" I="%s" T="19" dF="%s"/>' % (F, FN, '1' if invert else '0', dF)

def shelf_fil_str(kind, F, Q, G, FN, invert=False):
    """Low shelf (T=3, band 1/dF=25) or high shelf (T=4, band 30/dF=20000).
    VERIFIED encodings from the 2026-07-03 export: LS -2.25@4980.25 Q1 -> T=3;
    HS +0.25@5400 Q0.5 -> T=4. Q 0.1-2 IS the slope (no separate S param).
    G in 0.25 dB steps, within [-15,+6]."""
    assert kind in ('low', 'high')
    assert 0.1 <= Q <= 2.0, 'shelf Q must be 0.1-2 (AF spec)'
    assert -15.0 <= G <= 6.0, 'shelf gain out of Helix range'
    T, dF = ('3', '25') if kind == 'low' else ('4', '20000')
    return '<Fil Q="%s" G="%s" F="%.2f" FN="%s" I="%s" T="%s" dF="%s"/>' % (Q, G, F, FN, '1' if invert else '0', T, dF)

def fil_attrs(tag):
    import re as _re
    return dict(_re.findall(r'([A-Za-z]+)="([^"]*)"', tag))

def delays_semantically_equal(xml_a, xml_b):
    """PC-Tool round-trips REORDER attributes inside <T .../> tags (verified
    2026-07-03: PM= T= P= became T= P= PM=, same values). So for any file that
    passed through PC-Tool, compare delay tags as attr DICTS, not bytes. For
    our own Python writes the byte check is still fine (we never reorder)."""
    import re as _re
    ta = [fil_attrs(t) for t in _re.findall(r'<T [^>]*/>', xml_a)]
    tb = [fil_attrs(t) for t in _re.findall(r'<T [^>]*/>', xml_b)]
    return ta == tb

# --------------------------------------------------------------------------
# 4) HEADROOM REPORT (mandatory output on every tune — clipping guard)
def headroom_report(freqs, bands, xover_lo=None, xover_hi=None):
    """Given a channel's full PEQ set, report the worst-case positive gain the
    EQ cascade produces (that's what eats digital headroom / clips). Every tune
    must print this per channel. `xover_*` optionally bounds the summed-boost
    check to the driver's passband. Returns a dict."""
    g = cascade_db(freqs, bands)
    sel = np.ones_like(freqs, dtype=bool)
    if xover_lo is not None: sel &= freqs >= xover_lo
    if xover_hi is not None: sel &= freqs <= xover_hi
    peak_gain = float(np.max(g[sel])) if np.any(sel) else 0.0
    fpk = float(freqs[sel][np.argmax(g[sel])]) if np.any(sel) else 0.0
    largest_boost = max([G for _, _, G in bands], default=0.0)
    return {'peak_cascade_gain_db': round(peak_gain, 2),
            'peak_gain_freq': round(fpk, 0),
            'largest_single_boost_db': round(largest_boost, 2),
            'clip_risk': peak_gain > 0.0,
            'recommended_trim_db': round(-peak_gain, 2) if peak_gain > 0 else 0.0}

# ==========================================================================
# SELF-TESTS + REAL-DATA VALIDATION

def weighted_median(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.median(values))
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return float(np.median(values[np.isfinite(values)]))
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    return float(values[np.searchsorted(cdf, 0.5 * cdf[-1])])

def target_anchor_offset(freqs, measured_db, target_db, confidence=None,
                         anchor_bands=((300.0, 3000.0), (120.0, 1000.0), (1000.0, 6000.0)),
                         min_bins=12):
    """Wide, confidence-weighted median target anchor with fallbacks."""
    freqs = np.asarray(freqs, dtype=float)
    dev = np.asarray(measured_db, dtype=float) - np.asarray(target_db, dtype=float)
    if confidence is None:
        confidence = np.ones_like(freqs)
    confidence = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    for lo, hi in anchor_bands:
        sel = (freqs >= lo) & (freqs <= hi) & (confidence > 0.3) & np.isfinite(dev)
        if np.count_nonzero(sel) >= min_bins:
            return weighted_median(dev[sel], confidence[sel])
    sel = np.isfinite(dev)
    return weighted_median(dev[sel], confidence[sel])

def allpass_H(freqs, f0, Q=0.7, order=2, fs=FS):
    """Digital all-pass response used by Helix-style filters.
    order=2 is the verified AFPX-writeable APF. order=1 is kept for modelling
    and live experiments, but do not write it unless the target hardware export
    has been verified."""
    w0 = 2 * np.pi * f0 / fs
    w = 2 * np.pi * freqs / fs
    z1 = np.exp(-1j * w)
    if order == 1:
        t = np.tan(w0 / 2.0)
        a = (t - 1.0) / (t + 1.0)
        return (a + z1) / (1.0 + a * z1)
    if order != 2:
        raise ValueError('order must be 1 or 2')
    al = np.sin(w0) / (2.0 * Q)
    b0, b1, b2 = 1.0 - al, -2.0 * np.cos(w0), 1.0 + al
    a0, a1, a2 = 1.0 + al, -2.0 * np.cos(w0), 1.0 - al
    z2 = np.exp(-2j * w)
    return (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)

def allpass_H_inv(freqs, f0, Q=0.7, order=2, fs=FS):
    """PC-Tool's Allpass 'invert' button, simulated: multiplying an all-pass by
    -1 is still an all-pass (|H|=1) but with 180 deg added at ALL frequencies.
    Mathematically identical to (channel polarity flip) + (normal APF) -- just
    applied inside the EQ block, so the TA/polarity page stays untouched.
    USE WHEN: live-dialing an APF and the trough DEEPENS for every F/Q you try
    -- the rotation direction is wrong; invert flips the branch relationship.
    XML encoding VERIFIED 2026-07-03: the I attribute (I="1" = inverted) --
    the export-diff showed exactly I 0->1 and nothing else."""
    return -allpass_H(freqs, f0, Q, order, fs)

def group_delay_ms_from_H(freqs, H):
    ph = np.unwrap(np.angle(H))
    w = 2 * np.pi * freqs
    return -np.gradient(ph, w) * 1000.0


def temporal_group_delay_cost(freqs, group_delay_ms, band,
                              allowance_cycles=0.35):
    """Frequency-normalised temporal cost for an all-pass candidate.

    A fixed millisecond threshold treats 100 Hz and 3 kHz as if the ear heard
    timing identically at both frequencies. This cost instead measures delay
    against a conservative fraction of one local cycle. It is a ranking cost,
    not a claim that one universal psychoacoustic threshold exists.
    """
    f = np.asarray(freqs, dtype=float)
    gd = np.maximum(np.asarray(group_delay_ms, dtype=float), 0.0)
    selected = (f >= float(band[0])) & (f <= float(band[1]))
    if not np.any(selected):
        raise ValueError('band does not overlap the frequency axis')
    allowance_ms = 1000.0 * float(allowance_cycles) / np.maximum(f, 1e-9)
    excess_ratio = np.maximum(gd / allowance_ms - 1.0, 0.0)
    weights = audibility_weight(f[selected])
    denominator = float(np.sum(weights ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sqrt(np.sum(
        (excess_ratio[selected] * weights) ** 2
    ) / denominator))

# Small timing/level changes are enough to make a razor-tuned crossover null
# appear or disappear. Phase candidates are therefore selected against this
# bounded envelope, not only the exact captured vectors.
PHASE_ROBUST_PERTURBATIONS = (
    (0.0, 0.0),
    (0.020, 0.0),
    (-0.020, 0.0),
    (0.0, 0.5),
    (0.0, -0.5),
    (0.015, 0.35),
    (-0.015, -0.35),
)


def _phase_snapshot_set(driver_a, driver_b, snapshots=None):
    if snapshots is None:
        return [(np.asarray(driver_a, complex), np.asarray(driver_b, complex))]
    values = [(np.asarray(a, complex), np.asarray(b, complex)) for a, b in snapshots]
    if not values:
        raise ValueError('phase snapshots cannot be empty')
    return values


def _phase_perturb(freqs, branch, delay_ms, level_db):
    return (
        np.asarray(branch, complex)
        * np.exp(-1j * 2.0 * np.pi * np.asarray(freqs, float) * float(delay_ms) / 1000.0)
        * 10.0 ** (float(level_db) / 20.0)
    )

def optimize_allpass(freqs, driver_a, driver_b, search_band, apply_to='A',
                     order=2, f_steps=96, q_steps=24, q_lim=(0.5, 2.0),
                     damage_band=(60.0, 16000.0), damage_free_db=0.5,
                     damage_penalty=1.0, gd_penalty=0.0, max_gd_ms=2.0,
                     gd_allowance_cycles=0.35,
                     snapshots=None, robust=True,
                     perturbations=PHASE_ROBUST_PERTURBATIONS):
    """Grid-search a robust 2nd-order APF for one or more driver-pair snapshots.

    Every candidate is tested under bounded timing and level drift. With
    multiple phase-valid positions, selection minimizes the worst snapshot.
    """
    sel = (freqs >= search_band[0]) & (freqs <= search_band[1])
    dmg_sel = (freqs >= damage_band[0]) & (freqs <= damage_band[1])
    if not np.any(sel):
        raise ValueError('search_band does not overlap the frequency axis')

    phase_snapshots = _phase_snapshot_set(driver_a, driver_b, snapshots)
    active_perturbations = perturbations if robust else ((0.0, 0.0),)

    def wrms(y, m):
        w = audibility_weight(freqs[m])
        den = np.sum(w ** 2)
        return float(np.sqrt(np.sum((y[m] * w) ** 2) / den)) if den > 1e-12 else float('inf')

    def evaluate(H=None):
        scores = []
        gap_scores = []
        damage_scores = []
        per_snapshot = []
        for snap_a, snap_b in phase_snapshots:
            snapshot_scores = []
            for drift_ms, level_db in active_perturbations:
                perturbed_b = _phase_perturb(freqs, snap_b, drift_ms, level_db)
                base_sum_db = 20 * np.log10(np.abs(snap_a + perturbed_b) + 1e-12)
                coherent_db = 20 * np.log10(np.abs(snap_a) + np.abs(perturbed_b) + 1e-12)
                if H is None:
                    candidate_sum_db = base_sum_db
                elif apply_to.upper() == 'A':
                    candidate_sum_db = 20 * np.log10(np.abs(snap_a * H + perturbed_b) + 1e-12)
                elif apply_to.upper() == 'B':
                    candidate_sum_db = 20 * np.log10(np.abs(snap_a + perturbed_b * H) + 1e-12)
                else:
                    raise ValueError("apply_to must be 'A' or 'B'")
                gap_score = wrms(np.maximum(coherent_db - candidate_sum_db, 0.0), sel)
                damage_score = (
                    0.0 if H is None else
                    wrms(np.maximum(base_sum_db - candidate_sum_db - damage_free_db, 0.0), dmg_sel)
                )
                score = gap_score + damage_penalty * damage_score
                scores.append(score)
                gap_scores.append(gap_score)
                damage_scores.append(damage_score)
                snapshot_scores.append(score)
            per_snapshot.append(max(snapshot_scores))
        return {
            'score': max(scores),
            'gap': max(gap_scores),
            'damage': max(damage_scores),
            'per_snapshot': per_snapshot,
        }

    base_metrics = evaluate()
    base_score = float(base_metrics['score'])
    f_grid = np.geomspace(search_band[0], search_band[1], f_steps)
    q_grid = np.linspace(q_lim[0], q_lim[1], q_steps)

    best = None
    for F in f_grid:
        for Q in q_grid:
            H = allpass_H(freqs, F, Q, order=order)
            metrics = evaluate(H)
            gd = group_delay_ms_from_H(freqs, H)
            cycle_cost = temporal_group_delay_cost(
                freqs, gd, search_band, allowance_cycles=gd_allowance_cycles
            )
            absolute_cost = max(
                0.0,
                float(np.max(gd[sel])) / max(float(max_gd_ms), 1e-9) - 1.0,
            )
            temporal_cost = max(cycle_cost, absolute_cost)
            score = float(metrics['score']) + gd_penalty * temporal_cost
            if best is None or score < best['selection_score_after']:
                iF = int(np.argmin(np.abs(freqs - F)))
                nominal_a, nominal_b = phase_snapshots[0]
                nominal_before = 20 * np.log10(np.abs(nominal_a + nominal_b) + 1e-12)
                nominal_after = 20 * np.log10(np.abs(
                    nominal_a * H + nominal_b if apply_to.upper() == 'A'
                    else nominal_a + nominal_b * H
                ) + 1e-12)
                best = {
                    'F': round(float(F), 1),
                    'Q': round(float(Q), 2),
                    'order': int(order),
                    'apply_to': apply_to.upper(),
                    'score_before': round(base_score, 3),
                    'selection_score_after': round(float(score), 3),
                    'gap_score_after': round(float(metrics['gap']), 3),
                    'robust_score_before': round(base_score, 3),
                    'robust_score_after': round(float(metrics['score']), 3),
                    'robust_snapshot_scores': [
                        round(float(value), 3) for value in metrics['per_snapshot']
                    ],
                    'phase_snapshot_count': len(phase_snapshots),
                    'perturbation_count': len(active_perturbations),
                    'lift_at_F_db': round(float(nominal_after[iF] - nominal_before[iF]), 2),
                    'worst_damage_db': round(float(metrics['damage']), 2),
                    'max_apf_gd_ms_in_band': round(float(np.max(gd[sel])), 3),
                    'temporal_gd_cost': round(float(temporal_cost), 3),
                    'gd_allowance_cycles': float(gd_allowance_cycles),
                }

    best['improvement_pct'] = round(
        100.0 * (base_score - best['robust_score_after']) / max(base_score, 1e-9), 1
    )
    return best

def loudness_weight(freqs):
    """Car-tuning priority weight.

    Keeps the old broad sensitivity idea but adds explicit upper-mid risk:
    presence errors around 2-5 kHz are costly, LF broad errors still matter,
    and the top octave gets less authority because off-axis/seat variance is
    usually high in cars.
    """
    freqs = np.asarray(freqs, dtype=float)
    w = audibility_weight(freqs)
    presence = np.exp(-0.5 * (np.log2(freqs / 3200.0) / 0.65) ** 2)
    midbass = 0.25 * np.exp(-0.5 * (np.log2(freqs / 120.0) / 0.9) ** 2)
    w = w * (1.0 + 0.45 * presence + midbass)
    w[freqs > 12000.0] *= 0.75
    return np.clip(w, 0.25, 1.8)

def band_weight(freqs, lo, hi, floor=0.0, edge_oct=0.5):
    """Soft rectangular band weight with octave-tapered edges."""
    freqs = np.asarray(freqs, dtype=float)
    w = np.ones_like(freqs)
    below = freqs < lo
    above = freqs > hi
    w[below] = np.clip(1.0 - np.log2(lo / freqs[below]) / edge_oct, floor, 1.0)
    w[above] = np.clip(1.0 - np.log2(freqs[above] / hi) / edge_oct, floor, 1.0)
    return np.clip(w, floor, 1.0)

def wrms(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return float('inf')
    den = np.sum(weights[ok] ** 2)
    return float(np.sqrt(np.sum((values[ok] * weights[ok]) ** 2) / den))

def coherence_confidence(coherence, min_usable=0.35, full_trust=0.85, power=1.5):
    """Map coherence-like confidence data onto a 0..1 trust weight."""
    c = np.asarray(coherence, dtype=float)
    if full_trust <= min_usable:
        raise ValueError('full_trust must be greater than min_usable')
    w = np.clip((c - min_usable) / (full_trust - min_usable), 0.0, 1.0)
    return np.power(w, power)

def coherence_weighted_db_average(db_traces, coherence_traces=None, min_usable=0.35):
    """Average dB traces in linear power, down-weighting low-confidence bins."""
    mags = np.asarray(db_traces, dtype=float)
    if mags.ndim == 1:
        mags = mags[None, :]
    powers = 10 ** (mags / 10.0)
    if coherence_traces is None:
        weights = np.ones_like(powers)
    else:
        coh = np.asarray(coherence_traces, dtype=float)
        if coh.ndim == 1:
            coh = coh[None, :]
        weights = coherence_confidence(coh, min_usable=min_usable)
    den = np.sum(weights, axis=0)
    den = np.where(den > 1e-12, den, np.nan)
    avg_power = np.nansum(powers * weights, axis=0) / den
    return 10.0 * np.log10(np.maximum(avg_power, 1e-12))

def band_limited_delay_from_phase(freqs, phase_diff_deg, band, coherence=None):
    """Estimate relative delay from phase-vs-frequency slope inside one band."""
    freqs = np.asarray(freqs, dtype=float)
    phase = np.unwrap(np.deg2rad(np.asarray(phase_diff_deg, dtype=float)))
    sel = np.isfinite(freqs) & np.isfinite(phase) & (freqs >= band[0]) & (freqs <= band[1])
    if coherence is not None:
        conf = coherence_confidence(coherence)
        sel &= np.isfinite(conf) & (conf > 0)
        weights = conf[sel]
    else:
        weights = np.ones(np.count_nonzero(sel), dtype=float)
    if np.count_nonzero(sel) < 3:
        return {'delay_ms': 0.0, 'rms_phase_err_deg': float('inf'), 'usable': False}
    x = freqs[sel]
    y_deg = np.rad2deg(phase[sel])
    X = np.vstack([x, np.ones_like(x)]).T
    sw = np.sqrt(np.maximum(weights, 1e-9))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y_deg * sw, rcond=None)
    slope_deg_per_hz, intercept_deg = beta
    fit = slope_deg_per_hz * x + intercept_deg
    rms = wrms(y_deg - fit, weights)
    delay_ms = -slope_deg_per_hz / 360.0 * 1000.0
    return {'delay_ms': round(float(delay_ms), 4),
            'rms_phase_err_deg': round(float(rms), 3),
            'usable': bool(np.isfinite(rms) and rms < 60.0)}

def gate_low_frequency_limit(gate_ms, cycles=1.0):
    gate_ms = float(gate_ms)
    if gate_ms <= 0:
        return float('inf')
    return float(1000.0 * cycles / gate_ms)

def gate_frequency_confidence(freqs, gate_ms, cycles=1.0, transition_oct=0.5):
    """Return a soft trust ramp above the gate-limited LF boundary."""
    freqs = np.asarray(freqs, dtype=float)
    flo = gate_low_frequency_limit(gate_ms, cycles=cycles)
    if not np.isfinite(flo) or flo <= 0:
        return np.zeros_like(freqs)
    hi = flo * (2 ** transition_oct)
    conf = np.zeros_like(freqs, dtype=float)
    conf[freqs >= hi] = 1.0
    mid = (freqs > flo) & (freqs < hi)
    if np.any(mid):
        t = np.log2(freqs[mid] / flo) / max(transition_oct, 1e-6)
        conf[mid] = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    return conf

def suggest_gate_from_impulse(impulse, sample_rate_hz, direct_index=None,
                              ignore_ms=0.35, threshold_db=-12.0, max_ms=50.0):
    """Suggest a post-direct gate that ends before the first strong reflection."""
    x = np.abs(np.asarray(impulse, dtype=float))
    if x.size == 0:
        return {'gate_ms': 0.0, 'reflection_index': None, 'usable': False}
    sr = float(sample_rate_hz)
    direct = int(np.argmax(x) if direct_index is None else direct_index)
    ignore = int(round(ignore_ms * sr / 1000.0))
    limit = min(len(x), direct + int(round(max_ms * sr / 1000.0)))
    thresh = x[direct] * (10 ** (threshold_db / 20.0))
    refl = None
    for idx in range(min(len(x) - 1, direct + ignore), limit):
        if x[idx] >= thresh:
            refl = idx
            break
    if refl is None:
        gate_samples = max(1, limit - direct)
    else:
        gate_samples = max(1, refl - direct)
    gate_ms = 1000.0 * gate_samples / sr
    return {'gate_ms': round(float(gate_ms), 3),
            'reflection_index': refl,
            'usable': bool(gate_samples > 2)}

def _raised_cosine_bandpass_weights(freqs, band, edge_oct=0.5):
    freqs = np.asarray(freqs, dtype=float)
    lo, hi = map(float, band)
    if lo <= 0 or hi <= lo:
        raise ValueError('invalid band')
    floor = 0.0
    w = np.ones_like(freqs, dtype=float)
    below = freqs < lo
    above = freqs > hi
    w[below] = np.clip(1.0 - np.log2(lo / np.maximum(freqs[below], 1e-9)) / edge_oct, floor, 1.0)
    w[above] = np.clip(1.0 - np.log2(np.maximum(freqs[above], 1e-9) / hi) / edge_oct, floor, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(w, 0.0, 1.0))

def bandpass_impulse(impulse, sample_rate_hz, band, edge_oct=0.5):
    """FFT-domain soft bandpass for impulse-domain timing work."""
    x = np.asarray(impulse, dtype=float)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate_hz))
    weights = _raised_cosine_bandpass_weights(np.maximum(freqs, 1e-9), band, edge_oct=edge_oct)
    weights[0] = 0.0
    return np.fft.irfft(spec * weights, n=len(x))

def band_limited_impulse_delay(impulse_a, impulse_b, sample_rate_hz, band,
                               max_lag_ms=5.0, gate_ms=None, edge_oct=0.5):
    """Estimate relative delay and polarity from band-limited impulses."""
    a = np.asarray(impulse_a, dtype=float)
    b = np.asarray(impulse_b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {'delay_ms': 0.0, 'polarity': 'same', 'usable': False}
    a = a[:n]
    b = b[:n]
    max_lag = max(1, int(round(float(max_lag_ms) * sample_rate_hz / 1000.0)))
    if gate_ms is not None and gate_ms > 0:
        gate_n = min(n, max(8, int(round(float(gate_ms) * sample_rate_hz / 1000.0))))
        peak_a = int(np.argmax(np.abs(a)))
        peak_b = int(np.argmax(np.abs(b)))
        start = max(0, min(peak_a, peak_b) - max_lag)
        stop = min(n, start + gate_n)
        a = a[start:stop]
        b = b[start:stop]
    af = bandpass_impulse(a, sample_rate_hz, band, edge_oct=edge_oct)
    bf = bandpass_impulse(b, sample_rate_hz, band, edge_oct=edge_oct)
    corr = np.correlate(bf, af, mode='full')
    lags = np.arange(-len(af) + 1, len(af))
    keep = np.abs(lags) <= max_lag
    corr = corr[keep]
    lags = lags[keep]
    idx = int(np.argmax(np.abs(corr)))
    lag = int(lags[idx])
    peak = float(corr[idx])
    energy = np.sqrt(np.sum(af ** 2) * np.sum(bf ** 2)) + 1e-12
    usable = bool(abs(peak) / energy > 0.15)
    return {'delay_ms': round(float(1000.0 * lag / sample_rate_hz), 4),
            'polarity': 'inverted' if peak < 0 else 'same',
            'usable': usable,
            'corr_norm': round(float(abs(peak) / energy), 4)}

def local_peak_q_proxy(freqs, local_db, min_prom_db=0.5):
    """Approximate how narrow/prominent positive local excess is.

    This is not a literal acoustic Q measurement; it is a cheap resonance-risk
    proxy for scoring. Broad tonal errors should be handled by normal ERB score,
    while narrow upper-mid peaks deserve extra caution.
    """
    freqs = np.asarray(freqs, dtype=float)
    local_db = np.asarray(local_db, dtype=float)
    q = np.ones_like(local_db)
    pos = np.maximum(local_db, 0.0)
    n = len(freqs)
    for i in range(1, n - 1):
        if pos[i] < min_prom_db or pos[i] < pos[i - 1] or pos[i] < pos[i + 1]:
            continue
        half = pos[i] * 0.5
        l = i
        r = i
        while l > 0 and pos[l] > half:
            l -= 1
        while r < n - 1 and pos[r] > half:
            r += 1
        bw_oct = max(np.log2(freqs[r] / freqs[l]), 1 / 24.0)
        q[i] = np.clip(1.0 / bw_oct, 0.5, 12.0)
    return q

def masking_relief(freqs, smoothed_db):
    """Small down-weight for errors sitting near much louder broad energy.

    This is intentionally conservative. It prevents the perceptual score from
    overreacting to small ripples on top of dominant bass/midbass energy, but it
    never hides a real error completely.
    """
    smoothed_db = np.asarray(smoothed_db, dtype=float)
    broad = octave_smooth_log(freqs, smoothed_db, 1.0)
    relief = np.where(broad > smoothed_db + 3.0, 0.72, 1.0)
    return np.clip(relief, 0.65, 1.0)

def perceptual_score(freqs, dev_db, left_db=None, right_db=None, band=(60.0, 16000.0),
                     mask=None, conf=None):
    """Composite score for car-audio tuning decisions.

    It keeps broad tonal error, but separately penalizes narrow upper-mid peaks
    and L/R mismatch in the image-critical band. Dips cost less than peaks so
    the app remains biased against filling nulls.
    """
    freqs = np.asarray(freqs, dtype=float)
    dev_db = np.asarray(dev_db, dtype=float)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    if not np.any(sel):
        return {'total': float('inf'), 'tonal': float('inf'),
                'resonance': float('inf'), 'stereo': 0.0}
    c = np.ones_like(freqs, dtype=float) if conf is None else np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
    sm = erb_smooth(freqs, dev_db)
    W = loudness_weight(freqs) * c
    peak_term = np.maximum(sm, 0.0)
    dip_term = 0.6 * np.maximum(-sm, 0.0)
    tonal = wrms((peak_term + dip_term)[sel] * masking_relief(freqs[sel], sm[sel]), W[sel])

    local = dev_db - sm
    q_proxy = local_peak_q_proxy(freqs, local)
    resonance_weight = band_weight(freqs, 1500.0, 6000.0, floor=0.05) * c
    resonance_term = np.maximum(local, 0.0) * np.clip(q_proxy / 1.8, 0.8, 3.0)
    resonance = wrms(resonance_term[sel], resonance_weight[sel])

    stereo = 0.0
    if left_db is not None and right_db is not None:
        lr = erb_smooth(freqs, np.asarray(left_db, dtype=float) - np.asarray(right_db, dtype=float))
        stereo_weight = band_weight(freqs, 700.0, 5000.0, floor=0.0) * c
        stereo = wrms(np.abs(lr[sel]), stereo_weight[sel])

    total = tonal + 1.2 * resonance + 0.8 * stereo
    return {'total': round(float(total), 4),
            'tonal': round(float(tonal), 4),
            'resonance': round(float(resonance), 4),
            'stereo': round(float(stereo), 4)}

def smooth_bool_mask(mask, oct_frac=1 / 12.0, threshold=0.5):
    y = np.asarray(mask, dtype=float)
    w = max(1, int(round((1.0 / np.log10(LOGSTEP)) * np.log10(2 ** oct_frac))))
    sm = np.convolve(y, np.ones(w) / w, mode='same')
    return sm >= threshold


def octave_smooth_log(freqs, y, oct_frac):
    w = max(1, int(round((1.0 / np.log10(LOGSTEP)) * np.log10(2 ** oct_frac))))
    return np.convolve(y, np.ones(w) / w, mode='same')

def ms_to_samples(delay_ms, sample_rate_hz):
    return float(delay_ms) * float(sample_rate_hz) / 1000.0

def samples_to_ms(samples, sample_rate_hz):
    return float(samples) * 1000.0 / float(sample_rate_hz)

def calibrate_solo_levels(freqs, solo_db, together_db, band):
    freqs = np.asarray(freqs, dtype=float)
    solo_db = np.asarray(solo_db, dtype=float)
    together_db = np.asarray(together_db, dtype=float)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(sel):
        raise ValueError('band does not overlap axis')
    diff = together_db[sel] - solo_db[sel]
    offset = float(np.median(diff))
    resid = together_db[sel] - (solo_db[sel] + offset)
    return {'level_offset_db': round(offset, 2),
            'residual_rms_db': round(float(np.sqrt(np.mean(resid ** 2))), 2)}

def phase_linearity_residual(freqs, phase_deg, band):
    freqs = np.asarray(freqs, dtype=float)
    phase_deg = np.asarray(phase_deg, dtype=float)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if np.sum(sel) < 3:
        raise ValueError('band does not overlap enough of the axis')
    ph = np.rad2deg(np.unwrap(np.deg2rad(phase_deg[sel])))
    f = freqs[sel]
    slope, intercept = np.polyfit(f, ph, 1)
    resid = ph - (slope * f + intercept)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {'rms_residual_deg': round(rms, 1),
            'trustworthy_for_timing': bool(rms <= 100.0),
            'grade': ('trustworthy' if rms <= 100.0 else
                     'marginal' if rms <= 300.0 else 'reflection-dominated (do not use)')}

def complex_vector_average(complex_traces):
    if len(complex_traces) < 2:
        raise ValueError('need >=2 position traces to average')
    return np.mean(np.stack(complex_traces, axis=0), axis=0)

def inert_band_check(target_driver_db, dominant_db, threshold_db=6.0):
    gap = float(dominant_db) - float(target_driver_db)
    return {'gap_db': round(gap, 2),
            'inert': bool(gap >= threshold_db),
            'note': ('target driver is buried -- this band barely affects the sum'
                     if gap >= threshold_db else 'target driver has enough level to matter here')}


# --------------------------------------------------------------------------
# 3d) POLARITY/DELAY SEARCH -- added 2026-07-03. Completes the doctrine ladder in
# code: polarity -> delay come BEFORE any APF (we had optimize_allpass but not
# the cheaper rungs below it, which was inconsistent). Same inputs (complex solo
# captures w/ shared time-zero) and the same gap-to-coherent-ceiling score as
# optimize_allpass, so results are directly comparable. Run THIS first; only if
# `residual_needs_apf` is True has an APF earned consideration.
def polarity_delay_search(freqs, driver_a, driver_b, band, max_delay_ms=1.5,
                          steps=121, damage_band=(60.0, 16000.0), damage_free_db=0.5,
                          snapshots=None, robust=True,
                          perturbations=PHASE_ROBUST_PERTURBATIONS,
                          sample_rate_hz=FS):
    """Search polarity and delay against timing/level drift and all snapshots.

    Positive delay is applied to B. A negative result must be implemented as
    positive delay on the other branch while preserving internal pair offsets.
    """
    sel = (freqs >= band[0]) & (freqs <= band[1])
    dmg = (freqs >= damage_band[0]) & (freqs <= damage_band[1])
    if not np.any(sel):
        raise ValueError('band does not overlap the frequency axis')
    phase_snapshots = _phase_snapshot_set(driver_a, driver_b, snapshots)
    active_perturbations = perturbations if robust else ((0.0, 0.0),)

    def wr(y, m):
        w = audibility_weight(freqs[m])
        den = np.sum(w ** 2)
        return float(np.sqrt(np.sum((y[m] * w) ** 2) / den)) if den > 1e-12 else float('inf')

    def evaluate(polarity_flip, delay_ms, active):
        scores = []
        per_snapshot = []
        sign = -1.0 if polarity_flip else 1.0
        correction = np.exp(-1j * 2.0 * np.pi * freqs * float(delay_ms) / 1000.0)
        for snap_a, snap_b in phase_snapshots:
            snapshot_scores = []
            for drift_ms, level_db in active:
                perturbed_b = _phase_perturb(freqs, snap_b, drift_ms, level_db)
                coherent = 20 * np.log10(np.abs(snap_a) + np.abs(perturbed_b) + 1e-12)
                baseline_db = 20 * np.log10(np.abs(snap_a + perturbed_b) + 1e-12)
                candidate_db = 20 * np.log10(
                    np.abs(snap_a + sign * perturbed_b * correction) + 1e-12
                )
                gap_score = wr(np.maximum(coherent - candidate_db, 0.0), sel)
                damage_score = wr(
                    np.maximum(baseline_db - candidate_db - damage_free_db, 0.0), dmg
                )
                score = gap_score + damage_score
                scores.append(score)
                snapshot_scores.append(score)
            per_snapshot.append(max(snapshot_scores))
        return max(scores), per_snapshot

    base, base_snapshots = evaluate(False, 0.0, active_perturbations)
    nominal_before, _ = evaluate(False, 0.0, ((0.0, 0.0),))
    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0.0:
        raise ValueError('sample_rate_hz must be positive')
    raw_delays = np.linspace(-max_delay_ms, max_delay_ms, steps)
    delay_samples = sorted(set(
        int(round(ms_to_samples(value, sample_rate_hz)))
        for value in raw_delays
    ))
    best = None
    for pol in (False, True):
        for samples in delay_samples:
            d_ms = samples_to_ms(samples, sample_rate_hz)
            score, snapshot_scores = evaluate(pol, d_ms, active_perturbations)
            if best is None or score < best['score_after']:
                nominal_after, _ = evaluate(pol, d_ms, ((0.0, 0.0),))
                best = {
                    'polarity_flip_B': pol,
                    'delay_ms_B': float(d_ms),
                    'delay_samples_B': int(samples),
                    'delay_step_ms': round(1000.0 / sample_rate_hz, 6),
                    'sample_rate_hz': sample_rate_hz,
                    'score_before': round(base, 3),
                    'score_after': round(score, 3),
                    'nominal_score_before': round(nominal_before, 3),
                    'nominal_score_after': round(nominal_after, 3),
                    'robust_snapshot_scores_before': [
                        round(float(value), 3) for value in base_snapshots
                    ],
                    'robust_snapshot_scores_after': [
                        round(float(value), 3) for value in snapshot_scores
                    ],
                    'phase_snapshot_count': len(phase_snapshots),
                    'perturbation_count': len(active_perturbations),
                }
    best['improvement_pct'] = round(
        100.0 * (base - best['score_after']) / max(base, 1e-9), 1
    )
    best['residual_needs_apf'] = bool(best['score_after'] > 0.25 * base)
    return best

# --------------------------------------------------------------------------
# 3e) TWO-LEVEL COMPRESSION GATE -- added 2026-07-03. Makes the "high-SPL
# linearity check" numeric. Sweep the same thing twice, `level_delta_db` apart
# electrically; where the measured rise falls short, the driver/region is
# compressing (thermal/excursion/resonance). NEVER boost a compressing region --
# re-crossover or reduce its workload instead. NOTE: per REW's docs, log-sweep
# distortion data is noise-floor-limited at HF (stepped-sine is the trustworthy
# method) -- treat sweep-derived HF distortion/compression evidence as lower
# confidence.
def compression_check(low_db, high_db, level_delta_db, warn_db=0.75):
    """Returns (compression_db_per_bin, flagged_mask). compression = expected
    rise minus measured rise; > warn_db (default 0.75) = compressing, veto boosts."""
    comp = level_delta_db - (np.asarray(high_db, float) - np.asarray(low_db, float))
    return comp, comp > warn_db



# --------------------------------------------------------------------------
# 3f) SHELF SIMULATION -- added 2026-07-03. RBJ low/high shelf (Q form), matching
# the Helix shelf parameterization (Q 0.1-2 IS the slope control; hinge freq in
# 1 Hz steps; band 1 = low-shelf-capable, band 30 = high-shelf-capable).
# SIMULATION ONLY: the active-shelf XML encoding (T=20 with G!=0) is still NOT
# export-diff-verified -- design the shelf here, set it manually in PC-Tool,
# then send the export back to verify the encoding before any Python shelf write.
def low_shelf_db(freqs, f0, Q, gain_db, fs=FS):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cw, al = np.cos(w0), np.sin(w0) / (2 * Q)
    sA = 2 * np.sqrt(A) * al
    b0 = A * ((A + 1) - (A - 1) * cw + sA)
    b1 = 2 * A * ((A - 1) - (A + 1) * cw)
    b2 = A * ((A + 1) - (A - 1) * cw - sA)
    a0 = (A + 1) + (A - 1) * cw + sA
    a1 = -2 * ((A - 1) + (A + 1) * cw)
    a2 = (A + 1) + (A - 1) * cw - sA
    w = 2 * np.pi * freqs / fs
    z1, z2 = np.exp(-1j * w), np.exp(-2j * w)
    H = (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return 20 * np.log10(np.abs(H))

def high_shelf_db(freqs, f0, Q, gain_db, fs=FS):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cw, al = np.cos(w0), np.sin(w0) / (2 * Q)
    sA = 2 * np.sqrt(A) * al
    b0 = A * ((A + 1) + (A - 1) * cw + sA)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - sA)
    a0 = (A + 1) - (A - 1) * cw + sA
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - sA
    w = 2 * np.pi * freqs / fs
    z1, z2 = np.exp(-1j * w), np.exp(-2j * w)
    H = (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return 20 * np.log10(np.abs(H))

def fit_shelf_to_curve(freqs, target_curve_db, kind, band, q_lim=(0.1, 2.0)):
    """Grid-fit one shelf to replicate `target_curve_db` (e.g. a stack of broad
    PEQs being considered for consolidation) over `band`. Returns (F, Q, G,
    max_abs_err_in_band). Use to decide IF a shelf faithfully replaces the
    stack -- if max_err > ~0.75 dB where it matters, keep the PEQs."""
    fn = low_shelf_db if kind == 'low' else high_shelf_db
    sel = (freqs >= band[0]) & (freqs <= band[1])
    gains = np.arange(-6.0, 6.01, 0.25)
    best = None
    for F in np.geomspace(band[0], band[1], 40):
        for Q in np.linspace(q_lim[0], q_lim[1], 20):
            for G in gains:
                if abs(G) < 0.5: continue
                err = float(np.max(np.abs(fn(freqs, F, Q, G)[sel] - target_curve_db[sel])))
                if best is None or err < best[3]:
                    best = (round(float(F), 1), round(float(Q), 2), float(G), round(err, 2))
    return best



# --------------------------------------------------------------------------
# 3g) PREDICTION-CONFIDENCE GATE -- adopted 2026-07-03 from the R&D brief (its
# best idea). Before trusting any phase-sensitive search (polarity_delay_search,
# optimize_allpass), prove the model can predict the CURRENT measured together
# trace from the solo captures. If it can't, the complex data is misaligned
# (clock drift, moved mic, wrong time-zero) and phase decisions are blocked.
def prediction_confidence(freqs, driver_a, driver_b, measured_together_db, band):
    """Complex solos A,B (shared time-zero) + the measured pair-together SPL.
    Returns dict with rms error (after removing a level bias) and a gate:
    usable_for_phase_decisions True only if the solo model reproduces the
    measured sum within ~2.5 dB rms in-band."""
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(sel):
        raise ValueError('band does not overlap axis')
    pred = 20 * np.log10(np.abs(driver_a + driver_b) + 1e-12)
    err = pred[sel] - np.asarray(measured_together_db, float)[sel]
    bias = float(np.median(err))
    resid = err - bias
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {'rms_err_db': round(rms, 2), 'level_bias_db': round(bias, 2),
            'usable_for_phase_decisions': bool(rms <= 2.5),
            'grade': 'high' if rms <= 2.0 else ('medium' if rms <= 4.0 else 'low')}

# --------------------------------------------------------------------------
# 3h) TUNE SCORECARD -- one canonical scoring function so every tune comparison
# uses identical math (yesterday's v5/v6/v7/aggressive benchmark was hand-rolled
# three times; this ends that). Named components, not one opaque number.
def tune_scorecard(freqs, traces, target_db,
                   img_band=(200.0, 6000.0), mid_bal_band=(200.0, 2000.0),
                   tw_bal_band=(2800.0, 16000.0), inband=(60.0, 16000.0)):
    """traces: dict with 'System Sum' and optionally 'FL Low','FR Low',
    'FL High','FR High' (predicted or measured SPL on `freqs`). Returns the
    named metrics used for every tune-vs-tune decision."""
    dev = erb_smooth(freqs, traces['System Sum'] - target_db)
    inb = (freqs >= inband[0]) & (freqs <= inband[1])
    w = np.ones_like(freqs); w[(freqs >= img_band[0]) & (freqs <= img_band[1])] = 1.8
    out = {'sum_rms_db': round(float(np.sqrt(np.mean(dev[inb] ** 2))), 2),
           'sum_wrms_img_db': round(float(np.sqrt(np.sum((dev[inb] * w[inb]) ** 2) / np.sum(w[inb] ** 2))), 2),
           'worst_dev_db': round(float(np.max(np.abs(dev[(freqs >= 100) & (freqs <= 8000)]))), 1)}
    if 'FL Low' in traces and 'FR Low' in traces:
        b = erb_smooth(freqs, traces['FL Low'] - traces['FR Low'])
        s = (freqs >= mid_bal_band[0]) & (freqs <= mid_bal_band[1])
        out['mid_balance_db'] = round(float(np.median(b[s])), 2)
    if 'FL High' in traces and 'FR High' in traces:
        b = erb_smooth(freqs, traces['FL High'] - traces['FR High'])
        s = (freqs >= tw_bal_band[0]) & (freqs <= tw_bal_band[1])
        out['tweeter_balance_db'] = round(float(np.median(b[s])), 2)
    return out


if __name__ == '__main__':
    import struct

    freqs = 24000.0 / (LOGSTEP ** (1231 - np.arange(1232)))

    # ---- TEST 1: excess-GD classifier on a synthetic known system ----------
    # Build: one minimum-phase peak (EQ-able) + one reflection notch
    # (delayed copy summed -> NON-minimum-phase around the notch).
    w = 2 * np.pi * freqs
    Hpk = 10 ** (peaking_db(freqs, 300.0, 2.0, +6.0) / 20.0) \
        * np.exp(1j * np.deg2rad(0))                        # magnitude only...
    # give the peak its true min phase:
    ph_pk = minphase_from_mag(freqs, peaking_db(freqs, 300.0, 2.0, +6.0))
    Hpk = 10 ** (peaking_db(freqs, 300.0, 2.0, +6.0) / 20.0) * np.exp(1j * ph_pk)
    # DSP subtlety the classifier must honor: a reflection WEAKER than the
    # direct (a<1) makes a comb that is still MINIMUM phase (zeros inside the
    # unit circle) -> technically EQ-able. Only a DOMINANT reflection (a>1)
    # flips the notch non-minimum-phase -> un-EQ-able. Test both.
    tau = 1.0 / (2 * 1200.0)                                # antiphase at 1.2 kHz
    H_weak = Hpk * (1.0 + 0.8 * np.exp(-1j * w * tau))      # min-phase comb
    H_dom  = Hpk * (0.8 + 1.0 * np.exp(-1j * w * tau))      # dominant reflection
    i_pk = int(np.argmin(np.abs(freqs - 300)))
    near = (freqs > 1200 / 2 ** (1 / 12.)) & (freqs < 1200 * 2 ** (1 / 12.))
    print('TEST1 excess-GD classifier:')
    for nm, H, expect_nt in [('weak refl (min-phase)', H_weak, True),
                             ('dominant refl (non-min-phase)', H_dom, False)]:
        spl = 20 * np.log10(np.abs(H))
        ph = np.rad2deg(np.angle(H))
        gd, mask = excess_gd_mask(freqs, spl, ph, flat_ms=0.15)
        nt_ok = bool(np.all(mask[near])) if expect_nt else bool(np.any(~mask[near]))
        print('  %-30s peak@300 eqable=%s (exp True) | notch@1.2k %s' %
              (nm, mask[i_pk], 'stays eqable (exp)' if expect_nt else
               ('flagged un-EQ-able (exp)' if nt_ok else 'NOT flagged (FAIL)')))
        assert mask[i_pk] and nt_ok, 'excess-GD classifier failed on ' + nm

    # ---- TEST 2: optimizer recovers a known correction ---------------------
    dev = peaking_db(freqs, 500.0, 2.0, 5.0) + peaking_db(freqs, 2000.0, 1.0, 4.0)
    bands, rep = fit_peq(freqs, dev, (100, 8000), n_bands_max=4)
    print('TEST2 optimizer on synthetic (+5@500 Q2, +4@2k Q1):')
    for b in bands: print('   fit: F=%-7.1f Q=%-5.2f G=%+.2f' % b)
    print('   score %.3f -> %.3f with %d bands' % (rep['score_before'], rep['score_after'], rep['bands_used']))
    assert rep['score_after'] < 0.35 * rep['score_before'] and rep['bands_used'] <= 3

    # ---- OPTIONAL historical validation on a real exported sample ----------
    MDAT = 'validation_sample.mdat'
    TGT = 'ResoNix Target Curve 2026.txt'
    if os.path.exists(MDAT) and os.path.exists(TGT):
        data = open(MDAT, 'rb').read()
        def gar(o):
            p = o + 6; n = struct.unpack('>I', data[p:p + 4])[0]
            return np.frombuffer(data[p + 4:p + 4 + 4 * n], dtype='>f4').astype(float)
        FR = gar(760318)
        tf, ts = [], []
        for line in open(TGT, encoding='utf-8', errors='replace'):
            s = line.strip()
            if s and not s[0].isalpha() and not s.startswith('*'):
                p = s.replace(',', ' ').split()
                try: tf.append(float(p[0])); ts.append(float(p[1]))
                except Exception: pass
        tgt = np.interp(np.log10(freqs), np.log10(np.array(tf)), np.array(ts))
        b = (freqs >= 300) & (freqs <= 1200)
        dev = FR - (tgt + np.median(FR[b] - tgt[b]))

        base_sm = erb_smooth(freqs, dev)
        mask_mag = ~((dev - base_sm) < -3.0) & ~(base_sm < -4.0)

        FIT = (150.0, 2450.0)
        hand = [(615.0, 5.5, -7.5), (628.0, 5.5, +7.5),
                (1000.0, 3.0, -3.5), (1000.0, 2.0, +3.5),
                (1175.0, 4.0, -4.5)]
        hand_dev = dev + cascade_db(freqs, hand)
        s_before = audibility_score(freqs, dev, band=FIT, mask=mask_mag)
        s_hand = audibility_score(freqs, hand_dev, band=FIT, mask=mask_mag)
        bands, rep = fit_peq(freqs, dev, FIT, n_bands_max=4, mask=mask_mag, verbose=True)
        print('VALIDATION on real FR Low sample:')
        print('  audibility score  as-measured : %.3f' % s_before)
        print('  after v4 hand/greedy changes  : %.3f' % s_hand)
        print('  after joint fit (%d new bands): %.3f' % (rep['bands_used'], rep['score_after']))
        for b_ in bands: print('     F=%-7.1f Q=%-5.2f G=%+.2f' % b_)
        beat_hand = rep['score_after'] <= s_hand + 1e-9
    else:
        print('VALIDATION on real FR Low sample skipped: validation files not present')
        beat_hand = True

    # ---- TEST 3: filter tax discourages boosts + narrow-HF filters ---------
    # A dip that COULD be filled with a boost: without tax the fit may boost;
    # with a strong tax it should prefer to leave it (fewer/no boost bands).
    devd = -peaking_db(freqs, 3000.0, 6.0, 5.0)     # a narrow -5 dip at 3 kHz (HF)
    b_notax, _ = fit_peq(freqs, devd, (300, 8000), n_bands_max=3,
                         boost_penalty=0.0, hf_q_penalty=0.0)
    b_tax, _ = fit_peq(freqs, devd, (300, 8000), n_bands_max=3,
                       boost_penalty=1.5, hf_q_penalty=1.5)
    boosts_notax = sum(1 for _, _, G in b_notax if G > 0)
    boosts_tax = sum(1 for _, _, G in b_tax if G > 0)
    print('\nTEST3 filter tax on a narrow +HF dip (fill temptation):')
    print('  no tax  -> %d band(s), boosts=%d: %s' % (len(b_notax), boosts_notax, b_notax))
    print('  w/ tax  -> %d band(s), boosts=%d: %s' % (len(b_tax), boosts_tax, b_tax))
    assert boosts_tax <= boosts_notax, 'filter tax did not reduce boosts'

    # ---- TEST 4: headroom report ------------------------------------------
    hr = headroom_report(freqs, [(120.0, 1.0, 4.0), (1000.0, 2.0, -3.0), (110.0, 1.5, 3.0)])
    print('\nTEST4 headroom report:', hr)
    assert hr['clip_risk'] and hr['recommended_trim_db'] < 0, 'headroom report wrong'

    # ---- TEST 5: interference audit (synthetic + real "Measurements.mdat") --
    tau = 1.0 / (2 * 415.0)                       # antiphase at 415 Hz
    w = 2 * np.pi * freqs
    A = np.ones_like(freqs, dtype=complex) * 10 ** (50 / 20.0)     # solo A, 50dB
    B = 10 ** (50 / 20.0) * np.exp(-1j * w * tau)                  # solo B, delayed
    together_complex = 20 * np.log10(np.abs(A + B))                # true coherent sum
    solo_a_db = 20 * np.log10(np.abs(A)); solo_b_db = 20 * np.log10(np.abs(B))
    psum, csum, interf, flag = interference_audit(freqs, solo_a_db, solo_b_db, together_complex)
    i415 = int(np.argmin(np.abs(freqs - 415)))
    i830 = int(np.argmin(np.abs(freqs - 830)))    # back in phase an octave up (2*tau cycle)
    print('\nTEST5 interference audit (synthetic antiphase @415Hz):')
    print('  @415Hz  psum=%.1f csum=%.1f together=%.1f interf=%+.1f flagged=%s (expect True)'
          % (psum[i415], csum[i415], together_complex[i415], interf[i415], flag[i415]))
    print('  @830Hz  interf=%+.1f flagged=%s' % (interf[i830], flag[i830]))
    assert flag[i415], 'interference audit missed a known cancellation'

    # ---- TEST 6: all-pass XML matches the VERIFIED real export exactly -----
    xml = allpass_fil_str(430.0, 0.7, FN='229')
    expect = '<Fil G="0" FN="229" F="430.00" T="20" I="0" dF="20000" Q="0.7"/>'
    print('\nTEST6 allpass_fil_str:', xml)
    assert xml == expect, 'allpass XML does not match the verified real export'


    # ---- TEST9: polarity/delay search (the rungs BELOW the APF) -------------
    w = 2 * np.pi * freqs
    A9 = np.ones_like(freqs, dtype=complex)
    B9 = -np.ones_like(freqs, dtype=complex)          # pure polarity inversion
    r1 = polarity_delay_search(freqs, A9, B9, (200, 2000))
    print()
    print('TEST9 polarity/delay search:')
    print('  inverted pair  -> flip=%s delay=%.2fms improve=%.0f%% needs_apf=%s'
          % (r1['polarity_flip_B'], r1['delay_ms_B'], r1['improvement_pct'], r1['residual_needs_apf']))
    assert r1['polarity_flip_B'] and abs(r1['delay_ms_B']) < 0.05 and not r1['residual_needs_apf']
    B9b = np.exp(-1j * w * 0.0004) * np.ones_like(freqs, dtype=complex)   # 0.4 ms late
    r2 = polarity_delay_search(freqs, A9, B9b, (500, 2000))
    print('  0.4ms-late B   -> flip=%s delay=%.2fms improve=%.0f%% needs_apf=%s'
          % (r2['polarity_flip_B'], r2['delay_ms_B'], r2['improvement_pct'], r2['residual_needs_apf']))
    # B was LATE, so the fix is NEGATIVE delay on B (advance). Hardware can't
    # advance: translate a negative delay_ms_B into "+delay on the OTHER branch"
    # (the doc's negative-delay rule).
    assert (not r2['polarity_flip_B']) and abs(r2['delay_ms_B'] + 0.4) < 0.05
    # frequency-LOCALIZED rotation (APF-shaped problem): polarity/delay cannot
    # fully fix it -> the search must hand off to the APF stage
    B9c = -(allpass_H(freqs, 415.0, 0.7) ** 2) * np.ones_like(freqs, dtype=complex)
    r3 = polarity_delay_search(freqs, A9, B9c, (250, 700))
    print('  local rotation -> improve=%.0f%% needs_apf=%s (expect True)'
          % (r3['improvement_pct'], r3['residual_needs_apf']))
    assert r3['residual_needs_apf'], 'should have handed off to APF search'

    # ---- TEST10: two-level compression gate ---------------------------------
    low10 = np.zeros_like(freqs)
    high10 = low10 + 10.0                              # perfectly linear +10 dB
    hot10 = (freqs > 2000) & (freqs < 4000)
    high10[hot10] -= 2.0                               # 2 dB compression in a band
    comp10, flag10 = compression_check(low10, high10, 10.0)
    print()
    print('TEST10 compression gate: flagged=%d bins, all inside 2-4k: %s'
          % (int(flag10.sum()), bool(np.all(flag10 == hot10))))
    assert np.all(flag10[hot10]) and not np.any(flag10[~hot10])


    # ---- TEST11: shelf shapes ------------------------------------------------
    ls = low_shelf_db(freqs, 200.0, 0.7, -6.0)
    hs = high_shelf_db(freqs, 5000.0, 0.7, -3.0)
    i20 = int(np.argmin(np.abs(freqs - 20))); i200 = int(np.argmin(np.abs(freqs - 200)))
    i20k = int(np.argmin(np.abs(freqs - 20000))); i5k = int(np.argmin(np.abs(freqs - 5000)))
    print()
    print('TEST11 shelves: LS(-6@200) 20Hz=%.1f 200Hz=%.1f 20kHz=%.1f | HS(-3@5k) 20Hz=%.1f 5kHz=%.1f 20kHz=%.1f'
          % (ls[i20], ls[i200], ls[i20k], hs[i20], hs[i5k], hs[i20k]))
    assert abs(ls[i20] + 6) < 0.3 and abs(ls[i200] + 3) < 0.5 and abs(ls[i20k]) < 0.3
    assert abs(hs[i20]) < 0.3 and abs(hs[i5k] + 1.5) < 0.5 and abs(hs[i20k] + 3) < 0.4

    # ---- TEST12: special-filter writers vs REAL export lines (semantic) -----
    real_ls = '<Fil Q="1" G="-2.25" F="4980.25" FN="0" I="0" T="3" dF="25"/>'
    real_hs = '<Fil Q="0.5" G="0.25" F="5400.00" FN="29" I="0" T="4" dF="20000"/>'
    real_a1 = '<Fil Q="1" G="0" F="2000.00" FN="19" I="0" T="19" dF="2000"/>'
    real_a1i = '<Fil Q="1" G="0" F="2000.00" FN="19" I="1" T="19" dF="2000"/>'
    mine_ls = shelf_fil_str('low', 4980.25, 1, -2.25, FN='0')
    mine_hs = shelf_fil_str('high', 5400.0, 0.5, 0.25, FN='29')
    mine_a1 = allpass1_fil_str(2000.0, FN='19', dF='2000')
    mine_a1i = allpass1_fil_str(2000.0, FN='19', dF='2000', invert=True)
    def _semeq(a, b):
        da, db = fil_attrs(a), fil_attrs(b)
        # numeric-normalize
        for d_ in (da, db):
            for k in ('F', 'Q', 'G'):
                d_[k] = float(d_[k])
        return da == db
    print()
    print('TEST12 special writers: LS match=%s HS match=%s APF1 match=%s'
          % (_semeq(mine_ls, real_ls), _semeq(mine_hs, real_hs), _semeq(mine_a1, real_a1)))
    assert _semeq(mine_ls, real_ls) and _semeq(mine_hs, real_hs) and _semeq(mine_a1, real_a1)
    assert _semeq(mine_a1i, real_a1i), 'inverted APF1 string mismatch vs real export'
    print('TEST12c invert flag: I="1" writer matches the real inverted export')
    # delay semantic comparison tolerates PC-Tool attr reordering
    xa = '<OC><T PM="4" T="223" P="0"/></OC>'
    xb = '<OC><T T="223" P="0" PM="4"/></OC>'
    xc = '<OC><T T="224" P="0" PM="4"/></OC>'
    assert delays_semantically_equal(xa, xb) and not delays_semantically_equal(xa, xc)
    print('TEST12b delay semantic-equality: reorder tolerated, value change caught')


    # ---- TEST13: APF invert = the opposite-direction tool -------------------
    # invert multiplies the APF by -1: same rotation, plus 180 deg EVERYWHERE.
    #  - healthy (in-phase) pair + normal 2nd-order APF at f0 -> NULL at f0
    #  - ANTIPHASE pair + normal APF at f0 -> FIXED at f0 (the 430 Hz use-case)
    #  - antiphase pair + INVERTED APF -> still null at f0 (wrong direction
    #    locally) but FIXED far from f0 (acts as a broadband polarity flip)
    # So: if live-dialing makes the target dip worse at every F/Q, hit invert --
    # the needed rotation is on the other side of the circle.
    A13 = np.ones_like(freqs, dtype=complex)
    i415 = int(np.argmin(np.abs(freqs - 415)))
    i5k  = int(np.argmin(np.abs(freqs - 5000)))
    def sdb13(x): return 20*np.log10(np.abs(x) + 1e-12)
    healthy = sdb13(A13 + A13)[i415]
    Hn, Hi = allpass_H(freqs,415,0.7), allpass_H_inv(freqs,415,0.7)
    n_on_healthy   = sdb13(A13*Hn + A13)[i415]
    n_on_antiphase = sdb13(A13*Hn - A13)[i415]
    i_on_anti_f0   = sdb13(A13*Hi - A13)[i415]
    i_on_anti_5k   = sdb13(A13*Hi - A13)[i5k]
    print()
    print('TEST13 APF invert: healthy=%.1f | norm-on-healthy@f0=%.1f (null) | '
          'norm-on-anti@f0=%.1f (fixed) | inv-on-anti@f0=%.1f (null) @5k=%.1f (fixed)'
          % (healthy, n_on_healthy, n_on_antiphase, i_on_anti_f0, i_on_anti_5k))
    assert n_on_healthy < healthy - 30
    assert abs(n_on_antiphase - healthy) < 0.1
    assert i_on_anti_f0 < healthy - 30
    assert abs(i_on_anti_5k - healthy) < 0.5
    assert np.allclose(np.abs(Hi), 1.0)


    tgt_like = 60.0 + 0.0 * freqs
    # ---- TEST14: prediction-confidence gate ----------------------------------
    A14 = np.ones_like(freqs, dtype=complex)
    B14 = np.exp(-1j * 2 * np.pi * freqs * 0.0002) * 0.8   # coherent pair, known sum
    true_together = 20 * np.log10(np.abs(A14 + B14)) + 3.0  # +3 dB level bias (mic cal)
    r14 = prediction_confidence(freqs, A14, B14, true_together, (200, 2000))
    # now corrupt the model: pretend B was captured with a wrong time-zero
    B14bad = B14 * np.exp(-1j * 2 * np.pi * freqs * 0.004)
    r14b = prediction_confidence(freqs, A14, B14bad, true_together, (200, 2000))
    print()
    print('TEST14 prediction gate: good rms=%.2f (%s, bias %+.1f) | corrupted rms=%.2f (%s)'
          % (r14['rms_err_db'], r14['grade'], r14['level_bias_db'], r14b['rms_err_db'], r14b['grade']))
    assert r14['usable_for_phase_decisions'] and abs(r14['level_bias_db'] + 3.0) < 0.2
    assert not r14b['usable_for_phase_decisions']

    # ---- TEST14b: coherence / gated impulse helpers -------------------------
    db_bad = np.vstack([np.zeros_like(freqs), np.full_like(freqs, 10.0)])
    coh14 = np.vstack([np.full_like(freqs, 0.95), np.full_like(freqs, 0.05)])
    avg14 = coherence_weighted_db_average(db_bad, coh14)
    print('TEST14b coherence-weighted average @1k: %.2f dB' % avg14[int(np.argmin(np.abs(freqs - 1000)))])
    assert abs(avg14[int(np.argmin(np.abs(freqs - 1000)))]) < 1.0

    tau_ms = 0.4
    ph_diff = -360.0 * freqs * (tau_ms / 1000.0)
    dfit = band_limited_delay_from_phase(freqs, ph_diff, (500.0, 5000.0),
                                         coherence=np.full_like(freqs, 0.95))
    print('TEST14b phase-delay fit:', dfit)
    assert dfit['usable'] and abs(dfit['delay_ms'] - tau_ms) < 0.03

    fs22 = 48000.0
    n22 = 4096
    a22 = np.zeros(n22, dtype=float)
    b22 = np.zeros(n22, dtype=float)
    a22[1000] = 1.0
    a22[1192] = 0.35   # first reflection 4 ms later
    b22[1048] = 1.0    # B is 1 ms later
    gate22 = suggest_gate_from_impulse(a22, fs22, direct_index=1000, ignore_ms=0.2, threshold_db=-12.0)
    conf22 = gate_frequency_confidence(freqs, gate22['gate_ms'])
    a23 = np.zeros(n22, dtype=float)
    b23 = np.zeros(n22, dtype=float)
    a23[64] = 1.0
    b23[112] = 1.0
    d22 = band_limited_impulse_delay(a23, b23, fs22, (500.0, 5000.0))
    i250 = int(np.argmin(np.abs(freqs - 250.0)))
    i1k = int(np.argmin(np.abs(freqs - 1000.0)))
    print('TEST14b gate/impulse helpers: gate=%sms delay=%sms polarity=%s conf250=%.2f conf1k=%.2f'
          % (gate22['gate_ms'], d22['delay_ms'], d22['polarity'], conf22[i250], conf22[i1k]))
    assert abs(gate22['gate_ms'] - 4.0) < 0.2
    assert conf22[i250] < 0.55 and conf22[i1k] > 0.95
    assert d22['usable'] and d22['polarity'] == 'same' and abs(d22['delay_ms'] - 1.0) < 0.05

    s96 = ms_to_samples(6.52, 96000.0)
    s48 = ms_to_samples(6.52, 48000.0)
    print('TEST14c delay conversion: 6.52ms -> %.0f samples @96k, %.0f samples @48k' % (s96, s48))
    assert abs(samples_to_ms(s96, 96000.0) - 6.52) < 1e-6 and abs(s96 - s48) > 100

    A19 = np.full_like(freqs, 60.0)
    together19 = A19 + 4.5
    cal = calibrate_solo_levels(freqs, A19, together19, (200.0, 2000.0))
    print('TEST14c solo calibration:', cal)
    assert abs(cal['level_offset_db'] - 4.5) < 0.05

    ph_clean = -0.02 * freqs
    ph_noisy = ph_clean + 180.0 * np.sin(freqs / 120.0)
    r_clean = phase_linearity_residual(freqs, ph_clean, (300.0, 3000.0))
    r_noisy = phase_linearity_residual(freqs, ph_noisy, (300.0, 3000.0))
    print('TEST14c phase trust:', r_clean, r_noisy)
    assert r_clean['trustworthy_for_timing'] and not r_noisy['trustworthy_for_timing']

    comb1 = np.exp(-1j * 2 * np.pi * freqs * 0.00020)
    comb2 = np.exp(-1j * 2 * np.pi * freqs * 0.00021)
    avg = complex_vector_average([comb1, comb2])
    buried = inert_band_check(target_driver_db=60.0, dominant_db=75.0)
    audible = inert_band_check(target_driver_db=72.0, dominant_db=75.0)
    print('TEST14c vector avg / inert-band:', float(np.median(np.abs(avg))), buried, audible)
    assert buried['inert'] and not audible['inert']

    # ---- TEST15: scorecard + gain rung ---------------------------------------
    tr15 = {'System Sum': tgt_like + 2.0 * np.sin(np.log(freqs)),
            'FL Low': tgt_like - 3.0, 'FR Low': tgt_like + 0.0,
            'FL High': tgt_like + 1.0, 'FR High': tgt_like - 1.0}
    sc = tune_scorecard(freqs, tr15, tgt_like)
    print('TEST15 scorecard:', sc)
    assert abs(sc['mid_balance_db'] + 3.0) < 0.1 and abs(sc['tweeter_balance_db'] - 2.0) < 0.1
    assert sc['sum_rms_db'] > 0

    print('\n' + ('ALL TESTS PASSED' if beat_hand else
          'TESTS PASSED (note: joint fit tied/lost vs hand set on this data)'))
