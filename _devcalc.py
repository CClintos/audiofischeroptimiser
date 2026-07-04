import os
import struct
import numpy as np

try:
    from _tunefit import optimize_allpass, target_anchor_offset
except Exception:
    optimize_allpass = None
    target_anchor_offset = None

MDAT = os.environ.get('MDAT_PATH', 'System Sum.mdat')
TGT  = os.environ.get('TARGET_PATH', 'ResoNix Target Curve 2026.txt')

data = open(MDAT, 'rb').read()

def find_arrays(data, min_len=256):
    out, i = [], 0
    while True:
        j = data.find(b'\x75\x71\x00\x7e', i)
        if j < 0: break
        p = j + 6; n = struct.unpack('>I', data[p:p+4])[0]; body = p + 4
        if 0 < n < 5000000 and body + 4*n <= len(data):
            a = np.frombuffer(data[body:body+4*n], dtype='>f4').astype(float)
            if n >= min_len and np.isfinite(a).all(): out.append((j, n, a))
        i = j + 2
    return out

arrs = find_arrays(data)
off, n, mag = max(arrs, key=lambda t: t[1])
print('maglen', n, 'off', off, 'range', round(float(mag.min()),1), round(float(mag.max()),1))

# Axis: REW SPL data, 96 points/octave -> logStep = 2^(1/96); last point validated at 24kHz last session
logStep = 2 ** (1/96.0)
endFreq = 24000.0
freqs = endFreq / (logStep ** (n - 1 - np.arange(n)))
print('startFreq', round(float(freqs[0]),3), 'endFreq', round(float(freqs[-1])), 'logStep', round(logStep,8))

# VALIDATION: find the deepest dip 350-550Hz (the known cabin null) and the bass peak 30-60Hz
def at(f):
    return int(np.argmin(np.abs(freqs - f)))
nullband = (freqs >= 350) & (freqs <= 550)
ni = np.where(nullband)[0][np.argmin(mag[nullband])]
bassband = (freqs >= 30) & (freqs <= 60)
bi = np.where(bassband)[0][np.argmax(mag[bassband])]
print('VALIDATE: null at %.0f Hz (%.1f dB), bass peak at %.0f Hz (%.1f dB)' % (freqs[ni], mag[ni], freqs[bi], mag[bi]))

# target
tf, ts = [], []
for line in open(TGT, encoding='utf-8', errors='replace'):
    line = line.strip()
    if not line or line[0].isalpha() or line.startswith('*'): continue
    parts = line.replace(',', ' ').split()
    try: a = float(parts[0]); b = float(parts[1])
    except: continue
    tf.append(a); ts.append(b)
tf = np.array(tf); ts = np.array(ts)
tgt = np.interp(np.log10(freqs), np.log10(tf), ts)
# WIDE, MEDIAN-based level anchor: one local bump in a narrow window must not float the whole target.
band = (freqs >= 300) & (freqs <= 3000)
offset = target_anchor_offset(freqs, mag, tgt) if target_anchor_offset else float(np.median(mag[band] - tgt[band]))
dev = mag - (tgt + offset)
print('level offset applied to target: %+.1f dB (median over 300-3000 Hz)' % offset)
# Split-band smoothing: below the cabin transition the cabin is modal (narrow peaks are
# real -> keep 1/6 oct); above it reflections dominate (only broad trends are real -> 1/3 oct).
TRANSITION = 400.0
def smooth(y, frac):
    w = max(1, int(round((1.0/np.log10(logStep)) * np.log10(2**frac))))
    k = np.ones(w)/w
    return np.convolve(y, k, mode='same')
sd = np.where(freqs < TRANSITION, smooth(dev, 1/6.0), smooth(dev, 1/3.0))
print('--- deviation: 1/6oct <%.0fHz (modal), 1/3oct above : + = too loud (CUT) ---' % TRANSITION)
for f in [25,32,40,50,63,80,100,110,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]:
    i = at(f); print('%6dHz  dev %+5.1f   (raw %+5.1f)' % (f, sd[i], dev[i]))

# ---------------------------------------------------------------------------
# Filter-interaction check. Model the PROPOSED PEQ bands as RBJ peaking biquads,
# cascade them (product of transfer fns = sum in dB) and show the PREDICTED post-EQ
# deviation. Use this instead of trusting G = -dev per band: bands within ~1 octave
# have overlapping skirts that SUM, so naive per-band gains overshoot.
FS = 96000.0  # Helix internal rate
def peaking_db(freqs, f0, Q, gain_db, fs=FS):
    A  = 10 ** (gain_db / 40.0)
    w0 = 2*np.pi*f0/fs
    al = np.sin(w0)/(2*Q)
    b0, b1, b2 = 1+al*A, -2*np.cos(w0), 1-al*A
    a0, a1, a2 = 1+al/A, -2*np.cos(w0), 1-al/A
    w  = 2*np.pi*freqs/fs
    z1, z2 = np.exp(-1j*w), np.exp(-2j*w)
    H = (b0 + b1*z1 + b2*z2) / (a0 + a1*z1 + a2*z2)
    return 20*np.log10(np.abs(H))

# (F, Q, G) bands you intend to ADD to a pair. Edit and re-run BEFORE writing the .afpx.
PROPOSED = [
    (110.0, 1.3, -4.0),
    (2500.0, 2.5, -4.0),
]
if PROPOSED:
    eq = np.zeros_like(freqs)
    for F, Q, G in PROPOSED:
        eq += peaking_db(freqs, F, Q, G)
    pred  = dev + eq                  # predicted deviation from target AFTER the EQ
    preds = np.where(freqs < TRANSITION, smooth(pred, 1/6.0), smooth(pred, 1/3.0))
    print('--- PREDICTED post-EQ deviation (PROPOSED bands, interaction included) ---')
    for F, Q, G in PROPOSED:
        i = at(F)
        print('  %7.0f Hz  Q=%.2f G=%+.1f  ->  before %+5.1f   after %+5.1f' % (F, Q, G, sd[i], preds[i]))
    # NOTE: residual still includes the cabin null (~500 Hz) and tweeter rolloff (~20 kHz)
    # that you INTENTIONALLY do not EQ -- don't chase this to zero. The per-band before/after
    # numbers above are the real check; this is just a coarse "did anything blow up" guard.
    inband = (freqs >= 20) & (freqs <= 20000)
    print('  max |predicted residual| 20Hz-20kHz: %.1f dB  (incl. null/rolloff you leave alone)'
          % float(np.max(np.abs(preds[inband]))))

# ---------------------------------------------------------------------------
# PHASE / ALL-PASS toolkit. An all-pass changes phase only (|H| == 1), so it only
# affects the SUMMED response of two drivers. Use it to test whether an APF fills a
# crossover cancellation (e.g. the ~415 Hz mid<->midbass dip) BEFORE writing the .afpx.
# Put the APF on ONE acoustic branch of the summing pair. If the problem is present
# on both L/R sides, using the same APF on the same branch on both sides can be
# image-safe. What cancels out is putting the same APF on BOTH drivers being summed.
def allpass_H(freqs, f0, Q, order=2, fs=FS):
    w0 = 2*np.pi*f0/fs
    w  = 2*np.pi*freqs/fs
    z1 = np.exp(-1j*w)
    if order == 1:                       # 1st-order all-pass (90 deg at f0, |H|=1)
        t = np.tan(w0/2.0); a = (t-1)/(t+1)
        return (a + z1)/(1 + a*z1)
    al = np.sin(w0)/(2*Q)                 # 2nd-order RBJ all-pass (180 deg at f0)
    b0, b1, b2 = 1-al, -2*np.cos(w0), 1+al
    a0, a1, a2 = 1+al, -2*np.cos(w0), 1-al
    z2 = np.exp(-2j*w)
    return (b0 + b1*z1 + b2*z2)/(a0 + a1*z1 + a2*z2)

def load_complex_export(path, freqs):
    """REW text export (Freq, SPL dB, Phase deg) -> complex response on `freqs`."""
    f, spl, ph = [], [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line or line[0].isalpha() or line.startswith('*'): continue
        p = line.replace(',', ' ').split()
        try: f.append(float(p[0])); spl.append(float(p[1])); ph.append(float(p[2]))
        except Exception: continue
    if len(f) < 2:
        raise ValueError('No usable freq/SPL/phase rows found in %s' % path)
    f = np.array(f); spl = np.array(spl); ph = np.unwrap(np.deg2rad(np.array(ph)))
    lf = np.log10(freqs)
    mag = 10 ** (np.interp(lf, np.log10(f), spl) / 20.0)
    return mag * np.exp(1j * np.interp(lf, np.log10(f), ph))

# Set these to two per-driver exports (with phase) to evaluate an APF on driver A.
DRIVER_A = None   # e.g. r'C:\path\to\FL Low solo.txt'   (gets the APF)
DRIVER_B = None   # e.g. r'C:\path\to\FL High solo.txt'
APF      = (415.0, 2.0, 2)   # (F, Q, order) all-pass to try on driver A
APF_SEARCH_BAND = None       # e.g. (320.0, 520.0) to auto-search APF F/Q on driver A
if DRIVER_A and DRIVER_B:
    A = load_complex_export(DRIVER_A, freqs)
    B = load_complex_export(DRIVER_B, freqs)
    F0, Q0, O0 = APF
    sum0 = 20*np.log10(np.abs(A + B) + 1e-9)
    sum1 = 20*np.log10(np.abs(A*allpass_H(freqs, F0, Q0, O0) + B) + 1e-9)
    i = at(F0)
    print('--- ALL-PASS summation check (APF F=%.0f Q=%.1f order=%d on driver A) ---' % (F0, Q0, O0))
    print('  summed level @ %.0f Hz: %.1f dB -> %.1f dB  (%+.1f dB)'
          % (F0, sum0[i], sum1[i], sum1[i]-sum0[i]))
    print('  (positive = dip filled; sweep F/Q/order to maximize, then set on ONE channel in PC-Tool)')
    if APF_SEARCH_BAND and optimize_allpass:
        print('--- ALL-PASS grid search (candidate only; verify with summed re-measure) ---')
        print(' ', optimize_allpass(freqs, A, B, APF_SEARCH_BAND, apply_to='A'))

# ---------------------------------------------------------------------------
# PER-SIDE correction mode. The System Sum can't separate L from R, so EQ from it is
# inherently SHARED. To get legitimately different L/R EQ, measure each driver SOLO and
# decompose into two layers:
#   SHARED voicing tilt  = broad (~1 oct) Sum-vs-target deviation  -> identical both sides
#   PER-SIDE corrective  = each solo driver's LOCAL anomalies vs its OWN ~1 oct MEDIAN baseline
#                          (vs its own baseline, NOT the target, so the tilt isn't applied twice)
def load_spl_export(path, freqs):
    """REW text export (Freq, SPL[, Phase]) -> SPL(dB) interpolated onto `freqs`."""
    f, spl = [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line or line[0].isalpha() or line.startswith('*'): continue
        p = line.replace(',', ' ').split()
        try: f.append(float(p[0])); spl.append(float(p[1]))
        except Exception: continue
    if len(f) < 2:
        raise ValueError('No usable freq/SPL rows found in %s' % path)
    f = np.array(f); spl = np.array(spl)
    return np.interp(np.log10(freqs), np.log10(f), spl)

def octave_smooth(y, oct_frac):
    w = max(1, int(round((1.0/np.log10(logStep)) * np.log10(2**oct_frac))))
    return np.convolve(y, np.ones(w)/w, mode='same')

def octave_median(y, oct_frac):
    """MEDIAN baseline over an octave window -- ignores peaks (a mean baseline swallows
    the very peaks it should isolate, under-reporting anomaly height). Use for per-side trend."""
    w = max(3, int(round((1.0/np.log10(logStep)) * np.log10(2**oct_frac))))
    if w % 2 == 0: w += 1
    half = w // 2; out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i-half):min(len(y), i+half+1)])
    return out

def propose_cuts(freqs, resid, band, thresh=2.0, max_bands=6, min_sep_oct=1/3.0):
    """Local maxima of `resid` (dB above own trend) within `band` -> (F, Q, G) cut bands."""
    lo, hi = band; n = len(freqs); cand = []
    for i in range(1, n-1):
        if lo <= freqs[i] <= hi and resid[i] >= thresh and resid[i] >= resid[i-1] and resid[i] >= resid[i+1]:
            cand.append(i)
    cand.sort(key=lambda i: -resid[i])
    chosen = []
    for i in cand:
        if all(abs(np.log2(freqs[i]/freqs[j])) >= min_sep_oct for j in chosen):
            chosen.append(i)
        if len(chosen) >= max_bands: break
    out = []
    for i in sorted(chosen):
        half = resid[i]/2.0; l = i; r = i
        while l > 0 and resid[l] > half and freqs[l] > lo: l -= 1
        while r < n-1 and resid[r] > half and freqs[r] < hi: r += 1
        N = max(np.log2(freqs[r]/freqs[l]), 1/12.0)
        Q = round(min(max(float(np.sqrt(2**N)/(2**N - 1)), 0.5), 15.0), 2)   # AF Q range 0.5-15
        G = max(-round(float(resid[i]), 1), -15.0)                            # AF cut floor -15 dB
        out.append((round(float(freqs[i]), 1), Q, G))
    return out

# Config: set any to a REW text export path. PASSBAND = the driver pair's crossover band.
LEFT_SOLO  = None   # e.g. r'C:\path\to\FL Low solo.txt'
RIGHT_SOLO = None   # e.g. r'C:\path\to\FR Low solo.txt'
SUM_EXPORT = None   # e.g. r'C:\path\to\System Sum.txt'
PASSBAND   = (80.0, 2600.0)

if SUM_EXPORT:
    sm  = load_spl_export(SUM_EXPORT, freqs)
    off = float(np.median((sm - tgt)[(freqs>=300)&(freqs<=3000)]))
    tilt = octave_smooth(sm - (tgt+off), 1.0)     # broad only = tonal balance, not narrow wiggle
    print('\n=== SHARED voicing tilt (Sum vs target, ~1 oct; apply IDENTICALLY to both sides) ===')
    for f0 in [40,63,100,160,250,400,630,1000,1600,2500,4000,6300,10000,16000]:
        i = at(f0); print('   %6dHz  shared dev %+5.1f' % (f0, tilt[i]))

if LEFT_SOLO or RIGHT_SOLO:
    print('\n=== PER-SIDE corrective layer (each driver vs its OWN ~1/2 oct trend, %g-%g Hz) ==='
          % PASSBAND)
    side_bands = {}
    for label, path in [('LEFT', LEFT_SOLO), ('RIGHT', RIGHT_SOLO)]:
        if not path: continue
        m = load_spl_export(path, freqs)
        resid = m - octave_median(m, 1.0)        # local anomalies vs own median baseline (G runs ~1dB conservative)
        side_bands[label] = propose_cuts(freqs, resid, PASSBAND)
        print(' %-5s solo -> %d local cuts:' % (label, len(side_bands[label])))
        for F,Q,G in side_bands[label]:
            print('    F=%-8.1f Q=%-4.2f G=%+.1f' % (F,Q,G))
    if 'LEFT' in side_bands and 'RIGHT' in side_bands:
        # how different are the sides? bands within 1/6 oct = "agree" (could be shared)
        L, R = side_bands['LEFT'], side_bands['RIGHT']
        agree = sum(1 for f,_,_ in L if any(abs(np.log2(f/fr))<1/6 for fr,_,_ in R))
        print(' L/R agreement: %d of %d left bands have a right-side match within 1/6 oct'
              ' (low = sides genuinely differ -> keep per-side; high = mostly shareable)'
              % (agree, len(L)))

# ---------------------------------------------------------------------------
# RESEARCH-DERIVED REFINEMENTS (vetted subset, 2026-06-28). The rest of that
# digest was either corroboration of what's already here, hardware-uncertain, or
# (the "peer-review" personas + "SYSTEM RECORD" telemetry) fabricated -- ignored.

def erb_hz(fc):
    """Glasberg-Moore ERB bandwidth (Hz) of the auditory filter centred at fc."""
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)

def erb_smooth(freqs, y):
    """Variable smoothing matched to human cochlear bandwidth -- ~0.9 oct at 40 Hz,
    ~1/5 oct at 1 kHz, ~1/6 oct above. More perceptually honest than fixed 1/6-1/3 oct:
    it stops you 'fixing' narrow LF wiggles the ear integrates over, while keeping HF
    resolution where the ear is fussy. VERIFIED to give that octave-width profile."""
    dlog = np.log(logStep); out = np.empty_like(y)
    for i in range(len(y)):
        hb = max(1, int(round(np.log(1 + 0.5 * erb_hz(freqs[i]) / freqs[i]) / dlog)))
        out[i] = np.mean(y[max(0, i - hb):min(len(y), i + hb + 1)])
    return out

def spatial_consistency(arrays, volatile_db=2.5):
    """COHERENCE BLANKING adapted to a single-mic spatial grid -- the proper automated
    fix for the comb 'whack-a-mole'. Pass N position sweeps (same freq grid, level-aligned);
    returns (mean, std_across_positions, eqable_mask). Where std is high the feature MOVED
    between positions -> it's a position-dependent comb/null -> DO NOT EQ it (mask False).
    Only propose cuts where eqable_mask is True. NOTE: unproven on this car -- needs a real
    same-tune multi-position grid (never captured one); validate before trusting the mask."""
    M = np.vstack(arrays)
    return M.mean(0), M.std(0), (M.std(0) <= volatile_db)
