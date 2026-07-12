"""Optuna-based safe tuner search for REW TXT exports + Helix AFPX.

This runner is deliberately conservative about what it writes:
  - REW TXT input with optional phase/coherence columns;
  - PEQ added in free middle slots;
  - per-side front PEQ is allowed so L/R balance can be scored and corrected;
  - polarity/delay/APF writes only when the crossover ladder clears its gates;
  - no crossover, shelf, or level writes.

The optimizer searches extra filters on top of the supplied baseline tune, writes
ranked candidate AFPX files, and leaves all input files untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Keep each optimizer worker to one internal math-library thread. Without this,
# several Python workers can multiply into many OpenMP/BLAS threads, which can
# make Windows sluggish or fail to create new threads under load.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
import optuna
from scripts.make_measurement_manifest import build_manifest

from _make_v3 import (
    add_bands,
    afpx_roundtrip_lint,
    choose_free_slots,
    decode_afpx,
    encode_afpx,
    set_attr,
)
from _tunefit import (
    allpass_fil_str,
    audibility_score,
    audibility_weight,
    band_limited_impulse_delay,
    band_limited_delay_from_phase,
    cascade_db,
    erb_smooth,
    erb_hz,
    excess_gd_mask,
    gate_low_frequency_limit,
    headroom_report,
    interference_audit,
    LOGSTEP,
    phase_linearity_residual,
    peaking_db,
    ms_to_samples,
    optimize_allpass,
    polarity_delay_search,
    target_anchor_offset,
    tune_scorecard,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AFPX_DATA_ROOT", str(ROOT)))
DEFAULT_BASELINE = Path(os.environ.get("AFPX_BASELINE", str(DATA_ROOT / "baseline.afpx")))
DEFAULT_TARGET = Path(os.environ.get("AFPX_TARGET", str(ROOT / "ResoNix Target Curve 2026.txt")))
OBJECTIVE_PATH = ROOT / "objective_module" / "afpx_objective.py"


def _load_external_objective():
    if not OBJECTIVE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("afpx_objective", OBJECTIVE_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AFPX_OBJECTIVE = _load_external_objective()


def sync_external_objective(baseline: Path | None = None, target: Path | None = None,
                            level_calibration: Dict[str, float] | None = None) -> None:
    """Keep the imported objective pointed at this run's actual files.

    The worker scripts usually pass AFPX_* environment variables before Python
    starts. This also protects direct CLI runs where --baseline/--target are
    supplied after _optimizer has already imported objective_module.
    """
    if AFPX_OBJECTIVE is None:
        return
    changed = False
    if baseline is not None and hasattr(AFPX_OBJECTIVE, "BASELINE_AFPX"):
        new_baseline = Path(baseline)
        if Path(getattr(AFPX_OBJECTIVE, "BASELINE_AFPX")) != new_baseline:
            setattr(AFPX_OBJECTIVE, "BASELINE_AFPX", new_baseline)
            changed = True
    if target is not None and hasattr(AFPX_OBJECTIVE, "TARGET"):
        new_target = Path(target)
        if Path(getattr(AFPX_OBJECTIVE, "TARGET")) != new_target:
            setattr(AFPX_OBJECTIVE, "TARGET", new_target)
            changed = True
    if level_calibration is not None and hasattr(AFPX_OBJECTIVE, "LEVEL_CALIBRATION"):
        normalized = {str(key): float(value) for key, value in level_calibration.items()}
        if dict(getattr(AFPX_OBJECTIVE, "LEVEL_CALIBRATION", {})) != normalized:
            setattr(AFPX_OBJECTIVE, "LEVEL_CALIBRATION", normalized)
            changed = True
    if changed:
        for name, value in (("_F", None), ("_T", {}), ("_TGT", None), ("_NULL_MASK", None), ("_V5", None)):
            if hasattr(AFPX_OBJECTIVE, name):
                setattr(AFPX_OBJECTIVE, name, value)

HIGH_ALIASES_L = ("Front L High.txt", "Front L Tweeter.txt", "Front Left High.txt", "Front Left Tweeter.txt")
HIGH_ALIASES_R = ("Front R High.txt", "Front R Tweeter.txt", "Front Right High.txt", "Front Right Tweeter.txt")
MID_ALIASES_L = ("Front L Mid.txt", "Front L MID.txt", "Front L Midrange.txt", "Front Left Mid.txt")
MID_ALIASES_R = ("Front R Mid.txt", "Front R MID.txt", "Front R Midrange.txt", "Front Right Mid.txt")
LOW_ALIASES_L = ("Front L Low.txt", "Front L Midbass.txt", "Front L Mid Bass.txt", "Front Left Low.txt")
LOW_ALIASES_R = ("Front R Low.txt", "Front R Midbass.txt", "Front R Mid Bass.txt", "Front Right Low.txt")
MID_PAIR_ALIASES = ("Both Mids.txt", "Mids Together.txt", "Midrange Together.txt")
LOW_PAIR_ALIASES = ("Mid Bass Together.txt", "Both Midbass.txt", "Both Midbasses.txt", "Both Mid Bass.txt")
SUB_ALIASES = ("Sub.txt", "SUB.txt", "Subwoofer.txt")
SYSTEM_ALIASES = ("System Sum.txt", "SYSTEM SUM.txt")
HIGH_PAIR_ALIASES = ("Tweeters Together.txt", "Both Tweeters.txt")


def _has_any(data_root: Path, aliases: Tuple[str, ...]) -> bool:
    return any((data_root / alias).exists() for alias in aliases)


def detect_front_layout(data_root: Path = DATA_ROOT) -> str:
    has_mid = _has_any(data_root, MID_ALIASES_L) and _has_any(data_root, MID_ALIASES_R) and _has_any(data_root, MID_PAIR_ALIASES)
    has_low = _has_any(data_root, LOW_ALIASES_L) and _has_any(data_root, LOW_ALIASES_R) and _has_any(data_root, LOW_PAIR_ALIASES)
    return "3way" if has_mid and has_low else "2way"


FRONT_LAYOUT = detect_front_layout(DATA_ROOT)


def measurement_aliases_for_layout(layout: str) -> Dict[str, Tuple[str, ...]]:
    aliases = {
        "FL High": HIGH_ALIASES_L,
        "FR High": HIGH_ALIASES_R,
        "Sub": SUB_ALIASES,
        "System Sum": SYSTEM_ALIASES,
        "Tweeters Together": HIGH_PAIR_ALIASES,
    }
    if layout == "3way":
        aliases.update({
            "FL Mid": MID_ALIASES_L,
            "FR Mid": MID_ALIASES_R,
            "Mids Together": MID_PAIR_ALIASES,
            "FL Low": LOW_ALIASES_L,
            "FR Low": LOW_ALIASES_R,
            "Mid Bass Together": LOW_PAIR_ALIASES,
        })
    else:
        aliases.update({
            "FL Low": LOW_ALIASES_L + MID_ALIASES_L,
            "FR Low": LOW_ALIASES_R + MID_ALIASES_R,
            "Mid Bass Together": LOW_PAIR_ALIASES + MID_PAIR_ALIASES,
        })
    return aliases


MEASUREMENT_ALIASES = measurement_aliases_for_layout(FRONT_LAYOUT)


def resolve_measurement_files(data_root: Path = DATA_ROOT) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for name, aliases in MEASUREMENT_ALIASES.items():
        found = None
        for alias in aliases:
            p = data_root / alias
            if p.exists():
                found = p
                break
        files[name] = found if found is not None else data_root / aliases[0]
    return files


MEASUREMENT_FILES = resolve_measurement_files()


def groups_for_layout(layout: str, explore: bool = False) -> Dict[str, Dict[str, object]]:
    gain_sym = (-8.0, 0.0) if explore else (-6.0, 0.0)
    gain_side = (-8.0, 3.0) if explore else (-6.0, 3.0)
    q_sym = (0.5, 6.0) if explore else (0.5, 5.0)
    q_side = (0.5, 8.0) if explore else (0.5, 6.0)
    max_sym = 2 if explore else 1
    max_side = 3 if explore else 2
    groups: Dict[str, Dict[str, object]] = {
        "sub": {
            "channels": (6, 7),
            "branch": "sub",
            "range": (30.0, 90.0),
            "q_range": (0.5, 6.0 if explore else 5.0),
            "gain_range": (-8.0, 1.5) if explore else (-6.0, 0.0),
            "max_bands": 2,
        },
        "high_sym": {
            "channels": (0, 1),
            "branch": "high",
            "symmetric_tweeter": True,
            "range": (6000.0, 16000.0),
            "q_range": (0.5, 6.0 if explore else 4.5),
            "gain_range": gain_sym,
            "max_bands": 2 if explore else 1,
        },
        "fl_high": {
            "channels": (0,),
            "branch": "high",
            "trace": "FL High",
            "pair": "high",
            "side": "left",
            "range": (1800.0, 12000.0),
            "q_range": q_side,
            "gain_range": gain_side,
            "max_bands": max_side,
        },
        "fr_high": {
            "channels": (1,),
            "branch": "high",
            "trace": "FR High",
            "pair": "high",
            "side": "right",
            "range": (1800.0, 12000.0),
            "q_range": q_side,
            "gain_range": gain_side,
            "max_bands": max_side,
        },
    }
    if layout == "3way":
        groups.update({
            "mid_sym": {
                "channels": (2, 3),
                "branch": "mid",
                "range": (250.0, 3500.0),
                "q_range": q_sym,
                "gain_range": gain_sym,
                "max_bands": max_sym,
            },
            "fl_mid": {
                "channels": (2,),
                "branch": "mid",
                "trace": "FL Mid",
                "pair": "mid",
                "side": "left",
                "range": (250.0, 4000.0),
                "q_range": q_side,
                "gain_range": gain_side,
                "max_bands": max_side,
            },
            "fr_mid": {
                "channels": (3,),
                "branch": "mid",
                "trace": "FR Mid",
                "pair": "mid",
                "side": "right",
                "range": (250.0, 4000.0),
                "q_range": q_side,
                "gain_range": gain_side,
                "max_bands": max_side,
            },
            "low_sym": {
                "channels": (4, 5),
                "branch": "low",
                "range": (50.0, 500.0),
                "q_range": q_sym,
                "gain_range": gain_sym,
                "max_bands": max_sym,
            },
            "fl_low": {
                "channels": (4,),
                "branch": "low",
                "trace": "FL Low",
                "pair": "low",
                "side": "left",
                "range": (50.0, 600.0),
                "q_range": q_side,
                "gain_range": gain_side,
                "max_bands": max_side,
            },
            "fr_low": {
                "channels": (5,),
                "branch": "low",
                "trace": "FR Low",
                "pair": "low",
                "side": "right",
                "range": (50.0, 600.0),
                "q_range": q_side,
                "gain_range": gain_side,
                "max_bands": max_side,
            },
        })
    else:
        groups.update({
            "low_sym": {
                "channels": (2, 3),
                "branch": "low",
                "range": (70.0, 1800.0) if explore else (80.0, 1600.0),
                "q_range": q_sym,
                "gain_range": gain_sym,
                "max_bands": max_sym,
            },
            "fl_low": {
                "channels": (2,),
                "branch": "low",
                "trace": "FL Low",
                "pair": "low",
                "side": "left",
                "range": (70.0, 2200.0) if explore else (80.0, 2000.0),
                "q_range": q_side,
                "gain_range": gain_side,
                "max_bands": max_side,
            },
            "fr_low": {
                "channels": (3,),
                "branch": "low",
                "trace": "FR Low",
                "pair": "low",
                "side": "right",
                "range": (70.0, 2200.0) if explore else (80.0, 2000.0),
                "q_range": q_side,
                "gain_range": gain_side,
                "max_bands": max_side,
            },
        })
    return groups

GROUPS = groups_for_layout(FRONT_LAYOUT, explore=False)
SAFE_GROUPS = groups_for_layout(FRONT_LAYOUT, explore=False)
EXPLORE_GROUPS = groups_for_layout(FRONT_LAYOUT, explore=True)

if FRONT_LAYOUT == "3way":
    CH_TRACE = {
        0: "FL High",
        1: "FR High",
        2: "FL Mid",
        3: "FR Mid",
        4: "FL Low",
        5: "FR Low",
    }
else:
    CH_TRACE = {
        0: "FL High",
        1: "FR High",
        2: "FL Low",
        3: "FR Low",
    }

REPORT_FREQS = [
    31.5, 40, 50, 63, 80, 100, 125, 160, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 16000,
]

Band = Tuple[float, float, float]
GroupBands = Dict[str, List[Band]]
TraceMap = Dict[str, np.ndarray]
RichTrace = Dict[str, np.ndarray]
RichTraceMap = Dict[str, RichTrace]
ImpulseTrace = Dict[str, object]


def load_level_calibration(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Level calibration must be a JSON object of role/file -> dB offset")
    return {str(key): float(value) for key, value in payload.items()}


def calibration_offset(calibration: Dict[str, float], role: str, path: Path) -> float:
    for key in (role, path.name, path.stem, str(path)):
        if key in calibration:
            return float(calibration[key])
    return 0.0


def measurement_session_audit(manifest: Dict[str, object], calibration: Dict[str, float]) -> Dict[str, object]:
    metadata = dict(manifest.get("measurement_metadata", {}))
    signatures: Dict[str, Tuple[float | None, float | None]] = {}
    for role, raw in metadata.items():
        info = dict(raw)
        signatures[str(role)] = (
            None if info.get("source_volume") is None else float(info["source_volume"]),
            None if info.get("sweep_dbfs") is None else float(info["sweep_dbfs"]),
        )
    known = [sig for sig in signatures.values() if sig != (None, None)]
    reference = Counter(known).most_common(1)[0][0] if known else (None, None)
    outlier_roles = sorted(role for role, sig in signatures.items() if sig != (None, None) and sig != reference)
    unknown_roles = sorted(role for role, sig in signatures.items() if sig == (None, None))
    resolved = dict(manifest.get("resolved_roles", {}))
    missing_calibration = []
    for role in sorted(set(outlier_roles + unknown_roles)):
        path = Path(str(resolved.get(role, role)))
        if not any(key in calibration for key in (role, path.name, path.stem, str(path))):
            missing_calibration.append(role)
    tonal_valid = not manifest.get("measurements_missing") and not missing_calibration
    timing_references = list(dict(manifest.get("measurement_conditions", {})).get("timing_references", []))
    phase_valid = bool(tonal_valid and manifest.get("phase_available") and len(timing_references) == 1)
    warnings = list(manifest.get("warnings", []))
    if missing_calibration:
        warnings.append("uncalibrated_level_mismatch:" + ",".join(missing_calibration))
    if unknown_roles:
        warnings.append("source_level_metadata_missing:" + ",".join(unknown_roles))
    if len(timing_references) > 1:
        warnings.append("phase_writes_disabled_mixed_timing_references")
    elif not timing_references:
        warnings.append("phase_writes_disabled_timing_reference_missing")
    return {
        "tonal_valid": bool(tonal_valid),
        "phase_valid": phase_valid,
        "reference_level_signature": list(reference),
        "level_outlier_roles": outlier_roles,
        "unknown_level_roles": unknown_roles,
        "calibrated_roles": sorted(set(outlier_roles) - set(missing_calibration)),
        "missing_calibration_roles": missing_calibration,
        "timing_references": timing_references,
        "spatial_positions": sorted(dict(manifest.get("spatial_bundles", {}))),
        "warnings": sorted(set(warnings)),
    }


def prepare_measurement_session(baseline: Path, target: Path,
                                level_calibration_path: Path | None = None) -> Tuple[Dict[str, object], Dict[str, float]]:
    calibration = load_level_calibration(level_calibration_path)
    manifest = build_manifest(DATA_ROOT.resolve(), baseline, target)
    audit = measurement_session_audit(manifest, calibration)
    if not audit["tonal_valid"]:
        missing = ", ".join(audit["missing_calibration_roles"])
        raise SystemExit(
            "Measurement session gate failed: missing or inconsistent source/sweep levels require explicit dB calibration"
            + (" for " + missing if missing else "")
        )
    return {"manifest": manifest, "audit": audit}, calibration


def load_rew_export(path: Path) -> RichTrace:
    """Load REW-style text exports with optional phase/coherence/position columns.

    Accepted numeric layouts:
      freq, spl
      freq, spl, phase
      freq, spl, phase, coherence
      freq, spl, phase, coherence, position_id
    """
    cols: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("*") or s[0].isalpha():
            continue
        parts = s.replace(",", " ").split()
        row: List[float] = []
        for part in parts[:5]:
            try:
                row.append(float(part))
            except ValueError:
                break
        if len(row) >= 2:
            cols.append(row)
    if len(cols) < 16:
        raise ValueError(f"No usable frequency/SPL rows found in {path}")

    ncols = max(len(row) for row in cols)
    out = {
        "freq": np.asarray([row[0] for row in cols], dtype=float),
        "spl": np.asarray([row[1] for row in cols], dtype=float),
    }
    if ncols >= 3 and all(len(row) >= 3 for row in cols):
        out["phase"] = np.asarray([row[2] for row in cols], dtype=float)
    if ncols >= 4 and all(len(row) >= 4 for row in cols):
        out["coherence"] = np.asarray([row[3] for row in cols], dtype=float)
    if ncols >= 5 and all(len(row) >= 5 for row in cols):
        out["position_id"] = np.asarray([row[4] for row in cols], dtype=float)
    return out


def interp_rich_trace(trace: RichTrace, freqs: np.ndarray) -> RichTrace:
    src_f = trace["freq"]
    log_src = np.log10(src_f)
    log_dst = np.log10(freqs)
    out: RichTrace = {"freq": freqs.copy()}
    for key in ("spl", "phase", "coherence", "position_id"):
        if key in trace:
            values = trace[key]
            if key == "phase":
                # Interpolating wrapped degrees directly invents phase values at
                # every -180/+180 crossing. Unwrap first; complex conversion and
                # delay fitting both accept the resulting continuous degrees.
                values = np.rad2deg(np.unwrap(np.deg2rad(values)))
            out[key] = np.interp(log_dst, log_src, values)
    return out


def optimization_frequency_grid(freqs: np.ndarray, points_per_octave: int = 96) -> np.ndarray:
    """Normalize dense/linear REW exports to the scorer's logarithmic grid."""
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
    first = int(math.ceil(log_f[0] * points_per_octave))
    last = int(math.floor(log_f[-1] * points_per_octave))
    if last <= first:
        return freqs
    return 2.0 ** (np.arange(first, last + 1, dtype=float) / float(points_per_octave))


def load_txt_export(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    trace = load_rew_export(path)
    return trace["freq"], trace["spl"]


def load_measurements(level_calibration: Dict[str, float] | None = None) -> Tuple[np.ndarray, TraceMap, RichTraceMap]:
    level_calibration = level_calibration or {}
    missing = [str(p) for p in MEASUREMENT_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing measurement file(s):\n  " + "\n  ".join(missing))

    base_trace = load_rew_export(MEASUREMENT_FILES["System Sum"])
    base_f = optimization_frequency_grid(base_trace["freq"])
    base_trace["spl"] = base_trace["spl"] + calibration_offset(
        level_calibration, "System Sum", MEASUREMENT_FILES["System Sum"]
    )
    base_spl = np.interp(np.log10(base_f), np.log10(base_trace["freq"]), base_trace["spl"])
    traces: TraceMap = {"System Sum": base_spl}
    rich: RichTraceMap = {"System Sum": interp_rich_trace(base_trace, base_f)}
    log_base = np.log10(base_f)
    for name, path in MEASUREMENT_FILES.items():
        if name == "System Sum":
            continue
        measurement = load_rew_export(path)
        measurement["spl"] = measurement["spl"] + calibration_offset(level_calibration, name, path)
        traces[name] = np.interp(log_base, np.log10(measurement["freq"]), measurement["spl"])
        rich[name] = interp_rich_trace(measurement, base_f)
    return base_f, traces, rich


def _read_pcm_wav(path: Path) -> ImpulseTrace:
    """Read a mono/stereo PCM WAV impulse without adding a runtime dependency."""
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = float(wav.getframerate())
            frames = wav.readframes(wav.getnframes())
    except wave.Error:
        # REW may export IEEE-float WAV, which the stdlib wave module rejects.
        from scipy.io import wavfile
        sample_rate_raw, raw = wavfile.read(path)
        data = np.asarray(raw)
        if data.ndim > 1:
            data = data[:, 0]
        if np.issubdtype(data.dtype, np.integer):
            scale = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
            data = data.astype(float) / scale
        else:
            data = data.astype(float)
        return {"samples": data, "sample_rate": float(sample_rate_raw), "path": str(path)}
    if width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(float) - 128.0
        scale = 128.0
    elif width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(float)
        scale = float(2 ** 15)
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        data = np.where(values & 0x800000, values - 0x1000000, values).astype(float)
        scale = float(2 ** 23)
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(float)
        scale = float(2 ** 31)
    else:
        raise ValueError(f"Unsupported WAV sample width {width} in {path}")
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    return {"samples": data / scale, "sample_rate": sample_rate, "path": str(path)}


def _read_text_impulse(path: Path) -> ImpulseTrace:
    rows: List[Tuple[float, float]] = []
    header: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.replace(",", " ").split()
        try:
            row = (float(parts[0]), float(parts[1]))
        except (IndexError, ValueError):
            header.append(s.lower())
            continue
        rows.append(row)
    if len(rows) < 32:
        raise ValueError(f"No usable time/amplitude impulse rows found in {path}")
    time_axis = np.asarray([row[0] for row in rows], dtype=float)
    samples = np.asarray([row[1] for row in rows], dtype=float)
    steps = np.diff(time_axis)
    step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
    if step <= 0.0:
        raise ValueError(f"Impulse time axis is not increasing in {path}")
    header_text = " ".join(header[:20])
    milliseconds = "millisecond" in header_text or re.search(r"\bms\b", header_text) is not None
    sample_rate = (1000.0 if milliseconds or step >= 0.001 else 1.0) / step
    if not 4000.0 <= sample_rate <= 768000.0:
        raise ValueError(f"Implausible impulse sample rate {sample_rate:.1f} Hz in {path}")
    return {"samples": samples, "sample_rate": sample_rate, "path": str(path)}


def _impulse_candidates(measurement_path: Path, root: Path) -> List[Path]:
    stem = measurement_path.stem
    names = (
        f"{stem}.wav", f"{stem} Impulse.wav", f"{stem} IR.wav",
        f"{stem} Impulse.txt", f"{stem} IR.txt",
    )
    folders = (root, root / "impulses", root / "Impulse", root / "IR")
    return [folder / name for folder in folders for name in names]


def load_optional_impulses(root: Path | None = None) -> Dict[str, ImpulseTrace]:
    """Load companion impulse WAV/text files when supplied; magnitude TXT is untouched."""
    impulse_root = Path(root) if root is not None else DATA_ROOT
    found: Dict[str, ImpulseTrace] = {}
    for trace_name, measurement_path in MEASUREMENT_FILES.items():
        for path in _impulse_candidates(measurement_path, impulse_root):
            if not path.exists():
                continue
            try:
                found[trace_name] = _read_pcm_wav(path) if path.suffix.lower() == ".wav" else _read_text_impulse(path)
            except (ValueError, wave.Error):
                continue
            break
    return found


def load_target(path: Path, freqs: np.ndarray) -> np.ndarray:
    tf: List[float] = []
    ts: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("*") or s[0].isalpha():
            continue
        parts = s.replace(",", " ").split()
        try:
            tf.append(float(parts[0]))
            ts.append(float(parts[1]))
        except (IndexError, ValueError):
            continue
    if len(tf) < 4:
        raise ValueError(f"No usable target rows found in {path}")
    return np.interp(np.log10(freqs), np.log10(np.asarray(tf)), np.asarray(ts))


def power_sum_db(curves: Iterable[np.ndarray]) -> np.ndarray:
    total = None
    for curve in curves:
        p = 10.0 ** (curve / 10.0)
        total = p if total is None else total + p
    if total is None:
        raise ValueError("power_sum_db needs at least one curve")
    return 10.0 * np.log10(np.maximum(total, 1e-30))


def q_cap_for_band(F: float, G: float) -> float:
    """Single-position RTA/MMM data should not create needle filters."""
    cap = 2.5
    if G > 0.0:
        cap = min(cap, 2.0)
    return cap


def rounded_band(F: float, Q: float, G: float) -> Band | None:
    # Hardware/UI-friendly values. Gains are 0.25 dB steps; filters below
    # roughly half a dB are not worth burning a slot in safe mode.
    F = float(F)
    G = float(G)
    if F > 10000.0 and G > 0.0:
        return None
    Q = min(float(Q), q_cap_for_band(F, G))
    # The search is intentionally shallow by default. The objective still knows
    # how to punish deeper hand-supplied bands unless the solo trace supports it.
    G = max(G, -4.0)
    G = round(float(G) * 4.0) / 4.0
    if abs(G) < 0.5:
        return None
    F = round(F, 1)
    Q = round(Q, 2)
    return (F, Q, G)


def baseline_band_sets() -> List[List[Band]]:
    if AFPX_OBJECTIVE is not None and hasattr(AFPX_OBJECTIVE, "baseline_band_sets"):
        return AFPX_OBJECTIVE.baseline_band_sets()
    return [[] for _ in range(8)]


def groups_to_band_sets(groups: GroupBands) -> List[List[Band]]:
    band_sets = [list(bands) for bands in baseline_band_sets()]
    while len(band_sets) < 8:
        band_sets.append([])
    for group, bands in groups.items():
        if not bands:
            continue
        for channel in GROUPS[group]["channels"]:
            band_sets[channel].extend(bands)
            band_sets[channel].sort(key=lambda b: b[0])
    return band_sets


def suggest_group_bands(trial: optuna.Trial, group: str) -> List[Band]:
    cfg = GROUPS[group]
    out: List[Band] = []
    for idx in range(cfg["max_bands"]):
        if not trial.suggest_categorical(f"{group}_{idx}_on", [False, True]):
            continue
        F = trial.suggest_float(
            f"{group}_{idx}_freq",
            cfg["range"][0],
            cfg["range"][1],
            log=True,
        )
        Q = trial.suggest_float(f"{group}_{idx}_q", cfg["q_range"][0], cfg["q_range"][1])
        G = trial.suggest_float(
            f"{group}_{idx}_gain",
            cfg["gain_range"][0],
            cfg["gain_range"][1],
        )
        band = rounded_band(F, Q, G)
        if band is not None:
            out.append(band)
    out.sort(key=lambda b: b[0])
    return out


def trial_bands(trial: optuna.Trial) -> GroupBands:
    return {group: suggest_group_bands(trial, group) for group in GROUPS}


def bands_signature(groups: GroupBands) -> Tuple[Tuple[str, Tuple[Band, ...]], ...]:
    return tuple((name, tuple(groups.get(name, []))) for name in sorted(GROUPS))


def duplicate_penalty(groups: GroupBands) -> float:
    penalty = 0.0
    for bands in groups.values():
        for i, a in enumerate(bands):
            for b in bands[i + 1:]:
                if abs(math.log2(a[0] / b[0])) < 1 / 6:
                    penalty += 0.10
    return penalty


def filter_cost(groups: GroupBands) -> float:
    cost = 0.0
    for group, bands in groups.items():
        for F, Q, G in bands:
            cost += 0.050
            cost += 0.012 * abs(G) * Q
            cost += 0.120 * max(0.0, -G - 4.0)
            cost += 0.090 * max(0.0, Q - 2.5)
            if G > 0.0:
                cost += 0.030 * G * Q
            # Extra skepticism for narrow upper-mid/treble filters from one-seat data.
            if F >= 1000.0 and Q > 2.0:
                cost += 0.035 * (Q - 2.0)
    return cost + duplicate_penalty(groups)


def make_fast_smoother(freqs: np.ndarray):
    """Same rectangular ERB window as _tunefit.erb_smooth, but pre-indexed."""
    dlog = np.log(LOGSTEP)
    starts = []
    ends = []
    for i, f in enumerate(freqs):
        hb = max(1, int(round(np.log(1 + 0.5 * erb_hz(float(f)) / float(f)) / dlog)))
        starts.append(max(0, i - hb))
        ends.append(min(len(freqs), i + hb + 1))
    starts_a = np.asarray(starts, dtype=int)
    ends_a = np.asarray(ends, dtype=int)
    widths = (ends_a - starts_a).astype(float)

    def smooth(y: np.ndarray) -> np.ndarray:
        cs = np.empty(len(y) + 1, dtype=float)
        cs[0] = 0.0
        np.cumsum(y, out=cs[1:])
        return (cs[ends_a] - cs[starts_a]) / widths

    return smooth


def make_fast_audibility(freqs: np.ndarray, band: Tuple[float, float] = (20.0, 16000.0)):
    smooth = make_fast_smoother(freqs)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    w = audibility_weight(freqs)[sel]
    den = float(np.sum(w ** 2))

    def score(dev_db: np.ndarray) -> Tuple[float, np.ndarray]:
        sm = smooth(dev_db)
        if den <= 1e-12 or not np.any(sel):
            return float("inf"), sm
        return float(np.sqrt(np.sum((sm[sel] * w) ** 2) / den)), sm

    return score


if FRONT_LAYOUT == "3way":
    PAIR_DEFS = {
        "low": {
            "left": "FL Low",
            "right": "FR Low",
            "together": "Mid Bass Together",
            "branch_band": (50.0, 700.0),
            "balance_band": (80.0, 500.0),
        },
        "mid": {
            "left": "FL Mid",
            "right": "FR Mid",
            "together": "Mids Together",
            "branch_band": (250.0, 4500.0),
            "balance_band": (300.0, 3500.0),
        },
        "high": {
            "left": "FL High",
            "right": "FR High",
            "together": "Tweeters Together",
            "branch_band": (1800.0, 16000.0),
            "balance_band": (2500.0, 12000.0),
        },
    }
else:
    PAIR_DEFS = {
        "low": {
            "left": "FL Low",
            "right": "FR Low",
            "together": "Mid Bass Together",
            "branch_band": (80.0, 2200.0),
            "balance_band": (200.0, 2200.0),
        },
        "high": {
            "left": "FL High",
            "right": "FR High",
            "together": "Tweeters Together",
            "branch_band": (1800.0, 16000.0),
            "balance_band": (1800.0, 8000.0),
        },
    }


def weighted_rms(values: np.ndarray, weights: np.ndarray, sel: np.ndarray) -> float:
    if not np.any(sel):
        return 0.0
    w = np.asarray(weights, dtype=float)[sel]
    den = float(np.sum(w ** 2))
    if den <= 1e-12:
        return 0.0
    v = np.asarray(values, dtype=float)[sel]
    return float(np.sqrt(np.sum((v * w) ** 2) / den))


def channel_deltas(freqs: np.ndarray, groups: GroupBands) -> Dict[int, np.ndarray]:
    deltas: Dict[int, np.ndarray] = {}
    for group, bands in groups.items():
        if not bands:
            continue
        cfg = GROUPS.get(group)
        if not cfg:
            continue
        delta = cascade_db(freqs, bands)
        for channel in cfg["channels"]:
            if channel not in deltas:
                deltas[channel] = np.zeros_like(freqs)
            deltas[channel] = deltas[channel] + delta
    return deltas


def predict_traces(freqs: np.ndarray, traces: TraceMap, groups: GroupBands) -> TraceMap:
    pred: TraceMap = dict(traces)

    deltas = channel_deltas(freqs, groups)
    for channel, trace_name in CH_TRACE.items():
        pred[trace_name] = traces[trace_name] + deltas.get(channel, np.zeros_like(freqs))
    sub_delta = np.zeros_like(freqs)
    sub_count = 0
    for channel in (6, 7):
        if channel in deltas:
            sub_delta = sub_delta + deltas[channel]
            sub_count += 1
    if sub_count:
        # The sub trace is a combined branch capture, and we only write shared
        # sub filters. Average the duplicate channel deltas to model one shared
        # acoustic change instead of double-counting it.
        sub_delta = sub_delta / sub_count
    pred["Sub"] = traces["Sub"] + sub_delta

    branch_outputs = []
    for pair in PAIR_DEFS.values():
        pair_power = power_sum_db([traces[pair["left"]], traces[pair["right"]]])
        pair_residual = traces[pair["together"]] - pair_power
        pred[pair["together"]] = power_sum_db([pred[pair["left"]], pred[pair["right"]]]) + pair_residual
        branch_outputs.append(pred[pair["together"]])

    branch_power = power_sum_db([traces[pair["together"]] for pair in PAIR_DEFS.values()] + [traces["Sub"]])
    system_residual = traces["System Sum"] - branch_power
    pred["System Sum"] = power_sum_db(branch_outputs + [pred["Sub"]]) + system_residual
    return pred


def null_masks(freqs: np.ndarray, traces: TraceMap) -> Dict[str, np.ndarray]:
    masks = {name: np.zeros_like(freqs, dtype=bool) for name in PAIR_DEFS}
    for name, pair in PAIR_DEFS.items():
        try:
            masks[name] = interference_audit(
                freqs, traces[pair["left"]], traces[pair["right"]], traces[pair["together"]]
            )[3]
        except Exception:
            pass
    masks["system"] = np.zeros_like(freqs, dtype=bool)
    for name in PAIR_DEFS:
        masks["system"] |= masks[name]
    return masks


def pair_sum_validation(freqs: np.ndarray, traces: TraceMap, threshold: float = 2.5) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, pair in PAIR_DEFS.items():
        psum = power_sum_db([traces[pair["left"]], traces[pair["right"]]])
        diff = erb_smooth(freqs, traces[pair["together"]] - psum)
        lo, hi = pair["branch_band"]
        sel = (freqs >= lo) & (freqs <= hi)
        rms = float(np.sqrt(np.mean(diff[sel] ** 2))) if np.any(sel) else float("inf")
        rows.append({
            "pair": name,
            "together": pair["together"],
            "rms_db": round(rms, 2),
            "pass": bool(rms <= threshold),
            "threshold_db": float(threshold),
        })
    return rows


def _wrap_deg(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def _complex_from_rich(trace: RichTrace) -> np.ndarray | None:
    if "phase" not in trace:
        return None
    mag = 10.0 ** (trace["spl"] / 20.0)
    return mag * np.exp(1j * np.deg2rad(trace["phase"]))


def _band_rms(values: np.ndarray, sel: np.ndarray) -> float:
    ok = sel & np.isfinite(values)
    if not np.any(ok):
        return float("inf")
    return float(np.sqrt(np.mean(values[ok] ** 2)))


def _grade_from_rms(rms: float, good: float, medium: float) -> str:
    if not np.isfinite(rms):
        return "missing"
    if rms <= good:
        return "high"
    if rms <= medium:
        return "medium"
    return "low"


def optional_measurement(data_root: Path, aliases: Tuple[str, ...], freqs: np.ndarray) -> RichTrace | None:
    for alias in aliases:
        path = data_root / alias
        if path.exists():
            return interp_rich_trace(load_rew_export(path), freqs)
    return None


def crossover_specs(freqs: np.ndarray, rich: RichTraceMap) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    sub_front = optional_measurement(DATA_ROOT, (
        "Sub and Mids.txt", "Sub + Mids.txt", "Sub + Front Mids.txt",
        "Sub + Front L/R Midbass.txt", "Mids and Sub.txt", "Midbass and Sub.txt",
    ), freqs)
    if "Sub" in rich and "Mid Bass Together" in rich:
        specs.append({
            "name": "sub_to_front",
            "label": "Sub to front midbass",
            "a": "Sub",
            "b": "Mid Bass Together",
            "together": sub_front,
            "band": (50.0, 120.0),
        })
    if FRONT_LAYOUT == "3way":
        mid_left, mid_right = "FL Mid", "FR Mid"
    else:
        mid_left, mid_right = "FL Low", "FR Low"
    for side, mid, high, aliases in (
        ("left", mid_left, "FL High", (
            "Front L Mid Tweeter Together.txt", "Front L Mid+Tweeter.txt", "Front L Mid + Tweeter.txt",
            "Front Left Mid + Tweeter.txt", "Front L Together.txt",
        )),
        ("right", mid_right, "FR High", (
            "Front R Mid Tweeter Together.txt", "Front R Mid+Tweeter.txt", "Front R Mid + Tweeter.txt",
            "Front Right Mid + Tweeter.txt", "Front R Together.txt",
        )),
    ):
        if mid in rich and high in rich:
            specs.append({
                "name": f"{side}_mid_to_tweeter",
                "label": f"{side.title()} mid to tweeter",
                "a": mid,
                "b": high,
                "together": optional_measurement(DATA_ROOT, aliases, freqs),
                "band": (1800.0, 4500.0),
            })
    return specs


def _impulse_pair_result(a: ImpulseTrace | None, b: ImpulseTrace | None,
                         band: Tuple[float, float]) -> Dict[str, object]:
    if a is None or b is None:
        return {"available": False, "usable": False}
    fs_a = float(a["sample_rate"])
    fs_b = float(b["sample_rate"])
    if abs(fs_a - fs_b) > max(1.0, 0.001 * fs_a):
        return {
            "available": True,
            "usable": False,
            "warning": f"impulse sample rates differ ({fs_a:.1f} vs {fs_b:.1f} Hz)",
        }
    result = band_limited_impulse_delay(
        np.asarray(a["samples"], dtype=float),
        np.asarray(b["samples"], dtype=float),
        fs_a,
        band,
    )
    return {
        "available": True,
        "usable": bool(result.get("usable")),
        "arrival_delay_ms_B": result.get("delay_ms", 0.0),
        "correction_delay_ms_B": round(-float(result.get("delay_ms", 0.0)), 4),
        "polarity": result.get("polarity", "unknown"),
        "corr_norm": result.get("corr_norm", 0.0),
        "sample_rate": fs_a,
        "a_file": a.get("path", ""),
        "b_file": b.get("path", ""),
    }


def crossover_phase_diagnostics(freqs: np.ndarray, traces: TraceMap, rich: RichTraceMap,
                                impulse_root: Path | None = None) -> List[Dict[str, object]]:
    impulses = load_optional_impulses(impulse_root)
    rows: List[Dict[str, object]] = []
    for spec in crossover_specs(freqs, rich):
        a_name = str(spec["a"])
        b_name = str(spec["b"])
        band = tuple(spec["band"])  # type: ignore[arg-type]
        sel = (freqs >= band[0]) & (freqs <= band[1])
        a = rich[a_name]
        b = rich[b_name]
        phase_available = "phase" in a and "phase" in b
        coherence = None
        if "coherence" in a and "coherence" in b:
            coherence = np.minimum(a["coherence"], b["coherence"])

        row: Dict[str, object] = {
            "name": spec["name"],
            "label": spec["label"],
            "band": f"{band[0]:.0f}-{band[1]:.0f} Hz",
            "phase_available": phase_available,
        }
        impulse = _impulse_pair_result(impulses.get(a_name), impulses.get(b_name), band)
        row["impulse"] = impulse
        if phase_available:
            phase_diff = _wrap_deg(a["phase"] - b["phase"])
            delay = band_limited_delay_from_phase(freqs, phase_diff, band, coherence=coherence)
            polarity_cos = float(np.median(np.cos(np.deg2rad(phase_diff[sel])))) if np.any(sel) else 0.0
            phase_stability = phase_linearity_residual(freqs, phase_diff, band)
            try:
                _agd, a_eqable = excess_gd_mask(freqs, a["spl"], a["phase"])
                _bgd, b_eqable = excess_gd_mask(freqs, b["spl"], b["phase"])
                eqable_pct = 100.0 * np.count_nonzero(a_eqable & b_eqable & sel) / max(np.count_nonzero(sel), 1)
                egd_grade = "high" if eqable_pct >= 85.0 else "medium" if eqable_pct >= 60.0 else "low"
            except Exception:
                eqable_pct = float("nan")
                egd_grade = "missing"
            row.update({
                "relative_delay_ms": delay["delay_ms"],
                "delay_fit_rms_deg": delay["rms_phase_err_deg"],
                "delay_usable": delay["usable"],
                "polarity": "same" if polarity_cos >= 0 else "inverted/rotated",
                "polarity_cos": round(polarity_cos, 3),
                "phase_stability": phase_stability["grade"],
                "phase_rms_residual_deg": phase_stability["rms_residual_deg"],
                "excess_gd_stability": egd_grade,
                "minimum_phase_pct": "" if not np.isfinite(eqable_pct) else round(float(eqable_pct), 1),
            })
        else:
            row.update({
                "relative_delay_ms": "",
                "delay_fit_rms_deg": "",
                "delay_usable": False,
                "polarity": "phase missing",
                "phase_stability": "missing",
                "phase_rms_residual_deg": "",
                "excess_gd_stability": "missing",
                "minimum_phase_pct": "",
            })

        together = spec.get("together")
        try:
            if together is None:
                raise ValueError("no measured together trace")
            audit = interference_audit(freqs, traces[a_name], traces[b_name], together["spl"])
            interf = audit[2]
            null_pct = 100.0 * np.count_nonzero(audit[3] & sel) / max(np.count_nonzero(sel), 1)
            row["summation_quality"] = _grade_from_rms(_band_rms(np.minimum(interf, 0.0), sel), 1.5, 3.0)
            row["destructive_pct"] = round(float(null_pct), 1)
        except Exception:
            row["summation_quality"] = "missing"
            row["destructive_pct"] = ""

        ca = _complex_from_rich(a)
        cb = _complex_from_rich(b)
        if together is not None and ca is not None and cb is not None:
            predicted = 20.0 * np.log10(np.abs(ca + cb) + 1e-12)
            err = erb_smooth(freqs, together["spl"] - predicted)
            rms = _band_rms(err, sel)
            row["predicted_sum_rms_db"] = round(rms, 2)
            row["predicted_sum_match"] = _grade_from_rms(rms, 1.5, 2.5)
        elif together is not None:
            psum = power_sum_db([traces[a_name], traces[b_name]])
            err = erb_smooth(freqs, together["spl"] - psum)
            rms = _band_rms(err, sel)
            row["predicted_sum_rms_db"] = round(rms, 2)
            row["predicted_sum_match"] = _grade_from_rms(rms, 2.0, 3.5)
        else:
            row["predicted_sum_rms_db"] = ""
            row["predicted_sum_match"] = "missing together trace"

        # Active crossover ladder: polarity and delay first. APF only earns a
        # search when those cheaper corrections leave a meaningful residual.
        ladder: Dict[str, object] = {
            "available": False,
            "write_eligible": False,
            "source": "none",
            "reason": "phase and usable impulse timing are unavailable",
        }
        reference_valid = row.get("predicted_sum_match") in ("high", "medium")
        if ca is not None and cb is not None:
            max_delay_ms = 3.0 if spec["name"] == "sub_to_front" else 0.5
            pd = polarity_delay_search(freqs, ca, cb, band, max_delay_ms=max_delay_ms)
            gain = float(pd["score_before"]) - float(pd["score_after"])
            meaningful = (
                float(pd["score_before"]) >= 0.75
                and gain >= 0.25
                and float(pd["improvement_pct"]) >= 10.0
            )
            impulse_delay_agreement = None
            impulse_polarity_agreement = None
            if impulse.get("usable"):
                impulse_delay_agreement = abs(
                    float(pd["delay_ms_B"]) - float(impulse["correction_delay_ms_B"])
                ) <= 0.15
                impulse_polarity_agreement = bool(pd["polarity_flip_B"]) == (impulse.get("polarity") == "inverted")
            impulse_consistent = impulse_delay_agreement is not False and impulse_polarity_agreement is not False
            polarity_confident = (
                not bool(pd["polarity_flip_B"])
                or float(pd["improvement_pct"]) >= 25.0
                or impulse_polarity_agreement is True
            )
            ladder = {
                "available": True,
                "source": "complex_phase",
                "reference_valid": reference_valid,
                "score_before": pd["score_before"],
                "score_after_polarity_delay": pd["score_after"],
                "improvement_pct": pd["improvement_pct"],
                "polarity_flip_B": bool(pd["polarity_flip_B"]),
                "correction_delay_ms_B": float(pd["delay_ms_B"]),
                "residual_needs_apf": bool(pd["residual_needs_apf"]),
                "impulse_delay_agreement": impulse_delay_agreement,
                "impulse_polarity_agreement": impulse_polarity_agreement,
                "write_eligible": bool(reference_valid and meaningful and impulse_consistent and polarity_confident),
                "reason": (
                    "measured together trace supports the complex prediction"
                    if reference_valid and meaningful and impulse_consistent and polarity_confident else
                    "reference agreement, predicted improvement, polarity confidence, or impulse agreement did not clear the write gate"
                ),
            }
            if ladder["write_eligible"] and pd["residual_needs_apf"]:
                sign = -1.0 if pd["polarity_flip_B"] else 1.0
                b_after = sign * cb * np.exp(-1j * 2.0 * np.pi * freqs * float(pd["delay_ms_B"]) / 1000.0)
                apf_options = [
                    optimize_allpass(freqs, ca, b_after, band, apply_to=side, gd_penalty=0.5)
                    for side in ("A", "B")
                ]
                apf = min(apf_options, key=lambda item: float(item["selection_score_after"]))
                if float(apf["improvement_pct"]) >= 10.0 and float(apf["selection_score_after"]) < float(apf["score_before"]):
                    ladder["apf"] = apf
                    ladder["score_after_apf"] = apf["selection_score_after"]
                    ladder["apf_write_eligible"] = spec["name"] == "sub_to_front"
                    if not ladder["apf_write_eligible"]:
                        ladder["apf_note"] = "one-sided APF above 1 kHz is report-only; verify live before keeping"
                else:
                    ladder["residual_needs_apf"] = False
            if not ladder["write_eligible"] and not reference_valid and impulse.get("usable"):
                corr = float(impulse.get("corr_norm", 0.0))
                destructive = float(row.get("destructive_pct") or 0.0)
                impulse_polarity_supported = impulse.get("polarity") != "inverted" or corr >= 0.85
                impulse_delay_limit = 3.0 if spec["name"] == "sub_to_front" else 0.5
                impulse_delay_supported = abs(float(impulse.get("correction_delay_ms_B", 0.0))) <= impulse_delay_limit
                impulse_eligible = (
                    corr >= 0.75
                    and destructive >= 10.0
                    and row.get("summation_quality") in ("low", "medium")
                    and impulse_polarity_supported
                    and impulse_delay_supported
                )
                if impulse_eligible:
                    ladder = {
                        "available": True,
                        "source": "band_limited_impulse_fallback",
                        "reference_valid": False,
                        "polarity_flip_B": bool(impulse.get("polarity") == "inverted" and corr >= 0.85),
                        "correction_delay_ms_B": float(impulse.get("correction_delay_ms_B", 0.0)),
                        "residual_needs_apf": False,
                        "write_eligible": True,
                        "reason": "invalid complex reference replaced by strong impulse evidence plus measured destructive summation",
                    }
        elif impulse.get("usable"):
            corr = float(impulse.get("corr_norm", 0.0))
            destructive = float(row.get("destructive_pct") or 0.0)
            impulse_polarity_supported = impulse.get("polarity") != "inverted" or corr >= 0.85
            impulse_delay_limit = 3.0 if spec["name"] == "sub_to_front" else 0.5
            impulse_delay_supported = abs(float(impulse.get("correction_delay_ms_B", 0.0))) <= impulse_delay_limit
            eligible = (
                corr >= 0.75
                and destructive >= 10.0
                and row.get("summation_quality") in ("low", "medium")
                and impulse_polarity_supported
                and impulse_delay_supported
            )
            ladder = {
                "available": True,
                "source": "band_limited_impulse",
                "reference_valid": False,
                "polarity_flip_B": bool(impulse.get("polarity") == "inverted" and corr >= 0.85),
                "correction_delay_ms_B": float(impulse.get("correction_delay_ms_B", 0.0)),
                "residual_needs_apf": False,
                "write_eligible": bool(eligible),
                "reason": (
                    "strong impulse correlation plus measured destructive summation"
                    if eligible else "impulse evidence did not clear the conservative write gate"
                ),
            }
        row["crossover_ladder"] = ladder
        rows.append(row)
    return rows


def phase_diagnostic_fingerprint(measurement_session: Dict[str, object],
                                 impulse_root: Path | None = None) -> str:
    manifest = dict(measurement_session.get("manifest", {}))
    paths = [Path(value) for value in dict(manifest.get("resolved_roles", {})).values()]
    paths.extend(Path(value) for value in dict(manifest.get("impulse_files", {})).values())
    if impulse_root is not None:
        paths.extend(path for path in Path(impulse_root).rglob("*") if path.is_file())
    records = []
    for path in sorted(set(paths), key=lambda value: str(value).lower()):
        try:
            stat = path.stat()
            records.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
        except FileNotFoundError:
            records.append([str(path), None, None])
    payload = {
        "version": 1,
        "layout": manifest.get("detected_layout"),
        "paths": records,
        "audit": measurement_session.get("audit", {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cached_crossover_phase_diagnostics(cache_path: Path | None, freqs: np.ndarray,
                                       traces: TraceMap, rich: RichTraceMap,
                                       measurement_session: Dict[str, object],
                                       impulse_root: Path | None = None) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    fingerprint = phase_diagnostic_fingerprint(measurement_session, impulse_root)
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint:
                return list(payload.get("rows", [])), {
                    "source": "cache", "fingerprint": fingerprint, "path": str(path),
                }
        except (OSError, ValueError, TypeError):
            pass
    rows = crossover_phase_diagnostics(freqs, traces, rich, impulse_root)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "fingerprint": fingerprint, "rows": rows}
        tmp = path.with_suffix(path.suffix + ".tmp." + str(os.getpid()))
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    return rows, {"source": "computed", "fingerprint": fingerprint, "path": str(path or "")}


def apply_session_phase_validity(crossover_rows: List[Dict[str, object]],
                                 session_audit: Dict[str, object]) -> None:
    valid = bool(session_audit.get("phase_valid"))
    for row in crossover_rows:
        row["session_phase_valid"] = valid
        if valid:
            continue
        ladder = row.get("crossover_ladder")
        if isinstance(ladder, dict):
            ladder["write_eligible"] = False
            ladder["reason"] = "phase session invalid: " + "; ".join(
                str(item) for item in session_audit.get("warnings", [])
            )


def trace_channels(trace_name: str) -> Tuple[int, ...]:
    if trace_name == "Sub":
        return (6, 7)
    if trace_name == "Mid Bass Together":
        if FRONT_LAYOUT == "3way":
            return (4, 5)
        return (2, 3)
    return tuple(ch for ch, name in CH_TRACE.items() if name == trace_name)


def phase_write_plan(crossover_rows: List[Dict[str, object]], sample_rate_hz: float) -> List[Dict[str, object]]:
    """Build writes from the gated polarity -> delay -> residual APF ladder."""
    plan: List[Dict[str, object]] = []
    for row in crossover_rows:
        if row.get("session_phase_valid") is False:
            continue
        ladder = row.get("crossover_ladder")
        if not isinstance(ladder, dict) or not ladder.get("write_eligible"):
            continue
        try:
            correction_ms_b = float(ladder.get("correction_delay_ms_B", 0.0))
        except (TypeError, ValueError):
            continue

        name = str(row.get("name", ""))
        if name == "sub_to_front":
            a_trace, b_trace = "Sub", "Mid Bass Together"
        elif name == "left_mid_to_tweeter":
            a_trace, b_trace = ("FL Mid" if FRONT_LAYOUT == "3way" else "FL Low"), "FL High"
        elif name == "right_mid_to_tweeter":
            a_trace, b_trace = ("FR Mid" if FRONT_LAYOUT == "3way" else "FR Low"), "FR High"
        else:
            continue

        polarity_flip = bool(ladder.get("polarity_flip_B"))
        if name == "sub_to_front":
            # Flipping the sub is magnitude-equivalent to flipping the whole
            # front stage and preserves front-stage absolute polarity.
            polarity_channels = trace_channels(a_trace) if polarity_flip else ()
        else:
            polarity_channels = trace_channels(b_trace) if polarity_flip else ()

        delay_channels: Tuple[int, ...] = ()
        write_delay_ms = 0.0
        if abs(correction_ms_b) >= 0.02:
            if correction_ms_b > 0.0:
                if name == "sub_to_front":
                    delay_channels = tuple(range(0, 6 if FRONT_LAYOUT == "3way" else 4))
                else:
                    delay_channels = trace_channels(b_trace)
            else:
                delay_channels = trace_channels(a_trace)
            write_delay_ms = abs(correction_ms_b)

        apf = ladder.get("apf") if isinstance(ladder.get("apf"), dict) else None
        apf_channels: Tuple[int, ...] = ()
        if apf is not None and ladder.get("apf_write_eligible"):
            apf_channels = trace_channels(a_trace if apf.get("apply_to") == "A" else b_trace)

        if not polarity_channels and not delay_channels and not apf_channels:
            continue

        impulse = row.get("impulse") if isinstance(row.get("impulse"), dict) else {}
        impulse_agrees = (
            ladder.get("impulse_delay_agreement") is True
            and ladder.get("impulse_polarity_agreement") is True
        )
        phase_full = bool(
            row.get("predicted_sum_match") == "high"
            and (impulse_agrees or (not impulse.get("available") and row.get("phase_stability") == "trustworthy"))
        )
        operations = []
        if polarity_channels:
            operations.append("polarity")
        if delay_channels:
            operations.append("delay")
        if apf_channels:
            operations.append("apf")
        plan.append({
            "source": row.get("label", name),
            "kind": "_".join(operations),
            "ladder_source": ladder.get("source"),
            "polarity_toggle": polarity_flip,
            "polarity_channels": polarity_channels,
            "channels": delay_channels,
            "delay_ms": round(write_delay_ms, 4),
            "delay_samples": int(round(ms_to_samples(write_delay_ms, sample_rate_hz))),
            "apf_channels": apf_channels,
            "apf_f": None if apf is None else float(apf["F"]),
            "apf_q": None if apf is None else float(apf["Q"]),
            "apf": bool(apf_channels),
            "crossover_channels": tuple(sorted(set(trace_channels(a_trace) + trace_channels(b_trace)))),
            "crossover_band": tuple(
                float(value) for value in str(row.get("band", "")).replace("Hz", "").split("-")[:2]
            ),
            "confidence": "full" if phase_full else "warning",
            "warning": "" if phase_full else (
                "not full confidence; re-measure this crossover and verify polarity, arrival, image, and summed level"
            ),
        })
    return plan


def phase_peq_conflicts(freqs: np.ndarray, groups: GroupBands,
                        phase_plan: List[Dict[str, object]] | None,
                        threshold_db: float = 0.5) -> List[Dict[str, object]]:
    conflicts: List[Dict[str, object]] = []
    if not phase_plan:
        return conflicts
    for edit in phase_plan:
        try:
            lo, hi = (float(value) for value in edit.get("crossover_band", ()))
        except (TypeError, ValueError):
            continue
        selected = (freqs >= lo) & (freqs <= hi)
        if not np.any(selected):
            continue
        crossover_channels = {int(value) for value in edit.get("crossover_channels", ())}
        for group, bands in groups.items():
            group_channels = {int(value) for value in GROUPS.get(group, {}).get("channels", ())}
            affected = sorted(crossover_channels & group_channels)
            if not affected:
                continue
            for F, Q, G in bands:
                response = peaking_db(freqs, F, Q, G)
                maximum = float(np.max(np.abs(response[selected])))
                if maximum + 1e-12 < float(threshold_db):
                    continue
                conflicts.append({
                    "source": edit.get("source", "crossover"),
                    "crossover_band": [lo, hi],
                    "group": group,
                    "channels": affected,
                    "filter": {"F": float(F), "Q": float(Q), "G": float(G)},
                    "max_change_db": maximum,
                    "threshold_db": float(threshold_db),
                })
    return conflicts


def phase_safe_component_scorer(component_score, freqs: np.ndarray,
                                phase_plan: List[Dict[str, object]] | None,
                                threshold_db: float = 0.5):
    def score(groups: GroupBands) -> Dict[str, float]:
        comp = dict(component_score(groups))
        conflicts = phase_peq_conflicts(freqs, groups, phase_plan, threshold_db)
        comp["phase_peq_conflict_count"] = float(len(conflicts))
        comp["phase_peq_conflict_max_db"] = max(
            (float(item["max_change_db"]) for item in conflicts), default=0.0
        )
        if conflicts:
            comp["objective"] = 1e6 + 1000.0 * len(conflicts) + comp["phase_peq_conflict_max_db"]
        return comp
    return score


def add_allpass_to_oc(oc: str, f0: float, q: float = 0.7, invert: bool = False) -> str:
    slot = choose_free_slots(oc, 1)[0]
    fn = slot_attr(slot, "FN") or "0"
    df = slot_attr(slot, "dF") or "20000"
    new = allpass_fil_str(float(f0), float(q), fn, dF=df, invert=invert)
    return oc.replace(slot, new, 1)


def slot_attr(tag: str, key: str) -> str | None:
    import re
    m = re.search(r'(?<![A-Za-z])' + re.escape(key) + r'="([^"]*)"', tag)
    return m.group(1) if m else None


def apply_phase_writes(xml: str, phase_plan: List[Dict[str, object]] | None) -> str:
    if not phase_plan:
        return xml
    blocks = [m.group() for m in re.finditer(r"<OC\b.*?</OC>", xml, re.S)]
    new_blocks = list(blocks)
    delay_add: Dict[int, int] = {}
    polarity_toggle: Dict[int, bool] = {}
    for edit in phase_plan:
        channels = tuple(int(ch) for ch in edit.get("channels", ()))
        for ch in channels:
            delay_add[ch] = delay_add.get(ch, 0) + int(edit.get("delay_samples", 0))
        for ch in (int(value) for value in edit.get("polarity_channels", ())):
            polarity_toggle[ch] = not polarity_toggle.get(ch, False)
        if edit.get("apf"):
            for ch in (int(value) for value in edit.get("apf_channels", ())):
                if 0 <= ch < len(new_blocks):
                    new_blocks[ch] = add_allpass_to_oc(new_blocks[ch], float(edit["apf_f"]), float(edit["apf_q"]))
    out = replace_oc_blocks(xml, new_blocks)

    if delay_add or polarity_toggle:
        tags = list(re.finditer(r"<T [^>]*/?>", out))
        changed_channels = set(delay_add) | {ch for ch, toggle in polarity_toggle.items() if toggle}
        for ch in sorted(changed_channels, reverse=True):
            if ch >= len(tags):
                continue
            tag = tags[ch].group()
            new_tag = tag
            add_samples = delay_add.get(ch, 0)
            if add_samples:
                current = int(float(slot_attr(new_tag, "T") or "0"))
                new_tag = set_attr(new_tag, "T", str(current + add_samples))
            if polarity_toggle.get(ch):
                pm = slot_attr(new_tag, "PM")
                if pm not in ("1", "4"):
                    raise ValueError(f"delay tag {ch} has no confirmed PM=1/4 polarity value")
                new_tag = set_attr(new_tag, "PM", "1" if pm == "4" else "4")
            out = out[:tags[ch].start()] + new_tag + out[tags[ch].end():]
    return out


def gate_validity_notes(gate_ms: float | None) -> List[str]:
    if gate_ms is None or gate_ms <= 0:
        return ["No impulse/window gate length was supplied; low-frequency phase trust is unknown."]
    f_low = gate_low_frequency_limit(gate_ms)
    return [
        f"Gate/window `{gate_ms:.2f}` ms implies lowest trustworthy gated frequency around `{f_low:.0f}` Hz.",
        f"Do not trust gated response below about `{f_low:.0f}` Hz; use ungated or spatially averaged low-frequency measurements there.",
    ]


def trust_meter_lines(row: Dict[str, object], crossover_rows: List[Dict[str, object]], gate_ms: float | None) -> List[str]:
    sc = row["score"]
    comp = row.get("components", {})
    tonal_conf = "high" if float(sc.get("sum_rms_db", 99.0)) <= 3.0 else "medium"
    balance = float(sc.get("sum_wrms_img_db", 99.0))
    balance_conf = "high" if balance <= 2.0 else "medium" if balance <= 4.0 else "low"
    notes = [
        f"- Tonal improvement: {tonal_conf} confidence",
        f"- L/R balance improvement: {balance_conf} confidence",
    ]
    for item in crossover_rows:
        label = item["label"]
        match = item.get("predicted_sum_match", "missing")
        phase = item.get("phase_stability", "missing")
        summing = item.get("summation_quality", "missing")
        ladder = item.get("crossover_ladder") if isinstance(item.get("crossover_ladder"), dict) else {}
        ladder_status = "write" if ladder.get("write_eligible") else "report-only"
        notes.append(
            f"- {label} ({item['band']}): phase `{phase}`, excess-GD `{item.get('excess_gd_stability', 'missing')}`, "
            f"summation `{summing}`, acoustic-sum match `{match}`, ladder `{ladder_status}`"
        )
    if gate_ms is None or gate_ms <= 0:
        notes.append("- 70-95 Hz improvement: low confidence, gate/phase-valid measurement missing")
    else:
        f_low = gate_low_frequency_limit(gate_ms)
        if f_low > 95.0:
            notes.append(f"- 70-95 Hz improvement: low confidence below gated limit `{f_low:.0f}` Hz")
    if float(comp.get("unsupported_filter_penalty", comp.get("unsupported_filter_penalty_db", 0.0))) > 0.0:
        notes.append("- Some asymmetric/deep/narrow EQ was penalized because the solo traces did not justify it")
    return notes


def make_component_scorer(
    freqs: np.ndarray,
    traces: TraceMap,
    target: np.ndarray,
    filter_cost_scale: float = 0.1,
    worst_weight: float = 0.10,
):
    if AFPX_OBJECTIVE is not None:
        def external_score(groups: GroupBands) -> Dict[str, float]:
            comp = dict(AFPX_OBJECTIVE.score_bands(groups_to_band_sets(groups)))
            center_tonal = float(comp.get("tonal_masked", 0.0))
            tonal = float(comp.get("spatial_tonal_db", center_tonal))
            tonal_anchor = float(comp.get("sum_tonal_anchor_db", tonal))
            presence = float(comp.get("presence_error_db", tonal))
            peak = float(comp.get("spatial_peak_db", comp.get("peak_penalty_db", 0.0)))
            worst = float(comp.get("spatial_worst_db", comp.get("worst_masked", 0.0)))
            low_bias = float(comp.get("low_balance", comp.get("mid_balance", 0.0) if FRONT_LAYOUT != "3way" else 0.0))
            mid_bias = float(comp.get("mid_balance", 0.0))
            tweeter_bias = float(comp.get("tweeter_balance", 0.0))
            low_balance = float(comp.get(
                "low_balance_rms_db",
                comp.get("mid_balance_rms_db", abs(low_bias)) if FRONT_LAYOUT != "3way" else abs(low_bias),
            ))
            mid_balance = float(comp.get("mid_balance_rms_db", abs(mid_bias)))
            tweeter_balance = float(comp.get("tweeter_balance_rms_db", abs(tweeter_bias)))
            balance_terms = [tweeter_balance]
            if FRONT_LAYOUT == "3way":
                balance_terms.extend([low_balance, mid_balance])
            else:
                balance_terms.append(low_balance)
            reported_balance = math.sqrt(sum(x * x for x in balance_terms) / max(len(balance_terms), 1))
            balance = float(comp.get("balance_penalty_db", reported_balance))
            headroom = float(comp.get("headroom_peak", 0.0)) + float(comp.get("null_boost_avg", 0.0))
            comp.update({
                "objective": float(comp["objective"]),
                "tonal_error_db": tonal,
                "center_tonal_error_db": center_tonal,
                "sum_tonal_anchor_db": tonal_anchor,
                "presence_error_db": presence,
                "pareto_tonal_db": 0.40 * tonal + 0.35 * tonal_anchor + 0.25 * presence,
                "peak_penalty_db": peak,
                "balance_penalty_db": balance,
                "worst_presence_dev_db": worst,
                "positive_gain_penalty_db": headroom,
                "positive_gain_rms_db": float(comp.get("headroom_peak", 0.0)),
                "positive_gain_peak_db": float(comp.get("headroom_peak", 0.0)),
                "filter_count": float(comp.get("n_front_bands", 0.0)),
                "filter_cost_units": float(filter_cost(groups)),
                "guardrail_penalty_db": float(comp.get("guardrail_penalty", 0.0)),
                "shape_penalty_db": float(comp.get("shape_penalty", 0.0)),
                "unsupported_filter_penalty_db": float(comp.get("unsupported_filter_penalty", 0.0)),
                "wasted_band_penalty_db": float(comp.get("wasted_band_penalty", 0.0)),
                "asymmetric_eq_penalty_db": float(comp.get("asymmetric_eq_penalty", 0.0)),
                "n_added_front_bands": float(comp.get("n_added_front_bands", 0.0)),
                "low_balance_rms_db": low_balance,
                "mid_balance_rms_db": mid_balance if FRONT_LAYOUT == "3way" else 0.0,
                "high_balance_rms_db": tweeter_balance,
                "low_balance_median_db": low_bias,
                "mid_balance_median_db": mid_bias if FRONT_LAYOUT == "3way" else 0.0,
                "high_balance_median_db": tweeter_bias,
            })
            return comp

        return external_score

    smooth = make_fast_smoother(freqs)
    masks = null_masks(freqs, traces)
    audible = audibility_weight(freqs)
    vocal_weight = np.ones_like(freqs)
    vocal_weight[(freqs >= 200.0) & (freqs <= 6000.0)] = 1.8
    tonal_w = audible * vocal_weight
    tonal_sel = (freqs >= 25.0) & (freqs <= 16000.0) & ~masks["system"]
    tonal_anchor_sel = (freqs >= 25.0) & (freqs <= 16000.0)
    presence_sel = (freqs >= 200.0) & (freqs <= 6000.0)
    worst_sel = presence_sel & ~masks["system"]

    balance_w = audible.copy()
    balance_w[(freqs >= 700.0) & (freqs <= 5000.0)] *= 1.8

    headroom_w = audible.copy()
    headroom_sel_by_channel = {6: (freqs >= 25.0) & (freqs <= 120.0), 7: (freqs >= 25.0) & (freqs <= 120.0)}
    for name, pair in PAIR_DEFS.items():
        sel = (freqs >= pair["branch_band"][0]) & (freqs <= pair["branch_band"][1])
        for channel, trace_name in CH_TRACE.items():
            if trace_name in (pair["left"], pair["right"]):
                headroom_sel_by_channel[channel] = sel

    def score(groups: GroupBands) -> Dict[str, float]:
        pred = predict_traces(freqs, traces, groups)
        dev = smooth(pred["System Sum"] - target)

        tonal = weighted_rms(dev, tonal_w, tonal_sel)
        tonal_anchor = weighted_rms(dev, tonal_w, tonal_anchor_sel)
        presence = weighted_rms(dev, tonal_w, presence_sel)
        peak = weighted_rms(np.maximum(dev, 0.0), tonal_w, tonal_sel)
        worst = float(np.max(np.abs(dev[worst_sel]))) if np.any(worst_sel) else 0.0

        pair_scores = []
        pair_outputs: Dict[str, float] = {}
        for name, pair in PAIR_DEFS.items():
            diff = smooth(pred[pair["left"]] - pred[pair["right"]])
            lo, hi = pair["balance_band"]
            sel = (freqs >= lo) & (freqs <= hi)
            bal = weighted_rms(diff, balance_w, sel)
            pair_scores.append(bal)
            pair_outputs[f"{name}_balance_rms_db"] = bal
            pair_outputs[f"{name}_balance_median_db"] = float(np.median(diff[sel])) if np.any(sel) else 0.0
        balance = float(np.sqrt(np.mean(np.square(pair_scores)))) if pair_scores else 0.0

        pos_rms_scores = []
        pos_peak = 0.0
        for channel, delta in channel_deltas(freqs, groups).items():
            sel = headroom_sel_by_channel.get(channel, np.ones_like(freqs, dtype=bool))
            pos = np.maximum(delta, 0.0)
            pos_rms_scores.append(weighted_rms(pos, headroom_w, sel))
            if np.any(sel):
                pos_peak = max(pos_peak, float(np.max(pos[sel])))
        positive_gain = float(np.sqrt(np.mean(np.square(pos_rms_scores)))) if pos_rms_scores else 0.0

        total_filters = float(sum(len(bands) for bands in groups.values()))
        filter_units = float(filter_cost(groups))
        headroom = positive_gain + 0.12 * pos_peak
        pareto_tonal = 0.40 * tonal + 0.35 * tonal_anchor + 0.25 * presence
        objective = (
            1.35 * tonal
            + 1.00 * tonal_anchor
            + 0.60 * presence
            + 0.45 * peak
            + 0.28 * balance
            + float(worst_weight) * worst
            + 0.22 * headroom
            + 0.04 * total_filters
            + float(filter_cost_scale) * filter_units
        )
        out = {
            "objective": float(objective),
            "tonal_error_db": float(tonal),
            "sum_tonal_anchor_db": float(tonal_anchor),
            "presence_error_db": float(presence),
            "pareto_tonal_db": float(pareto_tonal),
            "peak_penalty_db": float(peak),
            "balance_penalty_db": float(balance),
            "worst_presence_dev_db": float(worst),
            "positive_gain_penalty_db": float(headroom),
            "positive_gain_rms_db": float(positive_gain),
            "positive_gain_peak_db": float(pos_peak),
            "filter_count": total_filters,
            "filter_cost_units": filter_units,
            "null_masked_bins": float(np.count_nonzero(masks["system"] & (freqs >= 25.0) & (freqs <= 16000.0))),
        }
        out.update(pair_outputs)
        return out

    return score


def make_objective(
    freqs: np.ndarray,
    traces: TraceMap,
    target: np.ndarray,
    filter_cost_scale: float = 1.0,
    worst_weight: float = 0.018,
    min_total_bands: int = 0,
    phase_plan: List[Dict[str, object]] | None = None,
):
    component_score = phase_safe_component_scorer(
        make_component_scorer(freqs, traces, target, filter_cost_scale, worst_weight),
        freqs,
        phase_plan,
    )

    def objective(trial: optuna.Trial) -> float:
        groups = trial_bands(trial)
        return component_score(groups)["objective"]

    return objective


def replace_oc_blocks(xml: str, blocks: List[str]) -> str:
    import re

    matches = list(re.finditer(r"<OC\b.*?</OC>", xml, re.S))
    if len(matches) < len(blocks):
        raise ValueError("AFPX contains fewer OC blocks than expected")
    out = xml
    for old_match, new_block in zip(matches, blocks):
        old = old_match.group()
        if old != new_block:
            out = out.replace(old, new_block, 1)
    return out


def write_candidate(base_xml: str, path: Path, groups: GroupBands,
                    phase_plan: List[Dict[str, object]] | None = None) -> Dict[str, object]:
    conflicts = phase_peq_conflicts(
        np.geomspace(20.0, 20000.0, 2048), groups, phase_plan, threshold_db=0.5
    )
    if conflicts:
        first = conflicts[0]
        raise ValueError(
            "PEQ/phase crossover conflict: %s %s changes %s by %.2f dB"
            % (first["group"], first["filter"], first["source"], first["max_change_db"])
        )
    blocks = [m.group() for m in re.finditer(r"<OC\b.*?</OC>", base_xml, re.S)]
    new_blocks = list(blocks)
    for group, bands in groups.items():
        if not bands:
            continue
        for channel in GROUPS[group]["channels"]:
            new_blocks[channel] = add_bands(new_blocks[channel], bands)

    new_xml = replace_oc_blocks(base_xml, new_blocks)
    new_xml = apply_phase_writes(new_xml, phase_plan)
    encode_afpx(new_xml, path)
    written = decode_afpx(path)
    allow_delay = any(int(edit.get("delay_samples", 0)) != 0 for edit in (phase_plan or []))
    allow_polarity = any(bool(edit.get("polarity_channels")) for edit in (phase_plan or []))
    allow_apf = any(bool(edit.get("apf_channels")) for edit in (phase_plan or []))
    lint = afpx_roundtrip_lint(
        base_xml,
        written,
        allow_delay_changes=allow_delay,
        allow_polarity_changes=allow_polarity,
        allowed_added_types=("17", "20") if allow_apf else ("17",),
    )
    if not lint["pass"]:
        raise AssertionError("AFPX lint failed: " + "; ".join(lint["errors"]))
    return lint


def mask_ranges(freqs: np.ndarray, mask: np.ndarray, band: Tuple[float, float]) -> List[Tuple[float, float]]:
    sel = mask & (freqs >= band[0]) & (freqs <= band[1])
    ranges: List[Tuple[float, float]] = []
    start = None
    last = None
    for f, on in zip(freqs, sel):
        if on and start is None:
            start = float(f)
        if on:
            last = float(f)
        elif start is not None:
            ranges.append((start, float(last)))
            start = None
            last = None
    if start is not None:
        ranges.append((start, float(last)))
    # Drop one-bin specks; they are report noise.
    return [(lo, hi) for lo, hi in ranges if hi / lo > 2 ** (1 / 48)]


def format_ranges(ranges: List[Tuple[float, float]], limit: int = 3) -> str:
    if not ranges:
        return ""
    parts = [f"{lo:.0f}-{hi:.0f} Hz" for lo, hi in ranges[:limit]]
    if len(ranges) > limit:
        parts.append(f"+{len(ranges) - limit} more")
    return ", ".join(parts)


def left_alone_note(freqs: np.ndarray, traces: TraceMap) -> str:
    masks = null_masks(freqs, traces)
    notes = []
    labels = {"low": "midbass", "mid": "midrange", "high": "tweeter"}
    for name, pair in PAIR_DEFS.items():
        pretty = format_ranges(mask_ranges(freqs, masks[name], pair["branch_band"]))
        if pretty:
            notes.append(f"{labels.get(name, name)} nulls {pretty}: flagged as destructive summing, not EQ-able")
    notes.append("sub low edge/top-octave rolloff/crossover skirts: treated as driver or phase behaviour")
    return "; ".join(notes)


def format_bands(groups: GroupBands) -> str:
    lines: List[str] = []
    for group in GROUPS:
        bands = groups.get(group, [])
        if not bands:
            lines.append(f"- {group}: no added filters")
            continue
        joined = ", ".join(f"{F:g} Hz Q{Q:g} {G:+g} dB" for F, Q, G in bands)
        lines.append(f"- {group}: {joined}")
    return "\n".join(lines)


def score_row(freqs: np.ndarray, traces: TraceMap, target: np.ndarray, groups: GroupBands) -> Dict[str, float]:
    pred = predict_traces(freqs, traces, groups)
    raw = tune_scorecard(freqs, pred, target)
    components = make_component_scorer(freqs, traces, target)(groups)
    raw["objective"] = round(float(components["objective"]), 4)
    raw.update({
        f"objective_{k}": round(float(v), 4)
        for k, v in components.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    })
    return raw


def format_component_summary(comp: Dict[str, float]) -> str:
    if "tonal_masked" in comp:
        return (
            f"objective `{comp.get('objective', 0.0):.3f}`, tonal_masked `{comp.get('tonal_masked', 0.0):.3f}`, "
            f"spatial `{comp.get('spatial_tonal_db', comp.get('tonal_masked', 0.0)):.3f}`/`{comp.get('spatial_position_count', 1):.0f}` positions, "
            f"anchor `{comp.get('sum_tonal_anchor_db', 0.0):.3f}`, presence `{comp.get('presence_error_db', 0.0):.3f}`, "
            f"peak `{comp.get('peak_penalty_db', 0.0):.3f}`, worst_masked `{comp.get('worst_masked', 0.0):.2f}`, "
            f"mid-bias `{comp.get('mid_balance', 0.0):+.2f}`, mid-RMS `{comp.get('mid_balance_rms_db', comp.get('low_balance_rms_db', 0.0)):.2f}`, "
            f"tweeter-bias `{comp.get('tweeter_balance', 0.0):+.2f}`, tweeter-RMS `{comp.get('tweeter_balance_rms_db', 0.0):.2f}`, "
            f"headroom_peak `{comp.get('headroom_peak', 0.0):.2f}`, "
            f"null_boost_avg `{comp.get('null_boost_avg', 0.0):.2f}`, guardrail `{comp.get('guardrail_penalty', 0.0):.3f}`, "
            f"unsupported `{comp.get('unsupported_filter_penalty', 0.0):.3f}`, wasted `{comp.get('wasted_band_penalty', 0.0):.3f}`, "
            f"asym `{comp.get('asymmetric_eq_penalty', 0.0):.3f}`, added_bands `{comp.get('n_added_front_bands', 0.0):.0f}`"
        )
    return (
        f"tonal `{comp.get('tonal_error_db', 0.0):.2f}`, "
        f"anchor `{comp.get('sum_tonal_anchor_db', 0.0):.2f}`, presence `{comp.get('presence_error_db', 0.0):.2f}`, "
        f"pareto-tonal `{comp.get('pareto_tonal_db', 0.0):.2f}`, peak `{comp.get('peak_penalty_db', 0.0):.2f}`, "
        f"balance `{comp.get('balance_penalty_db', 0.0):.2f}` "
        f"(low `{comp.get('low_balance_rms_db', 0.0):.2f}`, high `{comp.get('high_balance_rms_db', 0.0):.2f}`), "
        f"headroom `{comp.get('positive_gain_penalty_db', 0.0):.2f}`, filters `{comp.get('filter_count', 0.0):.0f}`"
    )


PARETO_KEYS = (
    "pareto_tonal_db",
    "peak_penalty_db",
    "balance_penalty_db",
    "positive_gain_penalty_db",
    "filter_count",
)


def row_metrics(row: Dict[str, object]) -> Dict[str, float]:
    return dict(row.get("components", {}))


def row_metric(row: Dict[str, object], key: str) -> float:
    return float(row_metrics(row).get(key, float("inf")))


def dominates(row_a: Dict[str, object], row_b: Dict[str, object], keys=PARETO_KEYS) -> bool:
    better = False
    for key in keys:
        va = row_metric(row_a, key)
        vb = row_metric(row_b, key)
        if va > vb + 1e-9:
            return False
        if va + 1e-9 < vb:
            better = True
    return better


def pareto_front_rows(rows: List[Dict[str, object]], keys=PARETO_KEYS) -> List[Dict[str, object]]:
    front = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if dominates(other, row, keys):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def pareto_rank_rows(rows: List[Dict[str, object]], keys=PARETO_KEYS) -> None:
    remaining = list(rows)
    rank = 1
    while remaining:
        front = pareto_front_rows(remaining, keys)
        if not front:
            break
        front_ids = {id(row) for row in front}
        for row in front:
            row["pareto_rank"] = rank
        remaining = [row for row in remaining if id(row) not in front_ids]
        rank += 1
    for row in rows:
        row.setdefault("pareto_rank", rank)


def normalized_metric(row: Dict[str, object], key: str, spans: Dict[str, Tuple[float, float]]) -> float:
    lo, hi = spans[key]
    val = row_metric(row, key)
    if not np.isfinite(val):
        return 1.0
    if hi <= lo + 1e-12:
        return 0.0
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def family_pick_scores(rows: List[Dict[str, object]], role: str) -> List[Tuple[float, Dict[str, object]]]:
    if not rows:
        return []
    span_keys = set(PARETO_KEYS) | {
        "tonal_error_db",
        "sum_tonal_anchor_db",
        "presence_error_db",
    }
    spans = {
        key: (
            min(row_metric(row, key) for row in rows),
            max(row_metric(row, key) for row in rows),
        )
        for key in span_keys
    }

    def score_of(row: Dict[str, object]) -> float:
        nm = lambda key: normalized_metric(row, key, spans)
        if role == "conservative":
            return (
                0.70 * nm("positive_gain_penalty_db")
                + 0.55 * nm("filter_count")
                + 0.35 * nm("peak_penalty_db")
                + 0.25 * nm("balance_penalty_db")
                + 0.20 * nm("sum_tonal_anchor_db")
                + 0.15 * nm("presence_error_db")
            )
        if role == "aggressive":
            return (
                0.95 * nm("sum_tonal_anchor_db")
                + 0.80 * nm("presence_error_db")
                + 0.65 * nm("tonal_error_db")
                + 0.30 * nm("peak_penalty_db")
                + 0.15 * nm("balance_penalty_db")
            )
        return (
            0.85 * nm("sum_tonal_anchor_db")
            + 0.75 * nm("presence_error_db")
            + 0.55 * nm("tonal_error_db")
            + 0.50 * nm("balance_penalty_db")
            + 0.30 * nm("positive_gain_penalty_db")
            + 0.20 * nm("peak_penalty_db")
            + 0.12 * nm("filter_count")
        )

    return sorted((score_of(row), row) for row in rows)


def select_family_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    for row in rows:
        row["family_role"] = ""
    if not rows:
        return {}
    pareto_rank_rows(rows)
    first_front = [row for row in rows if int(row.get("pareto_rank", 99)) == 1]
    pool = first_front if first_front else list(rows)
    chosen: Dict[str, Dict[str, object]] = {}
    used = set()
    for role in ("conservative", "balanced", "aggressive"):
        ranked = family_pick_scores(pool, role)
        pick = None
        for _score, row in ranked:
            if id(row) not in used:
                pick = row
                break
        if pick is None and ranked:
            pick = ranked[0][1]
        if pick is not None:
            chosen[role] = pick
            used.add(id(pick))
            pick["family_role"] = role
    return chosen


def write_family_aliases(out_dir: Path, rows: List[Dict[str, object]], base_xml: str,
                         phase_plan: List[Dict[str, object]] | None = None) -> Dict[str, str]:
    picks = select_family_rows(rows)
    aliases = {}
    for old in out_dir.glob("family_*.afpx"):
        old.unlink()
    for role, row in picks.items():
        file_name = f"family_{role}.afpx"
        write_candidate(base_xml, out_dir / file_name, row["groups"], phase_plan=phase_plan)
        aliases[role] = file_name
    return aliases


class StaticTrial:
    """Tiny Optuna-trial shim so one objective function can score fixed bands."""

    def __init__(self, groups: GroupBands):
        self.groups = groups

    def suggest_categorical(self, name, choices):
        group, idx, field = name.split("_", 2)
        return idx.isdigit() and int(idx) < len(self.groups.get(group, []))

    def suggest_float(self, name, low, high, log=False):
        group, idx, field = name.split("_", 2)
        F, Q, G = self.groups[group][int(idx)]
        if field == "freq":
            return F
        if field == "q":
            return Q
        if field == "gain":
            return G
        raise KeyError(name)


def file_fingerprint(path: Path) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def write_report(
    out_dir: Path,
    rows: List[Dict[str, object]],
    baseline_score: Dict[str, float],
    interference_notes: List[str],
    args: argparse.Namespace,
    family_rows: List[Dict[str, object]] | None = None,
    crossover_rows: List[Dict[str, object]] | None = None,
    phase_plan: List[Dict[str, object]] | None = None,
) -> None:
    md = out_dir / "optimizer_report.md"
    csv_path = out_dir / "optimizer_results.csv"
    json_path = out_dir / "optimizer_summary.json"
    assistant_path = out_dir / "assistant_summary.json"
    family_source = list(family_rows) if family_rows is not None else list(rows)
    family_picks = select_family_rows(family_source)
    crossover_rows = crossover_rows or []
    phase_plan = phase_plan or []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "file", "objective", "pareto_rank", "family_role", "sum_rms_db", "sum_wrms_img_db",
            "worst_dev_db", "mid_balance_db", "tweeter_balance_db",
            "tonal_error_db", "sum_tonal_anchor_db", "presence_error_db", "pareto_tonal_db",
            "peak_penalty_db", "balance_penalty_db",
            "low_balance_rms_db", "high_balance_rms_db", "positive_gain_penalty_db",
            "filter_count", "tonal_masked", "worst_masked", "mid_balance", "tweeter_balance",
            "headroom_peak", "null_boost_avg", "n_front_bands", "left_alone", "bands",
        ])
        for row in rows:
            sc = row["score"]
            comp = row.get("components", {})
            writer.writerow([
                row["rank"], row["file"], row["objective"], row.get("pareto_rank", ""), row.get("family_role", ""),
                sc["sum_rms_db"],
                sc["sum_wrms_img_db"], sc["worst_dev_db"],
                sc.get("mid_balance_db", ""), sc.get("tweeter_balance_db", ""),
                comp.get("tonal_error_db", ""), comp.get("sum_tonal_anchor_db", ""),
                comp.get("presence_error_db", ""), comp.get("pareto_tonal_db", ""),
                comp.get("peak_penalty_db", ""),
                comp.get("balance_penalty_db", ""), comp.get("low_balance_rms_db", ""),
                comp.get("high_balance_rms_db", ""), comp.get("positive_gain_penalty_db", ""),
                comp.get("filter_count", ""),
                comp.get("tonal_masked", ""), comp.get("worst_masked", ""),
                comp.get("mid_balance", ""), comp.get("tweeter_balance", ""),
                comp.get("headroom_peak", ""), comp.get("null_boost_avg", ""),
                comp.get("n_front_bands", ""), row.get("left_alone", ""),
                row["signature"],
            ])

    summary_rows = []
    for row in rows[:10]:
        sc = row["score"]
        comp = row.get("components", {})
        summary_rows.append({
            "rank": row["rank"],
            "file": row["file"],
            "objective": round(float(row["objective"]), 4),
            "pareto_rank": row.get("pareto_rank", ""),
            "family_role": row.get("family_role", ""),
            "sum_rms_db": sc.get("sum_rms_db", ""),
            "image_weighted_db": sc.get("sum_wrms_img_db", ""),
            "worst_dev_db": sc.get("worst_dev_db", ""),
            "filters": comp.get("filter_count", comp.get("n_front_bands", "")),
            "headroom_penalty_db": comp.get("positive_gain_penalty_db", comp.get("headroom_peak", "")),
            "components": {
                key: comp.get(key, "")
                for key in (
                    "objective",
                    "tonal_error_db",
                    "sum_tonal_anchor_db",
                    "presence_error_db",
                    "peak_penalty_db",
                    "balance_penalty_db",
                    "positive_gain_penalty_db",
                    "filter_count",
                    "tonal_masked",
                    "worst_masked",
                    "mid_balance",
                    "tweeter_balance",
                    "low_balance_rms_db",
                    "mid_balance_rms_db",
                    "tweeter_balance_rms_db",
                    "guardrail_penalty",
                    "unsupported_filter_penalty",
                    "wasted_band_penalty",
                    "asymmetric_eq_penalty",
                )
                if key in comp
            },
            "left_alone": row.get("left_alone", ""),
        })
    summary_payload = {
        "run_folder": str(out_dir),
        "data_root": str(DATA_ROOT.resolve()),
        "front_layout": FRONT_LAYOUT,
        "baseline": str(args.baseline),
        "target": str(args.target),
        "input_fingerprints": {
            "baseline": file_fingerprint(args.baseline),
            "target": file_fingerprint(args.target),
            "level_calibration": (
                file_fingerprint(args.level_calibration)
                if getattr(args, "level_calibration", None) else None
            ),
        },
        "run_config": {
            key: (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key))
            for key in (
                "seed", "profile", "sampler", "proposal", "jobs", "seconds",
                "max_trials", "filter_cost_scale", "worst_weight",
                "validation_threshold", "gate_ms", "sample_rate", "phase_writes", "impulse_root",
                "level_calibration",
                "phase_cache",
                "beam_width", "beam_pool_limit",
                "refine_top", "refine_passes",
            )
            if hasattr(args, key)
        },
        "refinement": getattr(args, "refinement", None),
        "beam": getattr(args, "beam", None),
        "trials": getattr(args, "trials", ""),
        "candidate_count": len(rows),
        "top_candidates": summary_rows,
        "family_picks": {
            role: {
                "file": row.get("file", ""),
                "objective": round(float(row.get("objective", 0.0)), 4),
                "pareto_rank": row.get("pareto_rank", ""),
            }
            for role, row in family_picks.items()
        },
        "validation": getattr(args, "validation", []),
        "measurement_session": getattr(args, "measurement_session", {}),
        "phase_diagnostic_cache": getattr(args, "phase_diagnostic_cache", {}),
        "objective_cache": (
            AFPX_OBJECTIVE.cache_stats()
            if AFPX_OBJECTIVE is not None and hasattr(AFPX_OBJECTIVE, "cache_stats") else {}
        ),
        "crossover_phase_confidence": crossover_rows,
        "written_phase_plan": phase_plan,
        "phase_peq_rejections": getattr(args, "phase_peq_rejections", [])[:20],
        "generated_files": {
            "report": str(md),
            "csv": str(csv_path),
            "summary_json": str(json_path),
            "assistant_summary_json": str(assistant_path),
        },
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    component_keys = (
        "objective", "tonal_error_db", "sum_tonal_anchor_db", "presence_error_db",
        "peak_penalty_db", "balance_penalty_db", "low_balance_rms_db",
        "mid_balance_rms_db", "high_balance_rms_db", "positive_gain_penalty_db",
        "filter_count", "guardrail_penalty", "center_tonal_error_db",
        "spatial_tonal_db", "spatial_peak_db", "spatial_worst_db",
        "spatial_fragility_penalty", "spatial_position_count",
    )
    base_components = dict(baseline_score.get("components", {}))
    best_row = rows[0] if rows else None
    best_components = dict(best_row.get("components", {})) if best_row else {}
    def compact_components(values):
        return {
            key: round(float(values[key]), 4)
            for key in component_keys
            if key in values and isinstance(values[key], (int, float, np.integer, np.floating))
        }
    baseline_core = compact_components(base_components)
    best_core = compact_components(best_components)
    deltas = {
        key: round(float(best_core[key]) - float(baseline_core[key]), 4)
        for key in sorted(best_core.keys() & baseline_core.keys())
    }
    assistant_warnings = list(
        dict(getattr(args, "measurement_session", {})).get("audit", {}).get("warnings", [])
    )
    assistant_warnings.extend(
        str(edit.get("warning")) for edit in phase_plan if edit.get("warning")
    )
    compact_inputs = {}
    for name, fingerprint in summary_payload["input_fingerprints"].items():
        if fingerprint:
            compact_inputs[name] = {
                "file": Path(str(fingerprint.get("path", ""))).name,
                "sha256": fingerprint.get("sha256", ""),
            }
        else:
            compact_inputs[name] = None
    session_audit = dict(getattr(args, "measurement_session", {})).get("audit", {})
    compact_session = {
        key: session_audit.get(key)
        for key in (
            "tonal_valid", "phase_valid", "reference_level_signature",
            "missing_calibration_roles", "timing_references", "warnings",
            "spatial_positions",
        )
        if key in session_audit
    }
    compact_phase_actions = [{
        key: edit.get(key)
        for key in (
            "source", "kind", "polarity_channels", "channels", "delay_samples",
            "apf_channels", "apf_f", "apf_q", "crossover_band", "confidence", "warning",
        )
        if edit.get(key) not in (None, "", [], (), False)
    } for edit in phase_plan]
    rejection_source = getattr(args, "phase_peq_rejections", [])
    compact_rejections = [{
        "source": item.get("source"),
        "band_hz": item.get("crossover_band"),
        "group": item.get("group"),
        "filter": item.get("filter"),
        "change_db": round(float(item.get("max_change_db", 0.0)), 3),
    } for item in rejection_source[:3]]
    assistant_payload = {
        "schema": "audiofischer-assistant-summary-v1",
        "run_folder": str(out_dir),
        "candidate_count": len(rows),
        "inputs": compact_inputs,
        "gates": {
            "measurement_session": compact_session,
            "pair_validation": getattr(args, "validation", []),
            "phase_diagnostic_cache": getattr(args, "phase_diagnostic_cache", {}),
        },
        "baseline": baseline_core,
        "baseline_position_tonal_db": base_components.get("spatial_position_tonal_db", {}),
        "best": None if best_row is None else {
            "file": best_row.get("file"),
            "objective": round(float(best_row.get("objective", 0.0)), 6),
            "components": best_core,
            "delta_vs_baseline": deltas,
            "left_alone": best_row.get("left_alone", ""),
            "position_tonal_db": best_components.get("spatial_position_tonal_db", {}),
            "spatial_hold_pass": best_components.get("spatial_hold_pass", True),
        },
        "families": {
            role: {
                "file": f"family_{role}.afpx",
                "objective": round(float(row.get("objective", 0.0)), 6),
            }
            for role, row in family_picks.items()
        },
        "search": {
            "proposal": getattr(args, "proposal", ""),
            "beam": getattr(args, "beam", None),
            "refinement": getattr(args, "refinement", None),
        },
        "phase_actions": compact_phase_actions,
        "phase_peq_rejections": {
            "reported_count": len(rejection_source),
            "may_be_truncated": len(rejection_source) >= 20,
            "examples": compact_rejections,
        },
        "warnings": list(dict.fromkeys(assistant_warnings))[:12],
        "remeasure": (
            ["Re-measure every crossover pair changed by polarity, delay, or APF and verify summed level and image focus."]
            if phase_plan else []
        ),
        "details": {
            "optimizer_summary": json_path.name,
            "report": md.name,
        },
    }
    assistant_path.write_text(json.dumps(assistant_payload, indent=2), encoding="utf-8")

    base_comp = baseline_score.get("components", {})
    baseline_line = (
        f"- sum RMS `{baseline_score['sum_rms_db']}` dB, "
        f"image-weighted `{baseline_score['sum_wrms_img_db']}` dB, "
        f"worst `{baseline_score['worst_dev_db']}` dB"
    )
    if base_comp:
        baseline_line += f"; {format_component_summary(base_comp)}"

    lines = [
        "# Optimizer Report",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Target: `{args.target}`",
        f"- Trials: `{args.trials}`",
        "- Mode: PEQ from magnitude data; optional polarity/delay/APF from gated crossover evidence; no crossover/shelf/level writes",
        "- Objective: imported `afpx_objective.score_bands(band_sets)['objective']`; lower is better. Search-space guardrails restrict what candidates are generated, but no extra target-matching term is added.",
        "",
        "## Validation Gate",
        "",
    ]
    validation = getattr(args, "validation", [])
    if validation:
        for item in validation:
            verdict = "PASS" if item.get("pass") else "FAIL"
            lines.append(
                f"- {item['pair']} solo power-sum -> `{item['together']}`: "
                f"`{item['rms_db']}` dB RMS vs `{item['threshold_db']}` dB gate: {verdict}"
            )
    else:
        lines.append("- Solo/together validation was not provided for this run.")
    session_audit = dict(getattr(args, "measurement_session", {})).get("audit", {})
    if session_audit:
        lines.append(
            "- Measurement session: tonal `%s`; phase `%s`; warnings `%s`"
            % (
                "PASS" if session_audit.get("tonal_valid") else "FAIL",
                "PASS" if session_audit.get("phase_valid") else "DISABLED",
                ", ".join(session_audit.get("warnings", [])) or "none",
            )
        )
    lines.extend([
        "",
        "## Baseline Score",
        "",
        baseline_line,
        "",
        "## Rejected / Flagged Regions",
        "",
    ])
    if interference_notes:
        lines.extend(f"- {note}" for note in interference_notes)
    else:
        lines.append("- No strong L/R destructive-summing regions were flagged by the TXT magnitude audit.")
    lines.extend([
        "- Destructive pair-summing regions are masked from tonal scoring so EQ cannot win by filling phase nulls.",
        "- Polarity, delay, and all-pass writes require validated fixed-position crossover evidence; crossovers remain untouched.",
        "",
        "## Crossover Phase Confidence",
        "",
    ])
    for note in gate_validity_notes(getattr(args, "gate_ms", None)):
        lines.append(f"- {note}")
    if crossover_rows:
        for item in crossover_rows:
            ladder = item.get("crossover_ladder") if isinstance(item.get("crossover_ladder"), dict) else {}
            ladder_action = (
                f"flip-B `{ladder.get('polarity_flip_B', False)}`, correction-B "
                f"`{ladder.get('correction_delay_ms_B', '')}` ms, improvement "
                f"`{ladder.get('improvement_pct', '')}`%, write `{ladder.get('write_eligible', False)}`"
            )
            impulse = item.get("impulse") if isinstance(item.get("impulse"), dict) else {}
            impulse_text = (
                f"; impulse correction `{impulse.get('correction_delay_ms_B')}` ms, "
                f"polarity `{impulse.get('polarity')}`, correlation `{impulse.get('corr_norm')}`"
                if impulse.get("available") else "; impulse `missing`"
            )
            lines.append(
                f"- {item['label']} `{item['band']}`: phase-slope delay `{item.get('relative_delay_ms', '')}` ms; "
                f"polarity `{item.get('polarity', '')}`; phase `{item.get('phase_stability', '')}`; "
                f"excess-GD `{item.get('excess_gd_stability', '')}` ({item.get('minimum_phase_pct', '')}% minimum-phase bins); "
                f"summation `{item.get('summation_quality', '')}`; predicted-sum `{item.get('predicted_sum_match', '')}` "
                f"({item.get('predicted_sum_rms_db', '')} dB RMS); ladder {ladder_action}{impulse_text}."
            )
    else:
        lines.append("- No crossover-band phase diagnostics were available from the supplied measurement set.")
    lines.extend([
        "",
        "## Written Polarity / Delay / APF Changes",
        "",
    ])
    if phase_plan:
        for edit in phase_plan:
            channels = ", ".join("ch%d" % int(ch) for ch in edit.get("channels", ()))
            polarity = ""
            if edit.get("polarity_channels"):
                polarity_channels = ", ".join("ch%d" % int(ch) for ch in edit.get("polarity_channels", ()))
                polarity = f"toggled polarity on {polarity_channels}; "
            apf = ""
            if edit.get("apf"):
                apf_channels = ", ".join("ch%d" % int(ch) for ch in edit.get("apf_channels", ()))
                apf = f"; APF T=20 at `{edit.get('apf_f')}` Hz Q`{edit.get('apf_q')}` on {apf_channels}"
            warning = edit.get("warning") or "full-confidence phase write"
            delay_text = (
                f"added `{edit.get('delay_ms')}` ms (`{edit.get('delay_samples')}` samples) to {channels}"
                if edit.get("channels") else "no delay change"
            )
            lines.append(
                f"- {edit.get('source')}: {polarity}{delay_text}{apf}. "
                f"Confidence `{edit.get('confidence')}`: {warning}."
            )
        lines.append("- After loading the tune, re-measure the affected crossover pairs and confirm centre image, summed level, and that no new hole appeared either side of the crossover.")
    else:
        lines.append("- No polarity/delay/APF writes cleared the crossover-ladder confidence and improvement gates.")
    lines.extend(["", "## Rejected PEQ / Phase Interactions", ""])
    phase_rejections = getattr(args, "phase_peq_rejections", [])
    if phase_rejections:
        for item in phase_rejections[:20]:
            fil = item.get("filter", {})
            lines.append(
                f"- Rejected `{item.get('group')}` filter `{fil.get('F')} Hz Q{fil.get('Q')} {fil.get('G'):+} dB`: "
                f"it changes `{item.get('source')}` by `{item.get('max_change_db'):.2f}` dB "
                f"inside `{item.get('crossover_band')}` (limit `{item.get('threshold_db')}` dB)."
            )
    else:
        lines.append("- No candidate PEQ conflicted with an attached polarity/delay/APF correction.")
    lines.extend([
        "",
        "## Family Picks",
        "",
    ])
    if family_picks:
        for role in ("conservative", "balanced", "aggressive"):
            row = family_picks.get(role)
            if row is None:
                continue
            sc = row["score"]
            comp = row.get("components", {})
            alias = f"family_{role}.afpx"
            lines.extend([
                f"### {role.title()}: `{row['file']}`",
                "",
                (
                    f"- alias `{alias}`; Pareto rank `{row.get('pareto_rank', '?')}`; "
                    f"sum RMS `{sc['sum_rms_db']}` dB; image-weighted `{sc['sum_wrms_img_db']}` dB"
                ),
                f"- components: {format_component_summary(comp)}",
                *trust_meter_lines(row, crossover_rows, getattr(args, "gate_ms", None)),
                f"- left alone: {row.get('left_alone', '')}",
                "",
            ])
    else:
        lines.append("- No Pareto family picks were available.")
    lines.extend([
        "",
        "## Candidates",
        "",
    ])
    for row in rows:
        sc = row["score"]
        comp = row.get("components", {})
        component_line = ""
        if comp:
            component_line = f"- components: {format_component_summary(comp)}"
            lines.extend([
            f"### Rank {row['rank']}: `{row['file']}`",
            "",
            (
                f"- objective `{row['objective']:.4f}`; Pareto rank `{row.get('pareto_rank', '?')}`; "
                f"family `{row.get('family_role', '-') or '-'}`; sum RMS `{sc['sum_rms_db']}` dB; "
                f"image-weighted `{sc['sum_wrms_img_db']}` dB; worst `{sc['worst_dev_db']}` dB"
            ),
            component_line,
            *trust_meter_lines(row, crossover_rows, getattr(args, "gate_ms", None)),
            f"- left alone: {row.get('left_alone', '')}",
            format_bands(row["groups"]),
            "",
        ])
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe Optuna PEQ optimizer for Helix AFPX + REW TXT exports.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None, help="Optional optimizer timeout in seconds.")
    parser.add_argument("--sampler", choices=("random", "tpe"), default="random")
    parser.add_argument("--profile", choices=("safe", "explore"), default="safe")
    parser.add_argument("--filter-cost-scale", type=float, default=None)
    parser.add_argument("--worst-weight", type=float, default=0.10)
    parser.add_argument("--min-total-bands", type=int, default=0)
    parser.add_argument("--gate-ms", type=float, default=None, help="Optional impulse/window gate length in milliseconds for confidence warnings.")
    parser.add_argument("--sample-rate", type=float, default=96000.0, help="DSP internal sample rate used for delay writes.")
    parser.add_argument("--impulse-root", type=Path, default=None,
                        help="Optional folder containing companion WAV/text impulse exports; defaults to the measurement folder.")
    parser.add_argument("--phase-cache", type=Path, default=None,
                        help="Shared fingerprinted crossover diagnostic cache.")
    parser.add_argument("--level-calibration", type=Path, default=None,
                        help="JSON object mapping measurement role/file names to dB offsets for mixed-level sessions.")
    parser.add_argument("--phase-writes", choices=("auto", "off"), default="auto",
                        help="Use 'off' to report the crossover ladder without writing polarity/delay/APF changes.")
    parser.add_argument("--progress", action="store_true", help="Show Optuna's progress bar.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.measurement_session, level_calibration = prepare_measurement_session(
        args.baseline, args.target, args.level_calibration
    )
    sync_external_objective(args.baseline, args.target, level_calibration)

    global GROUPS
    GROUPS = {k: dict(v) for k, v in (EXPLORE_GROUPS if args.profile == "explore" else SAFE_GROUPS).items()}
    if args.filter_cost_scale is None:
        args.filter_cost_scale = 0.25 if args.profile == "explore" else 1.0

    freqs, traces, rich_traces = load_measurements(level_calibration)
    raw_target = load_target(args.target, freqs)
    target = raw_target + target_anchor_offset(freqs, traces["System Sum"], raw_target)
    base_xml = decode_afpx(args.baseline)
    args.validation = pair_sum_validation(freqs, traces)
    crossover_rows, args.phase_diagnostic_cache = cached_crossover_phase_diagnostics(
        args.phase_cache, freqs, traces, rich_traces, args.measurement_session, args.impulse_root
    )
    apply_session_phase_validity(crossover_rows, args.measurement_session["audit"])
    phase_plan = [] if args.phase_writes == "off" else phase_write_plan(crossover_rows, args.sample_rate)
    failed_validation = [item for item in args.validation if not item["pass"]]
    if failed_validation:
        details = "; ".join(
            f"{item['pair']} {item['rms_db']} dB > {item['threshold_db']} dB"
            for item in failed_validation
        )
        raise SystemExit("Measurement validation gate failed: " + details)

    baseline_groups: GroupBands = {group: [] for group in GROUPS}
    baseline_pred = predict_traces(freqs, traces, baseline_groups)
    baseline_score = tune_scorecard(freqs, baseline_pred, target)
    component_score = phase_safe_component_scorer(
        make_component_scorer(
            freqs,
            traces,
            target,
            filter_cost_scale=args.filter_cost_scale,
            worst_weight=args.worst_weight,
        ),
        freqs,
        phase_plan,
    )
    baseline_score["components"] = component_score(baseline_groups)
    objective = make_objective(
        freqs,
        traces,
        target,
        filter_cost_scale=args.filter_cost_scale,
        worst_weight=args.worst_weight,
        min_total_bands=args.min_total_bands,
        phase_plan=phase_plan,
    )
    baseline_objective = objective(StaticTrial(baseline_groups))

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs,
        show_progress_bar=args.progress,
    )

    out_dir = args.out or ROOT / ("Optimizer_Output_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_rows: List[Dict[str, object]] = []
    args.phase_peq_rejections = []
    seen = set()
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    candidate_groups = [(float(baseline_objective), baseline_groups)]
    candidate_groups.extend(
        (float(trial.value), trial_bands(trial))
        for trial in complete
        if trial.value is not None
    )
    for candidate_value, groups in sorted(candidate_groups, key=lambda item: item[0]):
        sig = bands_signature(groups)
        if sig in seen:
            continue
        seen.add(sig)
        conflicts = phase_peq_conflicts(freqs, groups, phase_plan)
        if conflicts:
            args.phase_peq_rejections.extend(conflicts)
            continue
        rank = len(ranked_rows) + 1
        file_name = f"candidate_{rank:02d}_objective_{candidate_value:.4f}.afpx"
        out_path = out_dir / file_name
        lint = write_candidate(base_xml, out_path, groups, phase_plan=phase_plan)
        pred = predict_traces(freqs, traces, groups)
        score = tune_scorecard(freqs, pred, target)
        components = component_score(groups)
        hr = {}
        for group, bands in groups.items():
            hr[group] = headroom_report(freqs, bands)
        ranked_rows.append({
            "rank": rank,
            "file": file_name,
            "path": str(out_path),
            "objective": candidate_value,
            "score": score,
            "components": components,
            "groups": groups,
            "signature": sig,
            "lint": lint,
            "headroom": hr,
            "left_alone": left_alone_note(freqs, traces),
        })
        if len(ranked_rows) >= args.top:
            break
    args.phase_peq_rejections = args.phase_peq_rejections[:20]

    low_audit = interference_audit(freqs, traces["FL Low"], traces["FR Low"], traces["Mid Bass Together"])
    tw_audit = interference_audit(freqs, traces["FL High"], traces["FR High"], traces["Tweeters Together"])
    notes: List[str] = []
    for label, audit, band in [
        ("Midbass L/R", low_audit, (80.0, 1200.0)),
        ("Tweeter L/R", tw_audit, (2200.0, 16000.0)),
    ]:
        ranges = mask_ranges(freqs, audit[3], band)
        if ranges:
            pretty = ", ".join(f"{lo:.0f}-{hi:.0f} Hz" for lo, hi in ranges[:8])
            if len(ranges) > 8:
                pretty += ", ..."
            notes.append(f"{label} destructive-summing audit flagged: {pretty}.")

    write_family_aliases(out_dir, ranked_rows, base_xml, phase_plan=phase_plan)
    write_report(out_dir, ranked_rows, baseline_score, notes, args, family_rows=ranked_rows,
                 crossover_rows=crossover_rows, phase_plan=phase_plan)

    print("Optimizer complete")
    print("Output:", out_dir)
    print(
        "Baseline: objective %.4f | sum RMS %.2f | image %.2f | worst %.1f"
        % (
            baseline_objective,
            baseline_score["sum_rms_db"],
            baseline_score["sum_wrms_img_db"],
            baseline_score["worst_dev_db"],
        )
    )
    for row in ranked_rows:
        sc = row["score"]
        print(
            "#%d %-36s objective %.4f | sum RMS %.2f | image %.2f | worst %.1f"
            % (
                row["rank"],
                row["file"],
                row["objective"],
                sc["sum_rms_db"],
                sc["sum_wrms_img_db"],
                sc["worst_dev_db"],
            )
        )
        print(format_bands(row["groups"]))
    print("Report:", out_dir / "optimizer_report.md")
    print("CSV:", out_dir / "optimizer_results.csv")


if __name__ == "__main__":
    main()
