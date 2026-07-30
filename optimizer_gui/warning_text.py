from __future__ import annotations

from typing import Any


WARNING_TEXT: dict[str, dict[str, str]] = {
    "missing_required_measurements": {
        "severity": "error",
        "title": "Required measurements are missing",
        "meaning": "The optimizer cannot identify every trace needed for this speaker layout.",
        "remedy": "Map the files to speaker roles, or export the missing measurements from REW.",
    },
    "optional_pair_measurements_missing": {
        "severity": "warning",
        "title": "Optional pair measurements are missing",
        "meaning": "PEQ will use the individual drivers and measured System Sum, but it cannot verify how the missing left-and-right pair sums acoustically.",
        "remedy": "You may continue with PEQ. Export the Together traces from REW later to restore pair-null and summation validation.",
    },
    "baseline_missing": {
        "severity": "error",
        "title": "Baseline tune is missing",
        "meaning": "There is no AFPX tune matching the measurements.",
        "remedy": "Choose the AFPX that was loaded when these measurements were captured.",
    },
    "target_missing": {
        "severity": "error",
        "title": "Target curve is missing",
        "meaning": "The optimizer has no tonal target to score against.",
        "remedy": "Choose a valid two-column target-curve TXT file.",
    },
    "measurement_source_volume_changed": {
        "severity": "error",
        "title": "Measurement volume changed",
        "meaning": "The REW exports were captured at different source-volume settings, so raw levels cannot be compared safely.",
        "remedy": "Re-measure the whole set without touching the volume knob, or provide explicit level calibration.",
    },
    "mixed_timing_references": {
        "severity": "error",
        "title": "Timing references do not match",
        "meaning": "The sweeps used different acoustic timing references, so their phase cannot be compared coherently.",
        "remedy": "Re-measure every phase trace with one fixed reference speaker and unchanged microphone position.",
    },
    "measurement_frequency_grids_differ": {
        "severity": "warning",
        "title": "Measurement frequency ranges differ",
        "meaning": "The exports do not share the same start, end, or point count.",
        "remedy": "Export every REW trace with the same frequency range and resolution.",
    },
    "phase_unavailable_impulse_timing_only": {
        "severity": "warning",
        "title": "Phase columns are unavailable",
        "meaning": "Impulse files may support limited arrival timing, but full complex crossover validation is unavailable.",
        "remedy": "Export phase with the REW sweeps and retain the same acoustic timing reference.",
    },
    "phase_unavailable_peq_only": {
        "severity": "info",
        "title": "PEQ-only measurement set",
        "meaning": "The files contain magnitude data but not usable phase.",
        "remedy": "Continue with PEQ, or take fresh timing-referenced sweeps before using the Phase workflow.",
    },
    "uncalibrated_level_mismatch": {
        "severity": "error",
        "title": "Level changes need calibration",
        "meaning": "Some traces were measured at different levels and no correction is available.",
        "remedy": "Re-measure at one level, or provide a role/file-to-dB calibration map.",
    },
    "source_level_metadata_missing": {
        "severity": "warning",
        "title": "Measurement level metadata is missing",
        "meaning": "The app cannot prove that every trace used the same source level.",
        "remedy": "Export REW headers with level information, or confirm the session with explicit calibration.",
    },
    "phase_writes_disabled_mixed_timing_references": {
        "severity": "error",
        "title": "Phase writes are disabled",
        "meaning": "Multiple timing references make delay, polarity, and APF predictions unsafe.",
        "remedy": "Re-measure with one fixed acoustic timing reference.",
    },
    "phase_writes_disabled_timing_reference_missing": {
        "severity": "error",
        "title": "Phase writes are disabled",
        "meaning": "No shared timing reference was found in the sweep metadata.",
        "remedy": "Re-measure using REW acoustic timing reference and keep the reference speaker unchanged.",
    },
    "phase_writes_disabled_pair_measurements_missing": {
        "severity": "warning",
        "title": "Phase writes are disabled",
        "meaning": "A measured Together trace is missing, so acoustic pair summation cannot be validated.",
        "remedy": "Capture each missing Together trace with the same microphone position and acoustic timing reference before applying delay, polarity, or APF changes.",
    },
    "large_boosts": {
        "severity": "warning",
        "title": "Large boosts need a headroom check",
        "meaning": "One or more candidate filters add substantial positive gain.",
        "remedy": "Check the reported headroom and re-measure at the intended listening level.",
    },
    "high_q_filters": {
        "severity": "warning",
        "title": "Narrow filters need verification",
        "meaning": "High-Q filters may be sensitive to seat position or measurement noise.",
        "remedy": "Re-measure around the listening area and reject changes that do not hold spatially.",
    },
    "deep_cuts": {
        "severity": "warning",
        "title": "Deep cuts need verification",
        "meaning": "A large attenuation may indicate a local peak or uncertain measurement.",
        "remedy": "Confirm the affected band with a fresh measurement before keeping the filter.",
    },
    "apf_present_verify_phase": {
        "severity": "warning",
        "title": "All-pass change requires re-measurement",
        "meaning": "The candidate contains a phase correction whose benefit is predicted.",
        "remedy": "Load the candidate and re-measure the affected crossover before accepting it.",
    },
    "many_filters": {
        "severity": "warning",
        "title": "Candidate uses many filters",
        "meaning": "A dense correction may be less robust than a simpler alternative.",
        "remedy": "Compare the restrained family candidate and verify the result across positions.",
    },
}

SEVERITY_COLOURS = {
    "error": "#a12622",
    "warning": "#9a6500",
    "info": "#2b6684",
}


def warning_info(token: Any) -> dict[str, str]:
    raw = str(token).strip()
    key, separator, detail = raw.partition(":")
    entry = WARNING_TEXT.get(key)
    if entry is None:
        readable = raw.replace("_", " ").strip().capitalize() or "Unspecified warning"
        return {
            "raw": raw,
            "severity": "warning",
            "title": readable,
            "text": readable,
            "colour": SEVERITY_COLOURS["warning"],
        }
    suffix = f" Details: {detail.replace(',', ', ')}." if separator and detail else ""
    text = (
        f"{entry['title']}. {entry['meaning']} "
        f"Fix: {entry['remedy']}{suffix}"
    )
    severity = entry["severity"]
    return {
        "raw": raw,
        "severity": severity,
        "title": entry["title"],
        "text": text,
        "colour": SEVERITY_COLOURS[severity],
    }
