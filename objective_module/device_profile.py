# device_profile.py -- consolidated hardware limits for supported Helix/
# Audiotec Fischer DSPs.
#
# Repo-review finding: the DSP's internal sample rate and its PEQ frequency/
# Q/gain range were hardcoded literals scattered across _tunefit.py
# (FS = 96000.0) and _make_v3.py (validate_peq_band's 20-20000 Hz/0.5-15 Q/
# -15..+6 dB range), with no single place documenting where those numbers
# came from or that they're specific to one verified model. Consolidated
# here as the one source of truth; the two call sites now read from it
# instead of repeating the literals (value-preserving - same numbers,
# same behaviour, just named and sourced in one place).
#
# This is deliberately NOT a device-selection feature yet - there is only
# one verified profile below. Introducing it is the first step toward
# broader Fischer-device support, not a claim that support already exists;
# using it for a different model without a controlled export-diff
# verification pass (see helix_hardware.md / afpx_format.md in the
# helix-rew-tuner skill) would be guessing, not verifying.
#
# GROUPS's own gain_range/q_range in _optimizer.py are a SEPARATE thing -
# deliberately conservative ACOUSTIC SEARCH POLICY (how wide the search is
# willing to go), not a hardware limit. Do not fold those into this file;
# they answer "should we," this file only answers "can the hardware."
from dataclasses import dataclass, field
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class DeviceProfile:
    model_id: str
    sample_rate_hz: float
    peq_frequency_range_hz: Tuple[float, float]
    peq_gain_range_db: Tuple[float, float]
    peq_q_range: Tuple[float, float]
    filter_slots_per_output: int
    supported_filter_types: FrozenSet[str] = field(default_factory=frozenset)
    container_formats: FrozenSet[str] = field(default_factory=frozenset)
    verified_pc_tool_versions: Tuple[str, ...] = ()


# Verified 2026-07 by controlled export-diff against a Helix P SIX DSP MK2
# (DSP PC-Tool 4/6.01.08/6.03.04) - see afpx_format.md / pct6_format.md in
# the helix-rew-tuner skill for the underlying per-attribute verification
# notes. NOT independently verified on other Helix models.
HELIX_P_SIX_MK2 = DeviceProfile(
    model_id="helix_p_six_mk2",
    sample_rate_hz=96000.0,
    peq_frequency_range_hz=(20.0, 20000.0),
    peq_gain_range_db=(-15.0, 6.0),
    peq_q_range=(0.5, 15.0),
    filter_slots_per_output=30,
    supported_filter_types=frozenset({"1", "3", "4", "9", "15", "16", "17", "19", "20"}),
    container_formats=frozenset({"afpx", "pct6"}),
    verified_pc_tool_versions=("4", "6.01.08", "6.03.04"),
)

DEFAULT_DEVICE_PROFILE = HELIX_P_SIX_MK2
