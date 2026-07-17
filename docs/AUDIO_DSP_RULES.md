# AUDIO_DSP_RULES.md

These rules are meant to make Codex faster and more accurate, not more timid.

## Non-Negotiable Safety Rules

- Do not overwrite the baseline tune.
- Write candidate files only.
- Do not change crossovers or polarity unless the user explicitly asks for that exact operation.
- Treat destructive summing/null regions as not EQ-fixable.
- Prefer fewer, wider, shallower filters.
- Prefer cuts over boosts.
- Permit a broad matched whole-front boost only when it follows the supplied
  target shape and a verified uniform attenuation preserves baseline headroom.

## PEQ Rules

- Base PEQ decisions on magnitude data.
- Penalise filters on drivers that are not materially contributing at that frequency.
- Penalise unnecessary asymmetry, wasted filters, very high Q, and excessive gain.
- Avoid single-seat vanity corrections unless the supporting solo measurements justify them.

## Phase / Delay / APF Rules

- Run the active crossover ladder in order: polarity, then relative delay, then APF only for a supported residual.
- Polarity/delay/APF changes are allowed only when the measured together trace validates the solo prediction, or strong band-limited impulse evidence plus destructive summation clears the fallback gate.
- One-sided APFs above 1 kHz are report-only, not automatic writes.
- Do not make strong phase claims from poor-confidence or mismatched-reference data.
- When confidence is not full, the report should say so and state what to re-measure.
- Conservative APF use is acceptable only when it solves a supported crossover-band problem; do not add APFs speculatively.

## Review Rules

- Start from `assistant_summary.json` before reading the full summary, reports, CSVs, or candidate files.
- Explain audible trade-offs, not just score deltas.
- Distinguish the safest candidate from the strongest correction candidate.
- Anchor the target once from the measured baseline and reuse it for every candidate. Never re-anchor before/after responses independently.
- Compare requested and delivered local target shape relative to the same fixed
  reference band; full-range RMS alone is not proof that a retarget happened.
- Fingerprint the exact baseline before assigning ownership to a changed filter.
- Include AFPX `<Vol>` deltas in prediction and verification.
- Verify candidate minus baseline as a raw delta. At every checkpoint, `candidate error - baseline error` must equal that raw delta.
- Recombine unilateral EQ through the solo pair and system model; a 3 dB filter on one side is not a 3 dB change to the combined response.
- Judge imaging from absolute L/R mismatch. Do not normalize each side to its own local trend and call the result balance.
- Treat tiny modeled differences as ties unless a named tonal, peak, spatial, or L/R component shows a meaningful advantage.

## Measurement Rules

- Use the measurement manifest first to confirm what is present.
- If phase/coherence columns are absent, stay in PEQ-first mode and avoid phase-driven edits.
- For front 3-way work, prefer separate branch measurements so the optimiser can score the real layout instead of guessing from a simpler 2-way model.

## AFPX / PCT6 Rules

- AFPX is the primary trusted automated write path.
- PCT6 support is beta and should be treated as careful utility work, not blind write automation.
- For PCT6, keep byte-preserving workflows intact and do not assume decoded channel order matches visible PC-Tool output tabs.
- Before trusting a PCT6 write-path change, decode original, write output, decode again, and verify that only intended fields changed.
