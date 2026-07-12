# Roadmap

Implementation order is based on audible leverage, correctness risk, local
compute benefit, and token savings. Detailed evidence is in
`docs/AUDIT_2026-07-12.md`.

## P0: Fix Before Adding Features

1. **Objective integrity release**
   - Add a real positive-deviation/peak term.
   - Replace signed-median-only L/R scoring with signed bias plus weighted
     absolute/RMS mismatch.
   - Return distinct tonal, presence, peak, and balance components.
   - Keep full internal objective precision; round only reports.
   - Add synthetic invariants and a modern TXT/AFPX golden benchmark.

2. **Protect combined PEQ and phase candidates**
   - Initially veto candidate PEQ that materially changes a crossover band when
     a polarity/delay/APF write is attached.
   - Report the rejected filter and affected crossover.
   - Replace the veto later only when full complex biquad prediction is tested.

3. **Enforce measurement-session validity**
   - Feed manifest metadata into the optimizer.
   - Tonal mode: require consistent source level or explicit calibration.
   - Phase mode: permit level-normalized solos but require timing/reference and
     measured-together validation.

4. **Emit `assistant_summary.json`**
   - Bound it to the decision core: fingerprints, gates, baseline/best deltas,
     family files, phase writes/warnings, and re-measure instructions.
   - Make AI guidance read this file before any full report.

## P1: Largest Audible And Local-Compute Wins

5. **Spatially robust objective**
   - Discover optional centre/left/right position bundles.
   - Score centre-weighted median plus high-percentile/worst-position error.
   - Require narrow and asymmetric corrections to hold across positions.

6. **Deterministic beam combinations**
   - Use the existing guided band pools and exact scalar.
   - Retain the best partial combinations per group, then run existing
     hardware-step coordinate refinement.
   - Compare against guided/CMA at equal wall time and seed.

7. **Cache immutable objective work**
   - Cache baseline cascades, masks, smooth solo references, contribution totals,
     and any reusable guided-band responses.
   - Require score equivalence before/after caching.

8. **Run phase diagnostics once per session**
   - Fingerprint and cache the crossover audit, or apply it only at final merge.
   - Workers should search PEQ, not repeat an identical APF grid at checkpoints.

9. **One-command local run**
   - Validate, choose bounded workers, run, merge, verify, and emit only the
     assistant-summary path.
   - Preserve resume/checkpoint behavior and silent long-run mode.

## P2: Audible Extensions With Explicit User Choice

10. **Same-level sub blend recommendation**
    - Recommendation-only first; require calibrated measurements and headroom.
    - Do not fake output trim with broad PEQ boost.

11. **Explicit voicing audition variants**
    - Keep the supplied target untouched.
    - Generate a small labelled set only when requested; never auto-claim a
      preferred tonal balance.

12. **Full complex PEQ/crossover prediction**
    - Model biquad magnitude and phase together.
    - Enable only for phase-valid solo/together sessions.
    - Then jointly verify PEQ with polarity/delay/residual APF.

13. **Unify normal phase entry points**
    - Keep specialist midbass/multinull engines experimental.
    - Expose one canonical phase API and report schema for routine use.

## P3: Maintainability And Distribution

14. De-duplicate the two `_tunefit.py` copies after objective fixtures exist.
15. Add dependency locking and one supported installation/launcher path.
16. Expand AFPX/PCT6 fixtures and property-style byte-preservation tests.
17. Add PCT6 optimization writes only after per-device channel maps are proven.

## Explicitly Rejected For Now

- Special +4 to +5 dB broad-LF boost exception.
- More APF sophistication before P0/P1.
- Blind crossover writes.
- Live analyzer/GUI as the next milestone.
- ML, cloud/API search, or a larger model for numerical optimization.
- Adding optimizer libraries before beam search is benchmarked.
