# Cross-Optimizer Comparison Lessons

This note records the methods that survived a direct comparison between two
independently generated retunes. It is about reusable arithmetic, not copying a
particular tune.

## What The Comparison Exposed

1. A fixed full-range RMS can miss a requested local contour. Compare target and
   candidate shape relative to one fixed reference band. For the Harman retarget,
   the requested rise from 1300 Hz was about `+3.4 dB` at 2 kHz and `+5.8 dB` at
   2.6 kHz. A candidate delivering almost zero rise was a bass-reduced version of
   the old voicing, not a strict retarget, even when its overall score looked good.
2. Never re-anchor baseline and candidate independently. Use one baseline-derived
   target anchor, and require `candidate error - baseline error == raw candidate
   response - raw baseline response` at every checkpoint.
3. Fingerprint the exact AFPX baseline before attributing changes. A filter that
   looks "modified" against one baseline can be an untouched inherited filter
   against the baseline actually used to build the candidate.
4. Model `<Vol>` tags. Uniform front attenuation can preserve headroom after a
   broad voicing boost and changes front-to-sub balance. Ignoring it makes both
   acoustic scoring and change ownership wrong.
5. Recombine unilateral filters through the measured pair and system model. A
   one-sided `-2.5 dB` PEQ does not move the combined trace by `-2.5 dB` when the
   other side still contributes.
6. Absolute L/R level controls image balance. Normalising each side to its own
   local trend can reverse which channel appears to need correction.

## Implemented Consequences

- Beam has a `front_voicing` family: one low-Q transfer is written identically to
  every front output. It can follow a supplied target through a crossover without
  changing L/R balance or relative electrical phase.
- The objective reports and scores `target_shape_error_db`, referenced to
  1000-1400 Hz and evaluated through 1300-5000 Hz. A constant level shift cannot
  improve this component.
- A matched positive front voicing transfer receives a calculated uniform front
  attenuation only when its full PEQ cascade raises the baseline peak. The trim
  is attenuation-only, limited to 6 dB, rounded to 0.25 dB, modelled in the score,
  written to `<Vol>`, and independently verified.
- Matched whole-front PEQ is one logical correction for parsimony and driver-share
  guardrails. It is not misclassified as several inert crossover filters.
- Null masking, peak asymmetry, spatial checks, and the deep/narrow/asymmetric
  guardrails remain active. The new path does not permit narrow one-seat boosts or
  choose a target for the listener.

## Review Checklist

- Confirm measurement and AFPX fingerprints match the session being scored.
- Inspect `target_shape_error_db` as well as full-range tonal RMS.
- Inspect `output_volume_changes_db`; every automatic change must be the same
  negative value on all front outputs and zero on sub/rear outputs.
- Use the raw fixed-anchor response audit when two analyses disagree.
- Treat tiny aggregate differences as ties only after checking local target
  coverage, L/R balance, headroom, and crossover safety separately.
