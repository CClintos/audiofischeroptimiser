# CLAUDE.md

This repository is a local AFPX optimisation engine for Helix / Audiotec Fischer DSP systems.

## Behaviour

- Read `AGENTS.md` first.
- Prefer generated JSON summaries over raw logs.
- Do not inspect every candidate AFPX unless verification fails or the user asks for a deep review.
- Do not rerun long optimisation unless explicitly asked.
- Do not change optimiser scoring without a benchmark or before/after score comparison.
- Do not make delay/APF claims unless the measurement exports include phase-valid data or the user explicitly requested those edits.

## Review Output

When reviewing results, return:

1. safest candidate
2. strongest correction candidate
3. risky/rejected candidate
4. filter risk summary
5. expected audible change
6. what to re-measure

## Coding Output

When modifying code, return:

- files changed
- tests run
- benchmark impact, if scoring changed
- safety risks
- next command to run

Keep it short.
