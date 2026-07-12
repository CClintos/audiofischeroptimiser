# CLAUDE.md

This repository is a local AFPX optimisation engine for Helix / Audiotec Fischer DSP systems.

## Behaviour

- Read `AGENTS.md` first.
- Read `docs/ai_context/CURRENT_STATE.md` next and treat it as the normal task checkpoint.
- Do not reread the full conversation or raw logs unless compact sources leave a real ambiguity.
- If the task is broad, read `docs/ai_context/PROJECT_MAP.md` before opening code.
- Use `docs/ai_context/TASK_TEMPLATE.md` to turn broad asks into bounded work.
- Prefer generated JSON summaries over raw logs.
- Read `assistant_summary.json` before `optimizer_summary.json` or the Markdown report.
- Do not inspect every candidate AFPX unless verification fails or the user asks for a deep review.
- Do not rerun long optimisation unless explicitly asked.
- Do not change optimiser scoring without a benchmark or before/after score comparison.
- Do not make delay/APF claims unless the measurement exports include phase-valid data or the user explicitly requested those edits.
- Use `docs/AUDIO_DSP_RULES.md` or `docs/ai_context/DSP_RULES.md` before changing DSP logic or tune-writing logic.

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
