# DSP_RULES.md

This is the short AI-context version of the domain rules.

Read `docs/AUDIO_DSP_RULES.md` if the task directly changes scoring, phase/delay/APF logic, or AFPX/PCT6 writing.

## Core Rules

- Do not aggressively boost narrow deep nulls.
- Do not treat low-confidence measurement dips as safely correctable.
- Prefer broad cuts over boosts.
- Preserve channel mapping and tune-write safety.
- Preserve polarity and delay semantics.
- Judge sub/front alignment around the crossover region, not full-band.
- Phase/delay/APF changes need before/after summation validation.
- Any AFPX/PCT6 writer change needs decode-write-decode verification.

## Scope Rules

- PEQ changes are safer than crossover or polarity changes.
- Delay/APF claims require usable phase evidence.
- If confidence is weak, report the limitation and what to re-measure.
