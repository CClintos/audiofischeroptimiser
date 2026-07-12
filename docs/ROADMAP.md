# ROADMAP.md

This is the short implementation-order version of the deeper research report.

## Easy Wins

- Keep treating the repo as an offline optimiser, not a live analyser clone.
- Prefer compact summaries and explicit validation before broad code review.
- Use project/session configuration instead of pushing more hard-coded workflow assumptions into prompts.
- Strengthen task scoping before implementation so broad requests do not trigger whole-repo exploration.

## Next Good Engineering Targets

1. Add a real automated test suite and fixtures around parser safety, AFPX write safety, and scoring invariants.
2. Add explicit project/session config for layout, aliases, crossover regions, and policy knobs.
3. Tighten delay/APF write gating so "write nothing" is the default when phase confidence is weak.

## Bigger But Valid Later Steps

1. De-duplicate `_tunefit.py` and `objective_module/_tunefit.py` into one shared package.
2. Improve prediction with complex summation when phase is present.
3. Add deterministic post-search refinement.
4. Build a review UI around the current optimiser core.

## Not The Default Direction

- Do not treat "be more like Smaart" as the immediate goal.
- Do not prioritise live capture before the optimiser/review workflow is trustworthy and testable.
- Do not start with a huge rewrite unless the task explicitly asks for architectural work.
