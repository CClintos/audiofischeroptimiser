# DECISIONS.md

Important repo decisions that should not be rediscovered every session.

- This repo is an offline optimiser and review tool, not a live measurement app.
- Keep the optimiser conservative rather than chasing the flattest possible graph.
- Use compact summaries before raw logs or bulk artifact inspection.
- Prefer deterministic or explainable behaviour where practical.
- Weight measurement trust and phase confidence when available; do not blindly assume all data is equally reliable.
- AFPX is the primary trusted automated write path.
- PCT6 support is useful but still more fragile than AFPX and must be treated carefully.
- Candidate-file output is preferred over overwriting baselines.
- Short-term product focus is measurement import, validation, optimisation, and review; live capture is a later phase, not the default direction.
