# P1 Search Benchmark

Equal search budget on a phase-valid three-position historical session:

- budget: 10 seconds per method
- seed: `20260712`
- profile: explore
- refinement: one candidate, one hardware-step pass
- phase writes: off, to isolate PEQ search quality

| Method | Objective | Trials | Wall time |
|---|---:|---:|---:|
| Beam | 10.530787 | 1928 | 14.288 s |
| Guided | 10.917834 | 1430 | 18.966 s |
| CMA-ES | 10.917834 | 1331 | 19.102 s |

Lower is better. Beam is the one-command default based on this result. The
other methods remain available; one session is not evidence that beam wins on
every system.

The beam candidate passed AFPX verification as PEQ-only. No delay, polarity,
APF, crossover, output attribute, time-alignment attribute, or unknown field
changed.
