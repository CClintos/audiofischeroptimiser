# Windows GUI

`AudioFischerOptimizer.exe` is the normal no-Codex interface for the local AFPX
optimizer. It is a native PySide6 application and does not use a cloud service.

## Workflow

1. In **PEQ / RTA**, drop in magnitude/RTA measurements and the current AFPX.
2. Validate and run. The app always uses deterministic Beam search and disables
   phase writes for this stage.
3. Load the selected PEQ result into the DSP and take fresh phase-valid sweeps.
4. In **Sweeps / Phase**, select the fresh sweep folder and use the PEQ result as
   the baseline. This stage preserves PEQ and searches no new PEQ filters.
5. Validate the measurement session. Missing files, tonal provenance, phase
   references, and solo/together gates are checked before workers start.
6. Start the run. Candidate count, objective, worker count, elapsed time, and
   process-tree memory are shown live.
7. Stop safely when needed. Workers save their current state and partial results,
   then the normal merge and AFPX verification path runs.
8. Review and export the verified candidate files from Results.

Alternative guided/CMA/random search methods remain developer CLI options for
benchmarking. They are intentionally hidden from the normal GUI.

The GUI assumes each folder is one fresh measurement session captured without
changing playback or input level. Advanced level-calibration files remain
available through the CLI only.

Runs are stored under `Documents\AudioFischer Optimizer Runs` by default. Each
contains `gui_job.json`, worker checkpoints, logs, merged results, verification
JSON, and `assistant_summary.json`. Open Existing Run resumes an incomplete run
or displays an already completed one.

## Resource Controls

- CPU target maps to a bounded worker count, never above 12.
- RAM limit measures the complete optimizer process tree, not only the GUI.
- Three consecutive over-limit samples request a graceful stop.
- If workers do not respond within 20 seconds, the process tree is terminated;
  the most recent disk checkpoint remains available.

## Development

```powershell
.\setup_gui.ps1
.\start_gui.ps1
```

Dependencies are pinned in `requirements-gui.lock.txt`.

## Build

```powershell
.\build_gui.ps1
```

The on-disk package is written to `dist\AudioFischerOptimizer`. It contains:

- `AudioFischerOptimizer.exe`: windowed desktop application.
- `AudioFischerOptimizerWorker.exe`: hidden command worker used by PowerShell.
- `_internal`: bundled Python/DSP runtime and optimizer scripts.

The two-executable design is intentional. Windows PowerShell waits for the worker
binary but does not open a console for the desktop interface.

Install for the current Windows user without administrator access:

```powershell
.\install_gui.ps1
```

This copies the built package to `%LOCALAPPDATA%\AudioFischerOptimizer` and
creates `AudioFischer Optimizer.lnk` on the desktop. Remove it with
`uninstall_gui.ps1`.
