# Windows GUI

`AudioFischerOptimizer.exe` is the normal no-Codex interface for the local AFPX
optimizer. It is a native PySide6 application and does not use a cloud service.

## Workflow

1. Drop in a measurement folder and baseline AFPX.
2. Validate the measurement session. Missing files, tonal provenance, phase
   references, and solo/together gates are checked before workers start.
3. Choose run time, CPU target, optimizer RAM limit, search method, and explicit
   phase/voicing/sub-blend options.
4. Start the run. Candidate count, objective, worker count, elapsed time, and
   process-tree memory are shown live.
5. Stop safely when needed. Workers save their current state and partial results,
   then the normal merge and AFPX verification path runs.
6. Review and export the verified candidate files from Results.

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
