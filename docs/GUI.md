# Windows GUI

`AudioFischerOptimizer.exe` is the normal no-Codex interface for the local AFPX
optimizer. It is a native PySide6 application and does not use a cloud service.

## Workflow

1. Start on **Home** and choose the workflow that matches the measurements available.
2. For a normal tune, open **PEQ / RTA**, drop in magnitude/RTA measurements and
   the current AFPX, then validate and run. Beam search is used with phase writes disabled.
3. Load the selected PEQ result into the DSP and take fresh phase-valid sweeps.
4. In **Sweeps / Phase**, select the fresh sweep folder and use the PEQ result as
   the baseline. This stage preserves PEQ and searches no new PEQ filters.
5. Validate the measurement session. Core files, tonal provenance, phase
   references, and available solo/together gates are checked before workers start.
   PEQ requires each individual driver plus Sub and System Sum. Together-pair
   traces are recommended but optional; missing pairs reduce the available
   diagnostics and disable phase writes without blocking PEQ.
   When filenames are unfamiliar, map every TXT file to its measurement role in
   the in-app dialog. The run stores `role_map.json`, and remembered names are
   suggested automatically in later sessions.
6. Start the run. Candidate count, objective, worker count, elapsed time, and
   process-tree memory are shown live.
7. Stop safely when needed. Workers save their current state and partial results,
   then the normal merge and AFPX verification path runs.
8. Review and export the verified candidate files from Results.

If PEQ is already dialled in, users may start directly at **Sweeps / Phase**
with the current tune and fresh phase-valid sweeps; a PEQ/RTA run is not required.

**Retarget** is the final tab. Use it later with fresh MMM/RTA measurements of
the current tune when changing to a different target curve; it preserves phase controls.
The selected target is previewed against the built-in reference with both curves
normalized to 0 dB at 1 kHz, making their tonal-shape difference visible.

Results leads with improvement versus the current tune and the same named metric
cards used by the PDF. The current baseline appears beside generated candidates,
and a clear banner warns when the best result does not clear the 1% modelled
improvement threshold. Added filters are shown in a driver/frequency/Q/gain table
and can be copied as DSP-entry text. Warnings, deliberately untouched regions,
and in-car checks have separate readable panels rather than a raw JSON dump.

The response chart has before/candidate/target and optional per-driver predicted
change toggles, hover frequency/dB readout, filter-centre markers, and
click-to-enlarge. The Retarget preview uses the same interactive chart. Loading
Results also creates `SQ_Tuning_Report.pdf` beside the merged candidate files.
The About tab explains the same scoring and safety model in-app.

Alternative guided/CMA/random search methods remain developer CLI options for
benchmarking. They are intentionally hidden from the normal GUI.

The GUI assumes each folder is one fresh measurement session captured without
changing playback or input level. Advanced level-calibration files remain
available through the CLI only.

Runs are stored under `Documents\AudioFischer Optimizer Runs` by default. Each
contains `gui_job.json`, an optional `role_map.json`, worker checkpoints, logs, merged results, verification
JSON, and `assistant_summary.json`. Open Existing Run resumes an incomplete run
or displays an already completed one.

Run folders are reserved atomically. If two GUI instances start in the same
second they receive different suffixed folders, and an active-run claim prevents
the same folder from being started twice. A stale claim from a process that no
longer exists is recovered automatically.

The PowerShell runner is detached from the GUI. Closing during a run offers
**Keep Running and Close**, **Stop Safely and Close**, or Cancel. Background runs
continue writing their normal checkpoints and can be reattached from Recent Runs.

Home shows a live required-measurement checklist and can create a safe 2-way or
3-way folder template with empty REW placeholders and instructions. Last-used
paths are remembered per workflow. Recent runs show their workflow, status, best
objective, and location, and completed entries open directly into Results.

Run remains unavailable until a workflow validates, and Results remains
unavailable until a completed run is loaded. Validation and reports translate
machine warning tokens into severity-coloured explanations with a concrete fix.
Validation also exposes Copy Diagnostics after an attempt; the copied bundle
contains preflight stderr/stdout, the exact GUI job configuration, and the
measurement manifest for support or bug reports.

The optional voicing-audition and sub-level controls include in-app explanations
and tooltips. Voicing files remain neutral listening alternatives. Sub level is
recommendation-only and requires calibrated measurement level plus declared
headroom; it never writes an output-level change.

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

The application/package version has one source in
`optimizer_gui/_version.py`. Packaging reads that value through `pyproject.toml`;
the same version appears in the window title and About tab for support.

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
