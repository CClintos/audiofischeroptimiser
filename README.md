# AudioFischer Optimiser

AudioFischer Optimiser is a Windows app for turning your REW measurements and
current Helix/Audiotec Fischer tune into sensible DSP tuning candidates.

You give it the tune that is already in the car, the measurements you took with
that tune, and a target curve. It creates new candidate `.afpx` files to try. It
does not overwrite your original tune, change crossover settings, or pretend a
measurement can replace listening and verification in the car.

Everything runs locally on your PC. You do not need Codex, Claude, Python, or a
cloud account to use the Windows app.

## Download the Windows app

Download the [latest Windows package](https://github.com/CClintos/audiofischeroptimiser/releases/latest/download/AudioFischerOptimizer-windows-x64.zip),
extract it, then open `AudioFischerOptimizer.exe`.

Keep the extracted `_internal` folder beside the EXE. It contains the bundled
runtime the app needs. No Python installation is required.

Every version, checksum, and release note is on the
[GitHub Releases page](https://github.com/CClintos/audiofischeroptimiser/releases).

## What you need before you start

- Your current Helix/Audiotec Fischer `.afpx` tune. This is the baseline the app
  measures every suggestion against.
- Fresh REW TXT exports from the car, captured with that same baseline tune.
- A target curve. The built-in ResoNix target is included, or you can select
  your own two-column target-curve TXT file.

For normal PEQ work, you need a System Sum, Sub, and a separate left/right
measurement for each front driver. The app recognises common file names and
offers a role-mapping screen when yours are different.

## Choose the job you want to do

### Improve tonal balance with PEQ / RTA

Use magnitude, RTA, or moving-mic measurements to create PEQ candidates. The
app compares your current response with the target, proposes additional filters
for the individual driver groups, and ranks the results against your existing
tune. This is the normal starting point.

You can run PEQ with the required individual-driver, Sub, and System Sum
measurements. Measured left-and-right `Together` traces are recommended but not
required. If they are missing, the app tells you which pair checks it cannot
perform instead of silently making them up.

### Check phase and timing after PEQ

After choosing and loading a PEQ result, take a new set of timing-referenced
sweeps. The separate Sweeps / Phase workflow can assess supported polarity,
relative delay, and residual all-pass changes around a crossover.

It only writes a phase-related change when the measurements support it. Without
matching timing references, usable phase data, and measured pair evidence, it
does not apply delay, polarity, or all-pass changes.

### Retarget a tune you already like

Measure the current tune again, choose a different target curve, and create PEQ
options for that target. Retargeting is for changing tonal preference; it does
not disturb the existing phase controls.

## What you get after a run

The Results screen gives you a practical comparison with your current tune:

- recommended AFPX candidate files, alongside the untouched baseline
- the exact filters added to each output, ready to copy into DSP PC-Tool if you
  prefer to enter them manually
- before, candidate, and target response curves, plus optional driver changes
- a plain-language summary of what improved, warnings, and what to check in the
  car
- `SQ_Tuning_Report.pdf` and `assistant_summary.json` saved with the run files

You can export any candidate to another folder without replacing an existing
export by accident.

## What the app will and will not change

The app is deliberately conservative because a tune file is not a simulator of
the car.

| It can do | It will not do |
| --- | --- |
| Add conservative PEQ filters to a copy of your AFPX baseline | Overwrite your baseline tune |
| Suggest and, in a fully validated phase session, write limited polarity, delay, or residual all-pass corrections | Change crossover frequencies or slopes |
| Keep a baseline candidate so a worse generated option is not selected | Boost an acoustic cancellation/null as if it were an EQ problem |
| Reject incomplete or inconsistent inputs and show the reason | Claim phase confidence from magnitude-only data |

Candidate AFPX files are checked after writing to make sure only allowed fields
changed. Load a candidate, listen, and re-measure before deciding it is your new
final tune.

## Measurement files

The GUI can map your own names, but these are the usual names.

### Required for a 2-way front stage plus subwoofer

- `System Sum.txt`
- `Sub.txt` or `Subwoofer.txt`
- `Front L High.txt` or `Front L Tweeter.txt`
- `Front R High.txt` or `Front R Tweeter.txt`
- `Front L Low.txt` or `Front L Mid.txt`
- `Front R Low.txt` or `Front R Mid.txt`

For a 3-way front stage, provide separate left/right mid and low measurements
as well. The Home page shows a live checklist for the layout it finds and can
create a labelled REW export folder for you.

### Recommended for pair and phase checks

- `Tweeters Together.txt` or `Both Tweeters.txt`
- `Mids Together.txt` or `Both Mids.txt` for a 3-way system
- `Mid Bass Together.txt` or `Both Mids.txt`

These left-and-right pair traces let the app test acoustic summation and detect
destructive interaction at crossover regions. They are optional for a
magnitude-only PEQ run, but required evidence for phase-related writes.

REW exports can contain frequency/SPL only, or include phase, coherence, and
position columns. For phase work, keep the microphone fixed and use the same
acoustic timing reference for the complete session.

## How suggestions are chosen

The app does not simply flatten a graph. It compares candidate changes with the
measured system response, target shape, left/right balance, audible peaks,
driver operating range, filter count, and available headroom. It prefers fewer,
wider and shallower corrections, with cuts favoured over boosts.

This makes the result a shortlist of measured, constrained options—not an
automatic promise that every curve will sound better in every car. The app shows
when an option is not meaningfully better than your current tune, so keeping the
baseline is always a valid result.

## AFPX and PCT6 support

AFPX is the main automated input and output format, and is the format the GUI
creates and verifies.

The repository also contains beta `.pct6` inspection and round-trip utilities
for newer DSP PC-Tool 6 saves. Treat that as an advanced developer tool and
verify any PCT6 file in PC-Tool before loading it into a DSP. See
[PCT6_SUPPORT.md](./PCT6_SUPPORT.md) for the limits.

## Help and development

For the app walkthrough, see [docs/GUI.md](./docs/GUI.md). To report a problem,
use **Copy Diagnostics** after validation and include that output with the
measurement-file names and app version.

If you want to run or modify the source code:

```powershell
.\setup_gui.ps1
.\start_gui.ps1
```

Build a local Windows package with:

```powershell
.\build_gui.ps1
```

The repository's developer reference is in [REPO_MAP.md](./REPO_MAP.md),
[docs/TEST_COMMANDS.md](./docs/TEST_COMMANDS.md), and the scripts under
[`scripts`](./scripts).
