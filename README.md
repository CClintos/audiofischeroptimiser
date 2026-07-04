# AudioFischer Optimiser

This repo is a measurement-driven AFPX optimizer for Helix / Audiotec Fischer DSP tunes.
It is designed as a tool that other people can run against their own files:

- their own baseline `.afpx`
- their own REW text exports
- their own target curve, or the included example target

The optimizer does not try to auto-magically flatten a mono sum. It scores candidate tunes with a perceptual objective that:

- improves tonal accuracy with extra weight in the vocal band
- improves left/right balance from solo-driver traces
- penalizes boosting into destructive-interference nulls
- penalizes unnecessary positive gain and filter clutter
- leaves delays, crossovers, polarity, and all-pass filters untouched

## What It Does

The long-run optimizer uses a constant-memory, multi-worker search architecture. It reads your baseline tune and REW measurements, generates plausible PEQ candidates in the relevant passbands, tests many combinations, and exports ranked AFPX files you can actually audition.

Outputs typically include:

- ranked `candidate_*.afpx` files
- `family_conservative.afpx`
- `family_balanced.afpx`
- `family_aggressive.afpx`
- `optimizer_report.md`
- `optimizer_results.csv`

## What You Need

Put your input files in one folder and point the optimizer at it.

Required measurement exports:

- `System Sum.txt`
- `Sub.txt`
- left high solo: `Front L High.txt` or `Front L Tweeter.txt`
- right high solo: `Front R High.txt` or `Front R Tweeter.txt`
- left low solo: `Front L Low.txt` or `Front L Mid.txt`
- right low solo: `Front R Low.txt` or `Front R Mid.txt`
- high pair trace: `Tweeters Together.txt` or `Both Tweeters.txt`
- low pair trace: `Mid Bass Together.txt` or `Both Mids.txt`

Required tune file:

- `baseline.afpx`

Target curve:

- by default the repo uses `ResoNix Target Curve 2026.txt`
- you can override it with your own target file

## Quick Start

1. Create a working folder with your REW exports and rename your baseline tune to `baseline.afpx`.
2. Run the worker launcher and point `-DataRoot` and `-Baseline` at your files.
3. Wait for the run to finish.
4. Run the merge script to build the final ranked AFPX outputs.

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_guided_stream_workers.ps1 `
  -Root "Optimizer_Run" `
  -Workers 8 `
  -Seconds 1200 `
  -Proposal guided `
  -ValidationThreshold 3.0 `
  -DataRoot "C:\path\to\your\measurements" `
  -Baseline "C:\path\to\your\measurements\baseline.afpx" `
  -Target ".\ResoNix Target Curve 2026.txt"
```

Then merge:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\merge_guided_stream_results.ps1 `
  -Root "Optimizer_Run" `
  -Top 30 `
  -ValidationThreshold 3.0 `
  -DataRoot "C:\path\to\your\measurements" `
  -Baseline "C:\path\to\your\measurements\baseline.afpx" `
  -Target ".\ResoNix Target Curve 2026.txt"
```

## Using It With Claude or Codex

You can also hand this repo to a coding model such as Claude or Codex and have it run the optimizer for you.

### What To Give The Model

Attach or point it to:

- this repo
- your measurement folder
- your `baseline.afpx`
- optionally your own target curve

Make sure your measurement folder contains the expected REW text exports listed above.

### Suggested Prompt

Use something close to this:

```text
Read the optimizer repo in this folder and use it as a local AFPX tuning tool.

I have attached or dragged in:
- my REW measurement text exports
- my baseline .afpx tune
- optionally my target curve text file

Please:
1. verify the required measurement files are present
2. run the streaming optimizer locally for 20 minutes
3. merge the worker outputs at the end
4. give me the best AFPX candidates and a short summary of what improved

Do not invent your own scoring method if the repo already defines one.
Do not change delays, crossovers, polarity, or all-pass filters.
Treat this as a local file-and-script workflow, not a rewrite project.
```

### What A Good Agent Run Should Do

- read the repo before making assumptions
- check that your measurement filenames match the accepted aliases
- launch `run_guided_stream_workers.ps1`
- wait for the run to finish
- run `merge_guided_stream_results.ps1`
- hand back the best `candidate_*.afpx` files or family picks

### What To Watch Out For

- If the solo/pair validation gate fails, the model should stop and tell you why instead of tuning anyway.
- If your sweep names differ from the supported aliases, the model should either map them or ask you to rename them.
- If a model starts proposing delay or APF changes automatically, it is going beyond what this repo is built to do.

## Main Files

- [_optimizer.py](./_optimizer.py): core scoring, prediction, AFPX writing, reporting
- [_optimizer_stream.py](./_optimizer_stream.py): constant-memory multi-worker optimizer
- [_merge_stream_results.py](./_merge_stream_results.py): merges worker archives into final outputs
- [run_guided_stream_workers.ps1](./run_guided_stream_workers.ps1): launches long local runs
- [merge_guided_stream_results.ps1](./merge_guided_stream_results.ps1): safe merge wrapper
- [objective_module/afpx_objective.py](./objective_module/afpx_objective.py): independent scalar objective used by the optimizer
- [objective_module/_tunefit.py](./objective_module/_tunefit.py): DSP/math helpers used by the objective module

## Safety / Scope

This tool is intentionally conservative.

- It optimizes PEQ only.
- It does not edit delay tags.
- It does not change crossovers.
- It does not write polarity or APF changes.
- It treats destructive summing regions as not EQ-fixable.

That means it is best for tonal work, not for automated phase alignment.
