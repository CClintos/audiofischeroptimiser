# AudioFischer Optimiser

This repo is a local AFPX tuning tool for Helix / Audiotec Fischer DSP systems.

It is meant to be used through Claude or Codex:

1. Drag in your REW measurement text exports.
2. Drag in your baseline `.afpx` tune.
3. Optionally drag in your target curve text file.
4. Ask Claude or Codex to use this repo as a local AFPX optimizer and run it.

Suggested prompt:

```text
Use this repo as a local AFPX tuning tool.

I have attached:
- my REW measurement text exports
- my baseline .afpx tune
- optionally my target curve

Please verify the files, run the optimizer locally, merge the results, and give me the best AFPX candidates with a short summary of what improved.
Do not change delays, crossovers, polarity, or all-pass filters.
```

The optimizer is designed to:

- improve tonal accuracy
- improve left/right balance
- avoid boosting into destructive nulls
- keep the tune conservative and PEQ-only

Expected measurement files:

- `System Sum.txt`
- `Sub.txt`
- `Front L High.txt` or `Front L Tweeter.txt`
- `Front R High.txt` or `Front R Tweeter.txt`
- `Front L Low.txt` or `Front L Mid.txt`
- `Front R Low.txt` or `Front R Mid.txt`
- `Tweeters Together.txt` or `Both Tweeters.txt`
- `Mid Bass Together.txt` or `Both Mids.txt`

Expected tune file:

- `baseline.afpx`

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
