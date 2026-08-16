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
- a Verify workflow that compares the predicted result with fresh measurements
  after you load a candidate

You can export any candidate to another folder without replacing an existing
export by accident. You can also record which candidate you loaded, what you
heard, and whether the post-load measurement confirmed the prediction.

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

### Optional: nearfield captures for null confirmation

- `Front L Nearfield.txt` or `Front Left Nearfield.txt`
- `Front R Nearfield.txt` or `Front Right Nearfield.txt`

A close-mic capture right at each front driver, with negligible room path. If
both are supplied, they let the app confirm whether a dip seen at the seat is
a room/summation artefact rather than a real driver problem - see
[how nulls are treated](#4-treat-nulls-and-missing-evidence-honestly) below.

REW exports can contain frequency/SPL only, or include phase, coherence, and
position columns. For phase work, keep the microphone fixed and use the same
acoustic timing reference for the complete session.

## How the tuning methodology and scoring work

The app does not try to make a line on a graph as flat as possible. It starts
with the tune and response you actually measured, predicts the effect of each
candidate filter on the relevant drivers and the combined system, then compares
every candidate against the same target and the same current-tune baseline.

The objective is a single score used throughout the search: lower is better.
The number is useful for comparing candidates from the same run; it is not a
universal sound-quality rating for different cars or measurement sessions.

### 1. Start from the measured baseline

The baseline AFPX is decoded so existing filters are part of the model. The
target is anchored once to the measured System Sum, then kept fixed while every
candidate is compared. This prevents a candidate from looking better only
because the target was moved to suit it.

The app models individual drivers first, then recombines their predicted change
through the measured pair and system response. A one-sided filter is therefore
not treated as though it changes the whole car by the same number of decibels.

### 2. Score the parts of a useful result

Each candidate is judged on the following evidence together:

| What is scored | Why it matters |
| --- | --- |
| System-to-target error | Reduces broad tonal error across the usable listening range. The vocal and presence region is weighted so a bass improvement cannot hide an obvious midrange problem. |
| Target-shape delivery | Checks whether the requested local contour, especially through the presence region, was actually delivered rather than only improving a full-range average. |
| Broad and narrow peaks | Penalises audible positive peaks, including narrow issues that broad smoothing can hide. |
| Left/right balance | Uses the separate driver traces to penalise level mismatch and image pull rather than letting opposite-side errors cancel out. |
| Worst-case response | Stops a candidate winning because it improves the average while leaving one obvious problem behind. |
| Optional seat measurements | When centre, left-ear, or right-ear system sums are supplied, the candidate must hold up across them instead of chasing one microphone position. |
| Headroom and filter cost | Penalises excessive combined boost, unnecessary output gain, too many filters, high-Q corrections, deep cuts, and filters outside a driver's useful range. |

Headroom accounting covers every output including the subwoofer, and any
active shelf filter already in your tune, not just the PEQ bands a candidate
proposes - a boost that would look safe by PEQ alone but actually pushes a
channel toward clipping once its existing shelf or subwoofer trim is counted
is rejected outright. Before a candidate is written, any proposed filter that
would not measurably change that channel's response on its own, or a pair of
filters that mostly cancel each other out, is dropped rather than left in
place using a slot for nothing.

The result is not simply the candidate with the most filters or the flattest
single trace. At final merge, appended filters are removed when the complete
acoustic component set remains within a cumulative 0.1 dB repeatability
envelope; removals cannot chain into a larger hidden change and are listed in
the report. Existing tune-slot edits are never silently simplified. The search
therefore favours the smallest correction that makes a meaningful improvement
across the evidence available.

### 3. Separate real L/R offsets from cabin combing

A one-sided cut is allowed only when the left-minus-right difference keeps one
dominant sign across roughly one octave around the proposed filter. If the sign
alternates more than once, the app treats the pattern as comb filtering,
driver aiming, or spatial interference rather than a correctable channel-level
offset.

Balance-only corrections are concentrated in the useful imaging band from
about 500 Hz to 8 kHz and are heavily de-weighted below 400 Hz. A one-sided cut
is also rejected when the measured System Sum is already below target at that
frequency, so a small image-score improvement cannot be bought by deepening a
tonal hole.

Every new filter must clear the assumed local measurement-repeatability floor.
The default model uses this rig's same-day MMM repeatability: approximately
0.1 dB at 400-500 Hz, 0.6-1.0 dB at 700-1400 Hz, up to 1.6 dB above that for
midbass, and 0.23-0.46 dB through the tweeter range. The required deviation is
2.5 times the local floor. The exact floor and threshold used are printed in
the tuning report so a user can judge the assumption rather than trusting a
hidden constant.

If you have a second same-day measurement session, the command-line runner can
use `--repeatability-folder` to calculate per-driver, frequency-dependent floors
from your own measurements. An optional `known_eq_delta.json` removes a known EQ
difference between the sessions before the floor is calculated. The empirical
values are saved in the run summary and report.

If you have more than one full measurement session for the same tune, the
command-line runner can also accept them with `--persistence-sessions`. A
correction is only proposed if its sign and size hold up across every supplied
session, not just the one used for scoring - this catches a deviation that
looked real in a single moving-mic pass but was actually run-to-run capture
noise. Each qualifying candidate in the report shows how many sessions
supported it, and sessions are compared by shape rather than absolute level,
so a louder or quieter capture on a different day does not throw off the
comparison.

When a proposed correction falls near a crossover, the search compares the
same region on the upper driver pair, the lower driver pair, and both together.
The full system score decides which scope earns the filter; it does not
automatically copy a tweeter correction onto the midbass or midrange.

### 4. Treat nulls and missing evidence honestly

If the measurements show a destructive acoustic cancellation, that region is
masked from the tonal reward and positive EQ there is penalised. The app does
not call a deep null a "peak to fix" and boost into it.

When close-mic nearfield captures are available for the front drivers, a dip
that looks deep at the seat but is nearly gone right at the driver is
confirmed as a room or summation artefact rather than a driver problem, and
boosting into it - even with a broad filter whose edge only reaches into the
dip rather than sitting on top of it - is blocked outright rather than merely
discouraged.

When individual drivers and System Sum are present but a `Together` trace is
missing, PEQ can still use the measured solo drivers plus system response.
Pair-summation and null checks for that missing pair are shown as unavailable.
The synthetic pair model contributes only its predicted change to the measured
System Sum, so the untouched baseline still reproduces the real system
measurement exactly. The app does not invent phase evidence from magnitude
data.

### 5. Use a stricter method for phase, delay, and all-pass changes

PEQ is a magnitude workflow. Phase-related changes need a fresh,
timing-referenced sweep session. Before the app can write polarity, relative
delay, or a residual all-pass filter, it checks that the measured solo phase can
reproduce the measured together/system behaviour in the crossover band.

Every supported phase candidate is then stress-tested against small timing and
level drift.

Delay search tests only integer samples at the selected DSP sample rate, so
the correction being scored is exactly the correction AFPX can write. Residual
all-pass candidates also pay a frequency-relative temporal cost: a concentrated
millisecond of group delay is treated as more intrusive in the upper mids than
around a subwoofer crossover.

If centre, left-ear, or right-ear crossover snapshots contain the
required solos and measured-together trace, share the same timing reference, and
pass acoustic-sum validation, the chosen hardware setting must improve the worst
validated position. Mixed references, fragile improvements, and conflicts with
impulse evidence are rejected rather than written.

If those gates do not pass, phase controls stay untouched. The app can still
complete a PEQ run; it simply does not make a phase claim it cannot support.

### 6. Direct search effort toward supported improvements

Before the timed search begins, the app builds a problem census. It ranks
measured regions by error, audibility, driver authority, and whether the
evidence says they are fixable. The report lists both the strongest supported
opportunities and the regions deliberately skipped, including the reason.

Search capacity is then allocated according to recoverable error instead of
giving every driver group the same candidate budget. Peak spacing becomes finer
where an issue is strong and well above the noise floor, which keeps nearby real
features such as the measured 2.67 kHz peak available without filling the pool
with measurement noise. The Run screen shows recent objective improvements and
whether the search is still improving or has stalled.

### 7. Keep the baseline and verify the output

Your original tune is always retained as a candidate. If a generated option is
not meaningfully better than the baseline, the results say so rather than
forcing a change. Every written AFPX candidate is decoded and checked against
the baseline to confirm that only the permitted fields changed.

After loading a candidate, use the Verify tab with fresh REW exports. It overlays
the response the model predicted with the response actually achieved for System
Sum and available drivers, reports their difference, and saves the verification
beside the run. A listening-decision entry can link the candidate, your notes,
and that measured result. Existing run folders can also be replayed under newer
app versions to show which old proposals now fail current guardrails and why.

This produces a constrained shortlist to test in the car, not an automatic
promise that a graph will sound better everywhere. Load a candidate, listen,
and re-measure before making it your final tune.

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
