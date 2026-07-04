# Tuning workflow: read REW .mdat → edit Helix .afpx (paste into a new chat)

This is a self-contained brief. Hand it to a fresh Claude/Codex chat along with **one `.mdat`** (the REW measurement) and **one `.afpx`** (the tune that was loaded *at the time of that measurement*). It explains how to decode both and edit the tune from the data instead of eyeballing a graph.

> **Scope:** this doc is ONLY the direct `.mdat → .afpx` editing workflow (no app needed). The separate browser-app project ("REW-less Tuner") has its own handoff at `C:\Users\Adroit\Downloads\rewless tuner\HANDOFF.md` — fork that one for app/code work. The two don't depend on each other.
>
> **Note on OneDrive:** files on the Desktop are often cloud-only and may not open in a sandboxed session. Put the `.mdat`/`.afpx` you want edited in `Downloads\` or `Documents\` (both reliably readable).
>
> **Note on model choice:** this doc carries the domain knowledge (hardware facts, physics, verified encodings, reusable code) — that part transfers fully to any fresh session that reads it. What does NOT transfer is reasoning quality on new, messy, real data: classifying which bump is real, iterating on a prediction that overshoots, catching a bug in your own math. Use a strong reasoning model (Opus/Fable-tier) for actual measurement-to-EQ judgment calls; a lighter model is fine only for pure mechanical edits with no judgment involved (e.g. "change this existing band's gain from −4 to −6").

---

## System facts (do not re-derive)

- **DSP:** Helix P SIX DSP MK2, 8 channels, in a **RHD VW Golf MK7**.
- **Source:** clean **optical (digital) input** — no OEM/factory head-unit processing in the chain. So **no input de-EQ / source-chain audit is needed** (§0's source-chain step is moot for this system), and there's no analog input-gain-sensitivity concern. The "source de-processing" gap both research digests flag as the #1 place a human beats an auto-tune does **not** apply here.
- **Tweeters:** A-pillar pods, asymmetrically aimed (one near-ear, one into the dash).
- **Mic is closer to the right side** (RHD). Two different kinds of EQ live in this file — keep them straight:
  - **Voicing corrections** (derived from the *System Sum* vs the target — the bumps you compute in §3): **apply IDENTICALLY to both L and R of the pair.** Same F/Q/G.
  - **Per-driver corrective EQ** (from individual driver sweeps — the dense low-mid bands that differ side-to-side): **legitimately different L vs R, do NOT flatten them.** Each driver was corrected to its own measured response; making them equal would un-correct the quieter/duller side. **To CREATE this layer you must measure each driver SOLO** — the System Sum can't separate L from R, so anything derived from it is unavoidably shared. A tell that "per-driver" bands were actually *mirrored* (not measured per side): identical high-precision auto-fit values on both channels (e.g. `277.0 Hz Q2.235 G−5.52` on L *and* R) — two independent solo measurements never coincide to two decimals. Leave existing per-driver bands alone unless the data says otherwise.
  - **Electrical gain (level/Vol):** equal per pair. Path/timing differences belong to delay (TA), never gain.
- **Target = ResoNix house curve:** ~+13 dB bass shelf at 20 Hz tapering to flat by ~300 Hz; flat mids 300 Hz–1 kHz; ~−7 dB presence scoop centered ~2.6 kHz; relaxed top (−2 to −3 dB) from 5–20 kHz.
- **Time alignment / delays:** ATM is the baseline; the user now **manually refines TA in PC-Tool** via the REW digital-routing procedure (§5). **My Python edits STILL never touch `<T .../>` delay tags** — TA is the user's job in the PC-Tool UI. The "delays byte-unchanged" check guards a single EQ edit from corrupting delays; it does NOT mean delays are frozen forever. **After a user TA session, the new `<T>` values are the legitimate new baseline** — treat a post-TA tune's delays as the reference and keep doing EQ-only edits on top.

---

## Non-negotiables (the 12 rules a fresh session must not miss)

1. **Never edit `<T>` delay tags in Python** — verify byte-identical after every save (§4). User changes them only via PC-Tool TA sessions (§5).
2. **PEQs are always `T="17"` in middle slots** (skip first/last/band-2). Special types are now fully mapped and writable — shelf `T=3`/`T=4` (end slots, defrag the squatter PEQ first), all-pass `T=19`/`T=20` (any slot) — but only via the verified writers in `_tunefit.py`, never hand-guessed (§1).
3. **Hardware limits:** G ∈ [−15, +6] (cap boosts at +3), Q ∈ [0.5, 15]; stacking same-freq bands OK for cuts, never boosts (§1).
4. **Anchor with `(?<![A-Za-z])F=`** — a naive `F="` regex grabs the one inside `dF=` (§1).
5. **Validate the mdat axis** against the 415 Hz null + 47 Hz peak before trusting any frequency; verify a measurement's *shape*, not its name (§2).
6. **Cut peaks, don't fill dips.** A null gets ≤+3 partial lift at most; a *one-sided* dip usually means cut the other side's peak instead (§3). **Traffic-light every bin GREEN/YELLOW/RED before fitting (§3b); RED never gets a filter.**
7. **Sum = shared EQ only** (both channels identical); different L/R EQ requires solo sweeps. Never invent L/R differences from a sum (§3).
8. **Fit bands JOINTLY (`_tunefit.py: fit_peq`), then simulate the cascade before writing** — greedy one-band-at-a-time fitting is the TuneEQ mistake; skirts interact within ~1 octave (§3, §3b). Report the audibility score before → after.
9. **Narrow features above ~400 Hz that move between sessions/positions are comb artifacts — don't EQ them.** Averaged (grid or MMM) data is magnitude-only; phase work needs fixed-position + acoustic timing ref + "Adjust clock" (§2).
10. **Close the loop:** predicted ≠ actual — re-measure after loading. Report MUST include audibility score before→after, **headroom/clip-risk**, and the **rejected-corrections list** with reasons (§4). Level-matched (0.2 dB) ear validation is the final ratify step.
11. **Always ask for the standard capture set (solo A, solo B, together) per summing pair, not just solos+Sum** — the together-trace enables `interference_audit` to catch L/R cancellation from magnitude alone, no phase capture needed (§2, §3).
12. **A `.afpx` returned from PC-Tool (not your own Python write) needs a FULL diff, not just the changed band** — PC-Tool round-trips renumber `FN` and can silently perturb unrelated filters (§1).

---

## 0. Preflight — measurement hygiene, metadata, source-chain (do this BEFORE §1)

*Added from research digest #2 (2026-06-28). A perfect EQ solver on a bad measurement — or one fighting a source-side artifact — loses to a careful human. These gates come first.*

**Measurement hygiene — don't generate a tune unless ALL are true:**
- UMIK-1 calibration file loaded; the **"MICROPHONE"** input selected (not Default Input).
- OS/driver **input AGC OFF** (Windows mic "enhancements" off) — AGC silently rides the level and ruins a sweep.
- Sweep/pink level in REW's **green** zone, never orange/red (clipping = harmonic mush no output EQ can fix). Real pink-noise/sweep file, not compressed audio.
- **TA + crossovers already set** (EQ is last). Gain structure set: **quietest channel at 0 dB, louder channels attenuated — never boost the master/output above 0 dB** (digital overmodulation is harsh and kills tweeters).

**Record this metadata with every measurement set (don't compare sets that differ):** seat position, windows/sunroof, doors, HVAC + fan level, cabin occupancy, engine on/off, playback volume, mic-grid coordinates. A dip at one HVAC/window/seat state is NOT comparable to another.

**Source-chain audit — NOT NEEDED for this system (confirmed 2026-06-28).** The P SIX is fed by a **clean optical (digital) input**, so there is no OEM EQ / HP-LP / delay / all-pass to detect or de-EQ, and no analog input-gain concern. **Skip this step.** The reference notes below are retained only in case the source ever changes:
- *OEM/factory head unit via high-level input (not this system):* the OEM may apply factory EQ, HP/LP, delay, even all-pass — you'd measure the input first and de-EQ at the Input-EQ layer, never letting output PEQ chase a source-side artifact. (The auto-detect "AISA" tooling the digests cite is a newer PC-Tool 5/6 / ACO feature that likely isn't on the P SIX MK2 anyway.)

---

## 1. Decode / encode the `.afpx` (VERIFIED on real files)

The file = **4-byte big-endian uint32 header that equals the uncompressed XML length** + standard zlib payload. The header is a length, **not** a magic number — recompute it on save.

```python
import zlib, struct

def read_afpx(path):
    raw = open(path, 'rb').read()
    return zlib.decompress(raw[4:]).decode('utf-8', 'replace')  # XML string

def write_afpx(path, xml_str):
    xml = xml_str.encode('utf-8')
    out = struct.pack('>I', len(xml)) + zlib.compress(xml, 9)
    open(path, 'wb').write(out)
```

XML shape: `<ATF ...> <OC ON="N" IS="0|1" ...> ... <Fil I="0" Q="1.3" T="17" dF="100" F="110.00" G="-4" FN="206"/> ... </OC> ... </ATF>` — one `<OC>` per output channel, ~30 filter slots each.

**`<Fil>` attributes — READ CAREFULLY:**
- `F` = **the real center frequency in Hz** (a float like `"13000.00"`, `"470.00"`). This is the one you edit.
- `dF` = the slot's nominal ISO label (25,32,40,…20k) — cosmetic, can differ from F (e.g. `dF="630" F="470.00"`). **Do NOT mistake `dF` for the frequency.** A regex for `F="..."` will wrongly grab the `F=` inside `dF=` — anchor it: `(?<![A-Za-z])F="([^"]*)"`.
- `Q`, `G` = Q and gain(dB). `T` = type. `FN` = filter id (keep unique). **`I` = INVERT flag (0/1) — NOT an index** (corrected 2026-07-03: pressing the Allpass panel's "invert" flipped exactly `I="0"→"1"` in the export-diff and nothing else). Present on every `<Fil>`; writers take `invert=True`.

**Filter types (`T=`) — COMPLETE VERIFIED MAP (updated 2026-07-03 by the "Test .afpx" export-diff, which CORRECTED the earlier "T=20 = shelf" inference — supersedes anything below that says otherwise):**

| T | Meaning | Slot | Status |
|---|---------|------|--------|
| `1` | free/off slot (`G="0"`) | any | verified |
| `17` | **Parametric EQ** (the normal band; obeys AutoSort) | any | verified |
| `15` / `16` | LP / HP crossover — don't touch | fixed | verified |
| `3` | **LOW SHELF** (active when `G≠0`; Q 0.1–2 IS the slope) | **band 1 (`dF="25"`) only** | **VERIFIED** (real export: `T="3" F="4980.25" Q="1" G="-2.25"`) |
| `4` | **HIGH SHELF** (active when `G≠0`) | **band 30 (`dF="20000"`) only** | **VERIFIED** (real export: `T="4" F="5400.00" Q="0.5" G="0.25"`) |
| `19` | **1st-order ALL-PASS** (−90° at corner; no Q — written as `Q="1"`) | **any slot incl. MIDDLE** (seen at `dF="2000"`) | **CONFIRMED** (export-diff + PC-Tool screenshot: "Q: N/A for 1st order", "1. Order" active) |
| `20` | **2nd-order ALL-PASS** (`G="0"`; Q 0.5–2 meaningful) | any (verified at band 30; middle plausible per T=19 finding) | **VERIFIED** (430 Hz Q0.7 export-diff) |

**Corrections this map forces:** (a) shelf and all-pass do NOT share a code — shelves are `T=3`/`T=4`, all-passes are `T=19`/`T=20`; (b) the old tunes' parked `T=20` bands (G=0, sometimes Q>2) were parked **all-passes**, not shelves — their out-of-range Q values are stale XML left over from the slot's previous PEQ life; (c) **all-passes can live in MIDDLE slots** — they never compete with shelves for the two end slots; (d) **switching band 1/30 into shelf mode CONSUMES whatever filter lived in that slot** (the Test export ate the legacy +1.97 @ 2835 PEQ that squatted band 1) — so DEFRAG FIRST: relocate the squatter PEQ to a free middle slot before a shelf goes in, or its correction is silently lost.

**FORMAT FULLY MAPPED (2026-07-03):** with the invert flag verified, there are no remaining unknown encodings. **Python may now write ALL of these** — `_tunefit.py`: `shelf_fil_str('low'|'high', F, Q, G, FN)`, `allpass_fil_str` (2nd-order), `allpass1_fil_str` (1st-order) — each semantically exact-tested against the real PC-Tool export lines.

**RULE — only ever write `T="17"` for PEQ.** Never introduce a special-type band (shelf `20`, or all-pass) unless its *function* is genuinely needed: specials don't AutoSort and clutter the list, which is exactly what makes a tune hard to adjust.

**Hardware gain ceiling (official, confirmed) — applies to every band, even Parametric:** **max boost +6 dB, max cut −15 dB, per band.** Deliberately asymmetric — Audiotec Fischer's own docs say boosts >6 dB are almost never needed and narrow dips are usually phase cancellation, which EQ can't fix regardless of how hard you boost (focus on cutting peaks, not filling dips). **Stacking multiple bands at the SAME frequency is explicitly fine — even recommended — for deep CUTS** (e.g. 4× bands at the same freq, Q15, −15dB each = −60dB total, their official example for killing a resonance/rattle) — **but never for boosts** at the same frequency: stacked boosts compound fast and risk real digital clipping ("can destroy your speakers in seconds," their wording). The app's own internal caps (`MAX_BOOST_DB=3`, `MAX_CUT_DB=12` in `tuning-session.html`) are already well inside this hardware ceiling — when writing gains directly via Python (bypassing the app), the hardware figures above (+6/−15) are the real ceiling to respect, not the app's tighter heuristics.

**Resolution note:** PC-Tool's default EQ slider step is 0.25 dB (configurable to 0.5/1 dB in DCM). Write gain values as clean multiples of 0.25 dB so nothing visually snaps when the user next opens PC-Tool.

**VERIFIED PC-Tool 4.x / P SIX MK2 filter & crossover spec** (official AF DSP PC-Tool 4 page + P SIX MK2 manual, 2026-06-28 — this resolves the version uncertainty for the filter set):
- **Four EQ filter modes:** Parametric EQ, **FineEQ** (a finer parametric variant — same Q range; we still only write standard PEQ `T="17"`), Allpass, and Low-/High-shelf.
- **Parametric / FineEQ:** Q **0.5–15** (confirms what we use).
- **All-pass:** **1st order (−90° at corner) or 2nd order (−180°)**, **Q 0.5–2** — this is the OFFICIAL range; it supersedes the earlier "freq-dependent max ~1.5" figure from a weaker source. (The low-Q *preference* still stands — see §3.)
- **Shelf:** **Q 0.1–2**, frequency in 1 Hz steps. That **Q is the slope control** — there is **no separate S/slope parameter** documented, and the active-shelf XML encoding is still unconfirmed (get it by export-diff before writing one).
- **Crossovers:** five families — **Butterworth, Bessel, Tschebyscheff, Linkwitz, Self-Define** — slope ceiling **−42 dB/oct** on this 64-bit / 96 kHz unit (not just LR4/24 dB/oct).
- **PC-Tool graph:** magnitude **+ phase**, RTA to **1/24 oct** — but **no group-delay display** (use REW for group delay).
- **I/O confirmed:** 6 RCA + 6 high-level + **1 optical SPDIF (12–96 kHz)** in; 6 speaker out + 2 processed pre-outs; **64-bit DSP @ 96 kHz** (so the biquad sim's `FS=96000` is officially correct), 20 Hz–44 kHz.

**Reserved band slots (PC-Tool 4.8 / P SIX) — don't burn these on narrow PEQs:**
- **First band** (lowest slot, `dF="25"`): can be switched to a **Low Shelf** — reserve for broad bass-shelf / house-curve shaping, not a narrow cut.
- **Last band** (highest slot, `dF="20000"`, or the sub's top slot): can be switched to a **High Shelf** (this *is* the `T="20"` we found) — reserve for broad treble tilt.
- **Band 2** (second slot, `dF="32"`): on **4.8, Tone Control / Remote Tone Control steals EQ band 2** of every output channel when enabled (a later PC-Tool version removed this). Keep **Tone Control OFF** (DCM → Remote / DIRECTOR / WIFI CONTROL) to keep band 2 usable; if band 2 looks greyed/missing, that's why.
- **AutoSort** (DCM → *EQ Band AutoSort*): reorders Parametric bands by frequency for **display only** (no sound change). Turn it ON when running many PEQs so the list stays readable.

So when converting free slots to PEQs, **prefer the middle slots and skip the first, the last, and (if Tone Control is on) the second** — preserving the shelves and the tone-control band.

**SHELF STRATEGY (audited + simulated 2026-07-03) — when to actually use band 1 / band 30:**
- **Current occupancy (audit of the live tune):** every channel's **band 1 is burned by a legacy PEQ** (placed before the reservation rule existed): tweeters 2835/2676, mids 277, rears 412/462, subs 38.9. Band 30 is free everywhere except FL Low (the 430 Hz APF). **Consequence: no low shelf can be placed anywhere today without a "slot defrag"** — move the squatter PEQ to a free middle slot (17–26 free per channel; audio-identical, verifiable with the round-trip lint since only slot labels change). Do the defrag lazily, only when a shelf is actually wanted on that channel.
- **Which channels even want which shelf:** tweeters → **high shelf only** (HP'd at 2600; their band-1 burn is irrelevant). Mids → low shelf plausible (80–300 region), high shelf pointless (LP'd at 2600 — which is also why FL Low's band-30 APF costs nothing). Subs → **low shelf only** (LP'd at 80; band 30 useless): after a defrag of the 38.9 PEQ, a sub low shelf is the natural **bass-level voicing knob** (§4 envelope trims) instead of re-touching the sub PEQs.
- **When a shelf beats PEQs (the policy):** (1) a **broad tilt anchored at a spectrum end** — relaxed top, bass shelf voicing; (2) **replacing ≥2 overlapping low-Q PEQs whose combined shape is genuinely shelf-like**; (3) as the **±0.75–1 dB voicing knobs** of §4 (one shelf move beats stacking bells). **When NOT:** any mid-band problem (that's a bell), and any hump/dip that returns to 0 on both sides — a shelf stays down.
- **Simulated consolidation verdicts (real tune, `fit_shelf_to_curve`):** tweeter pair −1.75@5k Q1.8 + −1.5@13k Q0.7 → **one high shelf −1.25 @ 3306 Hz Q1.7 replicates it within ±0.5 dB (3–20k)** — a valid consolidation freeing 2 slots/side + creating the top-end knob. Mid pair −3.25@100 + −4@110 → **NOT shelf-shaped** (best shelf misses by 1.3 dB; the stack is a bell) — keep the PEQs. Always simulate before converting.
- **ENCODING UNLOCKED (2026-07-03):** the user ran exactly the proposed verification — low shelf + middle-band all-pass + high shelf on FL High in one session. Result: **low shelf = `T=3`, high shelf = `T=4` (NOT T=20 as previously inferred), 1st-order APF = `T=19` and it sat in a MIDDLE slot** — so all-passes never compete with shelves for the end slots. `_tunefit.py` has the full chain: `low_shelf_db`/`high_shelf_db`/`fit_shelf_to_curve` for design + `shelf_fil_str` for writing (exact-tested vs the real export). **Shelf writes are now automatable** — with the defrag rule: shelf mode CONSUMES the slot's existing filter, so relocate any squatter PEQ to a middle slot first.

**SHELF PLACEMENT COOKBOOK (researched + verified 2026-07-03 — AF knowledge base + car-audio practice + RBJ shelf math):**
The parameter that actually matters on a shelf is the **hinge (corner) frequency `F`** — everything past the hinge (toward the spectrum end) is lifted/cut to the shelf plateau; everything on the near side stays put. `G` = plateau height. `Q` = the slope/knee (Helix 0.1–2).
- **AF's own rule:** shelves are for *"when a wide frequency range in the low or high frequency range is to be boosted or lowered"* — i.e. broad tonal balance, never a local feature.
- **Low shelf (band 1) hinge choices:** ~**60–80 Hz** = sub-only weight (lifts the very bottom, leaves midbass alone); ~**100–150 Hz** = whole-bottom "warmth/house-curve" tilt (the usual bass-voicing move). Car-audio practice: **link sub+midbass to the same low shelf so you change bottom-end balance without disturbing the crossover point/slopes** — this is exactly the sub-low-shelf voicing knob.
- **High shelf (band 30) hinge choices:** ~**8–10 kHz** = "air"/top-octave only; ~**3–5 kHz** = broad brightness/presence tilt across the whole top. **The single most useful car move is a gentle high-shelf CUT (−1 to −3 dB, hinge ~3–5 kHz)** to de-harsh the on-axis-tweeter + hard-reflective-cabin brightness in ONE filter instead of stacking upper-mid PEQ cuts (this is precisely the −1.25 @ 3306 Q1.7 consolidation).
- **`Q` on a shelf:** stay **low, ~0.1–0.7, for voicing** — a smooth monotonic transition. **`Q` > ~1 makes a *resonant* shelf** (a small overshoot bump/dip at the hinge); usually unwanted for tonal balance, occasionally handy to add a touch of "punch" at a bass-shelf knee. Amounts: voicing ±1–3 dB; a house-curve bass lift can run +3 to +6.
- **The decision, one line:** monotonic tilt anchored at an extreme → **shelf**; anything that returns to baseline on both sides → **bell** (`fit_shelf_to_curve` decides it numerically before you convert).

**APF PLACEMENT COOKBOOK (researched + verified 2026-07-03 — AF spec + Taylor phase-alignment work + sim):**
An APF is flat in magnitude and *only* earns its slot when the defect is **phase** (a summation null between two summing branches), never a magnitude bump.
- **Choose `F` (corner) = the frequency of the summation null / phase-crossing** you're trying to fill — for a crossover cancellation that's near the crossover frequency; for a measured null (e.g. the 415 Hz L/R dip) it's at the null.
- **Choose order + `Q` by how *sharp* the phase correction must be:** 1st-order (no Q, 90° at corner, gentle broad bend) for a slow phase divergence — good default for **sub↔midbass**. 2nd-order (180° at corner, Q 0.5–2) when you need to rotate more or more locally: **Q 0.5 = slow/broad bend, Q 2 = tight/fast bend at F.** **Use the LOWEST Q that fills the null** — higher Q = more group-delay ringing on transients = more audible cost.
- **Order the whole ladder first:** polarity → delay → crossover slope/family are cheaper and add no group delay — exhaust them (`polarity_delay_search`) before an APF is even a candidate.
- **Live-dial:** RTA + periodic pink noise, sweep F to the null, raise Q from 0.5 until it fills. If it only ever gets *worse* → hit **invert** (`I="1"`) and re-sweep — the needed rotation was on the other side of the circle.
- **Phase budget propagates UP the chain (subtle high-end technique from the research):** an APF you add to the midbass to fix sub↔midbass also changes the midbass's phase relative to the mid/tweeter *above* it — so a lower-crossover APF can require a matching APF cascaded up to keep the *whole* system coherent, not just the one crossover you were looking at. Re-check the next crossover up after any APF.
- **Imaging guardrail (unchanged, reconfirmed):** symmetric APF (same branch, both L/R) is summation-only and safe; **one-sided APF above ~1 kHz pulls the centre image** (interaural mismatch) — recommend-only + mono-vocal verify, never above ~2 kHz.

**To add a band:** take a free slot (`T="1"`) and rewrite `T`→`17` plus its `Q/F/G` (keep `I`/`FN`/`dF`). To change a band: edit `Q/F/G` on the matching `<Fil>`.

**Channel order (the OC blocks, 0-indexed):**
| idx | channel |
|----|----------|
| 0 | Front L High (tweeter) |
| 1 | Front R High (tweeter) |
| 2 | Front L Low (mid/midbass) |
| 3 | Front R Low (mid/midbass) |
| 4 | Rear L |
| 5 | Rear R |
| 6 | Sub 1 |
| 7 | Sub 2 |

**Pairs that must stay identical:** (0,1) (2,3) (4,5) (6,7).
**Delay/polarity:** `<T T="samples" PM="1|4"/>` @ 96 kHz internal (PM1 normal, PM4 inverted). `IS`/`CINV` on `<OC>` also encode polarity. **Python edits never touch these** — but they are NOT frozen: the user legitimately changes them in PC-Tool during TA sessions (§5), after which the new values are the baseline to preserve.

### Editing rules
- To **change** a band: find the right `<OC>`, edit `F`/`Q`/`G` on the matching `<Fil>`.
- To **add** a band: convert a free slot (`T="1"`) into `T="17"` with your `F/Q/G`, OR insert a new `<Fil .../>` if free slots exist. Don't exceed the channel's slot count. **Always `T="17"` (Parametric) so the band obeys AutoSort**; optionally set `dF` to the nearest ISO label of your `F` for a coherent slot read (AutoSort fixes order regardless). Never use a shelf/all-pass code for ordinary EQ. **Pick middle free slots — avoid the lowest-`dF` slot (reserve for low shelf), the highest-`dF` slot (reserve for high shelf), and band 2 / `dF="32"` (Tone-Control-reserved on 4.8).**
- **Apply every change to BOTH channels of the pair**, identical `F/Q/G`. **One deliberate exception: an all-pass filter (phase fix, §3).** An APF is **branch-relative, not inherently unilateral** (verified 2026-07-02): it changes the summation between the acoustic branch it's on and the branch it sums with (tweeter↔midbass, sub↔midbass). Put it on **one branch** of that pair (e.g. the tweeter branch), and — for a problem present on both sides — **on the SAME branch on BOTH L and R** (e.g. both tweeters). That fixes the crossover summation symmetrically and **preserves the stereo image.** The thing that actually "cancels out and does nothing" is the same APF on **both drivers being summed** (then their relative phase is unchanged) — NOT the same APF on both L and R. A **one-side-only** APF is the imaging-risk tool: use it only for a *proven side-specific* problem (§3). PEQ and level still stay pair-matched.
- After saving, re-decode and diff: confirm only the intended `<Fil>` lines and the header length changed; **all `<T .../>` delay tags identical.**
- **PC-Tool round-trip caution (caught 2026-07-03, real example):** whenever the USER sends back a file that was saved/reopened in PC-Tool (not one of my own Python writes), **diff the WHOLE file, not just the band they say they changed.** PC-Tool re-serializes on save: it **renumbers `FN` globally** (don't compare by FN — match by `(T,F,Q,G)` content instead), and it can **incidentally perturb an unrelated filter's parameter** — in the verified all-pass round-trip, both tweeter LP crossovers silently changed `Q="1"` → `Q="0.7"` even though the user only touched the FL Low all-pass. Always run the full per-channel diff (§4 verify pattern) before trusting "only X changed." **Also (verified 2026-07-03): PC-Tool round-trips REORDER attributes inside tags** (`<T PM= T= P=/>` came back as `<T T= P= PM=/>`, same values) — so for any PC-Tool-saved file, compare delay tags **semantically** (`_tunefit.py: delays_semantically_equal`, parses attrs to dicts), not byte-wise. Byte comparison stays valid only between my own Python writes.

---

## 2. Decode the `.mdat` (REW measurement)

REW `.mdat` = Java-serialized binary (`ac ed 00 05`), `roomeqwizard.MeasData`. **`javaobj-py3` fails** (`Invalid reference handle`) — scan raw bytes.

**Easiest path first:** if you can, ask the user to do **REW → File → Export → "Export measurement as text"** and send that instead. It's plain `freq, SPL` columns — zero ambiguity. Use the binary parse below only when you have just the `.mdat`.

**Preferred serious-workflow inputs (text export, not `.mdat` — `.mdat` binary parse is fallback only, validate identity+axis first):** System-Sum average (freq,SPL); **the individual 5–9 grid positions (freq,SPL each — send these, not only the average: they're what `spatial_consistency` needs to tell real problems from position artifacts)**; fixed-position **phase** exports (freq,SPL,phase) for min-phase / crossover work; solo-driver exports (freq,SPL[,phase]); a **distortion** export for any channel where a boost is being considered.

**Measure for averaging, not a single point (do this BEFORE export):** one mic position is full of position-specific comb filtering and nulls that look like real problems but move the instant the head moves — and the *entire* correction is computed from that one curve. **The tell that you're looking at a position artifact, not a real problem: the same tune, re-measured, shows the dip/peak at a different frequency or depth than last time** (this happened repeatedly with single-point captures — 612 Hz, ~800 Hz, and 1.7 kHz all shifted between two sessions of the *same* unchanged tune). That shift is the proof, not a measurement error — treat it as the signal to average, not to keep chasing the number.

**Verified REW steps:** capture a **grid of 5–9 fixed positions** around the listening headrest (center, then ~6 in / 15 cm left, right, up, down, forward), one sweep each. Then: open the **"All SPL"** graph tab → tick the checkbox on each of the position traces you want included → click **"Average the Responses."** Export *that* averaged trace as the System Sum (or per-driver layer). Real broad trends survive averaging; position artifacts wash out. The decode/deviation code below is unchanged — you just feed it an honest curve.

**Important correction — this average is magnitude-only.** REW's "Average the Responses" computes an **RMS average of SPL values; phase is explicitly discarded, traces are treated as incoherent.** So: use the spatial average for **target-matching / EQ placement only**. It is *not* usable for the phase-sensitive diagnostics (415 Hz cancellation-vs-null, all-pass placement, §3) — those still need a **single fixed position**, phase-coherent, with the acoustic timing reference, exactly as below. Don't average away the phase data you need for that step.

**Spatial average vs. Moving Mic Method (MMM) — not a strict "better," a different tradeoff. CORRECTION to MMM's mechanism (verified 2026-06-29):** MMM is **NOT** "one continuous sweep while moving the mic" (a log sweep assumes a fixed acoustic path at each instant — moving during one breaks that assumption). The actual, community-documented REW technique is **continuous pink noise + the RTA, time-averaged while the mic moves** — that's what makes MMM valid: the RTA keeps integrating magnitude across whatever positions the mic passes through. MMM is faster than a discrete grid and some report finer spatial resolution from the continuous path. But for this workflow, fixed multi-position still wins on fit: (1) **a car cabin is tight** — a smooth, deliberate 3D moving path is more awkward in a seat-bound space than discrete spots. (2) **You can inspect each position before averaging** — check whether a dip is consistent across all 7 positions (real, EQ it) or only 1 (comb artifact, ignore it) — averaging blind loses that diagnostic. (3) Neither method substitutes for the **per-driver solo + acoustic-timing-reference** phase work — that's always a separate, single fixed-position step. Default to fixed multi-position; MMM is a reasonable faster alternative for a quick magnitude/voicing check.

**Practical MMM with ONE UMIK-1 (no mic array needed — magnitude/voicing only):**
- **Capture:** REW Generator → **pink noise** (or Pink PN) playing continuously, then **RTA** with smoothing (~1/24–1/48 oct) and time-averaging running while you move the mic. Export the averaged RTA trace. (Community-documented setup: 64k FFT, Hann window, ~87.5% overlap, "Forever" smoothing — defaults are fine to start.)
- **Movement, scaled to a car cabin:** room literature uses a ~0.5 m (2 ft) radius "sweet spot" cloud — **too big for a seat**. In the cabin, shrink it to a **small head-cloud**: roughly ear-to-ear width, a little above/below ear height, a little forward/back — staying where ears actually sit, never sweeping down to the console or across the whole cabin. Move smoothly, ~30–60 s.
- **Repeatability check:** do 2–3 passes of the *same unchanged tune*; if the resulting curve/candidate corrections shift a lot between passes, the technique isn't converged yet for this session — redo rather than trust it.
- **Use it for:** bass shelf, midrange balance, presence energy, top-end tilt — broad tonal trends only. **Do not** treat a narrow dip/ripple in an MMM trace as EQ-able — MMM smears comb filtering by design, so anything narrow that survives is still not automatically fixable.
- **Hybrid workflow with one mic:** MMM for the main System Sum voicing pass, then separate fixed-position sweeps for: phase/ATR work, solo-driver per-side correction, and TA refinement (§5). MMM replaces the 5–9-position discrete grid; it does **not** replace any fixed-position phase measurement.

**Capture more than the sum if you want the high-end fixes (phase / distortion):**
- **Per-driver solo sweeps that share a common time-zero.** The user's mic is a **UMIK-1 (USB) → no electrical loopback** (single input, own clock). Use REW's **Acoustic Timing Reference** (Timing → "Use acoustic timing reference"; pick ONE driver — a tweeter/midrange, never the sub — as the Reference Output and keep it identical, mic unmoved, across every sweep). That preserves *relative* phase between drivers — which is what lets you (a) tell a crossover cancellation from a room null and (b) place an all-pass (§3). **Car caveat:** the cabin is reflective and REW can latch onto a reflection instead of the direct pulse — choose a reference driver with a clear direct path to the mic and adequate level. Relative-phase-between-drivers is more forgiving than absolute TA because the reference offset is common to all sweeps. Without phase data, the ~415 Hz dip can only be guessed at.

  **Verified REW capture discipline for the UMIK-1 (official REW help, 2026-06-28):**
  - **Single 256k sweeps — NOT repeated/averaged sweeps.** With a USB mic on a different clock than playback, multiple/long sweeps corrupt the impulse/phase. (Spatial averaging across *positions*, one sweep each, is fine — that's different from averaging repeated sweeps at one spot.)
  - **Enable Analysis → "Adjust clock with acoustic ref"** alongside the acoustic timing reference — *this* is what actually compensates the UMIK's clock drift and makes its phase usable. Without it, the timing reference alone isn't enough.
  - The timing-reference chirp is a **5–20 kHz sweep (~700 ms), so its output MUST be a tweeter-capable channel, never the sub.**
  - REW's **96-points-per-octave log spacing is official default behavior** (it 1/48-oct smooths, then converts) — which is exactly why §2's axis math anchors to 96 PPO. Validated, not a guess.
  - There is **no parseable coherence field** in a normal sweep export (confirmed against REW's help index) — so `spatial_consistency()` (cross-position std) remains the right proxy.
- **THD vs frequency per driver** (REW distortion graph). Any region over ~3% THD is distortion-limited — never EQ-*boost* into it; re-crossover or leave it.
- **FDW (frequency-dependent windowing)** on the measurement to separate direct sound from reflections — long window low (keep modes), short window high (keep only direct). **Decision rule:** a narrow HF feature that's in the in-seat average but **vanishes under FDW = a reflection → don't narrow-EQ it**; a feature present in **both** FDW and the average = likely direct/driver → EQ-able. Don't tune to the FDW curve itself — it's a diagnostic view, not the listening target.

**You may receive several files at once** — e.g. `L solo`, `R solo`, and the `System Sum`. Use each for its own layer: **Sum → shared voicing tilt** (both channels), **each solo → that side's per-driver corrective layer** (§3 "Per-side mode"). If you only get the Sum, you can only produce the shared layer — say so plainly rather than inventing L/R differences. Each can arrive as a `.mdat` or a REW text export.

**STANDARD CAPTURE SET (added 2026-07-03 — always ask for this, not just solos+Sum):** for each summing pair (front tweeters, front mids, sub+midbass), capture **THREE** traces at the same fixed mic spot: **solo A, solo B, and A+B playing together** (e.g. "Front L Low", "Front R Low", "Mid Bass Together"). Solos alone give per-side EQ; the **together trace unlocks the interference audit (§3)** — a cheap, no-phase-capture way to catch L/R destructive cancellation, which turned out to be the actual explanation for the long-standing 415 Hz mid-pair null (see §3 worked example). This is now the default ask, not an optional extra.

**Per-driver capture recipe (REW, UMIK-1) — the practical how-to:**
- **RTA vs Sweep — isolate at the DSP, not in REW.** REW's RTA/sweep shows whatever physically reaches the mic; it has no concept of "just this driver." If a solo capture shows full-range content, other channels are still playing — **mute everything except the driver(s) under test in PC-Tool/Helix routing** (leave TA/crossovers/EQ untouched, only routing changes), confirm by ear, then measure. Don't try to "fix" this by changing REW's display/graph limits — that only changes what you're looking at, not what's playing.
- **One-time setup:** REW → Preferences → Soundcard: Input = UMIK-1's **"MICROPHONE"** entry (not Default); Output = playback device. REW → Preferences → Analysis: tick **"Adjust clock with acoustic ref."** Load the UMIK-1 cal file. Route the **FL tweeter to always carry the timing chirp** (Ref Output). **Mic stays fixed for the whole batch** — moving it between drivers breaks the shared time-zero.
- **Per-driver loop:** mute all outputs except FL tweeter (chirp) + the driver under test (sweep). In REW's Measure window: **Method=Sweep, Length=256k, Repetitions=1, Start=10Hz, End=20000Hz, Timing=Use acoustic timing reference, Output=L, Ref output=R.** Start, then **rename the result immediately and clearly** (name strings aren't trustworthy per se — but a clear name still saves confusion; verify by shape regardless, per the rule above). Exception: measuring the FL tweeter itself — it's both reference and target, so route it 100% on both Digital In L and R.
- **QA gate:** measure the same driver twice without touching anything; if the reported delay disagrees by more than ~0.02–0.05 ms, the clock correction isn't locking — recheck routing/levels and redo (same repeatability gate as §5).
- **Restore normal playback routing when the batch is done.**
- If you only need magnitude (no phase/APF/TA work), the acoustic-timing-reference setup can be skipped for a simpler capture — but there's little downside to leaving it on, since it costs nothing extra once configured and unlocks more analysis.

Magnitude curve is a Java float32 array: marker `75 71` (TC_ARRAY + TC_REFERENCE) + 4-byte handle (`00 7e XX XX`) + 4-byte big-endian count + count × big-endian float32. Frequency axis is logarithmic: `freq[i] = startFreq * logStep**i`.

```python
import struct, numpy as np

def find_float_arrays(data, min_len=256):
    """Return list of (offset, count, np.float32 array) for Java float32 arrays."""
    out, i = [], 0
    while True:
        j = data.find(b'\x75\x71\x00\x7e', i)
        if j < 0: break
        # after 75 71 00 7e XX XX (6-byte handle) comes 4-byte BE length
        p = j + 6
        if p + 4 > len(data): break
        n = struct.unpack('>I', data[p:p+4])[0]
        body = p + 4
        if 0 < n < 5_000_000 and body + 4*n <= len(data):
            arr = np.frombuffer(data[body:body+4*n], dtype='>f4').astype(np.float64)
            if n >= min_len and np.isfinite(arr).all():
                out.append((j, n, arr))
        i = j + 2
    return out

data = open('System Sum.mdat','rb').read()
arrs = find_float_arrays(data)
for off,n,a in arrs:
    print(off, n, round(float(a.min()),1), round(float(a.max()),1))
# The SPL/magnitude curve is the long one (~1000-1300 pts) with dB-ish range.
```

**Axis (VALIDATED method — use this, it's robust):** REW's SPL data is **96 points per octave**, so `logStep = 2**(1/96) = 1.00724641` exactly (a constant, not per-file). This export's array **ends at 24000 Hz**. So with `n` = array length:

```python
logStep = 2 ** (1/96)
freqs = 24000.0 / (logStep ** (n - 1 - np.arange(n)))   # anchored at the 24 kHz top
```

For the known-good `System Sum.mdat`: `n=1232`, `startFreq≈3.31 Hz`, magnitude array at byte offset 104683. **n varies slightly per file (file size differs) — always use the actual array length, never hardcode n.**

**AXIS CORRECTION (2026-07-03, from REW .txt exports of the same session):** the RTA mdat frequency grid is **3.2958984 Hz .. 23369.487 Hz at 96 ppo** — NOT anchored at 24 kHz. The earlier 24k-anchor assumption ran **0.038 octave (2.7%) high**: the cabin null "at 427-430" is truly at **415.8 Hz** (exactly where the user sees it in REW — the validation should have demanded that, not accepted 350-550). Consequence: v7/v7.1's measurement-derived frequencies were all 2.7% high (v7.2 corrected them; gain-only edits to existing hardware bands were unaffected). **Rule: prefer REW .txt exports (they carry the exact axis); if parsing .mdat alone, use the 3.2958984 anchor for RTA captures and validate the null lands at 415-420 — a validation with 1/3-octave slack cannot catch a 4-bin axis error.**

**ALWAYS validate before trusting any frequency for EQ:** the deepest dip in 350–550 Hz must land on the cabin null the user sees (~**415 Hz**), and the 30–60 Hz peak on the bass peak (~**47 Hz**). This session both matched exactly. If they don't line up (different REW FFT/PPO settings), fall back to the REW text export. A working end-to-end copy of this lives in `C:\Users\Adroit\Downloads\Claude AF Tuner\_devcalc.py`.

**Don't trust a measurement's NAME string — verify its SHAPE.** A `.mdat` can contain a measurement literally named "Sub" that is actually full-range (content well past its supposed low-pass) — meaning it was captured with other channels still playing, not a clean solo. Before using any named array as a per-driver/per-band proxy, check it rolls off where that channel's crossover says it should (e.g. a real solo sub should be down hard above its LP frequency); if it doesn't, treat it as a system/full measurement instead and say so. Also: each named measurement in a `.mdat` typically has **two byte-identical copies** at different offsets — `np.array_equal` should be `True` between them; if it isn't, you may be looking at two different measurements, not duplicates of one.

---

## 3. Compute what to change

```python
# 1. measured = the long float32 array; freqs = startFreq*logStep**arange(count)
# 2. Read the ResoNix target (REW text export of the target, freq/SPL pairs).
# 3. Log-interpolate target onto the measurement freq grid.
# 4. Align levels: offset = MEDIAN(measured - target) over a WIDE band (300 Hz-3 kHz); target += offset.
#    (median over a wide band, NOT mean over 800-1200: one local bump must not be allowed
#     to float the whole target up/down and bias every band you compute downstream.)
# 5. dev = measured - target.  Positive dev = too loud -> CUT. Negative = too quiet.
```

**Only correct real, broad, fixable bumps** — peaks of meaningful width and height (a few dB over a third-octave-ish span). **Do NOT EQ:**
- Narrow cancellation **nulls** (e.g. the ~400–475 Hz cabin null) — can't fill a true null; a small partial lift (**≤ +3 dB**, same as the global boost cap — e.g. the +3 @ 160 Hz Q3.5 partial lift actually applied to FR) is the most that's ever worth it, and expect the shoulders to rise ~+2 as the cost.
- The tweeter top-octave rolloff (off-axis physics).
- Sub region if it's already heavily cut / excursion-limited.

**Split the band at the cabin transition (~300–400 Hz) — the two halves don't behave the same:**
- **Below ~400 Hz (modal):** narrow *peaks* are real and worth cutting; narrow *nulls* are not (can't fill a true null). High resolution (1/6 oct) is fine here.
- **Above ~400 Hz (reflection-dominated):** only *broad* trends are real. Smooth harder (1/3 oct / ERB) and EQ in broad strokes only — tight bands up here just chase comb ripple that moves with the listener's head.

**Is the ~415 Hz dip actually a fixable cancellation, not a cabin null?** A dip sitting in the mid↔midbass crossover region is often acoustic cancellation (phase/polarity/delay/slope between the two drivers), which EQ can NOT fix but TA/polarity/crossover CAN. Tell them apart from the **per-driver sweeps**: if each driver is individually flat through ~415 Hz but the *sum* dips → cancellation. If *both* drivers individually dip there → modal/boundary, leave it. Only treat it as an un-fixable null once the per-driver data confirms it.

**If it IS a cancellation → fix it with phase, not EQ (the high-end move auto-EQ never makes):** an **all-pass filter** rotates phase without touching magnitude, so it only changes the *summed* response of the two drivers. **Order matters and they are NOT interchangeable (official Audiotec Fischer behavior):**
- **1st-order APF: NO Q parameter at all** — only a corner frequency. Fixed, gentle 0°→−180° phase sweep, exactly −90° at the corner freq. Use this for the gentler, broader correction (e.g. midbass summation) — don't put a Q on it, the hardware doesn't have one to set. **Encoding: `T=19`, CONFIRMED (export-diff + screenshot); write via `allpass1_fil_str`; middle slots fine.**
- **2nd-order APF: HAS a Q parameter.** 0°→−360° sweep, exactly −180° at the corner freq. Higher Q = the 360° rotation happens more abruptly in a narrower band; lower Q = smoother/gentler. This is the one for a sharp crossover-region dip.
- **Official APF Q range is 0.5–2** (AF PC-Tool 4 page — supersedes the earlier "max ~1.5 at 50 Hz" claim from a weaker source). Within that range, **prefer LOW Q (≈0.5–1.0)**: the ear is far more sensitive to abrupt narrow-band phase shifts than gradual ones, and high-Q 2nd-order APFs ring on transients. So 0.5–2 is *allowed*, but reach for the low end unless a sharp dip genuinely needs the rotation focused.
- **Branch-relative, not unilateral (verified 2026-07-02).** An APF changes the sum between the branch it's on and the branch it sums with. For a crossover problem on **both sides**, put the same APF on the **same branch on both L and R** (e.g. both tweeters, or both subs) — fixes the summation symmetrically, imaging untouched. Only the same APF on **both drivers of the summing pair** truly cancels (relative phase unchanged) — don't do that.
- **UNILATERAL (one-side-only) APF is the imaging-risk tool — reserve it.** A one-side APF creates an interaural group-delay mismatch; above ~1 kHz the ear localizes by ITD and **>~35 µs pulls the phantom center off** (Blauert / Minnaar) — it *smears the stage* even while "fixing" a dip. Use a one-side APF ONLY when fixed-position solo measurements **prove the problem is side-specific**, only below ~1 kHz, and only after polarity, delay, crossover family/slope, and physical aiming have been tried. **Write-eligibility tiers (added 2026-07-03):** *auto-write OK* = symmetric APF (same branch, both L/R) or a proven-side-specific one-sider below ~500 Hz; *recommend-only, live-verify before keeping* = one-sided 500 Hz–1 kHz; *never* = one-sided above ~1 kHz.
- Recipe: start with a **2nd-order APF (180° rotation) centered in the dip, Q ≈ 0.7–1.0 (consistent with the low-Q preference above; raise toward 2 only if the dip is sharp and the sum demands it), on ONE channel only — and per the unilateral rule above, only below ~1 kHz**, then re-measure the sum and nudge F/Q. `_devcalc.py` now has an all-pass + two-driver summation model: feed it the per-driver complex sweeps and it shows the dip depth **with vs without** a proposed APF before you commit anything. Note the cabin-mode check: the lowest Golf MK7 axial modes land ~60–70 Hz (length) and ~120 Hz (width), so a **415 Hz feature is too high to be a fundamental room mode** — that alone leans it toward cancellation/SBIR, i.e. APF-fixable, not "leave forever."

**Interference / summation audit (`_tunefit.py: interference_audit`, added 2026-07-03) — detects L/R cancellation WITHOUT a phase capture.** If you have the standard capture set (solo A, solo B, A+B together — see §2), you can catch destructive interference from **plain magnitude data alone**, no acoustic timing reference needed:
```python
psum = 10*log10(10**(solo_a_db/10) + 10**(solo_b_db/10))   # incoherent (power) sum — the floor
csum = 20*log10(10**(solo_a_db/20) + 10**(solo_b_db/20))   # fully coherent (voltage) sum — the ceiling
interference_db = together_db - psum   # large NEGATIVE = the pair is cancelling, not just quiet
```
If the measured "together" trace reads **below the incoherent power-sum floor**, the two sources are physically working against each other at that frequency — a phase-relative problem, not a level or EQ-able magnitude one. Self-tested and unit-verified.

**Worked example — this is how the 415 Hz mid-pair null was finally explained (real data, 2026-07-03):** each mid *solo* measured healthy through 415–450 Hz, but the "Mid Bass Together" trace read **~3 dB below even the incoherent sum** in that band (and again, more mildly, near 1 kHz). That's the signature of L/R destructive interference, not a modal/boundary null — six sessions of "narrow dip, leave it" turned out to have an actual fixable mechanism once the together-trace was checked against the solos. This reclassified the dip from RED (permanent) to an **all-pass candidate**, and the value was quantified before touching anything: fixing it was worth as much audibility-score improvement as the entire rest of that session's EQ pass combined.

**Verified REW-only phase-integration workflow (APF is the LAST resort, not the first).** Captured with UMIK-1 + acoustic timing reference (see §2). Measure **sub-only and midbass-only separately from the same fixed mic position**, then use REW's **All SPL → alignment tool** at a cursor near the crossover, in this exact order:
1. **Impulse alignment first** — REW 1/3-oct zero-phase filters both traces at the cursor and cross-correlates. If it wants one trace inverted → that's your **polarity** answer; if invert + a small delay sums best → fix it electrically (polarity/TA).
2. **Align phase slopes at cursor** — matches group delay around the crossover.
3. **Align phase at cursor** — brings the two phase traces together.
4. Still dipped? Try a **different crossover family/slope** (Butterworth/Bessel/Tschebyscheff/Linkwitz/Self-Define, up to −42 dB/oct).
5. **Only now** an all-pass — and only if the residual is a broad phase-rotation issue, not a level mismatch or a geometric/modal cancellation.

**The first two rungs are now code (`_tunefit.py: polarity_delay_search`, self-tested 2026-07-03):** given the two complex solo captures, it searches polarity × local delay with the same gap-to-coherent-ceiling score as `optimize_allpass`, and returns `residual_needs_apf` — only if polarity/delay can't close the gap has an APF earned a look. (Sign note: a negative best delay on B means "delay the OTHER branch instead" — hardware can't advance.) Run it before any APF search; results are directly comparable since both use the same score.

**The APF "invert" button (PC-Tool Allpass panel) — semantics verified by simulation (`allpass_H_inv`, TEST13):** invert multiplies the all-pass by −1 — same rotation, plus 180° at ALL frequencies. It is mathematically identical to (channel polarity flip) + (normal APF), just applied inside the EQ block so the TA/polarity page stays untouched. **When to press it:** while live-dialing an APF, if the target dip gets WORSE at every F/Q you try, the needed rotation is on the other side of the circle — invert flips the branch relationship and re-sweep. (Numerically: a normal 2nd-order APF at f0 *fixes* an antiphase pair at f0; the inverted one *nulls* it at f0 but fixes it broadband — they are complements, not variants.) **XML encoding VERIFIED (2026-07-03): `I="1"` = inverted** — the export-diff showed exactly `I` 0→1 on the test all-pass and nothing else in the entire file. The `I` attribute on every `<Fil>` is the invert flag (the original "index" reading was wrong). With this, **the .afpx format is 100% mapped — every filter type, parameter, and flag is verified and writable.**

So the order is **polarity → delay → phase-slope → phase → crossover family/slope → APF.** REW warns its impulse-alignment result **struggles near strong modes** — if it does, that's evidence the feature is **modal/position-sensitive → reclassify and leave it, don't EQ or APF it.**

**Manual all-pass placement protocol — VALIDATED 2026-07-03, this is the reliable way to dial one in (better than writing a guessed F/Q straight to XML and hoping).** All-pass tuning is done live, by ear/RTA, in PC-Tool — not blind-computed and written:
1. **PC-Tool must be CONNECTED to the DSP** (not offline file editing) — live parameter tweaks only push to the hardware, and the live RTA only updates, while connected.
2. Route the summing pair under test (e.g. "Mid Bass Together") solo, mic fixed, and get REW's **RTA running continuously** (Generator → pink noise, RTA tab — not a single sweep capture). **Freeze/store the "before" trace** as a reference overlay.
3. **Sweep F first** (~10–20 Hz steps across the suspect range) watching the live RTA — note where the trough fills the most relative to the frozen baseline.
4. **Then sweep Q** (0.5–2 range, small steps) at that F, watching the same. Do one more quick F pass afterward since F and Q interact a little.
5. **Check the shoulders didn't collapse** — the bands just outside the target null should stay roughly where they were on the "before" trace. If filling the target dip pushed a shoulder down, back off.
6. **Confirm by ear on the FULL system** (restore normal routing) with a **mono vocal**, focused on the register around the target frequency — the image must stay dead-center. If it smears or pulls, the all-pass is working against imaging even if the RTA looked good; back it down or revert.
7. **Save & Store** in PC-Tool to commit, export the `.afpx`, and send it back — that gives a controlled export-diff to confirm the exact encoding (§1) and lets the interference audit or a re-measurement quantify what changed.

**Joint balance/tonality lesson (v7 build, 2026-07-03):** two traps found while re-tuning v5 from the 8-trace RTA set. (1) **Softening a quiet side's "overcuts" raises the SUM nearly 1:1 wherever the pair sums coherently** — FR's 4-6k cut-softenings looked right per-solo but pushed the sum to +4 @ 4k; and check WHICH side is actually louder before calling a solo-vs-target cut "excessive" (FR read below target at 4-6.3k yet was still the LOUDER side there). Sum tonality outranks solo shape in 2-6k. (2) **Balance lifts and tonality cuts interact:** lifting the far side re-fills the exact energy the near side's cut removed (630 Hz: FR −2.25 cut + FL +2 lift ⇒ sum −0.4 net). Compute pair edits JOINTLY against the predicted sum, never sequentially.

**APF roadmap — live status (2026-07-03):**
- **430 Hz on FL Low: IN THE TUNE, NOT YET VERIFIED.** Highest-priority action in the whole system: re-measure "Mid Bass Together" (+ System Sum) with the APF in and compare against the pre-APF capture — fill confirmed, shoulders intact, mono-vocal centered. If it didn't fill, try FR Low instead (rotation belongs on whichever side needs it).
- **~1 kHz mid-pair interference (−2.9 dB below power-sum): NEXT CANDIDATE** once 430 is settled. Same live protocol, centered ~1000–1100, Q 0.7–1. Borderline for the unilateral zone — be extra strict on the mono-vocal check.
- **2.6 kHz mid↔tweeter crossover: REJECTED** — measured summation is already near the coherent ceiling (51.9 vs 53.2 dB); nothing for an APF to recover.
- **Sub↔mid 80 Hz: REJECTED** — measured sum EQUALS the coherent ceiling (62.4 = 62.4 @100 Hz); textbook-perfect, do not touch.

**Model filter interaction — don't just set G = −dev per band:** independent `G ≈ −(peak dev)` is fine when bands are >~1 octave apart (e.g. 110 vs 2500). When two bands land within ~1 octave their skirts SUM, so the naive gains overshoot. `_devcalc.py` now has an RBJ peaking-biquad simulator — put the proposed bands in its `PROPOSED` list and read the **predicted post-EQ deviation** to confirm the combination actually lands on target before you write the `.afpx`.

**Per-side mode — producing legitimately different L/R EQ (needs solo sweeps):** the System Sum is the *shared* layer only; it physically can't separate L from R. To get correct L/R differences, decompose into two layers:
- **Shared voicing tilt** = the *broadly smoothed* (≈1 oct) System-Sum-vs-target deviation — the tonal balance / house-curve tilt. Apply IDENTICALLY to both channels of a pair.
- **Per-side corrective** = each driver's *local* anomalies, measured **solo**, relative to *its own* **~1 oct median baseline** (median, not mean — a mean baseline rises under the very peaks it should isolate and under-reports them) — **not** the target, so the tilt isn't applied twice. These bands differ L vs R exactly as much as the two sides actually measure differently. Stay inside the driver's crossover band. Proposed cut gains run ~1 dB conservative (median grazes the peak flanks) — finalize off the curve and let the close-the-loop re-measure trim.

After per-side correction, **re-check that the broadband balance of L still matches R** — correct local wiggles, but never let one side end up overall brighter/darker (that smears the center image). `_devcalc.py` has this mode: set `LEFT_SOLO` / `RIGHT_SOLO` / `SUM_EXPORT` and `PASSBAND`; it prints the shared tilt plus each side's candidate cuts. If only the Sum is available, build the shared layer and say you can't differentiate L/R without solo sweeps — don't invent differences.

**When L has a peak and R has a dip at the same frequency, cut the peak — don't chase the dip.** This came up repeatedly (e.g. FL +4.8 / FR −3.6 at 1.9 kHz): the dip side is almost always a one-sided cancellation null (can't be filled — see above), but the peak side is a genuine, EQ-able excess. Cutting FL's peak there did more for the center image (halved the L/R gap) than any attempt to lift FR's null ever could, at zero risk. Rule of thumb: **if a feature is one-sided, check whether the OTHER side has the opposite shape at the same frequency** — if so, fix the peak side, leave the dip side alone.

**EQ craft rules (the fundamentals the auto-EQ button gets wrong):**
- **Cut, don't boost.** Pull a peak down; don't raise everything around a dip. Cap any boost at **+3 dB** — boosting eats headroom and excites distortion/excursion.
- **Match Q to the cause, not a habit.** Narrow resonance/breakup → **Q 4–8**; broad tonal hump → **Q 0.7–1.5**. Q≈1 for everything is the classic weak-tune tell.
- **THD gate before any boost.** If the per-driver THD trace is >~3% in that region, do not boost — re-crossover or leave it.
- **Don't EQ a clean crossover dip.** An LR4 (24 dB/oct) pair sums flat in correct polarity; a notch right at the crossover is usually minimum-phase and fixes itself with polarity/TA, not a filter.
- **Iterate.** Filter skirts interact; re-measure after each pass (§4 close-the-loop).

**Translate dev → PEQ:** center `F` at the bump, `G ≈ −(peak dev)`, `Q` from width (narrow spike → Q 3–5; broad hump → Q 0.7–1.5). Apply to both pair channels.

---

## 3a. Research-derived refinements (vetted 2026-06-28) — and what was rejected

A research digest was run against this method. Most of it **corroborated** what's already here (band 1/2/30 reservations, polar/magnitude averaging, IIR-over-FIR, don't-boost-nulls, FDW ≈ Smaart's MTW). The genuinely new, **verified** additions:

- **ERB-variable smoothing > fixed 1/6–1/3 oct.** Human cochlear bandwidth is ~0.9 oct at 40 Hz, narrowing to ~1/6 oct above 1 kHz (Glasberg-Moore). Smoothing the deviation this way stops you "correcting" narrow LF wiggles the ear integrates over, while keeping HF resolution where the ear is fussy. `_devcalc.py` → `erb_smooth()` (verified to give that exact octave-width profile). Prefer it over the split-band smoother for deciding what's even audible.
- **Coherence blanking / spatial consistency — the automated comb whack-a-mole killer.** With a real multi-position grid, compute **std across positions per frequency**; where it's high (>~2.5 dB) the feature moved between positions → position-dependent comb/null → **do NOT EQ it**. `_devcalc.py` → `spatial_consistency()`. This automates the "is it real or comb?" call we kept making by hand (the 612/800/1.7 k saga). *Unproven on this car — needs a true same-tune multi-position grid we never captured; validate the mask before trusting it.*
- **High-SPL linearity check.** Re-measure at a high listening level, not just measurement level. Any tonal shift / new harshness / panel rattle that only appears loud = compression or resonance the quiet sweep hid (MECA scores exactly this). **Now numeric (`_tunefit.py: compression_check`, 2026-07-03):** sweep the same thing at two levels X dB apart; where the measured rise falls >0.75 dB short of X, the region is compressing → **veto boosts there** (re-crossover / reduce workload instead). Caveat from REW's own docs: log-sweep distortion data is noise-floor-limited at HF — stepped-sine is the trustworthy method up high, so treat sweep-derived HF compression evidence as lower confidence.
- **Sub excursion / port unloading.** Never boost at or below a ported sub's tuning frequency — the port unloads, cone excursion spikes, suspension at risk. The 20 Hz HP already IS the subsonic protection (which is why refusing to boost <25 Hz was correct). Full T/S-parameter excursion modelling is possible but needs the sub's BL/Mms/Cms/Re/Xmax + box tuning — we don't have them.
- **Evaluate tonality in 4 competition bands:** sub-bass 10–60, mid-bass 60–200 (front↔sub integration + impact/transient), **midrange 200–3k (vocals/timbre/imaging — the critical, most-weighted one)**, highs 3k+ (air without sibilance).

**Rejected / held back (do NOT adopt blindly):**
- **"Write voicing to virtual channels, not output channels."** Our output-channel edits are *verified working* — your measurements responded to them every pass. The virtual-channel routing idea is unverified against the actual XML; leave the working approach alone until a controlled test proves it.
- **Input de-EQ / AISA / IGS / VCP signal-flow acronyms.** The universal principle (set input gain so the DSP can't clip *before* you EQ) is real and worth doing. The specific named registries are uncertain (possible AI confabulation) and only matter if you run an OEM high-level input — flag, don't code.
- **MDCF = 150 Hz transition.** Directionally consistent with our pain (narrow EQ above ~150–200 Hz in this cabin mostly chases comb), but one paper doesn't justify hard-swapping the 400 Hz number. Treat **150–400 Hz as a grey zone** — increasingly distrust narrow EQ as you climb, and fully trust only spatial-averaged broad trends above it.
- **FABRICATED — ignore entirely:** the digest's "Peer Review Debates" forum personas and the "SYSTEM RECORD" telemetry block (NOISE_FLOOR 42.1 dBA, T60 0.08 s, DETECTED_MDCF 165.2 Hz, "VERIFICATION_PASS: TRUE") are invented — nothing ran. Don't let those numbers into the method as if they were measured.

**Second digest (ChatGPT, 2026-06-28) — additional vetted items.** (~Half of it restated what digest #1 already added above — ERB smoothing, coherence/spatial-consistency, high-SPL linearity, sub-excursion, competition tonal bands — not re-listing those.) Genuinely new and kept:
- **Excursion boost-budget (relative) — THD gating is NOT a mechanical safety check.** A boost that's low-distortion at test level can over-excurse loud. A **+X dB boost ×10^(X/20) on cone voltage**; at constant SPL, excursion demand rises ~**×4 per octave down**, so a LF boost's mechanical cost compounds fast. Rules: **no boost at/below a ported sub's tuning (or sealed Fc); require a protective HPF; cap LF boost harder the lower you go; re-measure at realistic SPL.** An *absolute* safe/unsafe gate needs the driver's Xmax + box tuning + amp voltage — supply those and it becomes real; without them, default conservative.
- **Group-delay audibility numbers (when phase work is worth it):** 300 Hz–1 kHz GD **<~1 ms inaudible, 1–2 ms sometimes, >2 ms often**; below ~100 Hz far more tolerant. Intervene (all-pass/crossover) only when GD/timing exceeds these *and* a **transient/click listening check** confirms — impulse-like material has the lowest detection thresholds, so judge phase fixes on transients, not on a tidy phase plot.
- **First-arrival / precedence check (this is TA, not EQ):** view REW's **impulse/ETC** per driver pair at the seat; check **L/R first-arrival symmetry** and early-reflection strength. Stage height/width and center-focus problems usually live here, invisible to a smoothed FR — hand them to TA/aiming, not PEQ.
- **TuneEQ over-fit guard (self-check before writing):** reject a band set that's chasing **<~1 dB residuals**, stacking **many narrow bands above the transition**, **filling dips**, or converging toward visual flatness. Restraint is the whole point of beating the auto-EQ button.
- **Cumulative-gain / digital-headroom report:** sum every gain stage per path (input + group + output EQ + channel gain) and flag **DSP clipping risk separately from acoustic target error** — two different failure modes.
- **Cabin-mode estimate is a PRIOR, not a verdict** (CORRECT): use the Golf MK7 mode frequencies to *suspect* a modal feature, but confirm modal-vs-comb with **spatial variance (`spatial_consistency`), decay, and solo-vs-sum** before classifying. The transition itself can be *measured* (spatial variance rising) rather than fixed — default ~300–400 Hz but say why this session supports it.

**VERSION CAVEAT (important — applies to BOTH digests):** the P SIX MK2 runs **PC-Tool 4.x**. Both digests leaned on features documented for **PC-Tool 5/6 / the newer ACO platform** (AISA auto-detection, specific TuneEQ 5/6 behaviors, Virtual Channel routing internals). Do **not** assume those map one-for-one onto the P SIX MK2 — verify against a real export-diff before relying on any of them.

**Third pass (targeted, 2026-06-28) — primary-sourced; what it RESOLVED vs. what's still open.** This pass pinned the **PC-Tool 4.x / P SIX MK2 filter & crossover spec** (now in §1 — APF Q 0.5–2, shelf Q 0.1–2, 5 crossover families to −42 dB/oct, no group-delay display) and the **verified REW UMIK-1 capture + All-SPL phase-alignment workflow** (§2/§3). **Still UNRESOLVED — keep current behavior, do not hard-code from memory:**
- **Toole/Olive resonance-audibility tables** — exact numbers still not found in primary sources (AES-paywalled; a 2026-07-02 pass confirmed only the *orderings* via secondaries: pink noise is the most revealing program material, low-Q resonances are detectable at smaller dB than high-Q). Keep ranking corrections by *broadness + magnitude + cross-position persistence* + the provisional `audibility_weight` in `_tunefit.py`; do **not** hard-code an "inaudible below X dB at Q=Y" gate until the real tables are in hand. **RESOLVED on the same date: the "which dips can EQ fix" question** — now answered rigorously by REW's primary-source minimum-phase doctrine + the `excess_gd_mask` classifier (§3b), superseding the eyeball heuristic.
- **Exact ResoNix / Harman-automotive target points** — no public primary found. The curve in §System-facts is the only verified target; keep it, don't overwrite it with an unsourced "official" clone.
- **Numeric off-axis A-pillar tweeter target** — not found. Keep the seat-referenced shared-voicing + per-side-solo decomposition (don't force mirrored HF EQ from the sum).

---

## 3b. Joint optimizer + minimum-phase classifier (`_tunefit.py`, added 2026-07-02 — SELF-TESTED)

`C:\Users\Adroit\Downloads\Claude AF Tuner\_tunefit.py` (run it: all self-tests + a real-data validation execute).

**Traffic-light gate — classify EVERY frequency bin BEFORE fitting (this restraint IS the thing that beats auto-EQ).** Auto-EQ sees an error and fills it; a good engineer first asks *is it real, audible, stable, and EQ-able?* Tag each bin:
- **GREEN (EQ allowed):** spatially consistent (low `spatial_consistency` std) · minimum-phase (or no phase conflict) · inside the driver's passband · not THD-limited · not a narrow null · enough audible payoff (`audibility_weight`) to earn a filter.
- **YELLOW (broad low-Q voicing only):** in the average but not perfectly consistent · above the transition · reflection-influenced. Correct only with wide, gentle filters — never narrow.
- **RED (no EQ):** high spatial variance · non-minimum-phase / excess-GD swing · narrow dip/null · high THD/compression · outside passband · crossover cancellation (→ polarity/delay/slope/APF, §3) · top-octave off-axis rolloff.

Feed GREEN as `fit_peq`'s `mask=True`; feed a continuous confidence (`1/(1+std)` × min-phase × distortion × passband) as `conf=` so YELLOW bins are **down-weighted, not chased**. RED never gets a filter — and every RED region is named in the report's rejected-corrections list (§4).

Three tools that structurally out-do TuneEQ / REW's EQ window:

1. **`excess_gd_mask(freqs, spl_db, phase_deg)` — the EQ-ability physics test (PRIMARY-SOURCED, REW's own minimum-phase doctrine).** From a single-position export **with phase**, it computes the minimum phase implied by the magnitude (cepstral method), subtracts to get **excess group delay**, and flags: *flat excess GD = minimum-phase = EQ works there; wild excess-GD swings (sharp dips) = non-minimum-phase = EQ cannot fix, period.* This replaces our heuristics ("narrow dip probably a null") with the actual physical criterion. **Verified subtlety the tests encode:** a reflection *weaker* than the direct sound makes a comb that is still minimum-phase (technically EQ-able); only a **dominant** reflection flips a notch non-minimum-phase. So not every comb notch is untouchable — the classifier, not the eyeball, decides. Needs the 3-column REW text export (freq, SPL, phase) from a fixed position with acoustic timing ref + clock adjust (§2).
2. **`audibility_score(freqs, dev)` — one number for "how audibly wrong."** ERB-smoothed residual, weighted by ear sensitivity (full weight 200 Hz–6 kHz, tapered below/above — provisional pending the still-unfound Toole/Olive tables). Use it to (a) rank candidate corrections by *audible* payoff, and (b) prove each pass improved: report score before → after for every tune written.
3. **`fit_peq(freqs, dev, band, mask=...)` — joint fit of the whole band set (scipy), not greedy.** TuneEQ/REW fit one band at a time against raw magnitude; this fits all bands **simultaneously** against the audibility-weighted error, with the discipline built in as constraints: masked bins (nulls/non-min-phase/volatile) can never attract a filter; boosts capped +3 / cuts −15; Q 0.5–8; **parsimony gate** — each added band must improve the weighted score ≥6% or fitting stops (the anti-TuneEQ rule: never chase sub-dB residuals with more filters); gains rounded to 0.25 dB.

**Validated on real data (New.mdat FR Low):** as-measured score 2.251 → v4 hand/greedy edits 1.912 → **joint fit 1.870 with ONE band** (vs five incremental hand edits). Same discipline, fewer filters, better weighted result — fewer filters also means less phase rotation.

**Filter tax + confidence (params added 2026-07-02, self-tested):** `fit_peq` now **taxes boosts and narrow-HF filters** — it only boosts / goes high-Q up top when the audible payoff clearly beats the tax (verified: on a narrow HF dip it dropped fitted boost from +5 dB/Q6.8 to +4 dB/Q4.8). It also accepts `conf=` for continuous down-weighting (the YELLOW tier). `headroom_report()` gives the mandatory clip-risk numbers (§4).

**Minimum-phase classifier guardrails (don't let a good tool become an overfitter):** run `excess_gd_mask` ONLY on a **fixed-position export with phase** — never on a spatial average or MMM trace (they have no coherent phase). Prefer full-range measurements; **re-run it after changing REW's IR window** (the result depends on windowing); treat its output as an EQ-**permission** test, not a correction generator. **If the min-phase mask and the spatial-consistency mask disagree at a frequency, do NOT EQ it — re-measure.** (REW's own docs: room responses are minimum-phase only in some regions, mainly LF.)

**Workflow from now on:** §3 finds/classifies the problems → **traffic-light every bin** (above) → **`fit_peq` allocates the band budget** with `mask` = GREEN and `conf` = graded confidence → biquad-simulate → write → report audibility score before/after + headroom + rejected corrections (§4).

---

## 4. Save & verify

```python
xml = read_afpx('Your Tune.afpx')
# ...string-edit the <Fil> lines for both channels of each pair...
write_afpx('Your Tune_v2.afpx', xml)

# verify
import re
a = read_afpx('Your Tune.afpx'); b = read_afpx('Your Tune_v2.afpx')
assert re.findall(r'<T [^>]*>', a) == re.findall(r'<T [^>]*>', b), 'delays changed!'
```
**Every tune report MUST include (not optional):**
- **Changes made** — per channel-pair: F/Q/G and the deviation each corrects.
- **Audibility score before → after** (`_tunefit.py audibility_score`) — the proof it actually improved, not just "looks flatter."
- **Headroom** (`headroom_report`) — peak EQ-cascade gain per channel, largest single boost, clip-risk flag, recommended trim if net boost is unsafe.
- **Rejected corrections + WHY** — list every candidate problem you did NOT EQ and the reason (null · non-minimum-phase · high spatial variance · crossover cancellation · off-axis rolloff · THD/compression · insufficient audible payoff). This is the RED tier from §3b made visible. **Restraint is the whole competitive advantage over auto-EQ — so show it, don't hide it.** Never silently change something the data doesn't justify.

**Close the loop — this is a 2-pass process, not 1:** measured response + cabin interaction means predicted ≠ actual. After the user loads the new tune, have them **re-measure (same averaged grid)** and re-run §3 against the new trace. The first pass gets ~90% there; the second catches overshoot/undershoot and filter interaction. Then the final ±1 dB is a *listening* decision — the data defines the target, the ears ratify it.

**Listening validation (how the ears ratify — do it, don't hand-wave):** first **level-match old vs new within 0.2 dB** — louder always "wins," so an unmatched A/B is worthless (PC-Tool setup switching gives instant A/B). Then map tracks to regions: **mono vocal** → center-image lock & L/R symmetry; **kick/bass** → sub↔midbass handoff; **male vocal** → 120–300 Hz warmth/boom; **female vocal** → 1–4 kHz presence; **sibilants/cymbals** → 5–10 kHz harshness; **dense rock/electronic** → compression/congestion. Never finalize from one track. **The target is an ENVELOPE, not a line:** final voicing = only broad, pair-matched trims (≈±0.75–1 dB per region: sub shelf, midbass punch, lower-mid warmth, presence, air) — never a narrow subjective filter. The mathematically-closest trace is rarely the best-sounding tune.

**After an all-pass edit, verify differently:** an APF changes nothing on a single channel's magnitude, so the delay-tag diff still applies but the proof is in the **summed** response of the two drivers — re-measure the sum to confirm the dip actually filled, not the per-channel curve.

**Four runnable, verified scripts live in `C:\Users\Adroit\Downloads\Claude AF Tuner\`** (real working code, not pseudocode):
- `_devcalc.py` — parses `.mdat`, validates the axis, computes deviation vs the ResoNix target; also holds `erb_smooth`, `spatial_consistency`, the APF summation model, per-side mode.
- `_tunefit.py` — **the solver layer (§3b–§3e):** joint PEQ optimizer (`fit_peq`), audibility + perceptual scores, minimum-phase/excess-GD classifier, **`interference_audit()`**, **`polarity_delay_search()`** (the doctrine rungs below the APF, as a search), **`optimize_allpass()`** (APF F/Q candidate finder), **`compression_check()`** (two-level linearity gate), **`prediction_confidence()`** (gate: solo model must reproduce the measured together trace before phase decisions are trusted), **`tune_scorecard()`** (canonical named-metric scoring), **`low_shelf_db`/`high_shelf_db`/`fit_shelf_to_curve`** (shelf design/simulation — write path still gated on the export-diff), `target_anchor_offset`, `headroom_report`, and **`allpass_fil_str()`** (verified all-pass XML writer). `python _tunefit.py` runs its self-tests + a real-data validation.
- `_benchmark.py` — **tune-vs-tune scoreboard:** score N `.afpx` files against the target on equal footing (each applied as EQ-delta-from-baseline to the measured traces, scored via `tune_scorecard`). `python _benchmark.py <mdat> <baseline.afpx> <tune...>`. Magnitude-only: APF/delay effects are NOT modeled — live re-measure for those.
- `_make_v3.py` — the actual edit: converts free slots to PEQs, pair-matches L/R, re-zips, and asserts the delay tags are byte-identical. Copy its `add_bands()` / `edit_tweeter()` / verify pattern.

**Worked example (this is what the method produced, end-to-end):** from `System Sum.mdat` the three real bumps were ~110 Hz (+3.8), 2500 Hz (+4.0), 12.5 kHz (+3.9). Edit applied to `New Tune.afpx → New Tune_v3.afpx`: mids got `-4 @ 110 Q1.3` and `-4 @ 2500 Q2.5` (both sides); tweeters' existing `13000 +2.5` became `-1.5` (both sides); null/rolloff/sub/per-driver bands untouched; all delays preserved.

**Worked example 2 — v6, an 8-trace RTA capture set (2026-07-02/03):** a fresh `Measurements.mdat` with FL/FR High, FL/FR Low, Sub, System Sum, "Tweeters Together", "Mid Bass Together" revealed the v3 sub cuts were confirmed working (47 Hz boom gone), but the **tweeters were carrying stale legacy EQ** — FR (the near-ear side) was 2–5 dB darker than FL through 2.8–8 kHz from old over-cuts, not acoustics. Fix: walked back two over-cut FR bands, added a small **shared** cut at 2600/5000 Hz to both tweeters (Sum's scoop-center overshoot), and a shared 100 Hz mid cut for a broad hump — **10 edits total, zero new boosts.** Audibility score 2.251→1.849 (null-masked); optimizer confirmed nothing further passed the parsimony gate (best candidate only 5.3% improvement, below the 6% threshold) — **proof v6 was at the EQ ceiling for that measurement set, not just an assertion.** The interference audit on the same data then found the real next gain (415 Hz worked example above) — a phase problem no amount of additional EQ would have reached.

---

## 5. Time-alignment refinement (REW + Helix digital-routing trick) — VALIDATED 2026-06-28

The user now **manually refines ATM's time alignment** using REW. This happens entirely **in PC-Tool's UI (delays + polarity) — I never edit `<T>` tags in Python.** The full step-by-step lives with the user; the validated essence + the watch-outs:

**The clever core (correct):** route through the Helix **digital input matrix** so REW's **timing chirp (REW Ref Output → Digital In R) always feeds the FL tweeter at 100%**, and the **sweep (REW Output → Digital In L) feeds only the driver/group under test**. Every measurement is then referenced to the same FL-tweeter arrival → valid *relative* timing across drivers, from one fixed mic position. Needs **"Adjust clock with acoustic ref"** + single 256k sweeps (the verified UMIK-1 discipline, §2). The timing chirp is 5–20 kHz, so the reference MUST be a tweeter — FL tweeter is the right pick.

**Alignment order (matches §3's verified workflow):** measure each branch solo from the fixed position, then in REW's **All SPL → Alignment Tool**: Level phase → Align phase slopes → Align phase → Aligned sum, **trying inverted polarity** (inverted + small delay often beats normal + large delay). Apply the winning **polarity/delay in PC-Tool**, then re-measure the *sum* to confirm. System order: **sub↔front first, then center-image trim by ear, then mid↔tweeter.**

**Verified-correct details:**
- **Negative delay → delay the OTHER group.** Helix can't do negative delay; if REW says the sub should arrive earlier, add that delay to the whole **front stage** (keep the front's internal relative delays intact). Correct.
- **Center image (precedence): the image pulls toward the EARLIER side — delay that side to recenter** ("pulls right → add delay to right"). Correct. Trim by ear with a mono vocal, 0.01–0.05 ms steps.
- **Phase math checks out:** at 2.5 kHz, 0.10 ms ≈ 90°, so mid↔tweeter steps must be tiny.

**My added watch-outs:**
0. **UMIK-1 clock drift is real and verified — this is the actual risk in this whole procedure.** A USB mic runs its own clock, independent of the playback device's; a documented case measured drift of **~0.1 ms per second** of measurement time. That's exactly why REW mandates single short (256k) sweeps (a long one drifts *during itself*) and why "Adjust clock with acoustic ref" exists — it's the documented correction for a documented weakness, not optional polish. **Repeatability gate (do this before trusting any session's data): re-measure the same thing twice in a row, nothing moved — if the reported delay disagrees by more than ~0.02–0.05 ms between the two runs, the clock correction isn't locking; stop and recheck routing/levels rather than acting on that data.** At sub↔front scale (ms-level corrections) the residual risk is a rounding error; at mid↔tweeter scale (0.1 ms ≈ 90° at 2.5 kHz) the risk is the same size as the answer — verify those with the *measured sum* and by ear, never the phase tool's number alone.
1. **Verify the optical L/R mapping ONCE** before trusting anything: run the FL-tweeter reference measurement and confirm the chirp physically comes from the **FL tweeter**. If the playback device / optical link swaps L/R, the whole routing inverts silently.
2. **TA is more position-stable than magnitude** (arrival times don't comb like nulls do), so a single mic position is fine here — *unlike* EQ, which needed spatial averaging. The ear-based center trim covers head-movement variance.
3. **Sanity-check the phase-tool result against the raw impulse delay** (REW Estimate IR delay): phase alignment at a crossover can lock onto a ±360° **cycle-slip**. If the phase-tool delay disagrees wildly with the impulse arrival difference, you slipped a cycle — redo.
4. The "front midbass pair" trace is already a blend of FL+FR arriving slightly differently at your off-center RHD mic — fine for sub *bulk* alignment, not surgical.
5. **Keep ATM as the starting point** (the guide does) — this *refines* ATM, doesn't replace it. **Restore normal playback routing when done.**

---

## 6. Reviewed external "improved copies" (2026-07-03) — what was ADOPTED vs REJECTED

A ChatGPT-generated set of "improved" copies (in `TuningApp_ImprovedCopies/`, 60 functions / 1637 lines) was audited function-by-function. It **ran and passed its self-tests**, but "runs" ≠ "improves" — much of it was tuned-constant heuristics with no validation. Only the genuinely-good, verifiable pieces were ported into the real files (each re-tested here). **Do not re-import the rejected pile.**

**ADOPTED (ported into the real `_tunefit.py` / `_devcalc.py` / `_make_v3.py`, all re-tested):**
- **`_make_v3.py` hardened writer** — `afpx_roundtrip_lint` (semantic filter diff that **ignores PC-Tool `FN` renumbering**, flags forbidden added types, catches delay/crossover side-effects), `validate_peq_band` (enforces G∈[−15,+6]/Q∈[0.5,15] before writing), `choose_free_slots` (reserves the shelf/APF/tone-control slots), and fail-loud `edit_tweeter`/`add_bands`. This is the biggest win — it hardens the actual thing that produces files loaded into the car, and codifies the §1 round-trip discipline as runnable checks.
- **`optimize_allpass(freqs, driver_a, driver_b, search_band)`** — grid-searches a 2nd-order APF F/Q, scored by gap-to-coherent-ceiling in the band **plus an off-band-damage penalty** (so it won't fix the target dip by wrecking a shoulder). Candidate finder, not a blind finalizer — still verify live (§3 manual APF protocol). Auto-finds what I'd been sweeping by hand.
- **`target_anchor_offset`** — confidence-weighted median with fallback anchor bands; a robust superset of the plain 300–3k median. Now wired into `_devcalc.py`.
- **`fit_peq` refinement** (`selection_tax_weight`) — a lighter tax on the parsimony gate than on the fit itself. Net effect: **more correctly restrained** — it now refuses to fit a narrow HF dip at all (the right call) where the old version fit two gentle boosts (subtly wrong: that's filling a dip).
- **`perceptual_score`** — additional scorer that separates broad tonal error, upper-mid resonance risk, and **L/R mismatch in the 700 Hz–5 kHz image band** (weights peaks over dips). Kept alongside `audibility_score` for final ranking when L/R data exists — valuable given how often L/R balance has been the issue here.
- **`interference_audit` smoothing** + defensive REW-import guards (`_devcalc`) + the corrected branch-relative all-pass comment.

**REJECTED (over-engineered, unvalidated, or premature — reasons logged so they don't creep back):**
- **`eqability_classifier`** — bundles 5 evidence streams into one opaque "GREEN/YELLOW/RED" score. The transparent traffic-light discipline (§3b) + the individual masks (`spatial_consistency`, `excess_gd_mask`) already do this *and show their reasoning*. A single fused number hides why a bin was rejected — the opposite of the rejected-corrections-list goal.
- **`optimize_allpass_v2`** — four extra tuned penalty weights (GD/shoulder/high-Q/imaging) on top of v1. v1 already penalizes off-band damage; the rest is unvalidated magic-number tuning, and live re-measurement is the real gate anyway. v1 is the keeper.
- **The confidence-component zoo** (`spatial_confidence`, `mmm_repeatability_confidence`, `snr_confidence`, `minphase_confidence`, `distortion_confidence`, `passband_confidence`, `combine_confidence_maps`) + `traffic_light_mask` monolith — the *concepts* are already the documented traffic-light; bundling them into a stack of tuned-threshold functions adds surface area without adding trustworthy signal. `fit_peq` already accepts a `conf=` array; build it explicitly when there's real grid data.
- **`fit_target_knobs` / `target_knob_curve`** — "target is an envelope" is a *listening-stage* discipline (§4); coding preference-knob fitting is premature and invites auto-generated voicing drift.
- **`lopo_filter_stability`** (leave-one-position-out) — genuinely good *idea*, but needs a real multi-position grid we've never captured. Revisit when grid data exists.
- **`crossover_summation_optimizer`** — polarity/delay-before-APF is already the **documented live REW workflow** (§3, §5), done by ear/RTA with the actual DSP in the loop; a Python search over synthetic sums is weaker than the real thing. Skipped.
- **`headroom_report_v2`** — needs channel-gain/shelf/crossover gain curves rarely passed in; the simpler `headroom_report` covers the main PEQ-cascade clip risk.
- **`TraceType`/`qa_trace`/manifest-metadata guards** — reasonable in principle but heavy ceremony; the disciplines (MMM is magnitude-only, phase work needs fixed-position, etc.) are already enforced in prose (§2) and by the model's own reasoning.

**Second external drop reviewed (2026-07-03, "Deep Research Backlog"):** better-grounded than the first (real REW/AF citations, honest "engineering gates, not psychoacoustic laws" framing), but ~70% was already implemented or already rejected here. **Adopted:** `polarity_delay_search` (it exposed a real inconsistency — we had the APF search but not the cheaper polarity/delay rungs the doctrine puts FIRST); `compression_check` (makes the high-SPL gate numeric, 0.75 dB threshold); the **APF write-eligibility tiers** (§3); the stepped-sine-vs-sweep distortion-confidence caveat. **Already had:** perceptual score w/ peak-vs-dip asymmetry + L/R image term, FDW decision rule, rejected-corrections reporting, coherent-ceiling scoring, MMM-vs-grid roles, headroom, metadata provenance. **Still rejected (same reasons as before):** the fused `eqability_classifier` evidence model (opaque weights), full crossover family/slope/frequency auto-search (crossovers stay hands-off in the .afpx; family changes need live measurement anyway), `lopo_stability` (no grid data yet), target-adaptation knobs (§4's tighter listening envelope stands — the backlog's ±3 dB ranges are looser than ours, keep ours). Its unverified hardware claims (Self-Define fixed 12 dB/oct Q0.5–2, Linkwitz even-orders-only, "PC-Tool phase control = 2nd-order APF") are plausible but NOT export-diff-verified — don't rely on them without checking.

**Third external brief reviewed (2026-07-03, "R&D Upgrade brief"):** the most architecturally ambitious yet — and the highest already-implemented ratio (~70%: complex prediction model, excess-GD masks, ERB/asymmetry/image/GD/headroom scoring, the pair action-ladder, joint fitting, parsimony, the write linter all predate it). **Adopted (4):** `prediction_confidence()` — its best idea: before trusting any phase-sensitive search, prove the solo model reproduces the CURRENT measured together trace (rms ≤2.5 dB in-band after level-bias removal), else block APF/delay decisions and re-measure; `tune_scorecard()` — one canonical named-components scoring function (sum RMS, image-weighted RMS, worst, mid/tweeter balance) so every tune comparison uses identical math; `_benchmark.py` — the comparison runner as a permanent tool (validated: reproduces the v5/v6/aggressive/v7/v7.1 scoreboard exactly); delay-policy nuance — delay writes are no longer doctrinally forbidden (encoding is fully known: `<T T="samples">` at 96 kHz, samples = ms×96) but remain user-initiated only, never optimizer-written. **Adopted-then-REVERTED (1, instructive):** the "gain trim rung" below polarity in the ladder — a failing self-test proved it ill-posed: the ladder's score is gap-to-coherent-ceiling with the ceiling fixed from the input solos, so a level change can push the sum past the ceiling and game the metric. Level mismatch belongs to the balance metrics, not the phase ladder. **Still rejected (same reasons as rounds 1-2):** Trace/MeasurementGraph dataclass ceremony, fused null_likelihood weights, target-family knobs, crossover variant search, the optimizer zoo (beam/DE/CMA-ES/Bayesian/Pareto — wildly oversized for a 30-band problem), safe-vs-aggressive dual mode (empirically falsified: the external "aggressive" tune LOST the benchmark — restraint won), ML anything. 

**Meta-lesson (kept because it recurs):** an AI-generated "improvement" pile that *passes its own tests* is not self-validating — the tests were written by the same pass that wrote the code. Adopt by **auditing each function for whether it's principled or just tuned constants**, port the principled ones, and reject fused black-box scorers even when they run. Restraint in the *toolkit* is the same virtue as restraint in the *tune*.

---

## Ready-to-paste prompt for a new chat

> I'm tuning a Helix P SIX DSP MK2 (8-ch) in a RHD Golf MK7. I'm attaching a REW `.mdat` measurement and the `.afpx` tune that was loaded when I measured. Read `C:\Users\Adroit\Downloads\Claude AF Tuner\MDAT_AFPX_INSTRUCTIONS.md` first — it has the exact decode/encode for both formats, the channel map, the ResoNix target, and the editing rules.
>
> Decode the `.mdat`, compute deviation from the ResoNix target, and edit the `.afpx` directly (Python, not the browser app): correct only real broad bumps, leave nulls/rolloff/delays alone, and apply identical shared-voicing EQ to both L/R of each pair. Save as `<name>_v2.afpx` and verify the delay tags are byte-unchanged.
>
> The System Sum I send is either a **fixed-grid spatial average (5–9 positions)** or an **MMM averaged-RTA trace** — both are **magnitude/voicing data only, never phase-valid**. If it's a single-position sweep instead, treat narrow features skeptically and say so. Treat **<400 Hz as modal** (cut peaks, leave nulls) vs **>400 Hz as broad-only**, **model filter interaction** (predicted post-EQ curve) rather than G=−dev per band, and **flag anything that needs fixed-position phase / solo-driver data** (e.g. the ~415 Hz dip) before calling it a null or proposing an APF.
>
> Allocate bands with the **joint optimizer** (`_tunefit.py: fit_peq`, §3b) with nulls/volatile bins masked — not greedy per-peak picking — and report the **audibility score before → after**. If I send a fixed-position export **with phase**, run `excess_gd_mask` first so only minimum-phase regions get filters.
>
> If I send the **standard capture set** (solo A, solo B, AND the pair together, for a summing pair like the front mids or tweeters), run the **interference audit** (`_tunefit.py: interference_audit`, §3) — a "together" trace reading below the incoherent power-sum of the solos means L/R destructive interference, not an EQ-able magnitude problem; flag it as an all-pass candidate instead. I can now write a **verified 2nd-order all-pass** directly into the `.afpx` (`allpass_fil_str`, §1) if you give me the F/Q you've already dialed in live in PC-Tool — I won't guess F/Q blind, that's a by-ear/live-RTA process (§3 manual APF protocol).
>
> If the `.afpx` I'm given was last saved by PC-Tool itself (not one of my own prior writes), **diff the WHOLE file against the prior version**, not just the part you say changed — PC-Tool round-trips can renumber filter IDs and occasionally perturb an unrelated band.
>
> List every change with the dev it fixes, and remind me to **re-measure to close the loop**. If I send **L-solo / R-solo** sweeps alongside the Sum, build the **per-side corrective layer from the solos** and the **shared voicing tilt from the Sum**; if I send only the Sum, apply shared EQ to both sides and tell me you can't differentiate L/R without solo sweeps.
