# `UNKNOWN` (-1) vs `BAD_ACQ_SKIP` — and how to compute TST / WASO

Investigated 2026-08-12 using AT36 (elderly_control) as the worst case. Handoff note for
other sessions. **Nothing in the codebase was changed** — this is a findings + policy note.

## TL;DR

- `-1` in the scoring file and the `UNKNOWN` annotation are **the same information**, not a
  duplication. `-1` is the source; `UNKNOWN` is its annotation representation.
- `BAD_ACQ_SKIP` is a **separate, independent** annotation created by MNE for skipped
  acquisition buffers (recording paused). It overlaps `UNKNOWN` only because both were caused
  by the same real-world event (subject out of bed → tech paused acquisition, scorer marked
  the epochs unscorable).
- For TST/WASO: use the **hypnogram only**, and treat `-1` as a third category
  (*not sleep, not wake, not recorded*) — exclude it from numerator **and** denominator.
  Never use `BAD_ACQ_SKIP` for sleep statistics.

## Where each label comes from

### `-1` → `UNKNOWN` (scorer)
- Source: `I:\Shaked\ISO_data\scoring\{group}\{sub}.txt` (or `{sub}_hypno.txt`).
- **AT36's hypnogram is 1 Hz**, not 30 s/line: 31,675 lines vs 31,674.1 s of recording.
  (Beware — this varies by subject; see `code/inspect_hypnos.py` and the `hypno_freq` gotcha.)
- `add_sleep_scoring()` in `code/step0_mff_cleaning.py:230` maps `-1 → 'UNKNOWN'`.
- Exact 1:1 correspondence in AT36: 6030 lines of `-1` ⇄ 6030.0 s of `UNKNOWN` (4 blocks).

### `BAD_ACQ_SKIP` (amplifier / MNE)
- MNE emits this automatically when the source recording has skipped acquisition buffers.
- AT36's FIF has only **3**, all long: 4555.8 s, 826.7 s, 618.3 s (total 6000.8 s).
- The data in them is literally zeros: max |x| = 6.4e-08 V inside a skip vs 5e-05 V during
  real EEG. That's the completely flat trace seen in the MNE browser.
- By the time the file is saved as FIF the skips are no longer skip-buffers
  (`ent is None` count = 0) — they are written-out zero samples carrying the annotation.

### Why they coincide (and why they are still independent)
- 5990 of 6030 s of `-1` fall inside a `BAD_ACQ_SKIP`; 40 s of `-1` are outside and 14 s of
  skip are scored Wake. The mismatch is pure grid rounding: scoring sits on a 1-s grid while
  skip onsets are sub-sample (118.7425, 9951.788, 25544.439).
- Independence is visible cohort-wide: EL3011, EL3012, EL3013, RD43, TZ7, SM09, VZ9, YS0 all
  have `UNKNOWN` blocks with **zero** `BAD_ACQ_SKIP`.

## Trap: `BAD_ACQ_SKIP` is an overloaded label in `*_cleaned_annotations.txt`

`AT36_cleaned_annotations.txt` contains **61** `BAD_ACQ_SKIP` entries, not 3:

| kind | n | durations | origin |
|---|---|---|---|
| real acquisition gaps | 3 | 618–4556 s | MNE, from the recording |
| manual artifact marks | 58 | 4.8–162 s, arbitrary float onsets | hand-drawn in the step1 notebook while `BAD_ACQ_SKIP` was the selected label in the MNE browser |

Consequences:
- Harmless for ISFS bout splitting (`mult_chan.py` cuts on any `BAD*` either way).
- **You cannot use the label name to identify real data gaps.** Discriminate by duration
  (real gaps here are all > 600 s) or, better, by checking flatness of the data.
- Other subjects show the same pattern (LS56 has 94, IS74 has 35, EF58 has 7).

Also noticed: 3 N2 annotations in `AT36_cleaned_annotations.txt` were nudged during that manual
editing (onsets 5168.356 / 6779.349 / 6991.465 instead of 5160 / 6780 / 6990). Durations are
intact, so stage totals are unaffected — N2 = 16380 s in both the hypnogram and the file.

## Recommended TST / WASO policy

Derive everything from the 1-Hz (or upsampled) hypnogram array `h`:

```python
valid  = h != -1                      # recorded & scorable
sleep  = np.isin(h, [1, 2, 3, 4])     # N1, N2, N3, REM
idx    = np.flatnonzero(sleep)
onset, offset = idx[0], idx[-1]       # first / last sleep second
spt    = h[onset:offset + 1]          # sleep period time

TST  = sleep.sum()                    # seconds
WASO = np.sum(spt == 0)               # wake inside SPT, -1 NOT counted
SOL  = onset
SE   = TST / valid.sum()              # denominator excludes -1
```

Rules:
1. `-1` is **never** wake. Folding it into WASO invents wake time that the amplifier caused.
2. Exclude `-1` from the denominator of sleep efficiency (whether TIB- or SPT-based).
3. Report an explicit **"unscorable minutes"** column per subject so large gaps stay visible
   rather than silently absorbed.
4. Do not touch `BAD_ACQ_SKIP` for these metrics — overloaded label, and it is sample-accurate
   rather than epoch-aligned so it will not sum cleanly against stage seconds.

## AT36 numbers (policy comparison)

| metric | value |
|---|---|
| Recording length | 527.9 min |
| TST (N1+N2+N3+REM) | **343.5 min** — N1 31.5 / N2 273.0 / N3 5.5 / REM 33.5 |
| Sleep onset latency | 84.0 min |
| SPT (onset → last sleep epoch) | 442.0 min |
| `-1` total | 100.5 min → 76.0 pre-onset, 24.5 inside SPT, 0 post-offset |
| **WASO, `-1` excluded** | **74.0 min** ← recommended |
| WASO, `-1` counted as wake | 98.5 min (+24.5 min of fake wake, +33%) |
| SE = TST / TIB (raw) | 65.1% |
| **SE = TST / (TIB − `-1`)** | **80.4%** ← recommended |
| SE = TST / SPT, `-1` excluded from SPT | 82.3% |

The three `-1` blocks: 2.0–78.0 min (pre-onset, 76.0 min), 166.0–180.0 min (inside SPT),
425.5–436.0 min (inside SPT).

Counting `-1` as wake inflates WASO by 33% and drops SE by 15 points from a paused amplifier
alone. Because 76 of the 100.5 min precede sleep onset, TIB-based SE takes the worst hit.
AT36's TST of 343.5 min clears the 210-min inclusion rule under every policy.

## Subjects with `UNKNOWN` blocks (from `*_cleaned_annotations.txt`)

`n` = number of annotation blocks, not duration. AT36 is the extreme; most others have a
single block (often a short trailing one).

| group | subjects (UNKNOWN blocks / BAD_ACQ_SKIP entries) |
|---|---|
| control_clean | EL3010 (1/2), EL3011 (1/0), EL3012 (1/0), EL3013 (1/0), EL3019 (1/1), RD43 (2/0) |
| elderly_control_clean | **AT36 (4/61)**, CH53 (1/4), DS6 (1/2), EF58 (1/7), GZ8 (3/3), IS74 (1/35), JR9 (2/2), LS56 (1/94), ME5 (1/1), RB88 (1/1), RP8 (1/1), TZ7 (4/0) |
| MCI_clean | ED9 (3/1), MR5 (1/1), SM09 (1/0), VZ9 (1/0), YS0 (3/0) |

## Open item

Not yet done: quantify how many minutes of `-1` sit **inside SPT** for the other 22 subjects,
to know whether the `-1`-as-wake vs `-1`-excluded choice changes anyone else's TST/WASO
materially, or whether AT36 is the only subject where it matters.
