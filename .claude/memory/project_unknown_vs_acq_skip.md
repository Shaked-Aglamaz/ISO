---
name: project_unknown_vs_acq_skip
description: "UNKNOWN(-1) is the scorer, BAD_ACQ_SKIP is the amplifier — independent labels that coincide; TST/WASO policy + the overloaded-BAD_ACQ_SKIP trap, full note in notes/tst_waso_unknown_handling.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: fb33fe0f-15a0-46bd-bfb7-e7b4297619c0
  modified: 2026-08-13T12:48:47.568Z
---

Investigated 2026-08-12 on AT36 (elderly). Full write-up: `notes/tst_waso_unknown_handling.md`.

- `-1` in `ISO_data/scoring/{group}/{sub}.txt` and the `UNKNOWN` annotation are the **same
  information** (mapped in `code/step0_mff_cleaning.py:230`), not a duplication.
- `BAD_ACQ_SKIP` is **independent** — MNE emits it for skipped acquisition buffers; data there
  is written-out zeros (6e-08 V vs 5e-05 V real EEG) → flat trace. It coincides with `UNKNOWN`
  only because the same event caused both (subject out of bed). Several subjects have
  `UNKNOWN` with zero `BAD_ACQ_SKIP`.
- **Trap:** in `*_cleaned_annotations.txt` the `BAD_ACQ_SKIP` label is overloaded — short
  entries (<200 s, arbitrary float onsets) are manual artifact marks drawn in the step1
  notebook while that label was selected in the MNE browser. AT36 has 3 real gaps + 58 manual;
  LS56 has 94, IS74 35. Harmless for ISFS bout splitting, fatal if you use the name to find
  real gaps.
- Hypnograms can be 1 Hz, not 30 s/line (AT36: 31,675 lines = 31,674 s) — see
  [[project_hypnospectrogram]] for the `hypno_freq` gotcha.

**Why:** counting `-1` as wake invents wake time the amplifier caused — AT36 WASO 74.0 →
98.5 min (+33%), SE 80.4% → 65.1%.

**How to apply:** compute TST/WASO from the hypnogram only, never from `BAD_ACQ_SKIP`. Treat
`-1` as a third category (not sleep, not wake, not recorded): exclude it from numerator AND
denominator, and report unscorable minutes as its own column. AT36 = TST 343.5 min, WASO 74.0,
SE 80.4% (clears the 210-min rule of [[project_cohort_change_mr5_tst210]]).

**Open item CLOSED 2026-08-13:** unscorable minutes are now a per-subject column
(`unscorable_min`) in `results/demographics_V4/sleep_statistics_per_subject.csv` for all 104.
24 subjects carry unscorable time; AT36 (100.7 min) is the only large one.

**Which source to read — settled empirically 2026-08-13 ([[project_c429_c390_results]]).** Rebuilding
the hypnogram at 1 Hz from `*_cleaned_annotations.txt` is **equivalent to reading the scoring file**,
because `add_sleep_scoring` writes the stage annotations as a plain run-length encoding of the
scoring array. Verified over the cohort: of the 93/104 subjects that have a locatable scoring file,
**89 are 100.000% identical second-by-second**, all 93 ≥99.2%; SOL and REM latency differ by
**0.0000 min for every one**, WASO for a single subject (RS5) by 20 s. So "use the hypnogram only"
means *stage info, not `BAD_ACQ_SKIP`* — it does not require reading `ISO_data/scoring/`.

**Prefer the annotations, because the scoring tree is a trap.** 11 cohort subjects have **no scoring
file at all** (EL3031/34/35/36/37/44, SM0016/18/19/20, SM09). Resolution needs ~8 filename patterns
plus per-subject epoch inference (68 subjects at 1 s/line, 25 at 30 s/line). Files are **misfiled
across group subdirs — LS56 and SB00 are elderly subjects whose correct scoring is under
`scoring/young_control/`** — and the loose `scoring/` root holds stale duplicates that agree with the
real scoring at only **60.3%** and **84.8%**, so a name-based lookup there silently uses the wrong
scoring. Also 6 scoring files contain `?`, which step0 maps to **Wake** (not unscorable); the
annotations bake that decision in, so reading raw files and treating `?` differently would diverge
from every other analysis in the repo.
