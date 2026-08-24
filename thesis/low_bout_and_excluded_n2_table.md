# Low-bout subjects and N2-related exclusions

## TST inclusion criterion (minimum 210 min)

To make the exclusion criteria align with standard sleep-research practice, we adopt a minimum
**total sleep time (TST) of 210 min (3.5 h)** as an inclusion threshold. This is a commonly cited
PSG/spectral-study cutoff (e.g. insomnia phase-3 studies require TST > 210 min with time-in-bed
450–550 min; the "≥ 2 sleep cycles" structural rule gives a comparable floor). TST = sum of all
sleep-stage durations (N1+N2+N3+REM) = recording minus Wake, matching `subjects.tst_sec`.

**Included subjects below 210 min — EXCLUDED (rule applied uniformly, 2026-06-16):**

| Subject | Group | TST (min) | Disposition |
|---|---|---|---|
| MCI13 | MCI | 96.5 | excluded (TST < 210) |
| HG78 | Young | 162.5 | excluded (TST < 210) |
| SM07 | MCI | 206.1 | excluded (TST < 210; borderline, applied for consistency) |

All retained subjects are now ≥ 216.5 min (next lowest: MCI16 216.5, YC8 235.0, RS5 256.7).
Cohort after all 2026-06-16 changes (MR5 in, DS6 re-included, TST<210 out): **YA 35, HE 39, MCI 30** (total 104).

**Excluded side vs 210 min:** TST was computed for every excluded subject and written to the
`excluded` sheet tab (`tst_sec`, plus `tst_note` for partial-data cases). The rule is consistent
there: **SM0017 (41 min)** is excluded and **NT3 (132 min)** was removed from the cohort entirely;
both had data-quality caveats (NT3 = truncated FIF/MFF; SM0017 = only Wake/N1/N2 scored). Every
other excluded subject slept ≥ 252 min, so TST is an **additive** criterion that formalizes the
no-sleep cases rather than re-deriving the N2-based exclusions.

---

## N2-based exclusions — decision table

The Results report N2 bouts (≥ 300 s clean NREM2) ranging 2–21 per subject. The question for the
"not enough N2" exclusions is whether they are genuinely worse on N2 than the *retained* low-bout
subjects. This compares them on the metric that actually gates ISFS analysis — the number of
**clean N2 bouts ≥ 300 s** (BAD-split + merged, the ISFS-pipeline definition) — alongside TST and
N2 % of TST. Low-TST exclusions (SM0017) are omitted: they are out on the TST rule regardless of N2.

- **N2 %** = N2 ÷ TST (all sleep, excluding Wake).
- **Clean bouts** = NREM2 segments ≥ 300 s after splitting around BAD epochs and merging.

### Retained low-bout subjects (still in cohort)

| Subject | Group | TST (min) | N2 % | Clean bouts |
|---|---|---|---|---|
| EL3003 | YA | 394.0 | 13.4 | 3 |
| EL3020 | YA | 395.0 | 21.7 | 3 |
| EL3006 | YA | 350.3 | 23.2 | 3 |
| el3007 | YA | 386.5 | 25.1 | 3 |
| RY42 | HE | 357.5 | 29.1 | 3 |

The former 2-bout retained subjects (HG78, MCI13) are now TST-excluded, so the **retained floor is
3 clean bouts**.

### "Not enough N2" exclusions (TST ≥ 210; SM0017 omitted), ordered by clean-bout count

| Subject | Group | TST (min) | N2 % | Clean bouts | Sheet reason |
|---|---|---|---|---|---|
| EL3026 | YA | 366.0 | 8.6 | 0 | 8% N2 |
| EL3032 | YA | 344.5 | 18.7 | 1 | 17% N2, 1 bout |
| SG27 | HE | 361.0 | 24.1 | 1 | not enough N2 |
| NE32 | HE | 374.5 | 48.5 | 1 ‡ | re-referenced in advance, not enough N2 |
| EL3040 | YA | 384.5 | 18.1 | 2 | 17% N2, 2 bouts |
| EL3030 | YA | 411.0 | 25.1 | 2 | 2 long N2 bouts |
| MCI12 | MCI | 272.5 | 32.8 | 2 ‡ | not enough N2 |
| AH3 | MCI | 361.0 | 43.7 | 2 ‡ | small N2 + all sweaty |
| MCI11 | MCI | 329.0 | 43.6 | 2 ‡ | stages too fragmented |
| EG5 | HE | 322.9 | 34.9 | 3 ‡ | not enough N2 + a lot of bad channels |
| DS6 | HE | 367.5 | 40.5 | 7 ‡ | not enough N2 |

‡ Re-counted 2026-06-16 from the subject's `cleaned_annotations.txt` using the full ISFS-pipeline
definition (NREM2 split around BAD epochs, merged adjacent segments, kept if ≥ 300 s). MCI11 / MCI12
now have proper cleaned-annotation files (BAD + scoring), so their counts are no longer hypno-only
upper bounds. This corrects the prior values (NE32 3→1, AH3 4→2, MCI11 5→2, MCI12 3→2; EG5 and DS6
unchanged at 3 and 7).

## Interpretation / open decisions

- **Clearly justified (clean bouts < 3, below the retained floor):** EL3026 (0), EL3032 (1),
  SG27 (1), NE32 (1), EL3040 (2), EL3030 (2), MCI12 (2), AH3 (2), MCI11 (2). Fewer usable clean bouts
  than any retained subject → stay excluded on N2 grounds. After the BAD-split re-count, NE32, AH3,
  MCI11 and MCI12 join this group — they had appeared to clear the floor only because of inflated
  (non-BAD-split / hypno-only) counts. EL3026 (8.6 % N2) also has genuinely little N2.
- **At/above the retained floor (clean bouts ≥ 3):** EG5 (3) and DS6 (7). **RESOLVED 2026-06-16:**
  EG5 relabelled to "too many bad channels" (stays excluded on that ground, not N2); **DS6 re-included**
  (7 clean bouts, 40.5 % N2 — no N2 shortage) and added to the HE group.
- **Outcome:** after the BAD-split re-count + these two decisions, every remaining "not enough clean
  bouts" exclusion is genuinely below the 3-bout floor, and the other excluded subjects carry a
  non-N2 reason (bad channels / bad epochs / TST < 210). The exclusion categories are now consistent.
