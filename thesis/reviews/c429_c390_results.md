# C429 + C390 — results, conclusions and thesis-ready facts

Run 2026-08-13. Answers two items from `thesis/reviews/yuval_review_triage.md`:
**C429** (are the ISFS group differences driven by differing amounts of N2 sleep?) and
**C390** (extend the sleep statistics with sleep efficiency, WASO and REM latency).

Cohort: **YA 35 / HE 39 / MCI 30 = 104**, mean ages 27 / 66 / 67. Stats only — no prose, no
figures. Sources of truth for everything else remain `three_groups_V10`, `demographics_V3`,
`moca_correlation_V3`.

---

## 0. Bottom line

**Generally good news.** The headline result is unaffected by the reviewer's confound, and three
of the four new sleep metrics land where the literature says they should.

| | Outcome |
|---|---|
| C429 — does N2 amount explain the ISFS differences? | **No.** Peak frequency survives the covariate essentially unchanged (p 0.0026 → 0.0034). |
| C429 — bandwidth | Group effect weakens (p 0.061 → 0.206), but it was **already non-significant**, so no published claim changes. One new caveat to state (§1.3). |
| C429 — AUC | Unchanged, ns before and after. |
| C390 — WASO, sleep onset latency, sleep efficiency | Extracted, compared, and **consistent with the aging literature** in direction and roughly in magnitude. |
| C390 — REM latency | Solid within our data (p = 0.005) but sits in **contested territory** in the literature (§3.3). Needs one honest sentence, not an apology. |
| MCI vs Elderly, all sleep metrics | Directions match the aMCI meta-analysis; none reach significance — underpowered at n = 30/39, and our MCI group is mixed rather than aMCI-only. |

Two things that must not be left silent in the thesis:
1. **ISFS bandwidth is duration-dependent** (§1.3) — a methodological caveat, not a finding.
2. **REM latency diverges from the most-cited aging trend** (§3.3) — flag it, don't hide it.

---

## 1. C429 — ANCOVA with analyzed N2 duration as covariate

Output: `results/group_comparison_results/three_groups_V11/`
(`three_group_ancova_statistics.txt`, `three_group_ancova_per_subject.csv`)
Script: `code/ancova_n2_duration.py`

### 1.1 Design

Per-subject whole-scalp ISFS values (mean across all channels, channels without a detected ISFS
skipped by the mean) regressed on group plus each subject's **total analyzed N2 bout duration**,
mean-centred:

```
parameter ~ C(group) + n2_min_centred        Type-II ANOVA, N = 104, residual df = 100
```

Covariate from `demographics_V3/n2_bouts_per_subject.csv` (`total_dur_min`), grand mean
**93.09 min**:

| Group | Analyzed N2 duration (min) |
|---|---|
| Young | 80.70 ± 44.23 |
| Elderly | 106.01 ± 50.05 |
| MCI | 90.75 ± 39.89 |

Correlation of the covariate with each parameter across all 104 subjects:
peak frequency **r = +0.064**, bandwidth **r = +0.386**, AUC **r = +0.090**.

### 1.2 Results

| Parameter | Unadjusted omnibus | ANCOVA: group | ANCOVA: covariate | Verdict |
|---|---|---|---|---|
| Peak frequency | ANOVA F = 6.3153, **p = 0.0026**, η² = 0.1112 | F = 6.0279, **p = 0.0034**, partial η² = 0.1076 | F = 0.0060, p = 0.9384, partial η² = 0.0001 | survives |
| Bandwidth | ANOVA F = 2.8739, p = 0.0611, η² = 0.0538 | F = 1.6036, p = 0.2063, partial η² = 0.0311 | F = 14.7300, **p = 0.0002**, partial η² = 0.1284 | ns before and after |
| AUC | Kruskal-Wallis H = 1.9572, p = 0.3758, η² = 0.0194 | F = 0.8153, p = 0.4454, partial η² = 0.0160 | F = 0.4794, p = 0.4903, partial η² = 0.0048 | ns before and after |

**Raw vs covariate-adjusted group means** (Young / Elderly / MCI):

| Parameter | Raw | Adjusted at grand-mean covariate |
|---|---|---|
| Peak frequency (Hz) | 0.0199 / 0.0226 / 0.0232 | 0.0199 / 0.0226 / 0.0232 |
| Bandwidth (Hz) | 0.0236 / 0.0281 / 0.0276 | 0.0244 / 0.0272 / 0.0278 |
| AUC (AU) | 6.4574 / 7.3137 / 7.4780 | 6.5178 / 7.2508 / 7.4894 |

**Peak frequency post-hoc on adjusted means** (Holm-corrected pairwise contrasts):

| Contrast | p (Holm) |
|---|---|
| Young vs Elderly | **0.0130** |
| Young vs MCI | **0.0057** |
| Elderly vs MCI | 0.5944 |

Identical in structure to V10's Tukey result (0.0129 / 0.0048 / 0.8562). Covariate slope on peak
frequency is +7.0 × 10⁻⁷ Hz per minute of analyzed N2 — numerically nil.

**Assumption checks.** Homogeneity of regression slopes holds for all three (group × covariate
interaction p = 0.1104 / 0.6909 / 0.8121). Residual normality: peak frequency p = 0.7640 and
bandwidth p = 0.9682 pass; **AUC residuals fail (W = 0.9624, p = 0.0048)**, consistent with V10
using Kruskal-Wallis for AUC. The AUC ANCOVA is therefore a covariate-adjusted supplement, and
the unadjusted KW stays primary for that parameter.

### 1.3 Conclusions

**Claimable.** Entering the amount of analyzed N2 sleep as a covariate leaves the peak-frequency
effect intact: F drops from 6.32 to 6.03, p from 0.0026 to 0.0034, partial η² from 0.111 to 0.108,
adjusted means identical to raw means to four decimals, and both significant pairwise contrasts
survive Holm correction. The covariate itself explains no peak-frequency variance whatsoever
(p = 0.94). Combined with the descriptive argument already established — Young adults contributed
the **least** analyzed N2 (80.7 vs 106.0 and 90.8 min) yet showed the strongest central-parietal
hotspot, so the confound runs **opposite** to the effect — C429 is answered on two independent
grounds.

**Must be stated, not omitted.** Analyzed N2 duration is the single strongest predictor of ISFS
**bandwidth** in the model (partial η² = 0.128, p = 0.0002; r = +0.386), larger than the group
effect. Longer analysed recordings yield wider fitted ISFS peaks — almost certainly a
spectral-resolution / averaging property of the estimate rather than biology. No published claim
changes, because bandwidth was already non-significant in V10 (p = 0.0611). But **bandwidth must
not be presented anywhere as an aging effect**, and the duration dependence belongs in the
Methods or Limitations as a property of the bandwidth estimator.

**Not claimable.** Nothing about AUC changes; it remains ns under both tests.

---

## 2. C390 — extended sleep statistics

Output: `results/demographics_V4/`
(`sleep_statistics_stats.txt`, `sleep_statistics_per_subject.csv`,
`sleep_statistics_table.csv` — the last is the tidy per-metric table for the figures session,
same column shape as `demographics_V3/n2_bouts_table.csv`)
Script: `code/sleep_statistics_extended.py`

### 2.1 Results

All four metrics failed the per-group normality gate, so all are Kruskal-Wallis with Dunn (Holm),
n = 35 / 39 / 30 throughout. Values are mean ± SD (median).

| Metric | Young | Elderly | MCI | Omnibus | Y–E | Y–MCI | E–MCI |
|---|---|---|---|---|---|---|---|
| WASO (min) | 23.17 ± 19.56 (16.5) | 51.87 ± 30.02 (40.5) | 70.98 ± 46.25 (59.8) | H = 31.436, **p < .001**, η² = 0.291 | **.0001** | **<.0001** | .160 |
| Sleep onset latency (min) | 17.40 ± 17.08 (11.5) | 15.59 ± 15.05 (12.5) | 19.87 ± 18.35 (14.0) | H = 1.933, p = .380, η² ≈ 0 | — | — | — |
| REM latency (min) | 92.76 ± 45.91 (78.3) | 125.44 ± 63.11 (114.5) | 130.26 ± 55.90 (108.2) | H = 10.527, **p = .005**, η² = 0.084 | **.020** | **.008** | .549 |
| Sleep efficiency (%) | 89.23 ± 6.71 (90.7) | 83.63 ± 8.49 (84.2) | 78.56 ± 11.68 (81.3) | H = 19.193, **p < .001**, η² = 0.170 | **.008** | **.0001** | .106 |

Supporting values for context (group means): TST 390.8 / 371.4 / 353.6 min;
N1 14.8 / 42.4 / 45.6 min; N3 121.9 / 81.0 / 86.1 min.

### 2.2 Conclusion

Three of the four metrics reproduce the same pattern as the ISFS story: **Young differs from both
older groups, Elderly and MCI are statistically indistinguishable.** Sleep onset latency is flat
across all three. This is a useful independent corroboration of the aging-not-MCI framing — the
sleep-architecture differences behave the same way the ISFS parameters do.

---

## 3. Literature check on the C390 numbers

### 3.1 Aging (Young vs Elderly)

Benchmarked against per-decade slopes from Boulos et al. 2019. Our Young→Elderly gap spans
**3.9 decades** (mean age 27 → 66).

| Metric | Literature slope | Predicted change | Observed change | Verdict |
|---|---|---|---|---|
| WASO | +9.7 min/decade | +37.8 min | **+28.7** (23.2 → 51.9) | right order, slightly conservative |
| Sleep onset latency | +1.1 min/decade | +4.3 min | **−1.8** (ns) | both negligible |
| Sleep efficiency | −2.1 %/decade | −8.2 pts | **−5.6** (89.2 → 83.6) | right direction, somewhat smaller |
| REM latency | ↓ with age (Ohayon 2004) | shorter | **+32.6 min longer** | contested — see §3.3 |

Young-group absolutes fall inside published normative ranges: sleep efficiency 89.2 % (≈85 %
typical for healthy adults) and REM latency 92.8 min (normal band 70–100 min). That is a useful
sanity check on the whole extraction pipeline.

### 3.2 MCI vs Elderly

Benchmarked against the amnestic-MCI PSG meta-analysis.

| Metric | Meta-analysis (PSG subgroup) | Ours (MCI vs HE) | Agreement |
|---|---|---|---|
| Sleep efficiency | lower in aMCI, SMD −1.83, p = 0.002 | 78.6 vs 83.6, p = 0.106 | direction yes, significance no |
| WASO | higher in aMCI, SMD +1.29, p = 0.002 | 71.0 vs 51.9, p = 0.160 | direction yes, significance no |
| Sleep onset latency | ns, SMD −0.02, p = 0.97 | ns, p = 0.380 | full agreement |
| REM latency | not pooled; narrative reviews report longer in MCI | 130.3 vs 125.4, ns | flat in ours |

Every direction matches. None reaches significance, which is unsurprising at n = 30 vs 39, and
our MCI group is **mixed rather than aMCI-only**, which would dilute effects relative to the
aMCI-restricted meta-analysis. Safe framing: directionally consistent with the MCI literature but
underpowered to confirm.

### 3.3 REM latency — the one contested row

The claim that REM latency shortens with age is **not settled**, and should not be treated as the
benchmark our data failed.

- **Ohayon et al. 2004** does report REM latency significantly decreasing with age in adults — but
  its own headline caveat is that **only sleep efficiency continued to significantly decrease after
  age 60.** The REM-latency trend is carried by the younger part of the lifespan, so that
  meta-analysis establishes no expectation *within* older adults, and none at all for our
  MCI-vs-Elderly comparison.
- The standard aging-and-sleep review discusses REM **percentage** (which plateaus after 60) and
  does not address latency at all. REM latency is not a headline aging parameter.
- A 40-year longitudinal PSG follow-up found REM latency **does not change** significantly by age
  group. General reviews describe REM changes as mixed, becoming clear only past ~80 or with
  pathology.
- An active literature runs the **opposite** way, specifically in aging and cognitive impairment:
  prolonged REM latency is associated with poorer cognition in older adults (adjusted for age, sex,
  education); **tau deposition in the pedunculopontine nucleus correlates with increased REM
  latency** (p < 0.001; p = 0.014 controlling for global tau and amyloid; p = 0.001 in
  amyloid-positive, p = 0.123 in amyloid-negative individuals); and the MCI systematic review lists
  longer REM latency among MCI's macro-architectural changes.

**Our observation is not an artifact.** Medians shift the same way as means (78.3 → 114.5 →
108.2 min), 18/39 Elderly and 12/30 MCI exceed 120 min, 8 Elderly exceed 180 min, and the
correlation with fragmentation is only weak (WASO r = +0.28 HE, +0.32 MCI), so WASO does not
explain it.

**Suggested honest position:** report the values, note they align with reports of prolonged REM
latency in older and cognitively impaired samples, and acknowledge they run counter to the
most-cited meta-analytic aging trend — whose own analysis does not extend past 60. Add that a
**single-night design cannot separate this from a first-night effect**, which is known to prolong
REM latency and may affect older participants more. The tau/amyloid interpretation is **not
available to us** — this cohort has no biomarkers.

### 3.4 References used

| Source | Used for |
|---|---|
| Ohayon MM, Carskadon MA, Guilleminault C, Vitiello MV. *Sleep* 2004;27(7):1255–73. PMID 15586779 | lifespan normative trends; the "only SE continues past 60" caveat |
| Boulos MI et al. *Lancet Respir Med* 2019;7(6):533–43. PMID 31006560 | per-decade slopes: WASO +9.7 min, SOL +1.1 min, SE −2.1 % |
| Sleep structure in amnestic MCI: a meta-analysis. PMC7689212 | aMCI vs control SMDs for SE, SOL, WASO |
| Objective measurement of sleep in MCI: systematic review and meta-analysis. *Sleep Med Rev* 2020. PMID 32302775 | MCI macro-architecture incl. longer REM latency |
| Aging and Sleep: Physiology and Pathophysiology. PMC3500384 | REM % plateaus after 60; latency not addressed |
| Sleep and Aging: a polysomnographic 40-year follow-up. *J Sleep Res* 2025 | REM latency unchanged by age group |
| Tau in the pedunculopontine nucleus and REM sleep alterations. PMC11715004 | PPN-tau ↔ increased REM latency |

> These are **not yet in `references/library.bib`** — adding them is the prose/citations session's
> call, and several would also count toward the ≥50-reference requirement (email item #9).

---

## 4. Methods facts needed for the write-up

- **ISFS per-subject values** are the mean across all channels, with undetected channels skipped by
  the mean — identical to the raw whole-scalp values already in V10. The ANCOVA adds nothing to the
  ISFS pipeline itself.
- **Hypnograms** were rebuilt at **1 Hz** from each subject's `*_cleaned_annotations.txt`. Second
  resolution is required: annotation onsets and durations are not 30-s multiples, so a 30-s grid
  would round.
- **Unscorable epochs.** `UNKNOWN` (scorer `-1`) is treated as a third category — not sleep, not
  wake, not recorded — excluded from both the numerator and the denominator of sleep efficiency and
  never folded into WASO. 24 subjects carry unscorable time (AT36 the extreme at 100.7 min).
  Rationale and the worked AT36 comparison are in `notes/tst_waso_unknown_handling.md`.
- **`BAD` / `BAD_EPOCH` / `BAD_ACQ_SKIP` annotations are not used** for sleep statistics. In
  `*_cleaned_annotations.txt` the `BAD_ACQ_SKIP` label is overloaded with hand-drawn artifact marks
  from the manual cleaning step (AT36 has 3 real gaps plus 58 manual marks; LS56 has 94), so the
  label cannot identify real recording gaps.
- **WASO, sleep onset latency and REM latency** come from `yasa.sleep_statistics` (yasa 0.6.5),
  which already honours the policy above — its WASO counts only wake epochs inside the sleep period
  time, so unscorable epochs are excluded.
- **REM latency is measured from sleep onset (AASM convention)**, i.e. yasa's `Lat_REM − SOL`.
  yasa's own `Lat_REM` is from the start of the recording and is retained as a separate column.
  State the convention explicitly — the two differ by the sleep onset latency.
- **Sleep efficiency** is the subjects-sheet column `sleep_efficiency_pct`, not recomputed.
- **Statistics.** Shapiro-Wilk per group → one-way ANOVA if all three normal, else Kruskal-Wallis →
  Tukey HSD or Dunn (Holm) only when the omnibus reaches p < 0.05, with η². Identical to the scheme
  behind `demographics_V3/sleep_stage_stats.txt`. ANCOVA post-hoc uses **Holm-corrected pairwise
  contrasts of adjusted means** rather than Tukey HSD, because Tukey on adjusted means requires
  `emmeans`, which is unavailable in this environment. statsmodels 0.14.6.
- **Three omnibus tests on the ISFS parameters remain uncorrected across parameters** — unchanged
  from V10, and already on the pending list as S12/S13.

---

## 5. Verification record

Everything below was checked before the numbers above were accepted.

1. **ANCOVA anchors to V10 exactly.** The unadjusted block reproduces V10 line for line:
   N = 35/39/30; peak frequency F = 6.3153, p = 0.0026; bandwidth F = 2.8739, p = 0.0611; AUC
   H = 1.9572, p = 0.3758; Tukey 0.8562 / 0.0129 / 0.0048.
2. **Covariate anchors to V3.** Group means 80.7 / 106.0 / 90.8 min; merge 104/104, zero missing,
   no group-label mismatch.
3. **AT36 regression test** against the hand-computed values in
   `notes/tst_waso_unknown_handling.md`: TST 343.3 min (note 343.5), WASO 74.0, SOL 84.0,
   SPT 442.0, unscorable 100.7 (note 100.5), policy SE 80.4 %. A WASO near 98.5 would have meant
   `-1` leaked into wake; it did not.
4. **Hypnogram source cross-check.** Rebuilding from the raw scoring files instead of the
   annotations changes nothing: of the **93 of 104** subjects that have a locatable scoring file,
   **89 are 100.000 % identical second-by-second**, 91 are ≥99.9 %, all 93 are ≥99.2 %. Sleep onset
   latency and REM latency differ by **0.0000 min for every one of the 93**; WASO differs for a
   single subject (RS5) by **20 seconds**. The only sub-99.9 % cases, ON68 and RS5, are the
   181-s uncovered gap and 29-s annotation overlap already recorded as QC columns.
5. **TST cross-check** against the sheet's `tst_sec`: agreement within **0.5 min for all 104**.
6. **Nothing outside the two new output dirs was written**; no commits.

### Why the annotations, not the scoring files

Worth recording, because the scoring tree is a trap. Resolving it needs eight filename patterns
plus per-subject epoch-length inference (68 subjects at 1 s/line, 25 at 30 s/line), **11 cohort
subjects have no scoring file at all** (EL3031, EL3034, EL3035, EL3036, EL3037, EL3044, SM0016,
SM0018, SM0019, SM0020, SM09), and files are **misfiled across group subdirectories** — LS56 and
SB00 are elderly-group subjects whose correct scoring sits under `scoring/young_control/`. Stale
duplicates in the `scoring/` root agree with the real scoring at only **60.3 %** and **84.8 %**, so
a name-based lookup of that directory would silently use the wrong scoring for those two. Also, six
scoring files contain `?`, which the preprocessing maps to **Wake**, not unscorable; the
annotations bake in that decision, so reading the raw files and treating `?` differently would
diverge from every other analysis in the repo.

---

## 6. Open flags (reported, not acted on)

- **EL3027 and EL3033** have scoring that ends **74** and **37 min** before their recordings do.
  Their sheet sleep-efficiency values divide by full recording length, so a scored-span denominator
  would read **+13.0** and **+8.6** points higher. Every other subject agrees within 0.2 points.
  If either subject's sleep efficiency is quoted individually, that number reflects a denominator
  choice, not a sleep difference.
- The shared Kruskal-Wallis η² formula returns a small **negative value (−0.001)** on the
  non-significant sleep-onset-latency row, since (H − k + 1)/(n − k) can go below zero when
  H < k − 1. Inherited from `n2_bouts_table.run_metric_stats`, which `demographics_V3` already
  uses. Cosmetic; on an ns row. Left unchanged.
- **First-night effect** is unaddressed and unaddressable — single-night design, no adaptation
  night. Most relevant to the REM latency row (§3.3).
- The remaining open item in `notes/tst_waso_unknown_handling.md` (unscorable minutes inside SPT for
  the other 22 subjects) is now covered by the `unscorable_min` column in
  `sleep_statistics_per_subject.csv`.
- **Not run, deliberately:** any rank-based or sustained-REM-criterion sensitivity analysis. Both
  are outside what the reviewer asked for.

---

## 7. File inventory

| Path | Contents |
|---|---|
| `code/ancova_n2_duration.py` | C429 ANCOVA |
| `code/sleep_statistics_extended.py` | C390 extraction + group comparison |
| `results/group_comparison_results/three_groups_V11/three_group_ancova_statistics.txt` | full ANCOVA report incl. unadjusted baseline, Type-II tables, adjusted means, post-hoc, assumption checks |
| `results/group_comparison_results/three_groups_V11/three_group_ancova_per_subject.csv` | subject, group, 3 ISFS parameters, `total_dur_min` |
| `results/demographics_V4/sleep_statistics_stats.txt` | per-metric report in `sleep_stage_stats.txt` format |
| `results/demographics_V4/sleep_statistics_per_subject.csv` | all metrics + TST/TIB/SPT, unscorable min, policy SE, QC columns |
| `results/demographics_V4/sleep_statistics_table.csv` | **tidy table for the figures session** |

Both scripts are re-runnable from the repo root:

```bash
source eeg_clean/Scripts/activate
PYTHONIOENCODING=utf-8 python code/ancova_n2_duration.py
PYTHONIOENCODING=utf-8 python code/sleep_statistics_extended.py
```
