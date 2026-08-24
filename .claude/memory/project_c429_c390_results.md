---
name: project_c429_c390_results
description: "C429 ANCOVA + C390 extended sleep stats RUN 2026-08-13 → three_groups_V11 + demographics_V4; peak-freq survives the N2-duration covariate, bandwidth is largely a duration artefact"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6349a5e9-7b93-4a57-af24-3a6b05200f69
  modified: 2026-08-13T12:43:13.691Z
---

Both analyses Yuval asked for are done (2026-08-13). **Full write-up with all numbers, the
literature check and thesis-ready framing: `thesis/reviews/c429_c390_results.md`** — read that
rather than re-deriving. Scripts: `code/ancova_n2_duration.py`,
`code/sleep_statistics_extended.py`. Outputs are **new dirs only** — V10 and demographics_V3 are
untouched and remain source-of-truth for everything else, so the new dirs are *not*
self-contained (the "always use the highest V dir" habit does not apply here).

**C429 — ANCOVA, `results/group_comparison_results/three_groups_V11/`.** Whole-scalp ISFS
parameter ~ group + total analyzed N2 bout duration (mean-centred, from
`demographics_V3/n2_bouts_per_subject.csv`). Unadjusted block reproduces V10 exactly.

| parameter | unadj. p | ANCOVA group p | covariate p |
|---|---|---|---|
| peak frequency | 0.0026 | **0.0034** | 0.9384 |
| bandwidth | 0.0611 | 0.2063 | **0.0002** |
| AUC | 0.3758 (KW) | 0.4454 | 0.4903 |

**The headline finding survives** — peak frequency is untouched by the covariate (Holm pairwise
Y-vs-E p = 0.013, Y-vs-MCI p = 0.006, E-vs-MCI ns; adjusted means identical to raw to 4 dp).
**Bandwidth is the one that moves:** it correlates with analyzed duration at r = 0.386 and its
already-ns group effect drops to p = 0.21 — the age difference in bandwidth is substantially a
how-much-N2-you-analyzed effect. Do not present bandwidth as an aging effect. Slope homogeneity
holds for all three; AUC residuals are non-normal (Shapiro p = 0.0048) so its KW stays primary.

**C390 — extended sleep stats, `results/demographics_V4/`** (per-subject CSV, stats txt, and
`sleep_statistics_table.csv` shaped like `n2_bouts_table.csv` for the figures session). All four
rows Kruskal-Wallis: WASO 23.2 / 51.9 / 71.0 min (p < .001), REM latency from sleep onset
92.8 / 125.4 / 130.3 min (p = .005), sleep efficiency 89.2 / 83.6 / 78.6 % (p < .001) — all three
Young-vs-both significant, Elderly-vs-MCI ns, same Elderly = MCI pattern as the ISFS story.
Sleep onset latency is flat (p = .38).

**Why:** C429 was the only reviewer point that could have overturned the main result, and it
does not — but it does quietly demote bandwidth.

**How to apply:** hypnograms are rebuilt at **1 Hz from `*_cleaned_annotations.txt`**, not from
`ISO_data/scoring/` (mixed naming / letter-vs-int / 1 Hz vs 30 s). yasa's WASO, SOL and Lat_REM
already honour the `-1` policy of [[project_unknown_vs_acq_skip]]; **yasa's SE does not** (it
divides by full hypnogram length) so SE comes from the sheet's `sleep_efficiency_pct`. REM
latency is reported AASM-style (`Lat_REM - SOL`); yasa's from-recording-start value is a separate
column. Validation: hypnogram TST matches the sheet to ≤ 0.5 min for all 104.

**Quirk found:** EL3027 and EL3033 have scoring that ends 74 and 37 min before the recording
does, so a scored-span SE denominator gives +13.0 and +8.6 points vs the sheet's
recording-length denominator. Everyone else agrees to ≤ 0.2 points. Note also that the shared
KW eta² formula in `n2_bouts_table.run_metric_stats` can return a small negative value when
H < k−1 (sleep onset latency: −0.001).

Related: [[project_yuval_review]], [[project_unknown_vs_acq_skip]], [[project_v10_regeneration]],
[[project_scientific_story]], [[feedback_versioned_output_dirs]]
