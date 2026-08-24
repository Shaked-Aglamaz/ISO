---
name: project_missing_channel_handling
description: "Three different NaN policies across the ISFS analyses, which one feeds which result, and the 2026-08-08 decision to delete the manuscript claim rather than explain it"
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-08T18:39:53.646Z
  originSessionId: 9a7160eb-8c21-4c70-87df-8a7b6ba0b71a
---

A channel is NaN in `{subject}_all_channels_summary.csv` when its Gaussian fit failed the acceptance
checks. **Three different policies exist** — the pipeline never had one rule, and the manuscript used to
describe only one of them:

1. **Subject-mean fill → ≈1.0.** `step4_distribution_analysis.py::normalize_subject_channels` (~L83-87)
   replaces NaN with the subject's mean over accepted channels, then divides by the 3σ-trimmed scalp mean,
   so imputed cells land at the neutral value ≈1.0. This feeds **every topographic statistic**: cluster
   permutation + electrode-wise ANOVA + post-hoc Tukey (`step5_topo_comparison.py:74`), the normalized maps
   in F4B/S1, and the **ROI violin F5** (`step6_groups_comparison.py:990-991`).
2. **Neighbor fill — display only.** `utils/topo_aggregation.py::neighbor_impute`, raw topos (F4A). Enters
   no statistic. See [[project_raw_topo_aggregation_consistency]] (the V5 decision that created this split).
3. **No fill.** Plain `nanmean` / pandas `.mean()`: raw whole-scalp violins (F3), the displayed group means,
   and `moca_correlation.py`.

**Key numbers (final cohort YA35/HE39/MCI30, see [[project_v10_regeneration]]):**
- Whole-scalp NaN fraction differs by group — young **25.5%**, elderly **14.4%**, MCI **18.0%** (exactly the
  complement of the detection rates 74.5 / 85.6 / 82.0 already reported in Results 4.2).
- **At the 9 AUC-cluster electrodes it does NOT** — 11.7 / 10.8 / 14.1%, per-electrode validity 80–94% in
  every group. The imbalance is peripheral, so the cluster is not an imputation artefact. All 9 survive even
  an 80%-valid-in-every-group channel filter (70% keeps 108/176 channels, 80% keeps 45).
- **The fill is conservative, not inflationary:** re-running the F5 ROI ANOVA over accepted ROI channels only
  moves the means apart (Y 1.099→1.118, E 1.040→1.017, M 1.012→1.003) and p from 0.143 → **0.126**.

**Decision 2026-08-08 (final-check item B2, user's call): delete the incorrect claim, don't explain the
mechanism.** Results 4.3 said channels were "interpolated from their neighbours for visualization purposes
only, never for the statistics" — the neighbour half was true, "never for the statistics" was false. Replaced
with *"The value above each map in Figure 4A is the mean of the per-subject means."* (kept short rather than
deleted outright: F4A prints `mean=6.457/7.314/7.478` in its panel titles and this is the only text explaining
them; it is also where Flavio's comment #39 asked the convention to live). **No Methods paragraph was added
and no sensitivity run was made** — the user judged the full explanation overkill for the thesis. The numbers
above are the backup if the PI asks. Applied to `thesis/chapters/04_results.md` **and** the Google Doc.

**Second error found in the same check:** the F5 caption said "mean across the **fitted** electrodes of the
ROI". After normalization no channel is missing, so all 36 ROI electrodes are averaged — the word was removed.
The identical phrase in the **Figure 3 / whole-scalp** caption is correct and was left alone (that path really
does average only accepted-fit channels).

**Still live in code, deliberately not touched:** `step5_topo_comparison.py` passes `min_detection_rate=0.2`,
the criterion dropped thesis-wide ([[project_dropped_20pct_criterion]]). It excludes nobody in the final cohort
(lowest rates EL3033 20.5%, RY42 21.0%, MCI07 22.7%), so no reported result is affected.
