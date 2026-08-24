# Yuval revision — analysis work (C429 ANCOVA, S1 site check, C390 sleep statistics)

## Context

Yuval Nir returned a review of the thesis (triaged in `thesis/reviews/yuval_review_triage.md`).
Three of his items need new statistics rather than new prose, and two of them independently
reproduce findings already flagged in our own pre-send check (`thesis/final_check_report.md`):

- **C429 / S7** — Results 4.1 currently argues that group differences in ISFS "do not reflect
  differing amounts of N2 sleep" from the fact that N2 is the largest stage. That is a
  non-sequitur, and N2 share *does* differ across groups (KW p = 0.0009). Yuval asks whether we
  "tried to equate this, or include this as a factor". Answer it empirically: ANCOVA with
  analyzed N2 duration as covariate.
- **S1** — group and recording site are entangled (Young 35 TASMC / 0 Sydney, Elderly 30 / 9,
  MCI 14 / 16). No site check has ever been run. The two older groups are the natural test beds.
- **C390** — he wants sleep efficiency, WASO and REM latency added to the sleep-architecture
  table. Sleep efficiency already exists in the subjects sheet; WASO, sleep-onset latency and
  REM latency have never been extracted.

Intended outcome: three stats reports + tidy CSVs in a new versioned output directory, so the
prose session can quote numbers and the figures session can drop a table in. **Numbers only —
no prose, no figures, no commits.**

Cohort: YA 35 / HE 39 / MCI 30 (104). Sources of truth: `results/group_comparison_results/three_groups_V10`,
`results/demographics_V3`, `results/moca_correlation_V3`.

**Scope confirmed with the user:** exactly these three analyses (the "four" in the request was a
miscount — the C395 interpolation-order sensitivity is *not* in scope). The site check covers
**both** Elderly (30 vs 9) and MCI (14 vs 16), plus a pooled older-adult test.

### Ownership / do-not-touch

Owned here: new scripts under `code/`, new output dir `results/yuval_revision_V11/`.
Off limits (other sessions): `thesis/chapters/*.md`, `thesis/figures/`, `thesis/figure_manifest.md`,
`thesis/references/library.bib`, the Google Doc. Existing scripts are **read/import only** — no edits
to `step4/step5/step6`, `n2_bouts_table.py`, `sleep_stage_pies.py`, `demographics_table.py`.
No `git add/commit/push`.

### Environment

```bash
source eeg_clean/Scripts/activate          # run everything from I:/Shaked/ISO
PYTHONIOENCODING=utf-8 python code/<script>.py
```
`statsmodels` 0.14.6, `scipy` 1.15.3, `scikit-posthocs` 0.12.0, `yasa` 0.6.5 are installed.
`pingouin` is **not** — use `statsmodels.formula.api.ols` + `anova_lm`, don't install anything.

---

## Locked data-path decisions

These are the pieces the three analyses share. Getting them wrong is the only real risk, so each
one is pinned to an existing function plus a number it must reproduce.

| Need | Use | Must reproduce |
|---|---|---|
| Cohort roster (104, with group) | `results/demographics_V3/n2_bouts_per_subject.csv` (`subject_id`,`group` ∈ YA/HE/MCI) | 35 / 39 / 30 |
| Per-subject **whole-scalp** ISFS scalar | `step4_distribution_analysis.load_and_process_all_channels_data(subjects, dir_path)[0]` — `groupby.mean()`, i.e. nanmean over all 176 rows, **no imputation** | V10 group means + F/H, p, η² |
| Per-subject **normalized extended-ROI AUC** | `step6_groups_comparison.load_and_process_roi_data_normalized(subjects, dir, EXTENDED_CENTRAL_PARIETAL_ROI)` — normalizes over **all** channels then restricts to ROI (the rule) | ANOVA F = 1.986, p = 0.1426, η² = 0.038 |
| ISFS results dirs | `results/sigma_fix_YA` / `_HE` / `_MCI` (subject subdirs) | dir count 35 / 39 / 30 |
| N2 covariates | same `n2_bouts_per_subject.csv`: `total_dur_min` (primary), `bout_count`, `prop_of_n2_pct` | YA 80.7 / HE 106.0 / MCI 90.8 min |
| Site | `demographics_table._site(subject_id)` — `"MCI" in id.upper()` → Sydney | HE 30/9, MCI 14/16, YA 35/0 |
| Normality-gated omnibus | **import** `n2_bouts_table.run_metric_stats(df_long, metric)` — it is already generic (`group` column + metric column) and returns Shapiro / ANOVA-or-KW / Tukey-or-Dunn(Holm) / η² | matches `sleep_stage_stats.txt` scheme |

Two gotchas: the roster label is `YA/HE/MCI` while step6 uses `Young/Elderly/MCI` — map once;
and one young subject id is lowercase (`el3007`) — merge case-insensitively.

**NaN policy:** whole-scalp scalars use nanmean over fitted channels only (no imputation) — this is
what `three_group_statistics.txt` tests, so the ANCOVA must use the same DV to be comparable. The
subject-mean imputation only happens inside `normalize_subject_channels`, i.e. only on the ROI path.

---

## Files to create

```
code/revision_stats_common.py      # shared: roster, scalars, site, report writer
code/ancova_n2_duration.py         # C429
code/site_check.py                 # S1
code/sleep_statistics_extra.py     # C390
results/yuval_revision_V11/        # all outputs (new dir)
```

### `code/revision_stats_common.py`

Small shared module so the three scripts agree on the cohort. Follows the `code/`-level import
convention (`sys.path` parent prepend, `from utils.config import ...`).

- `load_roster()` → DataFrame `subject_id, group (YA/HE/MCI), group_dir, site, total_dur_min, bout_count, prop_of_n2_pct`
  from `n2_bouts_per_subject.csv`; adds `site` via `demographics_table._site`. Asserts 35/39/30.
- `load_whole_scalp_scalars()` → per-subject `peak_frequency, bandwidth, auc` by calling
  `load_and_process_all_channels_data` once per group against `results/sigma_fix_*`.
- `load_roi_normalized_auc()` → per-subject `auc_roi_norm` via `load_and_process_roi_data_normalized`
  with `EXTENDED_CENTRAL_PARIETAL_ROI`.
- `build_subject_table()` → the merged 104-row frame all three scripts start from; written once to
  `results/yuval_revision_V11/subject_scalars_V11.csv`.
- `format_omnibus_block(...)` → the `--- <var> ---` / `Group means:` / `Shapiro p:` / `Omnibus:` /
  `Post-hoc:` text block, byte-for-byte in the style of `demographics_V3/sleep_stage_stats.txt`
  (`%.2f` means, `%.3f` Shapiro, `stat=%.3f, p=%.4f, eta2=%.3f`, `p=%.4f (*|ns)`).

### 1. `code/ancova_n2_duration.py` — C429

Outcomes (4): whole-scalp `peak_frequency`, `bandwidth`, `auc`, plus secondary
`auc_roi_norm` (the Figure-5 statistic).

Per outcome:
1. **Unadjusted reference** — re-run `run_metric_stats` so the report shows the V10 result next to
   the adjusted one (must match `three_group_statistics.txt`).
2. **Primary ANCOVA** — `ols("y ~ C(group) + total_dur_min").fit()` → `anova_lm(typ=2)`.
   Report: group F/p, partial η², covariate slope, covariate t/p, model R².
3. **Adjusted (estimated marginal) group means** at the grand-mean covariate, with SE.
4. **Covariate-adjusted post-hoc** — pairwise contrasts from the fitted model
   (`t_test` on contrast vectors), Holm-corrected across the 3 pairs.
5. **Assumption diagnostics** — homogeneity of regression slopes (`y ~ C(group)*total_dur_min`,
   report the interaction p), Shapiro on residuals, Levene on residuals by group.
6. **Rank fallback** — repeat the ANCOVA on rank-transformed y when residuals fail Shapiro
   (matters for AUC, which was KW in V10), so the conclusion doesn't rest on normality.
7. **Secondary covariates** — same model with `prop_of_n2_pct` and with `bout_count`, reported
   compactly (group F/p only). `prop_of_n2_pct` matters because it is the N2 variable that
   actually differs across groups.

Also emit the descriptive block that already answers the objection: covariate group means ±SD and
their own omnibus test (total analyzed duration KW p = 0.081; proportion of N2 retained ANOVA
p = 0.98), plus the direction note — young contribute the *least* analyzed N2 yet carry the
hotspot, so the confound runs opposite to the effect.

Outputs: `ancova_n2_duration.txt`, `ancova_n2_duration.csv` (one row per outcome × model:
`outcome, model, covariate, group_F, group_p, partial_eta2, cov_beta, cov_p, slopes_interaction_p, resid_shapiro_p`),
`ancova_adjusted_means.csv`.

### 2. `code/site_check.py` — S1

Same 4 outcomes. Three tests each:

1. **Elderly within-group**, TASMC n = 30 vs Sydney n = 9.
2. **MCI within-group**, TASMC n = 14 vs Sydney n = 16.
3. **Pooled older adults** (HE + MCI, n = 69): `ols("y ~ C(site) + C(group)")` → site effect
   adjusted for diagnostic group. This is the actual confound test; the two within-group tests
   are the readable version.

Two-group tests follow the repo's existing two-sample helpers in `step6_groups_comparison`
(`test_normality_shapiro` → `test_ttest_independent` / `test_mannwhitneyu`, gated the same way).
Because a null at n = 9 is only meaningful with an effect size, also report **Hedges' g with 95 %
CI** (rank-biserial r for the non-parametric branch) and the minimum detectable effect at 80 %
power for each n — a bare p > 0.05 will not satisfy a PI.

State explicitly in the report header that Young cannot be tested (0 Sydney), and record the
site composition table. Note for the prose session: the young-vs-elderly peak-frequency effect is
largely within-TASMC and therefore the least site-exposed contrast.

Outputs: `site_check.txt`, `site_check.csv`
(`outcome, comparison, n_tasmc, n_sydney, mean_tasmc, mean_sydney, test, stat, p, effect_size, ci_low, ci_high`).

### 3. `code/sleep_statistics_extra.py` — C390

**Hypnogram source: reconstruct from `{BASE_DIR}/{group_dir}/{sub}/{sub}_cleaned_annotations.txt`,
not from the raw scoring files.** The raw `ISO_data/scoring/` tree has inconsistent naming
(`{SUB}.txt` / `_hypno` / `_hypo` / `_hypnoWholeFile_revised` / `visFormatFromAlice`), ambiguous
multi-file variants, and no file at all under the group folders for ~20 of the 104 subjects. The
cleaned annotations are uniform across both sites, already length-matched to the recording, and
are the source of the `tst_sec` / `sleep_efficiency_pct` / `pct_*` numbers already published —
so the new metrics will be consistent with the existing table by construction.

Pipeline per subject:
1. `utils.cohort_overview.parse_annotations(path)` — reuses `subject_summary.parse_annotations`
   and applies `STAGE_ALIASES`, which already normalizes the two label dialects
   (`Wake/NREM2` vs `WAKE/N2`).
2. Rasterize stage spans onto a **1 s grid** (exact for both the TASMC 1 Hz and the Sydney 30 s
   scoring, since Sydney durations are multiples of 30). Ignore `BAD_*` spans — they overlap
   stages rather than tiling. `UNKNOWN` and gaps → `-2` (yasa "Unscored").
3. Map to yasa codes (`0 W, 1 N1, 2 N2, 3 N3, 4 REM, -2 Unscored`) — this project's integer
   dialect already is yasa's, so no remap beyond the unscored value.
4. `yasa.sleep_statistics(hypno, sf_hyp=1.0)` → `TIB, SPT, WASO, TST, SOL, Lat_REM, SE, …`
   (minutes).

Reported metrics: **Sleep efficiency (%)**, **WASO (min)**, **Sleep-onset latency (min)**,
**REM latency (min)** — reported *both* as yasa's `Lat_REM` (from recording start) and as
`Lat_REM − SOL` (from sleep onset), since the two conventions differ across papers and the prose
session will need to say which. Subjects with no REM → NaN, with the per-metric n stated.

Stats: `n2_bouts_table.run_metric_stats` on each metric — identical normality-gated scheme to
`demographics_V3/sleep_stage_stats.txt`.

Outputs:
- `sleep_statistics_per_subject.csv` — `subject_id, group, site, TIB, SPT, TST, WASO, SOL, Lat_REM, rem_lat_from_onset, SE` (104 rows).
- `sleep_statistics_stats.txt` — same format as `sleep_stage_stats.txt`.
- `sleep_statistics_table.csv` — **wide, matching `n2_bouts_table.csv`'s column shape**
  (`metric, {G}_n/{G}_mean/{G}_std, omnibus_test, omnibus_stat, omnibus_p, eta_squared, p_YA_vs_HE, p_YA_vs_MCI, p_HE_vs_MCI`)
  so the figures session can feed it straight into `combined_demographics_table.build_rows()`
  as a third section band without reshaping.

---

## Verification

Run in order; each step gates the next.

1. **Cohort** — `subject_scalars_V11.csv` has 104 rows, 35/39/30, site split 35/0, 30/9, 14/16.
2. **Scalar reproduction (blocking)** — before any new model, print the unadjusted 3-group tests
   for the 4 outcomes and diff against
   `three_groups_V10/three_group_statistics.txt` (peak freq F = 6.3153, p = 0.0026, η² = 0.111;
   bandwidth F = 2.8739, p = 0.0611; AUC KW H = 1.9572, p = 0.3758) and
   `three_group_statistics_extended_ROI_normalized_auc.txt` (F = 1.986, p = 0.1426).
   Any mismatch means the loader is wrong — stop and fix before interpreting anything.
3. **Covariate reproduction** — `total_dur_min` group means 80.7 / 106.0 / 90.8 and KW p = 0.0811,
   `prop_of_n2_pct` ANOVA p = 0.9808, against `demographics_V3/n2_bouts_table.txt`.
4. **Hypnogram reconstruction (blocking)** — for all 104 subjects, the rasterized hypnogram must
   reproduce the sheet/`overview/subjects.csv` values: `TST` within ±0.5 min of `tst_sec`,
   `SE` within ±0.5 pp of `sleep_efficiency_pct`, `%N2` within ±0.5 pp of `pct_n2`. Print a table
   of any subject that fails and resolve before running the stats. Also assert
   `WASO ≈ SPT − TST` and `SOL + SPT ≈ TIB − trailing wake` as internal sanity.
5. **Output completeness** — 8 files present in `results/yuval_revision_V11/`, no NaNs in any
   omnibus p, per-metric n printed wherever it is below the group n (REM latency).
6. **Report the numbers back in chat** — headline table of: whether each group effect survives the
   N2-duration covariate, the site p-values with effect sizes, and the four new sleep metrics with
   their omnibus results. Flag anything that changes a claim currently in the manuscript, but do
   not edit any prose.

## Out of scope (flag, don't do)

C395 interpolation-order sensitivity; B2 imputation sensitivity (AUC cluster on well-fitted
channels); the peak-frequency across-subject range for Results 4.3; anything under `thesis/`;
any figure; any commit.
