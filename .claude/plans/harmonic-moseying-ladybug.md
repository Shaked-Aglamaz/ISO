# C429 ANCOVA + C390 extended sleep statistics

## Context

Yuval's review (`thesis/reviews/yuval_review_triage.md`) left two items that need new numbers
rather than new prose:

- **C429** — he asked how we know the group differences in ISFS parameters are not driven by
  differing amounts of N2 sleep, and whether N2 amount can be entered as a factor. The
  descriptive answer is already established (N2 share differs, KW p = 0.0009; proportion of each
  subject's N2 that was analyzed does not, ANOVA p = 0.9808; total analyzed duration does not,
  KW p = 0.0811) but the "include it as a factor" half has never been run. An ANCOVA on the three
  whole-scalp ISFS parameters with analyzed N2 duration as covariate answers it empirically.
- **C390** — he wants the sleep-architecture statistics extended with sleep efficiency, WASO and
  REM latency. Sleep efficiency already exists in the subjects sheet; WASO, sleep-onset latency
  and REM latency have never been extracted.

Outcome: two new versioned output dirs with numbers the reply and the figures session can use.
This session produces **stats only** — no prose, no figures, no manuscript edits.

### Scope guard

Only the two analyses above. No sensitivity runs, no recording-site check, no re-derivation of
facts already in V10/V3. Anything else noticed gets reported to the user, not implemented.
Owned here: the two new scripts and the two new output dirs. Off limits:
`thesis/chapters/*.md`, `thesis/figures/`, `figure_manifest.md`, `library.bib`, the Google Doc,
and any existing `three_groups_V*` / `demographics_V*` dir. No commits.

## Verified during planning

- Re-running V10's whole-scalp pipeline reproduces V10 **exactly**: peak frequency
  0.0199 / 0.0226 / 0.0232, bandwidth 0.0236 / 0.0281 / 0.0276, AUC 6.4574 / 7.3137 / 7.4780
  for Young / Elderly / MCI at N = 35 / 39 / 30. So the ANCOVA is bolted onto the same
  per-subject values the manuscript already reports.
- The covariate merges cleanly: 104/104 subjects matched between the ISFS results dirs and
  `results/demographics_V3/n2_bouts_per_subject.csv`, group labels consistent, zero missing.
  Group means 80.7 / 106.0 / 90.8 min match the triage table.
- All 104 subjects have a `*_cleaned_annotations.txt`, second-resolution, and once `UNKNOWN` is
  included the stage annotations tile the recording: residual gaps only ON68 (181 s) and
  AT36 (10 s); overlaps only EL3005 (90 s), RS5 (29 s), AT36 (10 s).
- Exactly 23 subjects carry `UNKNOWN` blocks, matching `notes/tst_waso_unknown_handling.md`
  subject-for-subject, AT36 the worst at 100.5 min.
- All 104 subjects have REM sleep, so no metric row loses subjects to a missing stage.
- yasa 0.6.5 `sleep_statistics` is policy-compliant for WASO (`hypno_s == 0` inside SPT, so `-1`
  is not counted), SOL and `Lat_REM`. Its **SE is not** — it uses `TST / len(hypno)`, which
  includes unscorable seconds in the denominator. SE therefore gets recomputed (see below).
- `statsmodels` 0.14.6 supports the whole ANCOVA: `smf.ols` + `anova_lm(typ=2)` +
  `res.t_test_pairwise(..., method='holm')`.

## Part 1 — C429 ANCOVA

**New script:** `code/ancova_n2_duration.py`
**Output dir:** `results/group_comparison_results/three_groups_V11/` (new; V10 untouched and
remains the source of truth for the main statistics and all figures)

Reuse, do not reimplement:

- `code/utils/utils.py::get_all_subjects`, `code/utils/config.py::BASE_DIR`
- `code/step4_distribution_analysis.py::filter_subjects_by_detection_rate` and
  `::load_and_process_all_channels_data` — the same two calls
  `step6_groups_comparison.py::run_three_group_comparison` makes, so the per-subject
  whole-scalp values are identical to V10 by construction (per-channel `nanmean` via
  `groupby.mean()`, which is the existing NaN policy for the raw whole-scalp violins).

Covariate: `total_dur_min` from `results/demographics_V3/n2_bouts_per_subject.csv`, merged on
subject id. Assert 104/104 matched and fail loudly otherwise rather than silently dropping rows.
Mean-centre the covariate so the intercept and adjusted means sit at the grand mean.

Per metric (`peak_frequency`, `bandwidth`, `auc`), report in this order:

1. **Unadjusted baseline** — the V10 one-way result restated (Shapiro per group → ANOVA or KW),
   so the ANCOVA can be read as a delta rather than a standalone number.
2. **ANCOVA** — `ols('y ~ C(group) + n2_min_c')`, Type-II `anova_lm`: group F/p, covariate F/p,
   partial η² for both, residual df.
3. **Adjusted (marginal) group means** at the grand-mean covariate, next to the raw means.
4. **Post-hoc** — `t_test_pairwise('C(group)', method='holm')`: Holm-corrected pairwise
   contrasts of adjusted means. Note in the report header that this is Holm rather than the
   Tukey HSD used on the unadjusted means, because Tukey on adjusted means needs `emmeans`.
5. **Assumption checks** — homogeneity of regression slopes (`y ~ C(group)*n2_min_c`, report the
   interaction F/p) and Shapiro on the model residuals.

Two things to carry into the report text, both already visible in the planning data:

- The covariate correlates with bandwidth at r = 0.386 across all 104 subjects (peak frequency
  0.064, AUC 0.090). Bandwidth is the borderline parameter (V10 p = 0.0611), so it is the one
  most likely to move. Report whatever comes out; do not pre-judge the direction.
- AUC failed Shapiro in MCI (p = 0.0412) and V10 therefore used Kruskal-Wallis. ANCOVA is
  parametric, so the AUC ANCOVA is reported **with an explicit caveat** naming the unadjusted KW
  as the primary test for that parameter. No rank-based alternative is run — that would be an
  extra analysis, and it is outside scope. Flag to the user if the residual Shapiro also fails.

The normalization-order rule does not bite here: these are raw whole-scalp values with no ROI
restriction, so there is no normalize-then-restrict step to get wrong.

Files written:

- `three_group_ancova_statistics.txt` — the report above, mirroring the banner/section style of
  `three_groups_V10/three_group_statistics.txt`
- `three_group_ancova_per_subject.csv` — subject, group, the three metrics, `total_dur_min`

## Part 2 — C390 extended sleep statistics

**New script:** `code/sleep_statistics_extended.py`
**Output dir:** `results/demographics_V4/` (new; V3 untouched and remains the source of truth
for the pies, the demographics table and the N2 bout table)

Hypnogram source: each subject's `*_cleaned_annotations.txt`, reached with the same lookup
`code/n2_bouts_table.py::annotation_path_for` uses (sheet `group_dir`, `_cleaned_annotations.txt`
then `_annotations.txt`), parsed with `code/utils/subject_summary.py::parse_annotations`.
Chosen over `ISO_data/scoring/` because that tree has inconsistent naming, letter-vs-int codes
and mixed 1 Hz / 30 s resolutions across the three groups, while the annotations are the
normalized representation of exactly the same information — `notes/tst_waso_unknown_handling.md`
confirms the 1:1 correspondence on AT36.

Build a **1 Hz** integer hypnogram per subject (annotation durations are second-resolution, not
30 s multiples, so a 30 s grid would round):

- `Wake`/`WAKE` → 0, `N1`/`NREM1` → 1, `N2`/`NREM2` → 2, `N3`/`NREM3` → 3, `REM` → 4,
  `UNKNOWN` → -1. Both stage-label dialects must be handled; assert every non-`BAD*`
  description is in the map so a new dialect fails loudly.
- Never read `BAD_ACQ_SKIP` or `BAD*` for these metrics — per the note, that label is overloaded
  in `*_cleaned_annotations.txt` with hand-drawn artifact marks (AT36 has 3 real gaps + 58
  manual; LS56 has 94) and is sample-accurate rather than epoch-aligned.
- Overlaps resolved last-write-wins on onset-sorted annotations; residual uncovered seconds
  folded into the unscorable category. Both counted and reported as QC columns.

Metrics, following `notes/tst_waso_unknown_handling.md` — `-1` is a third category (not sleep,
not wake, not recorded), excluded from numerator **and** denominator, never folded into wake:

| metric | source |
|---|---|
| WASO (min) | `yasa.sleep_statistics` — its WASO already counts only `== 0` inside SPT, so `-1` is excluded |
| Sleep-onset latency (min) | yasa `SOL` |
| REM latency (min) | yasa `Lat_REM - SOL` (AASM, from sleep onset) as the compared row; yasa's raw `Lat_REM` (from recording start) kept as a second column |
| Sleep efficiency (%) | sheet `sleep_efficiency_pct`, authoritative, not recomputed |

Also computed for validation and QC, not as new claims: TST, TIB, SPT, unscorable minutes, and
a policy SE (`TST / (TIB − unscorable)`) to sit beside the sheet's value. Verify against the
sheet's `tst_sec` / `wake_sec` / `recording_sec` and report any subject that disagrees
materially — AT36 is the known extreme (note gives TST 343.5 min, WASO 74.0, policy SE 80.4%
vs a raw-TIB SE of 65.1%), so it doubles as the regression test.

Group comparison: the same normality-gated scheme as
`demographics_V3/sleep_stage_stats.txt` — Shapiro-Wilk per group → one-way ANOVA if all three
normal else Kruskal-Wallis → Tukey HSD or Dunn (Holm) only when omnibus p < 0.05, plus η².
Reuse `code/n2_bouts_table.py::run_metric_stats`, which is already the generic
(dataframe, metric) form of that scheme, rather than the per-stage variant in
`sleep_stage_pies.py`. Compared rows: **WASO, SOL, REM latency, sleep efficiency** (the user's
choice). TST/TIB stay as columns only.

Files written:

- `sleep_statistics_stats.txt` — same layout as `demographics_V3/sleep_stage_stats.txt`
  (header stating the per-subject definition and the pipeline, N per group, then a block per
  metric with group means, Shapiro, omnibus, post-hoc)
- `sleep_statistics_per_subject.csv` — one row per subject: id, group, WASO, SOL, both REM
  latencies, sheet SE, policy SE, TST, TIB, SPT, unscorable min, QC overlap/gap seconds
- `sleep_statistics_table.csv` — the tidy one-row-per-metric table for the figures session:
  metric, per-group n / mean / SD, omnibus test + statistic + p + η², and the three pairwise p's
  (same column shape as `demographics_V3/n2_bouts_table.csv` so it drops straight into a table)

## Verification

Both scripts run from the repo root with the venv active and `PYTHONIOENCODING=utf-8`:

```bash
source eeg_clean/Scripts/activate
PYTHONIOENCODING=utf-8 python code/ancova_n2_duration.py
PYTHONIOENCODING=utf-8 python code/sleep_statistics_extended.py
```

Checks before reporting numbers:

1. **ANCOVA anchors to V10.** The unadjusted block in `three_group_ancova_statistics.txt` must
   reproduce V10 line for line: N = 35/39/30, peak frequency F = 6.3153 / p = 0.0026,
   bandwidth F = 2.8739 / p = 0.0611, AUC KW H = 1.9572 / p = 0.3758. Any drift means the
   subject set or the NaN handling diverged — stop and diagnose rather than reporting.
2. **Covariate anchors to V3.** Group means of `total_dur_min` must print 80.7 / 106.0 / 90.8 min
   and the merge must report 104/104.
3. **AT36 regression test.** `sleep_statistics_per_subject.csv` must give AT36 TST ≈ 343.5 min,
   WASO ≈ 74.0 min, SOL ≈ 84.0 min, unscorable ≈ 100.5 min, policy SE ≈ 80.4%. These are the
   hand-computed values in `notes/tst_waso_unknown_handling.md`. A WASO near 98.5 means `-1`
   leaked into wake.
4. **Cohort integrity.** Both scripts print N = 35/39/30, and every compared metric reports
   n = 35/39/30 (all 104 subjects have REM, so no row should be short).
5. **Cross-check against the sheet.** Per-subject TST from the hypnogram vs sheet `tst_sec`, and
   policy SE vs sheet `sleep_efficiency_pct`; print the largest discrepancies rather than
   asserting equality, since the sheet's SE uses a raw-recording denominator.
6. **Nothing outside the two new dirs is written.** `git status` should show only the two new
   scripts and the two new output dirs.

Then report the numbers in chat: whether group effects survive the covariate for each of the
three parameters (with the bandwidth result called out, since that is the parameter the
covariate actually tracks), and the four sleep-statistic rows with their omnibus and post-hoc
p-values. No prose rewriting.
