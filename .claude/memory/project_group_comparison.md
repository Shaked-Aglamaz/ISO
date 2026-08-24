---
name: Group Comparison Methods
description: Complete description of statistical and topographic methods used to compare Young, Elderly, and MCI groups on ISFS parameters
type: project
originSessionId: e53dd0be-6c8b-43fa-86a7-2b47d8ff2780
---
# Group Comparison Pipeline for ISFS Parameters

We compare three groups — **Young** (N=28), **Elderly** (N=29), **MCI** (N=24) — on three ISFS parameters: **Peak Frequency**, **Bandwidth**, and **AUC**. Two complementary approaches are used: global (violin) and spatial (topography).

## 1. Global Parameter Comparison (step6_groups_comparison.py)

**Entry point:** `run_three_group_comparison()`
**Output:** `results/group_comparison_results/three_groups/`

### Data Preparation
- Each subject's ISFS parameters are averaged across all channels (one value per subject per metric).
- Subjects with <20% ISFS detection rate are excluded.

### Statistical Pipeline (`run_three_group_tests`)

**Step 0 — Normality:** Shapiro-Wilk test per group per metric. Decides parametric vs non-parametric path.

**Step 1 — Omnibus test (`test_omnibus`):**
- All groups normal -> One-Way ANOVA (F-statistic)
- Any group non-normal -> Kruskal-Wallis (H-statistic)
- Effect size: eta-squared (0.01=small, 0.06=medium, 0.14=large)

**Step 2 — Post-hoc (`test_posthoc`, only if omnibus p<0.05):**
- After ANOVA -> Tukey's HSD
- After Kruskal-Wallis -> Dunn's test with Holm correction
- Three pairwise comparisons: Young-Elderly, Elderly-MCI, Young-MCI

### Visualization
- `plot_group_comparison`: Side-by-side violin+box+dots plots (3 groups x 3 metrics)
- Significance brackets drawn only for post-hoc significant pairs (not omnibus)
- Statistics text box per group showing N, mean, std, median

### Key Results (as of 2026-03-18)
- **Peak Frequency:** ANOVA p=0.003, eta2=0.14. Only MCI vs Young significant (p=0.002).
- **Bandwidth:** ANOVA p=0.021, eta2=0.09. Young differs from both Elderly (p=0.041) and MCI (p=0.044). Elderly vs MCI ns.
- **AUC:** Kruskal-Wallis p=0.152. No group differences.
- Bottom line: Young stands out; Elderly and MCI are similar.

### Interpretation pitfall: normalized topo vs raw violin (2026-04-18)

The default `three_group_topo_auc.png` is **per-subject normalized** (each subject's channel vector divided by its own outlier-trimmed scalp mean before group-averaging) — it shows the relative **spatial pattern**, not absolute magnitude. The default all-electrodes violin is **raw** AUC averaged over all 175 channels per subject — it shows absolute magnitude.

These two views can move in **opposite directions** and both be correct. That happened for AUC in V2: normalized topo shows a central hotspot that clearly weakens Young → Elderly → MCI, while the raw all-electrodes violin shows Young < Elderly < MCI (medians 5.76, 6.68, 7.05) and the raw ROI violin shows Young < Elderly > MCI. Mechanism: with aging/MCI, absolute AUC rises globally *and* the central concentration dissolves — two effects that partly cancel in the raw ROI average. To cleanly isolate "concentration at the hotspot" use the **normalized ROI** violin (same per-subject normalization as the topo), now wired into V2.

### New V2 outputs (2026-04-18)
- `three_group_topo_auc.png` now overlays the core `CENTRAL_PARIETAL_ROI` (green filled dots) and the extended ring (green `+`) on all three group subplots alongside the black post-hoc circles. AUC only.
- `group_comparison_violin_ROI_normalized_auc.png` + `..._extended_ROI_...png` are now produced by `run_three_group_comparison()` alongside stats files `three_group_statistics_ROI_normalized_auc.txt` / `..._extended_ROI_...txt`. Values > 1 = ROI is enhanced vs the rest of that subject's scalp.
- Normalized ROI AUC (V2, N=34/38/31): Young μ=1.131 M=1.175, Elderly μ=1.057 M=1.038, MCI μ=1.037 M=1.064. One-way ANOVA F=1.45, p=0.239, η²=0.028 — trend in the predicted direction (Young > Elderly ≥ MCI, consistent with the topo) but not significant at current N.
- Extended ROI AUC: Young μ=1.099 M=1.127, Elderly μ=1.047 M=1.010, MCI μ=1.019 M=1.034. Same ordering.

---

## 2. Topographic Comparison (step5_topo_comparison.py)

**Entry point:** `main_three_group()`
**Output:** `results/group_comparison_results/three_groups/`

### Data Preparation (Steps 1-4)
Uses per-subject normalization via shared `normalize_subject_channels()` from step4:
1. For each subject, extract ISFS parameter values at every electrode.
2. NaN (undetectable ISFS) imputed with subject's channel mean.
3. 3-sigma outlier detection; normalization mean computed excluding outliers.
4. All electrode values divided by cleaned mean -> relative spatial pattern.
5. Grand average: mean of normalized values across all subjects per group.

**Why:** Per-subject normalization removes global magnitude differences (e.g., age-related power shifts), isolating the *spatial pattern* for comparison.

### Statistical Pipeline

**Step 5 — Electrode-wise ANOVA (`electrode_wise_anova`):**
- One-way ANOVA at each of the 175 electrodes (uncorrected, descriptive only).

**Step 6 — Cluster Permutation Test (`cluster_permutation_anova`):**
- MNE's `spatio_temporal_cluster_test` with F-statistic (3 groups -> automatic F-test).
- F-threshold from F-distribution (dfn=2, dfd=N-3, alpha=0.05).
- `tail=1` (F is one-tailed), 5000 permutations.
- Spatial adjacency from Delaunay triangulation of electrode positions.
- Corrects for multiple comparisons across electrodes.

**Step 7 — Post-hoc Tukey-Kramer (`posthoc_tukey_at_clusters`):**
- Only at electrodes within significant clusters.
- `pairwise_tukeyhsd` at each electrode -> identifies which pair(s) differ.

### Visualization
- `plot_three_group_topos`: 3 normalized grand-average topomaps side by side (Young, Elderly, MCI).
  - Common color scale across groups.
  - Black hollow circles on electrodes where post-hoc found significant pairwise differences.
  - Per-group mask: union of all post-hoc pairs involving that group.
- F-statistic topography: RdBu_r colormap showing electrode-wise F-values, with all clusters (pre-permutation) visualized as enclosing circles (one circle per cluster, color-coded, solid lines, thicker for significant clusters). Legend shows cluster size and p-value.
- All plots saved even if no significant clusters (always see the spatial patterns).

### Key Results (as of 2026-03-18)
- **No significant topographic clusters** for any parameter after permutation correction.
- Uncorrected: Peak Freq 15/175, Bandwidth 8/175, AUC 20/175 electrodes with p<0.05.
- Interpretation: Group differences are in global magnitude (step4), not spatial distribution. The topographic pattern of ISFS is similar across groups.

---

## Shared Code

- `normalize_subject_channels(values)` in step4: NaN imputation + 3-sigma outlier-aware per-subject normalization. Used by both step4 (`compute_per_subject_normalized_averages` for topo plotting) and step5 (`create_evoked_array_for_subject`).
- step5 imports from step4 (not circular).

## Two-Group Comparison (step6_groups_comparison.py, legacy)

**Entry point:** `run_two_group_comparison()`
- Young vs Elderly only.
- Shapiro-Wilk → Levene → automatic test selection:
  - Both normal + equal variances → Student's t-test
  - Both normal + unequal variances → Welch's t-test
  - Either non-normal → Mann-Whitney U
- Effect size: Cohen's d (parametric) or rank-biserial correlation (non-parametric).

## Three-Group Global Comparison (step6_groups_comparison.py)

**Entry point:** `run_three_group_comparison()`
- Same data as step4 global: each subject's mean across all channels (raw, unnormalized).
- Pipeline: Shapiro-Wilk → Omnibus (ANOVA or Kruskal-Wallis) → Post-hoc (Tukey HSD or Dunn-Holm).
- Visualization: violin+box+dots with significance brackets.
- Note: Levene's test for variance homogeneity is NOT checked in the 3-group path, only normality.

---

## Known Issues & Design Decisions

**Double-normalization fix (2026-03-22):** step4's `plot_single_topography` previously divided data by mean when `normalize=True`, on top of already-normalized data from `compute_per_subject_normalized_averages`. Fixed by removing normalization from the plotting function — data must arrive ready-to-plot.

**Per-subject normalization hides global differences:** The normalization in step5 (and step4 topos) divides each subject by their channel mean. This erases global magnitude differences between groups. For AUC especially, if MCI has globally reduced ISFS power, normalization makes everyone look the same. The cluster test can only detect *spatial redistribution*, not uniform global reduction. A separate global mean comparison (step6) is needed to catch uniform differences.
