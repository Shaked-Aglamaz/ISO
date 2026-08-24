---
name: Raw group topography aggregation (V5 fix)
description: Single source of truth for raw group topo + displayed mean across step4/step5/step6 — how it's computed, why, and where to find it
type: project
originSessionId: 31c46e93-6be4-4f9f-b4dd-31cb8bb42140
---
As of 2026-04-29 (V5 iteration), step4 per-group raw topo, step5 three-group raw topo, and step6 violin all display the **same group-level scalar** for the same metric — the mean of per-subject means, which is step6's canonical group statistic (e.g. AUC: Young 6.5604, Elderly 7.2715, MCI 7.4476).

**Source of truth:** `code/utils/topo_aggregation.py`
- `build_raw_group_topo(subjects_data, metric)` returns `(group_topo, displayed_mean, available_channels, info)`.
- `neighbor_impute(values, adjacency)` fills per-subject NaN with the mean of spatial-neighbor channels that are non-NaN for that subject. Uses MNE's Delaunay-triangulation adjacency (~6.7 neighbors/channel on EGI_256). Iterates so cells with all-NaN neighbors fill on a later pass; falls back to subject `nanmean` if still NaN after 5 iterations.
- `displayed_mean` is computed independently from the topo image: it's `mean over subjects of np.nanmean(raw_full_subject_df)` (full df includes VREF / no-position channels — matches step6 violin byte-for-byte).
- The topo image is `mean across subjects of (per-subject neighbor-imputed vector restricted to 175 EGI-positioned channels)`.

**Callers:**
- `step4_distribution_analysis.py:plot_topographies` → multi-subject + `normalize=False` branch.
- `step5_topo_comparison.py:main_three_group` → raw AUC pass (uses `plot_three_group_raw_topos` helper).

**Why this exists:** Pre-V5, step4 used `groupby('channel').mean()` (per-channel-then-mean) + 3σ outlier trim, which gave a ~0.6 AU upward bias on Young AUC (7.170 vs the true 6.5604) when ~25% of (subject × channel) cells were NaN due to failed Gaussian fits. step5 used per-subject EvokedArrays with NaN imputed to subject mean, which was equivalent to step6's mean-of-subject-means (matched 6.560 ≈ 6.5604). The fix consolidated both onto the shared utility.

**Legacy paths kept on the old per-channel + 3σ-trim logic:**
- Single-subject calls to `plot_single_topography` (still uses `np.nanmean(cleaned_values)` after a 3σ trim of the topo).
- The normalize=True path in step4 and step5 — but normalization itself (`normalize_subject_channels`) forces every subject's vector to mean ≈ 1.0, so averaging-order bias collapses anyway. Normalized displayed scalars are all ≈ 1.0 across pipelines; ROI-restricted normalized violins (e.g. `group_comparison_violin_extended_ROI_normalized_auc.png`) are where normalized values become informative (Y 1.099, E 1.047, M 1.019 in V5).

**Stats pipeline unchanged.** `prepare_three_group_data(..., normalize=True)` still feeds the cluster-permutation ANOVA in step5 with subject-mean-imputed EvokedArrays — orthogonal to the display fix.
