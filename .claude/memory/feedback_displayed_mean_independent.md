---
name: Don't derive displayed group mean from imputed topo image
description: When showing "Mean = X" alongside a group topo, compute X from per-subject means independently — never from the topo image's pixels
type: feedback
originSessionId: 31c46e93-6be4-4f9f-b4dd-31cb8bb42140
---
When a group-level topographic plot displays a `Mean = X` scalar, X must be the **mean of per-subject means** (the canonical group statistic that matches the violin / stats tests). Do NOT compute it as `np.nanmean(group_topo_array)`.

**Why:** The topo image array is built via per-subject NaN imputation (subject-mean fill, neighbor fill, selection-biased per-channel means, etc.). Its cross-channel spatial average is sensitive to (a) the imputation choice and (b) per-channel NaN distribution. The mean-of-subject-means is invariant to those details. Pre-V5, this conflation produced a ~9% gap between step4's per-group topo title (7.170) and step6's violin μ (6.5604) on the same Young AUC data.

**How to apply:**
- In raw multi-subject plots: pass `displayed_mean` into the plotting function as an explicit scalar that was computed *outside* the topo construction. The topo array is for the image; the scalar is for the title; they don't share computation.
- See `step4_distribution_analysis.py:plot_single_topography(..., displayed_mean=)` and `step5_topo_comparison.py:plot_three_group_raw_topos` for the V5 implementation pattern.
- The shared utility `code/utils/topo_aggregation.py:build_raw_group_topo` returns both `group_topo` and `displayed_mean` as separate values precisely so callers can't accidentally derive one from the other.
- For 3σ outlier trim: in the new path, don't trim the topo before computing the displayed scalar. The displayed scalar is already trim-free (it's an aggregate of subject-level means). Trimming the topo is only for visual color-scale stability.
