---
name: Normalize over the full scope, not the subset you're comparing
description: When computing an ROI-vs-rest-of-scalp style metric, per-subject normalization must use ALL channels, then restrict to ROI. Normalizing inside the ROI makes the result ~1 by construction.
type: feedback
originSessionId: e53dd0be-6c8b-43fa-86a7-2b47d8ff2780
---
When computing a per-subject normalized metric intended to compare a subset (e.g. an ROI) against the larger whole (e.g. whole scalp), **always normalize using the full set before subsetting**. Filtering to the ROI *first* and then dividing by the mean makes every subject's ROI value identically ~1 by construction — you have destroyed the signal you wanted to measure.

**Why:** Hit this exact bug in `step6_groups_comparison.py::load_and_process_roi_data_normalized` on 2026-04-18. The function filtered to the 20-channel ROI, then called `normalize_subject_channels(roi_values)`, which divided each subject's ROI values by their own ROI mean. Every subject's ROI mean came out ≈1.000 with near-zero std across groups (Young/Elderly/MCI all 1.002-1.008 with σ≈0.02), hiding the real Young > Elderly ≥ MCI effect. Fix: normalize the full 175-channel vector first (same `normalize_subject_channels` the topo uses), then average the normalized values over ROI channels only. Correct values then ranged 1.02-1.13 across groups with a clear trend.

**How to apply:** Any time per-subject normalization is combined with a subset aggregation (ROI means, region comparisons, peak-channel stats, per-cluster averages, etc.), verify the normalization denominator covers the full reference scope — not the subset. A quick sanity check: if the means across subjects are all ~1.0 with tiny std, normalization was scoped too narrowly. The correct pattern mirrors `step5_topo_comparison.py::create_evoked_array_for_subject` and `step4_distribution_analysis.py::compute_per_subject_normalized_averages`: normalize once over the full channel set, then index/slice.
