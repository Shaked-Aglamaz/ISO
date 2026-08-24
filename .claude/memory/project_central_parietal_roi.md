---
name: Central-parietal ROI for AUC analysis
description: Two predefined ROIs (20-ch and 36-ch) in the 256-ch EGI system, reverse-engineered from a 128-ch YA_AUC hotspot. Used for ROI-restricted group comparisons.
type: project
originSessionId: b5e4b0d5-37ca-457e-af8c-10aeb0e211d6
---
Two ROI constants live in `code/utils/config.py`:
- `CENTRAL_PARIETAL_ROI` — 20 electrodes (core hotspot)
- `EXTENDED_CENTRAL_PARIETAL_ROI` — 36 electrodes (core + one surrounding ring)

**Why:** The 128-ch `YA_AUC.png` reference image shows a central-parietal AUC hotspot in young controls. We had no access to the original data, so the ROI was reverse-engineered by defining an ellipsoidal region around a Cz/Pz-weighted midpoint in MNE head coordinates and selecting electrodes inside it in both the 128-ch (`GSN-HydroCel-129`) and 256-ch (`EGI_256`) standard montages. Channel names are NOT shared across systems — mapping must go through 3D coordinates. The derivation script is `code/debug/roi_128_to_256_mapping.py` (tunable `ROI_CENTER_WEIGHT`, `ROI_RADIUS_X/Y/Z`, and `ring_scale`).

**How to apply:** When doing group comparisons on ISFS metrics (especially AUC), consider offering ROI-restricted versions alongside whole-head averages. Support both variants: `CENTRAL_PARIETAL_ROI` (tighter, stricter) and `EXTENDED_CENTRAL_PARIETAL_ROI` (includes one ring, more robust to small anatomical shifts). Current integrations:
- `code/step6_groups_comparison.py`: `plot_group_comparison(..., roi_only=True, roi_channels=..., roi_label=..., metrics_filter=...)` and ROI-aware loaders `load_and_process_roi_data[_normalized]`.
- `code/step5_topo_comparison.py`: `plot_three_group_topos_roi` overlays a dashed covariance-fitted ellipse around the core ROI on the AUC topo.
- Outputs live under `results/group_comparison_results/three_groups_V2/` (e.g., `three_group_topo_auc_ROI.png`, `group_comparison_violin_ROI_auc.png`, `group_comparison_violin_extended_ROI_auc.png`).

Verify ROI existence by reading `config.py` before recommending — the exact electrode list may evolve if the derivation script is re-tuned.
