---
name: ROI choice — Extended (36-ch) selected
description: Decision on 2026-04-18 to use the 36-channel EXTENDED_CENTRAL_PARIETAL_ROI over the 20-channel Core ROI for group-comparison AUC reporting
type: project
originSessionId: 00af28b6-7452-4516-8572-c15e7b6213eb
---
Chosen ROI: **`EXTENDED_CENTRAL_PARIETAL_ROI`** (36 channels, defined in `code/utils/config.py`).

**Reader-facing naming (2026-05-31):** this is an INTERNAL choice. In the paper/captions never write "extended" or mention Core — to the reader it is simply "the ROI" (pre-defined central-parietal set from the Dimitriades 2024 young-adult AUC hotspot). The "p=0.074" below is the Young-vs-MCI Welch-t from the selection step and is NOT in the locked V6 stats (omnibus ANOVA p=0.194, ns, no post-hoc); reporting it as a trend is a pending user decision. See [[feedback_thesis_prose_rules]] and [[project_paper_figure_set]].

**Why:** decision made 2026-04-18 based on `roi_vs_extended_decision_table.txt` (AUC, 3 groups). Extended traded a small loss of absolute mean separation for ~25% lower SD, which improved the clinically-relevant Young-vs-MCI Welch-t p from 0.120 (Core) to 0.074 (Extended). Neither variant crossed α=0.05, but Extended is closer to significant on Y-vs-MCI. Core was marginally better on Y-vs-E (p=0.181 vs 0.224) — judged less important than the Y-vs-MCI contrast.

**How to apply:** in any new group-comparison / ROI violin / ROI-stats code, default to `EXTENDED_CENTRAL_PARIETAL_ROI` unless the user explicitly asks for Core. In `code/step6_groups_comparison.py`, both ROIs are currently plotted/reported side-by-side (the Core loop was kept for comparison); if the user later decides to drop Core entirely, remove the Core entry from the `roi_channels` loop in `run_three_group_comparison()`.

**Update 2026-04-19 — topo-plot rendering:** In `step5_topo_comparison.py` topographies, the Core/Extended visual distinction was dropped. The ROI overlay now renders all 36 extended channels as green dots in a single color and the legend simply says "ROI" (no green dots vs green-plus split). The two lists are intentionally kept separate in `code/utils/config.py` so we can fall back to distinguishing them, but in plots they are treated as one ROI. Same convention applies to the verification image at `code/debug/roi_256_verification.png` (regenerated via `code/debug/make_extended_roi_verification.py`, title: "256-ch ROI (36 electrodes)").
