---
name: project_v11_figure_overhaul
description: "V11 figures DONE + IN THE DOC 2026-08-17: figure numbering swapped (methods flow = Fig 1), Table 2 is a native Docs table, all figures relabelled aMCI, new Fig S1 example spectra; stats still V10"
metadata: 
  node_type: memory
  type: project
  originSessionId: f88a89e9-6654-4ef5-ab43-812b23df669d
  modified: 2026-08-16T22:48:53.603Z
---

**Figures pass CLOSED 2026-08-17.** The 2026-08-13 V11 restyle plus the session that put it in the
Google Doc. Per-figure provenance is current in `thesis/figure_manifest.md` (read the box at the
top first); the before/after record is `thesis/reviews/figures_edits_before_after.md`. Yuval's
C389, C412, C432, C487, C502, C591, C592 and emails #5/#6 are all **closed**.

**No statistic was ever recomputed.** V10 remains the numbers of record
([[project_v10_regeneration]]); V11 differs in graphics only.

## THE NUMBERING CHANGED — this invalidates older notes

Moving the sleep overview into Results 4.1 (C412) put it behind the methods-flow figure, which
Methods 3.4 cites first. So:

| now | was | where |
|---|---|---|
| **Figure 1** ISFS concept / feature extraction / ROI | Figure 2 | Methods 3.5 · `thesis/figures/methods_flow_roi_v3.png` |
| **Figure 2** sleep overview | Figure 1 | Results 4.1 · `thesis/figures/hypno_sleep_stages_V11.png` |
| **Table 2** sleep architecture / continuity / N2 bouts | — | Results 4.1 · **native Docs table** |
| Figure 3 whole-scalp violins | same | `three_groups_V11/group_comparison_violin.png` |
| Figure 4 AUC topo composite | same | `thesis/figures/f4_auc_composite_V11.png` |
| Figure 5 ROI violin | same | `three_groups_V11/group_comparison_violin_extended_ROI_normalized_auc.png` |
| **Figure S1** example spectra | S3 (new) | `thesis/figures/s3_example_spectra_V11.png` — keeps the `s3_` filename |
| **Figure S2** peak-freq/BW topos | S1 | `thesis/figures/s1_topo_composite_V11.png` |
| **Figure S3** MoCA grid | S2 | `results/moca_correlation_V4/moca_correlations_grid.png` |

The supplementary set shifted because the example-spectra figure is cited at the end of Results 4.1,
ahead of the other two. Renumbering a cycle like this needs temporary tokens — no ordering of
find-and-replace avoids a collision.

## Four things worth not re-discovering

- **Every V11 asset drew "MCI" while every caption said "aMCI."** `grep -rn aMCI code/` returned
  nothing. Fixed by `GROUP_DISPLAY` + `group_label()` in `code/utils/config.py`, applied only at the
  six sites that *draw* text. **Never rename the palette dict keys**: `'MCI'` is also a colour-lookup
  key, a `results/sigma_fix_MCI` directory name, and a literal that `replot_topos_paper.py` regex-parses
  out of `three_groups_V10/three_group_topo_statistics.txt`. In `moca_correlation.py` the label was
  *also* the palette key, so relabelling the dict silently turned the aMCI series grey.
- **Table 2 went into the Doc as a native 18×10 Docs table, not the PNG.** At page width the 20.5 in
  PNG renders ~6 pt type, which is the same defect C389 raised about panel C. Table 1 was already
  native, so they match. The PNG remains the manifest preview and source of record.
- **Two traps in the per-channel ISFS outputs** (hit while building the example-spectra figure, full
  detail in the manifest's S1 block): the CSV column named `mean_power` is `mean_power_no_baseline`
  and is **not** baseline-corrected, while the Gaussian was fitted to the corrected spectrum; and the
  summary file's `Peak Frequency (μ)` is the grid-snapped `actual_pf`, **not** the fitted mu that
  centres the curve. `code/make_s3_figure.py` recovers the curve by repeating the pipeline's
  `curve_fit` and asserting it reproduces the recorded amplitude and sigma.
- **S2's exploratory colours are still inverted** (Elderly blue, MCI red in `moca_correlation.py`
  `GROUP_COLORS`). Only `paper_style=True` is fixed. Do not reuse the exploratory renders as reference.

## Page geometry, for any future paste

Text area **6.5 × 9.0 in** (612 × 792 pt page, 72 pt margins). Everything sits at 6.5 in wide except
Figure 5, which Shaked reviewed and deliberately kept at 3.1 × 3.6 in. Figure 3 had to be reshaped
from 7.5 × 16.2 to 9.5 × 11.4 in (`step6_groups_comparison.py:1120`) — at 2.13:1 it could only be
sized to the page height, which left a third of the column empty and pushed its caption to the next
page. Image geometry cannot be set through the Docs MCP; resizing is always a manual step.

## Known gaps

- **Hypnospectrogram internal fonts** (axis labels/ticks inside the three bitmaps in Figure 2A) are
  still small; baked into `results/hypnospectrograms/*.png` by `hypnospectrogram.py`
  ([[project_hypnospectrogram]]).
- **η² = −0.001 for sleep onset latency** is shown as-is in Table 2. Negative η² is a Kruskal-Wallis
  formula artefact (H < k−1, see [[project_c429_c390_results]]); flagged, no decision taken.
- **Table 1's PNG still renders "MCI"** in its header — **left as-is by Shaked's decision 2026-08-17.**
  Manifest preview only; the Doc's Table 1 is a native table whose header already reads aMCI. Do not
  "fix" it.
- `results_edits_before_after.md` still carries off-by-one caption line pointers (`:101`/`:112`). Left
  deliberately: it is a historical record of that pass and nothing points a session at it any more.

Related: [[project_yuval_review]], [[project_paper_figure_set]], [[project_c429_c390_results]],
[[reference_manuscript_gdoc]], [[feedback_doc_figure_legibility]], [[feedback_caption_conventions]],
[[feedback_versioned_output_dirs]]
