---
name: feedback_doc_figure_legibility
description: Paper figures must be portrait/large-font to stay legible when embedded in the Google Doc (page-width shrink); mechanisms used 2026-06-22
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e4373938-8e3c-4435-af4c-1c92698d7513
  modified: 2026-08-13T15:03:26.145Z
---

The manuscript figures are embedded in a portrait Google Doc, where a wide/landscape figure shrinks to ~1/3 at page width and small text (legends, table cells, axis labels) becomes unreadable. So paper figures should be built **portrait or roughly square with large fonts**, sized to fill the page width before shrinking.

**Why:** the root cause of "text too small in the Doc" is aspect ratio, not font points — a tall/narrow figure occupies more page width at full resolution. Fixed 2026-06-22 across F1–F5.

**The rule, stated exactly (2026-08-13):** on-page type size = `fontsize × (page_width / figsize_width)`. Only the *ratio* matters, so **narrowing the canvas is as effective as raising the font**, and raising the font alone on an 18 in canvas buys almost nothing. Useful check before shipping: downsample the PNG to 975 px wide (6.5 in at 150 dpi) and look at it — that is what the reader gets. Two corollaries found the hard way:
- Narrowing a canvas can make previously-fitting titles **collide** (the F4 raw-topo `(N=35, mean=6.457)` line had to be split over three lines).
- For **matplotlib tables** the trick does not work: `auto_set_column_width` sizes columns to their text, so table width scales with fontsize and the aspect is fixed by content. The only levers are shorter labels/abbreviations (`Kruskal-Wallis` → `KW`) and fewer columns. Note it resolves at *draw* time, so text written into a cell after `auto_set_column_width` still widens that column — a long section-band label was silently setting the whole first column's width.

**How to apply (mechanisms used this session):**
- **F1** (`make_f1_figure.py`): rebuilt portrait 13×16.3 — 3 rows of [hypno | group pie], full-width comparison table beneath.
- **F2** (`make_f2_figure.py`): rebuilt portrait 9.5×9.8 — A timeline / B traces full-width stacked, C+D bottom row; panel-C portrait with narrow names-only legend + small y-headroom; B row-labels wrapped 3 lines, centered.
- **F3/F5** violins (`plot_group_comparison` in `step6_groups_comparison.py`): added a `paper_style=True` flag (passed from `replot_f3_no_title.py` / `replot_f5_no_title.py`) → drops "Group" x-label, folds N into x-tick labels, removes on-plot stat boxes + bracket p-values (asterisks only), bumps fonts. Flag defaults False so other callers are unchanged.
- **F4/S1** topos (`step5_topo_comparison.py`, then `make_topo_composites.py`): one shared colorbar per panel on the right (was one per map), removed the "Significant pairs" footnote, bigger fonts (titles 20 / cbar label 16), tightened inter-topo `wspace` to 0.02 so maps grow.

**Second pass 2026-08-13 ([[project_v11_figure_overhaul]]), applying the ratio rule:**
- **F3** went from a 19.5 in wide row of 3 metrics to a **3×1 portrait column** (7.5×16.2) under `paper_style`; fonts 22/20/19/28. F5 was already narrow, so the two now render at matching sizes.
- **F4/S1** topo canvas 18 in → **12 in** (`TOPO_FIGSIZE`), fonts 26/22/20, group-coloured titles, and the bare `AU` / `Normalized` colorbar legends replaced with spelled-out ones ("ISFS strength (AUC, a.u.)"). **Regenerate with `code/replot_topos_paper.py`, NOT `main_three_group()`** — the latter re-runs the unseeded 5000-permutation cluster test; the replotter reuses the V10 post-hoc electrodes from `three_group_topo_statistics.txt`.
- **S2** (`moca_correlation.py` `plot_grid(paper_style=True)` via `code/replot_s2_moca.py`): 13×10 → 9.5×9.6, stats moved off the cramped title onto their own line, four duplicate per-panel legends replaced by one shared legend (at the bigger size `loc="best"` was covering data points).
- **T2** (`make_table2_sleep.py`): shorter labels, `row_scale` 2.4 → 3.3 for line spacing, and **no on-figure title or footnote** — both duplicate the caption. Same reason F3/F5/S2 carry no title.

Caption text wording is governed by [[feedback_caption_conventions]]; figure set + sources in [[project_paper_figure_set]] and `thesis/figure_manifest.md` (per-figure layout notes kept current). Doc location: [[reference_manuscript_gdoc]].
