# Figures session — plan

## Context

Yuval's review left the figures untouched while the Methods, Results, Introduction and Discussion passes were applied. Five of his items are figure items — C412 (move the sleep-overview figure into Results), C389 (the sleep table is unreadable as a pasted panel), C487 / C502 / C591 / C592 and email #6 (fonts and one shared palette across all figures). The V11 assets that answer the graphics complaints were built on 2026-08-13 but never went into the Doc, so the Doc still carries V10 images, a stale Figure 1 caption describing a panel that no longer exists, and two dangling "(Table 2)" references.

Two things surfaced during exploration and were decided by the user:

- **Every V11 asset still draws "MCI" on-figure** (x-ticks, topo panel titles, pie and legend labels, table headers). The aMCI relabel only ever reached the captions; `grep aMCI code/` returns zero hits. Decision: **regenerate in place, overwriting the V11 files** — no V12 folder.
- **Moving the sleep overview into Results puts it after the methods-flow figure**, which is first cited in Methods 3.4. Decision: **renumber** — methods flow becomes Figure 1, sleep overview becomes Figure 2.

Also decided: **Table 2 goes in as a native Docs table** (matching Table 1, which is already native) rather than a 20.5-inch PNG that would shrink to ~6 pt type — the same unreadability C389 objected to. And **C432 (supplementary example spectra) is deferred**; the user is choosing the example subjects separately.

Intended outcome: one paste pass by the user, then one text pass by me, after which every display item is current, correctly numbered, correctly captioned, and in the section Yuval asked for.

Scope rule honoured: everything below traces to C412, C389, C487, C502, C591, C592 or email #6, or is a direct consequence of them. Items found along the way that Yuval did not ask for are listed in §6 as **flagged, not applied**.

---

## 1. Final display-item order

| # | Item | Section | Asset | Change |
|---|---|---|---|---|
| 1 | **Table 1** | Methods 3.1 | native Docs table | none |
| 2 | **Figure 1** — ISFS concept, feature extraction, ROI *(was Figure 2)* | Methods 3.5 | `thesis/figures/methods_flow_roi_v3.png` | **image stays put**, caption renumbered only |
| 3 | **Figure 2** — sleep overview *(was Figure 1)* | **Results 4.1** | `thesis/figures/hypno_sleep_stages_V11.png` | **move out of Methods 3.2**, swap V10→V11, new 2-panel caption, renumber |
| 4 | **Table 2** — sleep architecture / continuity / N2 bouts | Results 4.1 | built by me from `results/demographics_V4/table2_sleep_architecture.csv` | **new** |
| 5 | **Figure 3** — whole-scalp violins | Results 4.2 | `results/group_comparison_results/three_groups_V11/group_comparison_violin.png` | swap V10→V11 |
| 6 | **Figure 4** — AUC topographies + cluster | Results 4.3 | `thesis/figures/f4_auc_composite_V11.png` | swap V10→V11 |
| 7 | **Figure 5** — ROI violin | Results 4.4 | `.../three_groups_V11/group_comparison_violin_extended_ROI_normalized_auc.png` | swap V10→V11 **and make inline** (currently a floating wrapped image, 226×259 pt) |
| 8 | **Figure S1** — peak-freq + bandwidth topos | Supplementary | `thesis/figures/s1_topo_composite_V11.png` | swap V10→V11 |
| 9 | **Figure S2** — MoCA grid | Supplementary | `results/moca_correlation_V4/moca_correlations_grid.png` | swap V10→V11, caption updated to match the restyled panels |
| — | *(Figure S3 — example spectra, C432)* | Supplementary | — | **deferred**, user is picking subjects |

Anchors in the Doc (verified by reading the document JSON):

- **Figure 2** goes immediately after the paragraph starting *"An overview of the recorded sleep across the three groups…"* (Results 4.1).
- **Table 2** goes immediately after the paragraph starting *"Sleep continuity showed the same pattern (Table 2)…"* (Results 4.1). **Leave this one to me** — I insert the native table, you paste nothing there.

Page geometry is 612×792 pt with 72 pt margins, so the text area is **468 × 648 pt (6.5 × 9 in)**. Paste every figure at **full 6.5 in width**, except **Figure 3**, which is a 7.6 × 16.1 in portrait stack — size that one **to height instead** (about 4.2 in wide), or it overruns the page. Exact per-figure heights go in the hand-off file after regeneration.

---

## 2. Regenerate the V11 assets with aMCI labels

The group name doubles as a dict key for colour lookup (`GROUP_COLORS`, `GROUP_COLORS_DARK` in `code/utils/config.py`) and is parsed out of `three_groups_V10/three_group_topo_statistics.txt` by a regex in `code/replot_topos_paper.py`. So **do not rename the keys** — add a display-only map applied at draw time.

- Add `GROUP_DISPLAY = {"Young": "Young", "Elderly": "Elderly", "MCI": "aMCI"}` to `code/utils/config.py`, next to the existing palette dicts.
- Apply it at the label sites only: `code/step6_groups_comparison.py:1212` (x-ticks, feeds F3 and F5), `code/step5_topo_comparison.py:1242` and `:1341` (topo panel titles, feeds F4 and S1), `code/sleep_stage_pies.py:39` `GROUP_TITLES` and the `HYPNOS` banner in `code/make_f1_figure.py` (feeds Figure 2), `code/moca_correlation.py:54` `PAPER_GROUP_LABELS` (feeds S2).
- Re-run, overwriting in place: `make_f1_figure.py`, `replot_f3_no_title.py`, `replot_f5_no_title.py`, `replot_topos_paper.py` then `make_topo_composites.py` (F4 + S1), `replot_s2_moca.py`. Optionally `make_table2_sleep.py` so the manifest preview agrees with the native table.
- No statistic is recomputed — every one of these reads the V10 stats files and re-renders.
- Verify `methods_flow_roi_v3.png` (new Figure 1) carries no group labels; it is a single-subject figure, so it should need nothing.
- Run with `PYTHONIOENCODING=utf-8` and the `eeg_clean` venv.

Verification: re-read each regenerated PNG with the Read tool and confirm the group labels read "aMCI", and that dimensions and layout are otherwise unchanged from the pre-run values recorded in the hand-off file.

---

## 3. Renumber (Doc + repo)

Five in-text mentions plus two caption titles. Anchor every replacement on surrounding words, never on the bare string, and do the Methods direction first.

**Figure 2 → Figure 1** (Doc, Methods): `"shown in Figure 2A–C"` (3.4), `"as illustrated in Figure 2C"` (3.4), `"electrodes (Figure 2D)"` (3.5), `"spindles illustrated in Figure 2A"` (3.7), and the caption title `"Figure 2. Characterization of infra-slow fluctuations of sigma power."`

**Figure 1 → Figure 2** (Doc, Results 4.1): `"is shown in Figure 1; group statistics"`, and the sleep-overview caption title (replaced wholesale, see §4).

Mirror into `thesis/chapters/03_methods.md` lines 39, 47, 51, 67 and `thesis/chapters/04_results.md` line 11. Swap the F1 and F2 blocks in `thesis/figure_manifest.md` so its numbering matches.

---

## 4. Caption work in the Doc

| Caption | Action |
|---|---|
| Table 1 | none — already matches the manifest |
| **Figure 1** (methods flow) | title renumbered only, body unchanged |
| **Figure 2** (sleep overview) | **replaced wholesale.** The Doc caption still describes panel C, which became Table 2 on 2026-08-13. New text is the two-panel version at `figure_manifest.md:124`, renumbered to Figure 2 |
| **Table 2** | **new**, text from `figure_manifest.md:113` |
| Figure 3, Figure 4, Figure 5, Figure S1 | none — already match the manifest |
| **Figure S2** | **updated.** The Doc carries an older short version; the manifest text describes the restyled V11 panels ("whole-scalp strength", "strength within the central-parietal ROI", the dashed pooled fit, the per-panel Pearson/Spearman line, per-participant colouring) |

Mirror all caption changes into `thesis/figure_manifest.md`. Chapter files hold no captions — they are prose plus citation pointers only — so only the renumbered citations touch `thesis/chapters/`.

---

## 5. Table 2 as a native Docs table

Source: `results/demographics_V4/table2_sleep_architecture.csv` (14 data rows, 3 section bands). Build with `insertTableWithData`, then style with `updateTableCellStyle` / `applyTextStyle` to match Table 1.

Columns, folded to fit portrait width: **Measure | Young (n=35) | Elderly (n=39) | aMCI (n=30) | Test | p | η²**, with the three post-hoc p-values folded into a single "Post-hoc" column (dash where the omnibus was not significant, matching the caption's existing convention). Section bands ("Sleep architecture (%)", "Sleep continuity", "N2 bout properties") become bold single-cell rows, as in the PNG. Bold marks p < 0.05, as the caption states.

Caption goes underneath, from `figure_manifest.md:113`.

---

## 6. Flagged, not applied — your call

1. **Two empty heading paragraphs.** An empty Heading 3 sits between the Figure 5 caption and section 4.5 (Yuval's own stray insertion, his ¶196), and an empty Heading 2 sits before References. Both will appear as blank entries in the table of contents he asked for in email #1.
2. **`figure_manifest.md` preamble contradicts the applied C429 decision.** Lines 93 and 95 still say statistics are V10-with-demographics_V3 and that bandwidth "must be reported as a trend, not a robust effect". Results 4.2 and the Discussion now say the opposite — that the bandwidth difference is largely an artefact of analyzed N2 duration. The F3 "Supports" bullet at line 146 repeats the trend wording. Manifest-only fix, no thesis text involved.
3. **Stale line pointers in the hand-off notes.** `yuval_review_triage.md` and `results_edits_before_after.md` point at `figure_manifest.md:101` for the Table 2 caption and `:112` for the Figure 1 caption; both are off by one block (the real ones are 113 and 124). Following them literally would paste the wrong captions.
4. **Table 1 and the Table 2 PNG still render "MCI"** in their column headers. The Doc's Table 1 is native and its header cell already reads aMCI, so this only affects the manifest previews.

---

## 7. Order of work

1. **Regenerate** the six assets with aMCI labels (§2), verify by reading each PNG.
2. **Write** `thesis/reviews/figures_edits_before_after.md`: the placement table with absolute paths and exact paste widths/heights, every caption before/after, the renumber list, and the Table 2 spec. **Stop there for approval.**
3. **You** do the manual pass in the browser: move the sleep-overview image to Results 4.1, swap the six V11 assets, make Figure 5 inline. Leave Table 2 alone.
4. **I** then do the text pass: renumber, captions, insert the native Table 2, mirror into `03_methods.md`, `04_results.md`, `figure_manifest.md`.
5. **Verify** and record carry-over.

## 8. Verification

- Re-read the Doc JSON and assert: 8 inline images in the expected order, no positioned (floating) objects left, image aspect ratios matching the regenerated PNGs, no image still at a V10 aspect (Figure 3's current 468×191 is the tell).
- Assert every figure/table citation resolves and that first-citation order is 1, 2, 3, 4, 5, S1, S2 — the invariant the renumber exists to restore.
- Re-check the reference-numbering invariant (order of first appearance == list order, 63 entries) since caption edits can span superscripts.
- Confirm `thesis/chapters/03_methods.md`, `04_results.md` and `figure_manifest.md` carry the same numbering as the Doc.
- Update `yuval_review_triage.md` §8.1 and the manifest "START HERE" box: close C412, C389, C487, C502, C591, C592 and email #6, and record C432 as the one figure item still open.

No commits. No deletions — regenerated assets overwrite the V11 files in place, as agreed.
