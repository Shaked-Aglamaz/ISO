# Figure overhaul for Yuval's review (C389, C390, C487, C502, C591, C592, email #6)

## Context

Yuval's review verdict is that every figure's graphics and fonts are below publication
standard (`thesis/reviews/yuval_review_triage.md` §2, §5). Two jobs in this pass:

1. **One group colour scheme and legible fonts across every figure** — reuse the
   green/red/blue already in Figure 3, and fix Figure 4's unreadable axis labels and
   colorbar legends. Figures must survive being shrunk to page width in a Google Doc.
2. **C389 + C390** — Figure 1 panel C is an unreadable pasted table. Promote it to a
   standalone **Table 2**, bigger and with more information, adding WASO, sleep onset
   latency, REM latency and sleep efficiency.

**C432 (example spectra from several subjects) is deferred at the user's request** — they
will pick the example subjects manually and commission the figure later. Remind them once
the two jobs above are finished.

No statistic is recomputed. Every number comes from files already on disk:
`three_groups_V10/`, `demographics_V3/`, `demographics_V4/sleep_statistics_table.csv`.

**Scope discipline:** only the two jobs above. No refactors, no drive-by improvements.

---

## Decisions taken

- **Output versioning** (per user): everything lands at **V11**, because nothing there
  collides. Regenerated group-comparison assets go to `three_groups_V11/`; the new Table 2
  assets go to `demographics_V4/` under new filenames. `thesis/figures/` is a flat
  directory that carries the version in the filename, so the composites are written as
  `*_V11.png` alongside the existing `*_V10.png`. Nothing existing is overwritten; if a
  target ever did already exist, it would move up one version instead.
- **Editing shared files** (per user): cosmetic edits to the plot functions inside
  `step6_groups_comparison.py` and `step5_topo_comparison.py` are allowed. No
  statistical code is modified.
- **`replot_roi_violins.py` is skipped** (per user). It writes to `three_groups_V5`,
  reads the pre-`sigma_fix` result dirs, and backs no locked figure — F5 comes from
  `replot_f5_no_title.py`. Recorded as dead in the manifest, not restyled.

## Problems found — flagged, not silently worked around

- The five scripts named in the brief hold almost no styling. `replot_f3_no_title.py` /
  `replot_f5_no_title.py` are thin drivers; the palette is one line
  (`step6_groups_comparison.py:1081`) and the fonts are `step6:1122-1125`.
  `make_topo_composites.py` only stacks PNGs; Figure 4's text is baked in by
  `step5_topo_comparison.py:1197-1305`.
- Re-running `step5_topo_comparison.main_three_group()` would re-run the unseeded
  5000-permutation cluster test. Avoided: `plot_three_group_topos()` already takes the
  post-hoc electrodes as an argument, and V10's are recorded in
  `three_groups_V10/three_group_topo_statistics.txt`.
- For the deferred C432: the triage's claim that it is "nearly free" is optimistic.
  `find_example_gaussians.py` emits 20 separate single-channel PNGs from
  `results/new_iso_results` (the **pre-sigma_fix** tree, i.e. not the paper's data), and
  `find_clean_gaussian.py` writes no figure at all. Whenever that figure is commissioned,
  a panel assembler has to be written and pointed at `sigma_fix_*`.
- The group palette is ColorBrewer Set3 pastels (`#8dd3c7` / `#fb8072` / `#80b1d3`).
  Fills stay exactly as Yuval asked; **text and marker edges use darker variants of the
  same three hues** (`#2f9c86` / `#d0402c` / `#2f7fae`), because pastel text is the
  legibility problem he is complaining about.

---

## Job 1 — one palette, legible fonts

### 1a. Promote the palette to a module constant
`code/step6_groups_comparison.py:1081` — lift the local `colors` list to a module-level
`GROUP_COLORS = {'Young': ..., 'Elderly': ..., 'MCI': ...}` plus
`GROUP_COLORS_DARK` for text/edges, keeping the existing positional list behaviour
intact. This is the single import point for `make_f1_figure.py`, the topo replotter and
the new spectra script, so no separate style module is introduced.

### 1b. F3 and F5 — `plot_group_comparison` (`step6_groups_comparison.py:1049-1310`)
Changes confined to the `paper_style` branch, so non-paper callers are unaffected:
- Font block `1122-1125`: `fs_title` 19→22, `fs_label` 16→20, `fs_tick` 15→19,
  `fs_ast` 24→28.
- **Layout to portrait.** Under `paper_style` with 3 metrics, switch the 1×3 grid to
  3×1, `figsize≈(7.5, 16)`. F3 is currently 19.5 in wide, so it shrinks ~3× at page
  width while F5 (7 in) barely shrinks — the same nominal font renders at very
  different sizes. Stacking equalises them and is what "favour portrait layouts" asks for.
- Colour the per-subject dots by group (`GROUP_COLORS_DARK`) instead of the single
  `darkblue` at line 1182, so the green/red/blue reads consistently.

Drivers `replot_f3_no_title.py:43` and `replot_f5_no_title.py:43`: repoint `output_dir`
to `three_groups_V11`. Both already re-run stats only to drive the brackets and write no
stats file (`replot_f3_no_title.py:56-59`); that stays.

### 1c. F4 and S1 topo panels — `plot_three_group_topos` (`step5_topo_comparison.py:1223`)
- Titles 20→26 (`:1300`), colorbar label 16→22 (`:1304`), colorbar ticks 14→20 (`:1305`);
  same for the raw path at `:1208/1212/1213`.
- **Informative colorbar legends** (this is the substance of C487): replace the bare
  `'Normalized'` / `'AU'` with the metric name and unit, e.g. "ISFS strength (AUC, a.u.)"
  and "Normalized ISFS strength", "Peak frequency (Hz)", "Bandwidth (Hz)".
- Colour each group's panel title with `GROUP_COLORS_DARK`.
- Add a large-font key on the normalized AUC panel for the yellow post-hoc rings and the
  green ROI dots.
- Add an optional `fstat_fig=True` parameter; when `False` the function returns after
  saving the topo row, so the replotter never needs `F_obs`.

New driver **`code/replot_topos_paper.py`** (modelled on `replot_f3_no_title.py`):
loads the group evokeds with step5's existing loaders, parses the V10 post-hoc electrode
lists out of `three_groups_V10/three_group_topo_statistics.txt`, and calls
`plot_three_group_topos(..., fstat_fig=False)` for auc-raw, auc-normalized,
peak_frequency and bandwidth into `three_groups_V11/`. **No permutation test runs.**

`code/make_topo_composites.py:22-28`: `SRC` → `three_groups_V11`, outputs →
`f4_auc_composite_V11.png` and `s1_topo_composite_V11.png`.

### 1d. F1 — `code/make_f1_figure.py`
- **Drop panel C** (see Job 2). Layout becomes 3 rows × [hypno | pie], portrait ≈ 13×11;
  panel letters reduce to A and B.
- Group banners (`:83`) 19→24 and coloured with `GROUP_COLORS_DARK`; pie labels
  (`:95`) 12.5→16.
- Replace the Google-Sheet pie loader with `results/demographics_V3/sleep_stage_means.csv`
  (already holds `n` + per-stage group means) so the figure rebuilds offline from the
  same V3 numbers.
- `OUT` → `thesis/figures/hypno_sleep_stages_V11.png`.

---

## Job 2 — Table 2 (C389 + C390 + email #5)

New script **`code/make_table2_sleep.py`**, reading only CSVs already on disk:

| Section | Rows | Source |
|---|---|---|
| Sleep architecture (% of recording) | Wake, N1, N2, N3, REM | `demographics_V3/combined_sleep_n2_table.csv` |
| Sleep continuity | WASO, Sleep onset latency, REM latency, Sleep efficiency | `demographics_V4/sleep_statistics_table.csv` |
| N2 bout properties (≥ 300 s) | count, mean length, total duration, relative location, proportion of N2 | `demographics_V3/combined_sleep_n2_table.csv` |

Sleep onset latency is reported despite p = 0.3804, as asked.

- Reuses `combined_demographics_table.render()` / `_format_p` by import rather than
  reimplementing. `render()` gains optional `fontsize` / `figsize` / extra-column
  arguments **defaulting to today's values**, so `demographics_V3/combined_sleep_n2_table.png`
  is reproducible unchanged.
- "More information" per C389: two extra columns, **Test** and **η²**, taken from
  `demographics_V4/sleep_statistics_table.csv` and from
  `demographics_V3/{sleep_stage_stats.txt, n2_bouts_table.csv}`.
  Final columns: Variable | Young (n=35) | Elderly (n=39) | MCI (n=30) | Test | Omnibus p |
  η² | Y vs E | Y vs MCI | E vs MCI.
- Table font 16→20, standalone (title/footnote allowed now that it is not embedded).
- Outputs `results/demographics_V4/table2_sleep_architecture.png` and `.csv` — new
  filenames, nothing in V4 is overwritten.

**Numbering note:** no "Table 2" exists in `thesis/chapters/` today (only Table 1 is
referenced). The manifest's planned T2/T3 (ISFS stats tables) were never built and are
unreferenced, so the new table takes **Table 2** and the manifest's planned entries shift
to T3/T4. The chapter text that cites it is owned by another session; I only record the
numbering in the manifest.

---

## Manifest updates — `thesis/figure_manifest.md`

Following the existing per-entry shape (`### ID — title · *section* · STATUS`, italic
provenance note, image link, `**Caption:**` blockquote, `- **Supports:**` bullet) and the
caption conventions (`*Figure N. Title.*`, `A)` panel letters, stats in / claims out, no
em-dashes):

- **F1** — new path, panel C removed, the C-clause dropped from the caption and its
  content pointed at Table 2.
- **F3, F5** — `three_groups_V11` paths, portrait restyle noted.
- **F4, S1** — `_V11.png` composites, new topo source, colorbar-legend and font restyle
  noted, and an explicit note that the V10 cluster statistics were reused, not re-run.
- **T2 (new)** — sleep architecture, continuity and N2 bout properties, with caption.
- **Tables section** — planned T2/T3 shifted to T3/T4; a line recording that
  `replot_roi_violins.py` is superseded and left untouched.
- A new source-of-truth line: figure assets are V11 / `demographics_V4`; the V10 stats
  remain the numbers of record.

---

## Files touched

| File | Change |
|---|---|
| `code/step6_groups_comparison.py` | `GROUP_COLORS(_DARK)` constants; `paper_style` fonts and portrait layout; group-coloured dots |
| `code/step5_topo_comparison.py` | topo fonts, informative colorbar labels, coloured titles, overlay key, `fstat_fig` flag |
| `code/replot_f3_no_title.py`, `code/replot_f5_no_title.py` | output dir → `three_groups_V11` |
| `code/replot_topos_paper.py` | **new** — regenerate topo panels using V10 post-hoc electrodes |
| `code/make_topo_composites.py` | source V11, outputs `*_V11.png` |
| `code/make_f1_figure.py` | drop panel C, portrait relayout, fonts, offline pie source, output `_V11` |
| `code/combined_demographics_table.py` | optional `fontsize`/`figsize`/extra-column args (defaults unchanged) |
| `code/make_table2_sleep.py` | **new** — Table 2 |
| `thesis/figure_manifest.md` | entries and captions above |

Not touched: `thesis/chapters/*.md`, `library.bib`, the stats scripts' statistical code,
the Google Doc, `replot_roi_violins.py`, anything under `three_groups_V10/` or
`demographics_V3/`. Nothing is committed.

---

## Verification

All runs from `I:/Shaked/ISO` with `eeg_clean` active and `PYTHONIOENCODING=utf-8`.

1. **Baseline snapshot** — record mtimes of `three_groups_V10/*`, `demographics_V3/*`
   and the three `thesis/figures/*_V10.png`; re-check at the end that none changed.
2. `python code/replot_f3_no_title.py` and `python code/replot_f5_no_title.py` →
   violins in `three_groups_V11/`; confirm N prints 35/39/30 and no `.txt` is written.
3. `python code/replot_topos_paper.py` → four topo PNGs in `three_groups_V11/`; confirm
   the log shows the post-hoc electrodes parsed from the V10 file
   (Young-vs-Elderly 5, Young-vs-MCI 7, Elderly-vs-MCI 1) and that no permutation test line
   is printed.
4. `python code/make_topo_composites.py` → `f4_auc_composite_V11.png`,
   `s1_topo_composite_V11.png`.
5. `python code/make_f1_figure.py` → `hypno_sleep_stages_V11.png`, two panels only.
6. `python code/make_table2_sleep.py` → `table2_sleep_architecture.png/.csv`; check each
   value against `combined_sleep_n2_table.csv` and `sleep_statistics_table.csv`
   (e.g. WASO 23.17 / 51.87 / 70.98, KW p ≈ 0.0000, η² 0.291; SOL p = 0.3804 present).
7. **Legibility check**, which is the actual acceptance test: render each new PNG scaled
   to 6.5 in page width and confirm the smallest text is still readable. Repeat for the
   Table 2 PNG.
8. Re-run `python code/combined_demographics_table.py` into a scratch output dir and diff
   against `demographics_V3/combined_sleep_n2_table.csv` to prove the `render()` signature
   change is backward compatible.
9. Once both jobs are verified, **remind the user that C432 (example spectra) is still
   pending their manual choice of example subjects.**
