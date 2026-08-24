# Figures pass — before / after

**Target:** Google Doc "Shaked's Thesis V2" (`1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`), every display
item: Table 1, Table 2, Figures 1–5, Figures S1–S3.
**Yuval's items covered:** C412 (move the sleep-overview figure to Results), C389 + email #5 (the sleep
table is unreadable as a pasted panel), C432 (supplementary example spectra), C487, C502, C591, C592 and
email #6 (fonts, legibility, one group palette across all figures). **All of them are now closed.**
**Written 2026-08-17 as the approval gate; APPLIED the same day — see §9 for the closing state and the
verification results.** §2 is the instruction list Shaked worked from and is kept as written.

Decisions you made before this file was written:

- Regenerate the figures **in place, overwriting the V11 files** rather than opening a V12.
- **Renumber**: the methods-flow figure becomes **Figure 1**, the sleep overview becomes **Figure 2**.
- **Table 2 goes in as a native Docs table**, not as a pasted PNG.
- C432 deferred.

---

## 0. Why the renumber

Moving the sleep overview into Results 4.1 (C412) moves it *behind* the methods-flow figure, which is first
cited in Methods 3.4 ("The conceptual steps and a worked single-channel example are shown in Figure 2A–C").
A reader would have met Figure 2 before Figure 1. Swapping the two numbers restores citation order and
costs five in-text mentions and two caption titles.

---

## 1. What was regenerated, and what did not change

Every V11 asset still drew **"MCI"** on-figure — x-ticks, topography panel titles, the hypnogram banner,
the MoCA legend, the Table 2 column header — while every caption in the Doc had already been relabelled
**aMCI**. `grep -rn aMCI code/` returned zero hits before this pass. That mismatch is now fixed.

**Code change — display-only.** The group name doubles as a colour-lookup key (`GROUP_COLORS`,
`GROUP_COLORS_DARK`) and appears literally inside `three_groups_V10/three_group_topo_statistics.txt`, which
`replot_topos_paper.py` parses. Renaming the keys would have broken both. So `code/utils/config.py` gained

```python
GROUP_DISPLAY = {'Young': 'Young', 'Elderly': 'Elderly', 'MCI': 'aMCI'}
def group_label(name): ...
```

and it is applied at the six places that draw text:

| File | Site | What it labels |
|---|---|---|
| `code/step6_groups_comparison.py:1212` | violin x-ticks | Figures 3 and 5 |
| `code/step5_topo_comparison.py:1242` | raw topo panel titles | Figure 4A |
| `code/step5_topo_comparison.py:1341` | normalized topo panel titles | Figure 4B, S1 |
| `code/make_f1_figure.py:107` | hypnogram group banner | Figure 2 |
| `code/moca_correlation.py:180-184` | scatter legend | Figure S2 |
| `code/combined_demographics_table.py:39,120` | table header + pair headers | Table 2 |

The MoCA site needed care: the label was also the palette key, so `PAPER_GROUP_LABELS` stays `MCI` and only
the printed string goes through `group_label()`. Relabelling the dict directly would have turned the aMCI
series grey. `GROUP_TITLES` in `combined_demographics_table.py` likewise stays `MCI`, because it also keys
the CSV columns — and the regenerated `table2_sleep_architecture.csv` header is byte-identical to the old
one (`section, variable, Young, Elderly, MCI, …`).

**No statistic was recomputed.** Every script re-renders from the same per-subject files and the same V10
stats reports. The console output confirms it: Young 6.457 / Elderly 7.314 / aMCI 7.478 whole-scalp AUC,
ROI 1.099 / 1.040 / 1.012, post-hoc electrode counts 5 / 7 / 1, N = 35 / 39 / 30 — all identical to V10.
Pixel dimensions are unchanged for every asset except Table 2's PNG, which grew 4102 → 4192 px because
"aMCI" is a wider header string than "MCI".

Files overwritten (same names, V11 preserved as the version label):

- `thesis/figures/hypno_sleep_stages_V11.png`
- `thesis/figures/f4_auc_composite_V11.png`
- `thesis/figures/s1_topo_composite_V11.png`
- `results/group_comparison_results/three_groups_V11/group_comparison_violin.png`
- `results/group_comparison_results/three_groups_V11/group_comparison_violin_extended_ROI_normalized_auc.png`
- `results/group_comparison_results/three_groups_V11/three_group_topo_{auc,auc_raw,peak_frequency,bandwidth}.png`
- `results/moca_correlation_V4/moca_correlations_grid.png`
- `results/demographics_V4/table2_sleep_architecture.png` (+ `.csv`)

`thesis/figures/methods_flow_roi_v3.png` (the new Figure 1) was **not** regenerated — it is a
single-subject methods figure and carries no group labels.

### 1a. Figure 3 reshaped (2026-08-17, after the first paste round)

The first V11 render was 7.56 × 16.11 in, an aspect of 2.13:1. That could only be sized to the 9 in page
height, which capped it at 4.22 in wide, left a third of the column empty, and pushed the caption onto the
next page. Two changes in `code/step6_groups_comparison.py`:

- **`:1120`** — the `paper_style` canvas goes from `(7.5, 5.4 * n_metrics)` to `(9.5, 3.8 * n_metrics)`,
  so the figure is 9.39 × 11.30 in, aspect 1.20:1. It now pastes at the **full 6.5 in text width** and
  stands **7.82 in** tall, leaving about 1.2 in for the caption on the same page. The shrink factor goes
  from 0.48 to 0.68, so the panel titles render at about 15 pt instead of 10.6 pt and the ticks at 13 pt
  instead of 9.2 pt.
- **`:1271`** — the top significance bracket gains headroom in `paper_style` (0.10 of the axes instead of
  0.04). On the shorter panels the 28 pt asterisk is about 0.14 of the axes height, so `**` was riding up
  out of the plot and reading as part of the "Peak Frequency" title.

Statistics untouched: the re-run prints the same means, the same two significant peak-frequency contrasts,
and the same N = 35 / 39 / 30.

---

## 2. What you do by hand, in order

Page geometry: 612 × 792 pt with 72 pt margins, so the text area is **6.5 × 9.0 in**. For each swap the
fastest route is right-click the existing image → **Replace image → Upload from computer**, which keeps the
image in place, then set the exact size in **Image options → Size & rotation**.

| Step | Where | Do this | File | Set size to |
|---|---|---|---|---|
| 1 | **Methods 3.2** | Select the sleep-overview image **and its caption paragraph** and **cut** them. Then in **Results 4.1**, click at the end of the paragraph beginning *"An overview of the recorded sleep across the three groups…"*, press Enter, and paste. | — | — |
| 2 | its new home in Results 4.1 | Replace image | `I:\Shaked\ISO\thesis\figures\hypno_sleep_stages_V11.png` | **6.50 × 5.60 in** |
| 3 | **Results 4.2**, Figure 3 | Replace image | `I:\Shaked\ISO\results\group_comparison_results\three_groups_V11\group_comparison_violin.png` | **6.50 × 7.82 in** — reshaped 2026-08-17, see §1a; use the current file, not the one you already pasted. |
| 4 | **Results 4.3**, Figure 4 | Replace image | `I:\Shaked\ISO\thesis\figures\f4_auc_composite_V11.png` | **6.50 × 6.36 in** |
| 5 | **Results 4.4**, Figure 5 | This image is currently **floating with text wrapped around it** at 3.1 × 3.6 in, which is why it looks wrong. Click it, set wrap to **In line**, centre it, then Replace image | `I:\Shaked\ISO\results\group_comparison_results\three_groups_V11\group_comparison_violin_extended_ROI_normalized_auc.png` | **6.50 × 7.46 in** |
| 6 | **Supplementary**, Figure S1 | Replace image | `I:\Shaked\ISO\thesis\figures\s1_topo_composite_V11.png` | **6.50 × 5.65 in** |
| 7 | **Supplementary**, Figure S2 | Replace image | `I:\Shaked\ISO\results\moca_correlation_V4\moca_correlations_grid.png` | **6.50 × 6.30 in** |
| 8 | **Supplementary**, after S2 | **New figure S3** — insert it below the Figure S2 caption, then leave a blank paragraph for the caption (§5.6) | `I:\Shaked\ISO\thesis\figures\s3_example_spectra_V11.png` | **6.50 × 6.50 in** |

**Do not touch:**

- **Figure 1** (the methods-flow figure in Methods 3.5). Its asset is unchanged and already sized correctly
  at 6.50 × 6.71 in. Only its caption number changes, and I do that.
- **Table 1.** Unchanged.
- **Table 2.** Leave the gap after the *"Sleep continuity showed the same pattern (Table 2)…"* paragraph
  empty. I insert the table and its caption.
- **Every caption.** I rewrite them all in the text pass, including the two that get new numbers.

---

## 3. Final order of display items

| Order | Item | Section | First cited in |
|---|---|---|---|
| 1 | Table 1 | Methods 3.1 | Methods 3.1 |
| 2 | **Figure 1** — ISFS concept, feature extraction, ROI *(was Figure 2)* | Methods 3.5 | Methods 3.4 |
| 3 | **Figure 2** — sleep overview *(was Figure 1)* | Results 4.1 | Results 4.1 |
| 4 | **Table 2** — sleep architecture, continuity, N2 bouts | Results 4.1 | Results 4.1 |
| 5 | Figure 3 — whole-scalp violins | Results 4.2 | Results 4.2 |
| 6 | Figure 4 — AUC topographies + cluster | Results 4.3 | Results 4.3 |
| 7 | Figure 5 — ROI violin | Results 4.4 | Results 4.4 |
| 8 | Figure S1 — peak-frequency and bandwidth topographies | Supplementary | Results 4.5 |
| 9 | Figure S2 — MoCA grid | Supplementary | Results 4.6 |
| 10 | **Figure S3** — example ISFS spectra, six participants | Supplementary | Results 4.1 (new pointer, §5.7) |

---

## 4. Renumbering — every edit

### 4.1 In the Doc, Methods: Figure 2 → Figure 1

Anchored on surrounding words, never on the bare string.

| # | Section | Before | After |
|---|---|---|---|
| R1 | 3.4 | "The conceptual steps and a worked single-channel example are shown in **Figure 2A–C**." | "…shown in **Figure 1A–C**." |
| R2 | 3.4 | "…reading the values off the fit, as illustrated in **Figure 2C**." | "…as illustrated in **Figure 1C**." |
| R3 | 3.5 | "…giving a fixed set of central-parietal electrodes (**Figure 2D**)." | "…(**Figure 1D**)." |
| R4 | 3.7 | "YASA was used only to detect the spindles illustrated in **Figure 2A**…" | "…illustrated in **Figure 1A**…" |
| R5 | caption | "**Figure 2.** Characterization of infra-slow fluctuations of sigma power." | "**Figure 1.** Characterization of infra-slow fluctuations of sigma power." |

### 4.2 In the Doc, Results: Figure 1 → Figure 2

| # | Section | Before | After |
|---|---|---|---|
| R6 | 4.1 | "…and the distribution of sleep stages, is shown in **Figure 1**; group statistics for every measure below are reported in Table 2." | "…is shown in **Figure 2**; group statistics…" |
| R7 | caption | replaced wholesale — see §5.2 | |

R1–R5 are applied before R6–R7, so no two edits ever compete for the same string.

### 4.3 Mirrored in the repo

- `thesis/chapters/03_methods.md` lines 39, 47, 51, 67 — the same four mentions.
- `thesis/chapters/04_results.md` line 11 — the one mention.
- `thesis/figure_manifest.md` — the F1 and F2 blocks swap places and their captions swap numbers.

---

## 5. Captions — before / after

Caption formatting in the Doc, to be preserved: whole paragraph justified, 12 pt, coloured `#0070C0`,
with the leading title sentence **bold** and the body not.

### 5.1 Figure 1 (methods flow) — number only

**Before:** `Figure 2. Characterization of infra-slow fluctuations of sigma power. A) Spindle timeline…`
**After:** `Figure 1. Characterization of infra-slow fluctuations of sigma power. A) Spindle timeline…`

Body text unchanged.

### 5.2 Figure 2 (sleep overview) — replaced wholesale

**Before** (still describes a panel C that stopped existing on 2026-08-13, when it became Table 2):

> **Figure 1. Sleep architecture across groups.** A) Whole-night hypnogram and spectrogram for one
> representative subject per group. B) Proportion of each sleep stage per group: deep (N3) and REM sleep are
> reduced in elderly and aMCI relative to young adults (both p < 0.001; per-stage statistics in panel C),
> whereas N2 sleep, the stage in which ISFS occurs, remains the largest stage in all three groups.
> C) Group comparison of sleep-stage proportions and N2 bout properties: both older groups have more N2
> bouts than young adults, and in aMCI these bouts are also shorter; the proportion of each subject's N2
> that entered the analysis as bouts is similar across groups.

**After:**

> **Figure 2. Sleep architecture across groups.** A) Whole-night hypnogram and spectrogram for one
> representative subject per group. B) Proportion of each sleep stage per group, as a percentage of total
> recording time: deep (N3) and REM sleep are reduced in elderly and aMCI relative to young adults (both
> p < 0.001), whereas N2 sleep, the stage in which ISFS occurs, remains the largest stage in all three
> groups. Group statistics for every stage are reported in Table 2.

*Reason:* C389. Panel C is gone from the image, so the sentence describing it and the "per-stage statistics
in panel C" pointer both have to go; the reader is sent to Table 2 instead. The "as a percentage of total
recording time" clause is added because it was the one qualifier panel C used to supply.

### 5.3 Table 2 — new

> **Table 2. Sleep architecture, sleep continuity and N2 bout properties by group.** Values are mean ±
> standard deviation. Sleep architecture is expressed as a percentage of total recording time, and a bout is
> a stretch of NREM2 of at least 300 s that survived artifact rejection. WASO is wake after sleep onset. The
> omnibus test for each row is a one-way ANOVA or a Kruskal–Wallis test (KW), chosen by normality
> (Shapiro–Wilk), and η² is the corresponding effect size; Tukey HSD or Holm-corrected Dunn post-hoc tests
> were run only where the omnibus test was significant, and a dash marks the comparisons that were therefore
> not run. Bold marks p < 0.05. Wake, N1, wake after sleep onset and the number of N2 bouts all increase
> with age, while N3, REM sleep and sleep efficiency decrease; sleep onset latency did not differ across
> groups (p = 0.38). Elderly and aMCI did not differ significantly on any measure.

### 5.4 Figure S2 — updated to match the restyled figure

**Before:**

> **Figure S2. ISFS metrics versus MoCA.** The sample is the pooled elderly and aMCI group (n = 44). Each
> panel plots one metric (peak frequency, bandwidth, whole-scalp AUC, or ROI AUC) against MoCA score. No
> metric correlated with MoCA (all |r| ≤ 0.15, all p ≥ 0.34; Pearson and Spearman).

**After:**

> **Figure S2. ISFS metrics versus MoCA.** The sample is the pooled elderly and aMCI group (n = 44), with
> each participant coloured by group. Each panel plots one metric (peak frequency, bandwidth, whole-scalp
> strength, or strength within the central-parietal ROI) against MoCA score, with the pooled linear fit
> shown as a dashed line and the Pearson and Spearman statistics above the panel. No metric correlated with
> MoCA (all |r| ≤ 0.15, all p ≥ 0.34; Pearson and Spearman).

*Reason:* C592. The restyled panels are now titled "ISFS strength (whole-scalp)" and "ISFS strength
(central-parietal ROI)", the four duplicate legends were replaced by one shared legend, and the Pearson and
Spearman statistics moved onto their own line above each panel. The old caption describes none of that, and
its panel names no longer match what is drawn.

### 5.5 Figure S3 — new (C432)

> **Figure S3. Example ISFS spectra from individual participants.** Two example channels are shown per
> group, drawn with the same conventions as Figure 1C. Each panel plots the mean baseline-corrected spectrum
> of the sigma amplitude envelope across that channel's clean N2 bouts (blue), the fitted Gaussian (purple)
> with its peak marked, the bandwidth (orange), the ±1σ area that defines the AUC, and the detection
> threshold (dashed line). The fitted values and the number of bouts entering each average are printed in
> each panel. The vertical scale is set separately in each panel, because the examples differ in absolute
> power.

Built by the new `code/make_s3_figure.py` from the channels you picked: Young `ON68/E144` and
`EL3004/E122`, Elderly `MCI43/E19` and `SL44/VREF`, aMCI `MCI04/E195` and `MCI41/E99`. Each panel is
redrawn from that channel's `_spectral_power.csv` and `_analysis_summary.txt`, so nothing is recomputed and
no existing PNG is cropped — which is how the per-channel titles come off cleanly.

**Subject codes are deliberately not drawn on the figure.** Two of the elderly examples are coded `MCI43`
and `SL44`, and two of the aMCI examples are coded `MCI04` and `MCI41`. Printing the raw IDs would tell a
reader that four of the six examples are aMCI, when only two are. The mapping is recorded in
`figure_manifest.md` and in the script.

### 5.6 Results 4.1 — one new sentence, so S3 is cited

A figure that nothing points at is an orphan, and C432 was raised on this paragraph.

**Before** (last paragraph of 4.1):

> The ISFS was present in the vast majority of the data: across the 1023 clean N2 bouts analyzed, a valid
> Gaussian fit was obtained in a mean of 80.8% of channels overall, and in 74.5% of channels in young
> adults, 85.6% in older adults, and 82.0% in aMCI.

**After:** the same paragraph, with one sentence appended:

> …85.6% in older adults, and 82.0% in aMCI. Example spectra and their fits, for two participants from each
> group, are shown in Figure S3.

Mirrored into `thesis/chapters/04_results.md` line 19.

### 5.7 Unchanged

Table 1, Figure 3, Figure 4, Figure 5 and Figure S1 already carry exactly the manifest text, including the
aMCI relabel and the de-trended bandwidth wording in Figure 3. Nothing to do.

---

## 6. Table 2 as a native Docs table

Built from `results/demographics_V4/table2_sleep_architecture.csv`, so no number is retyped by hand.
Ten columns, fourteen data rows, three band rows. Styling mirrors Table 1: grey `#D9D9D9` bold header row,
centred data cells, plain body text — but at **9 pt** rather than Table 1's 10 pt, because ten columns have
to fit the 6.5 in text width.

| Column | Width (pt) |
|---|---|
| Measure | 96 |
| Young (n=35) / Elderly (n=39) / aMCI (n=30) | 60 each |
| Test | 32 |
| Omnibus p | 40 |
| η² | 22 |
| Y vs E | 28 |
| Y vs aMCI / E vs aMCI | 44 each |

Section bands ("Sleep architecture (%)", "Sleep continuity", "N2 bout properties") become bold rows with a
pale blue `#DCE6F1` fill across the row, as in the PNG. The five sleep-stage rows keep their colour swatch
in the measure column, matching the pie wedges in Figure 2. Cell padding is cut to 2 pt so the widest cell,
"629.0 ± 174.5", fits on one line. Bold marks p < 0.05, which is what the caption already claims.

Content, verbatim from the CSV:

| Measure | Young | Elderly | aMCI | Test | p | η² | Y vs E | Y vs aMCI | E vs aMCI |
|---|---|---|---|---|---|---|---|---|---|
| *Sleep architecture (%)* | | | | | | | | | |
| Wake | 9.9 ± 6.6 | 15.2 ± 7.5 | 21.1 ± 11.8 | KW | **<.001** | .201 | **.004** | **<.001** | .079 |
| N1 | 3.4 ± 2.5 | 9.3 ± 6.2 | 10.0 ± 5.6 | KW | **<.001** | .363 | **<.001** | **<.001** | .576 |
| N2 | 34.1 ± 13.2 | 44.5 ± 11.7 | 39.0 ± 12.3 | KW | **<.001** | .118 | **<.001** | .113 | .113 |
| N3 | 27.9 ± 8.7 | 18.4 ± 9.7 | 19.3 ± 9.9 | ANOVA | **<.001** | .177 | **<.001** | **.001** | .929 |
| REM | 23.9 ± 10.4 | 11.4 ± 5.0 | 10.3 ± 5.2 | ANOVA | **<.001** | .421 | **<.001** | **<.001** | .813 |
| *Sleep continuity* | | | | | | | | | |
| WASO (min) | 23.2 ± 19.6 | 51.9 ± 30.0 | 71.0 ± 46.3 | KW | **<.001** | .291 | **<.001** | **<.001** | .160 |
| Sleep onset latency (min) | 17.4 ± 17.1 | 15.6 ± 15.1 | 19.9 ± 18.4 | KW | .380 | −0.001 | — | — | — |
| REM latency (min) | 92.8 ± 45.9 | 125.4 ± 63.1 | 130.3 ± 55.9 | KW | **.005** | .084 | **.020** | **.008** | .549 |
| Sleep efficiency (%) | 89.2 ± 6.7 | 83.6 ± 8.5 | 78.6 ± 11.7 | KW | **<.001** | .170 | **.007** | **<.001** | .106 |
| *N2 bout properties* | | | | | | | | | |
| Bout count | 7.66 ± 3.76 | 11.00 ± 4.17 | 10.83 ± 4.65 | KW | **.001** | .111 | **.002** | **.009** | .735 |
| Mean Bout Length (s) | 629.0 ± 174.5 | 561.1 ± 108.6 | 500.6 ± 76.7 | KW | **.002** | .102 | .154 | **.002** | .055 |
| Total Duration of Bouts (min) | 80.7 ± 44.2 | 106.0 ± 50.1 | 90.8 ± 39.9 | KW | .081 | .030 | — | — | — |
| Mean Relative Location (%) | 56.7 ± 12.9 | 54.9 ± 7.1 | 54.9 ± 9.3 | KW | .072 | .032 | — | — | — |
| Proportion of all N2 (%) | 51.9 ± 15.9 | 52.3 ± 17.4 | 51.5 ± 15.5 | ANOVA | .981 | .000 | — | — | — |

Once this lands, the two dangling "(Table 2)" references in Results 4.1 resolve.

---

## 7. Flagged, not applied — your call

Nothing here is a Yuval ask, so nothing here will be touched without a word from you.

1. **Two empty heading paragraphs.** An empty Heading 3 sits between the Figure 5 caption and section 4.5
   (Yuval's own stray insertion, his ¶196), and an empty Heading 2 sits immediately before References. Both
   will show up as blank lines in the table of contents he asked for in email #1.
2. **`figure_manifest.md` still tells the reader to hedge bandwidth.** Its preamble (lines 93 and 95) says
   statistics are "V10 / demographics_V3" and that bandwidth "must be reported as a trend, not a robust
   effect", and the Figure 3 *Supports* bullet at line 146 repeats it. Results 4.2 and the Discussion now
   say the opposite — that the group difference in bandwidth is largely an artefact of analyzed N2 duration
   and should not be read as an effect of age. Manifest-only fix; no thesis text involved.
3. **Two stale line pointers in the hand-off notes.** `yuval_review_triage.md` and
   `results_edits_before_after.md` send the figures session to `figure_manifest.md:101` for the Table 2
   caption and `:112` for the corrected Figure 1 caption. Both are off by one block — the real ones are 113
   and 124. Following them literally would have put Table 1's caption under Table 2.
4. **Table 1's PNG still renders "MCI"** in its column header (`results/demographics_V3/demographics_table.png`).
   The Doc's Table 1 is a native table whose header cell already reads aMCI, so this only affects the
   preview embedded in the manifest.

---

## 9. Applied 2026-08-17 — closing state

Everything in §2 to §6 is applied and verified in the Doc. Two things changed after the paste round:

**The supplementary set was renumbered.** Citing the example spectra at the end of Results 4.1 put them
ahead of the other two supplementary figures, which is the same defect the main-figure renumber had just
fixed. Shaked moved the figure to the head of the supplementary section and asked for the renumber:

| Was | Is | Cited in |
|---|---|---|
| Figure S3 — example spectra | **Figure S1** | Results 4.1 |
| Figure S1 — peak-frequency and bandwidth topographies | **Figure S2** | Results 4.5 |
| Figure S2 — MoCA grid | **Figure S3** | Results 4.6 |

Three captions and three in-text mentions, in the Doc and in `04_results.md` / `figure_manifest.md`. It is a
3-cycle, so no ordering of replacements avoids a collision; each label was routed through a temporary token
(`Figure SX1` and friends) and then resolved. The asset and its builder keep their `s3_example_spectra_V11.png`
and `make_s3_figure.py` names from when the figure was numbered S3.

**The example-spectra image had been anchored inside the "Supplementary figures" heading paragraph**, a side
effect of the paste landing on the heading. Split out into its own NORMAL_TEXT paragraph between the heading
and its caption.

**Figure 5's image stays at 3.1 × 3.6 in** — Shaked reviewed it on the page and kept it.

### Verified in the Doc

- No positioned (floating) objects. Every image is inline and immediately followed by its own caption.
- Display order: Table 1, Figure 1, Figure 2, Table 2, Figure 3, Figure 4, Figure 5, Figure S1, S2, S3.
- First-citation order: Table 1, Figure 1, Figure 2, Table 2, Figure S1, Figure 3, Figure 4, Figure 5,
  Figure S2, Figure S3 — monotonic within the main figures, the tables and the supplementary set.
- Image sizes match the regenerated assets: 6.50 in wide throughout except Figure 5.
- Reference-numbering invariant holds: 63 entries, 63 distinct citation numbers, order of first appearance
  exactly 1 to 63. The caption edits did not disturb it.
- Table 2 spot-checked: 9 pt throughout, header bold and centred, band labels bold, p-values below .05 bold,
  the minus sign in −0.001 and the em-dashes intact.

## 8. What happens after you approve

1. You do §2 in the browser.
2. I then apply, in the Doc: the renumbering (§4.1–4.2), the caption edits (§5.1–5.4), and the native
   Table 2 with its caption (§6).
3. I mirror into `thesis/chapters/03_methods.md`, `thesis/chapters/04_results.md` and
   `thesis/figure_manifest.md`.
4. I verify: eight inline images in the expected order with no floating objects left, image aspect ratios
   matching the regenerated PNGs, first-citation order 1 → 2 → 3 → 4 → 5 → S1 → S2, and the reference
   numbering invariant (order of first appearance == list order, 63 entries) still holding after the caption
   edits.
5. I close C412, C389, C487, C502, C591, C592 and email #6 in `yuval_review_triage.md` §8.1, and record
   C432 as the one figure item still open.
