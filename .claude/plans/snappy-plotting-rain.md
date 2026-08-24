# Defense deck — ISFS in aging and aMCI (~45 min, editable .pptx)

## Context

The thesis (V2) is written and sent to Yuval; the next milestone is the defense. The only existing deck,
`thesis/references/ISFS Project - Shaked Gotfrid.pdf` (14 slides, ~May 2026), predates the thesis: its flow is
still good but every number and figure is stale (N=34/38/31 and V7-era stats, bandwidth reported as
significant, "MCI" instead of "aMCI", no ANCOVA, no sleep-continuity results, no other-populations section).

Goal: a self-contained ~45 min defense deck built on the current numbers of record (**V10 statistics**,
final cohort **35 / 39 / 30**), with the figure set rebuilt at slide geometry, plus speaker notes that
double as the study material for the later Q&A-prep sessions.

Decisions already taken (this session): ~45 min / 35+ slides · editable **.pptx** via python-pptx ·
**regenerate landscape slide figures** rather than reuse the portrait page figures · **expanded intro**.

## Ground rules for this work

- **Numbers come from the thesis chapters and V10 stats files only.** Nothing is recomputed, no test is
  re-run. Every figure regeneration is a re-plot of existing per-subject outputs
  (`results/sigma_fix_{YA,HE,MCI}`) exactly as the V11 pass did.
- The manuscript figure assets (`thesis/figures/*_V11.png`, `three_groups_V10/V11/*`) are **not touched**.
  Slide assets go to a new versioned dir `results/defense_slides_V1/` (per the versioned-output-dir rule).
- Group palette and labels come from `code/utils/config.py` (`GROUP_COLORS`, `GROUP_COLORS_DARK`,
  `group_label()` → "aMCI"). Never rename the `'MCI'` palette/dir key.
- Prose rules for slide text follow the thesis: "peak frequency" not "speed", strength always named with
  AUC, ROI is only "the central-parietal ROI" (never "extended"), bandwidth is **not** a trend (C429).
- Run everything with `PYTHONIOENCODING=utf-8` and the `eeg_clean` venv.

## Deliverables

| Path | What |
|---|---|
| `thesis/defense/deck_outline.md` | Slide-by-slide outline: title, visual, on-slide text, speaker notes, source of every number. Written first, reviewed before the deck is built. |
| `code/defense/make_slide_assets.py` | Composites + two new small figures (below). Writes `results/defense_slides_V1/`. |
| `code/defense/make_methods_panels.py` | Re-emits the F1 methods panels at slide geometry by importing `panel_A/B/C/D` from `code/make_f2_figure.py`. |
| `code/defense/deck_spec.json` | Declarative deck (slides, images with inch coordinates, tables, shapes, notes). |
| `code/defense/render_deck.ps1` | Exports slide PNGs via the installed PowerPoint COM object, for visual QA. |
| `thesis/defense/ISFS_defense_V1.pptx` | The deck: 16:9, slide numbers, section dividers, speaker notes on every slide. |

## Slide plan (37 main + ~10 backup)

Section dividers before Background / Methods / Results / Discussion. One idea per slide; on-slide text is
a short claim plus labels, with the detail in the notes.

**Title (1)** — thesis title, name, "Supervised by Prof. Yuval Nir", Sagol School of Neuroscience / TAU,
date. Optional Hebrew title line.

**Background (2–9)** — 2 roadmap · 3 sleep stages and scalp EEG, why N2 (one hypnospectrogram) ·
4 spindles and the sigma band (`spindles_example.png`, `raw_sigma_env_example.png`) · 5 spindles cluster:
the ~0.02 Hz ISFS (`spindle_timeline.png`, 8.9:1 — full-width band) + fragile/stable substates
(Lecci 2017, Lázár 2019) · 6 LC-NE pacing of the cycle (**new schematic**, drawn by us) · 7 how the ISFS is
quantified — the Dimitriades 2024 framework and the three parameters (methods panel C) · 8 aging and aMCI:
spindles, LC vulnerability, the biomarker hope (Braak, Gorgoni, Zhang, André) · 9 gap → aims → hypotheses.

**Methods (10–17)** — 10 design and cohort (T1 as a native table, two sites) · 11 recording and scoring
(256-ch EGI PSG) · 12 preprocessing and automated cleaning (native pptx shape flow) · 13 clean N2 bouts
≥ 300 s and why (3–5 ISFS cycles) · 14 feature-extraction chain, raw → sigma → envelope → FFT (methods
panels A+B, landscape) · 15 Gaussian fit → peak frequency / bandwidth / AUC + detection criteria (panel C
large) · 16 ROI definition: Dimitriades young-adult hotspot → 256-ch map (`code/debug/YA_AUC.png` +
panel D) · 17 statistics: per-subject normalization, omnibus + post-hoc, cluster permutation, ANCOVA.

**Results (18–29)** — 18 sleep architecture across groups (hypnospectrograms + pies) · 19 sleep
architecture and continuity, native trimmed table from `demographics_V4/table2_sleep_architecture.csv`
(full PNG in backup) · 20 the analyzed N2 is matched — % surviving cleaning identical, imbalance runs
opposite, ANCOVA leaves peak frequency intact · 21 the ISFS is present in 80.8% of channels + example fits
(`s3_example_spectra_V11.png`) · 22 **peak frequency is higher in both older groups** (single-metric violin,
p = 0.0026, post-hoc .013 / .005) · 23 bandwidth does not differ and tracks analyzed N2 duration
(violin + the duration scatter, p 0.061 → 0.206) · 24 whole-scalp AUC unchanged (violin, KW p = 0.38) ·
25 raw AUC topographies — the young hotspot (`three_group_topo_auc_raw.png`) · 26 normalized maps +
cluster permutation (`three_group_topo_auc.png`; p = 0.023, 9 electrodes, post-hoc 5/9 and 7/9) ·
27 ROI averaging dilutes the focal effect (F5 violin, p = 0.143) · 28 peak-frequency and bandwidth
topographies, no cluster (`s1_topo_composite` panels, side by side) · 29 no MoCA association (2 of the 4
MoCA panels; n = 44, all |r| ≤ 0.15).

**Discussion (30–37)** — 30 findings on one slide (three-line summary graphic) · 31 loss of temporal and
spatial precision in fast-spindle organization · 32 the noradrenergic account and what these data can and
cannot show · 33 no effect of aMCI beyond age — the two readings, dilution of a mixed aMCI group,
contrast with slow-wave synchrony (Sharon 2025) and CAP · 34 the ISFS in other populations and why the
measures barely compare · 35 limitations · 36 future directions · 37 conclusion + acknowledgements.

**Backup (~10)** — full Table 1 and Table 2 PNGs · exclusion flow (104 in / 26 out with reasons) ·
per-group ISFS detection rates · cluster electrode list and per-electrode post-hoc counts · full ANCOVA
output · bandwidth × analyzed-N2 scatter · the no-naMCI sensitivity result (rejected, kept as an answer) ·
Grollero 2026 side-by-side (their peak *amplitude* vs our AUC) · Gabor-Morlet parameters and the fit
validation gates · hypnogram/annotation provenance (`*_cleaned_annotations.txt`, UNKNOWN handling).

## Figure work

Reuse as-is (already landscape and slide-legible): the four `three_groups_V11` topo PNGs (~2.1–2.4:1),
`spindle_timeline.png`, `spindles_example.png`, `raw_sigma_env_example.png`, the three
`results/hypnospectrograms/*.png`, `demographics_V3/sleep_stage_pies.png`, `s3_example_spectra_V11.png`,
`moca_correlation_V4` panels (cropped), `code/debug/YA_AUC.png`.

Rebuild into `results/defense_slides_V1/`:

1. **Single-metric violins** — `plot_group_comparison(..., metrics_filter=['peak_frequency'])` etc. already
   emits one panel per metric with `paper_style` fonts, so slides 22–24 need **no code change** to
   `step6_groups_comparison.py`; drive it from a copy of `code/replot_f3_no_title.py` pointed at the new
   output dir. A three-in-a-row overview (if wanted) is composed from those three PNGs with PIL.
2. **F5 ROI violin** — same driver pattern as `code/replot_f5_no_title.py`, new output dir.
3. **Methods panels** — import `prepare_data`, `prepare_panelC`, `panel_A/B/C/D` from
   `code/make_f2_figure.py` and compose three landscape figures: A+B stacked (~2.2:1), C alone (~1.4:1),
   D + `YA_AUC.png` side by side. No duplication of the panel code, and `methods_flow_roi_v3.png` is
   untouched.
4. **New: LC-NE schematic** — our own matplotlib drawing: anti-phase noradrenaline and sigma traces over
   ~3 cycles, fragile/offline substates shaded, spindle trains in the troughs. Own figure, cited to
   Lecci 2017 / Osorio-Forero 2021 in the caption, so nothing is lifted from a paper.
5. **New: bandwidth × analyzed-N2 scatter** — from
   `three_groups_V11/three_group_ancova_per_subject.csv` (`bandwidth`, `total_dur_min`, group colours,
   r = 0.386). Backs slide 23 and the C429 answer.
6. **Trimmed Table 2** as a native pptx table from `demographics_V4/table2_sleep_architecture.csv`
   (keep the 5 architecture rows + WASO + sleep efficiency; full PNG to backup).

## Build mechanics

1. `pip install python-pptx` into `eeg_clean` (new package, no version changes elsewhere).
2. Write `code/defense/deck_spec.json`, then
   `python ~/.claude/skills/powerpoint/scripts/pptx_create.py code/defense/deck_spec.json thesis/defense/ISFS_defense_V1.pptx`.
   16:9, `slide_number: true`, figure placement in inches, `notes` on every slide.
3. Design: white background, one accent colour, Segoe UI / Arial (installed), titles ≤ 8 words, body ≥ 20 pt,
   figure labels never below ~18 pt on screen, group colours used consistently for Young / Elderly / aMCI
   wherever the three groups appear in text or shapes.
4. Guidance sources: `~/.claude/skills/scientific-slides/references/presentation_structure.md` and
   `assets/timing_guidelines.md` for the 45-min structure and per-section timing (its default
   AI-image → PDF workflow is deliberately **not** used — the deck must stay editable and carry the real
   figures); `~/.claude/skills/powerpoint` for the .pptx mechanics.

## Verification

- `pptx_read.py thesis/defense/ISFS_defense_V1.pptx --outline` → confirm slide count, every image embedded,
  tables populated, notes present on all slides.
- `code/defense/render_deck.ps1` exports one PNG per slide through the installed PowerPoint
  (`C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE`; `soffice`/`pdftoppm` are absent, so
  `pptx_render.py` cannot be used). Read every PNG and fix overflow, clipped figures, unreadable axis text.
- Cross-check each number on a slide against `thesis/chapters/04_results.md` and
  `three_groups_V10/three_group_{statistics,topo_statistics}.txt` / `three_group_ancova_statistics.txt`;
  confirm no slide says "MCI" where the thesis says "aMCI", and no slide calls bandwidth a trend.
- Timing pass: read the notes end to end against `timing_guidelines.md` for a 45-min budget
  (~10 background / ~8 methods / ~18 results / ~9 discussion) and flag slides to cut if it runs long.

## Open items (not blocking — I will proceed with the default and flag it)

- **Defense date, and whether examiner names go on the title slide** — placeholder date otherwise.
- **A TAU/Sagol or lab .pptx template**, if one exists; otherwise a clean own design (default).
- **Hebrew title line on slide 1** — included by default, trivially removable.
