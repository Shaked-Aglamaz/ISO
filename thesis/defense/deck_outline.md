# Defense deck outline: ISFS in aging and aMCI

**Talk length:** ~45 min + questions · **Deck:** `thesis/defense/ISFS_defense_V2.pptx` (16:9, 54 slides:
42 main + 1 backup divider + 11 backup). V1 is kept as the first draft.
**Statistics of record:** V10, final cohort Young 35 / Elderly 39 / aMCI 30. Nothing on a slide is recomputed.
**Slide assets:** `results/defense_slides_V1/`, plus the reusable manuscript PNGs named per slide.

Time budget: background ~10 min · methods ~8 min · results ~18 min · discussion ~9 min.

## How the deck is built

| Step | Command |
|---|---|
| Panels cropped from published figures (Purcell, Lecci, Champetier, Lazar, Nir review) | `python code/defense/make_ref_figures.py` |
| Methods panels (A+B, C, ROI) at slide geometry | `python code/defense/make_methods_panels.py` (add `--refresh` to recompute the cached example-subject data) |
| One violin panel per metric, plus the ROI violin | `python code/defense/make_violin_panels.py` |
| LC-NE schematic, bandwidth scatter, MoCA and hypnogram crops | `python code/defense/make_slide_assets.py` |
| The deck itself (content lives in `SLIDES` in the script) | `python code/defense/build_deck.py` |
| Slide PNGs for visual checking (PowerPoint COM; LibreOffice is not installed) | `powershell -File code/defense/render_deck.ps1` |
| Speaker notes as markdown, for the question-prep sessions | `python code/defense/dump_notes.py` |

All with `PYTHONIOENCODING=utf-8` and the `eeg_clean` venv, from the repo root.

Conventions: the title of each slide is a claim or a plain label, never a sentence to be read aloud; on-slide
text is short prompts; numbers live in the speaker notes unless the slide is *about* the number. Group
colours throughout: Young green, Elderly red, aMCI blue, matching every figure and the table headers.

---

## Slide map

| # | Slide | Visual |
|---|---|---|
| 1 | Title (English title, supervisor, date) | — |
| 2 | Roadmap: the rhythm, what we measured, what changed with age, what aMCI added | 4 cards |
| 3 | **Section: Background** | — |
| 4 | Sleep spindles: subtypes, topography | Purcell 2017 slow/fast density maps + 3 labels |
| 5 | Spindles appear in a rhythm | our timeline + zoom, Dimitriades young-adult AUC map, 4 labels |
| 6 | Two substates, paced by the locus coeruleus | our schematic + the rodent cycle (Nir et al. review) |
| 7 | What aging does to spindles, and to the rhythm | Purcell age curves + Champetier 2023 Fig 2 |
| 8 | Amnestic MCI | 4 short lines |
| 9 | Open questions (aims) | 3 cards |
| 10 | **Section: Methods** | — |
| 11 | Cohort: 104 participants, two sites | native table |
| 12 | Recording and scoring | text |
| 13 | Preprocessing and artifact rejection | 6-step flow + text |
| 14 | Clean N2 bouts: the unit of analysis | text |
| 15 | From the EEG to the sigma envelope | `methods_panel_AB.png` |
| 16 | The fit, and when we accept it | `methods_panel_C.png` |
| 17 | The ROI was defined a priori | `methods_panel_ROI.png` |
| 18 | Statistics at three levels of spatial detail | text |
| 19 | **Section: Results** | — |
| 20 | Sleep changes with age, but N2 stays plentiful | `sleep_stage_pies.png` |
| 21 | One whole night per group | three cropped hypnospectrograms |
| 22 | Sleep architecture and continuity | native table (7 rows) |
| 23 | Is the N2 that entered the analysis comparable? | native table + the three arguments |
| 24 | The ISFS is present in nearly every channel | `s3_example_spectra_V11.png` |
| 25 | **The rhythm runs faster in both older groups** | peak-frequency violin |
| 26 | Bandwidth: no group effect, and it tracks analyzed N2 | bandwidth violin + duration scatter |
| 27 | Overall strength, averaged over the scalp, is unchanged | AUC violin |
| 28 | Where the strength sits: the young-adult hotspot | `three_group_topo_auc_raw.png` |
| 29 | **One significant central-parietal cluster** | `three_group_topo_auc.png` |
| 30 | Averaging over the whole ROI dilutes the effect | ROI violin |
| 31 | The temporal changes are diffuse, not regional | `s1_topo_composite_V11.png` |
| 32 | No association with cognitive score | two MoCA panels |
| 33 | **Section: Discussion** | — |
| 34 | What changed, and what did not | 3 cards |
| 35 | A loss of temporal and spatial precision | text |
| 36 | A noradrenergic account, and its limits | text |
| 37 | No evidence for an effect of amnestic MCI beyond age | text |
| 38 | The ISFS in other populations, and why comparison is hard | text |
| 39 | Limitations | text |
| 40 | Future directions | text |
| 41 | Conclusion | statement |
| 42 | Acknowledgements | text |
| 43 | **Section: Backup slides** | — |
| 44 | B1. Full demographics and data quality | `demographics_V3/demographics_table.png` |
| 45 | B2. Full sleep table with tests and effect sizes | `demographics_V4/table2_sleep_architecture.png` |
| 46 | B3. Exclusions: 26 recordings | native table |
| 47 | B4. ANCOVA with analyzed N2 duration as covariate | native table + detail |
| 48 | B5. The cluster, electrode by electrode | native table |
| 49 | B6. Bandwidth against analyzed N2 duration | `bandwidth_vs_n2_duration.png` |
| 50 | B7. Feature extraction, in detail | text |
| 51 | B8. Detection rates, and what a failed fit means | text |
| 52 | B9. Was the aMCI group too heterogeneous? | text |
| 53 | B10. Alongside Grollero et al. (2026) | native table |
| 54 | B11. Where the hypnograms and annotations come from | text |

Full speaker notes are in `thesis/defense/speaker_notes.md` (generated from the deck); examiner profiles
and the questions to prepare are in `thesis/defense/qa_prep.md`.

**Slide-text rule (2026-08-20):** as few full sentences on a slide as possible. Background slides carry
short phrases in pill boxes and the sentences live in the notes; the same sweep is still to be applied to
methods, results and discussion.

**Published panels** are cropped by `code/defense/make_ref_figures.py` into
`results/defense_slides_V1/refs/`, with the source named on the slide. The rodent-cycle panel on slide 6
comes from Yuval's own unpublished review, so it needs his agreement before the talk; the published
fallback (Lecci 2017 mouse and human spectra) is already in the same folder.

---

## Numbers on the slides, and where each comes from

| Slide | Claim | Source |
|---|---|---|
| 11 | 35 / 39 / 30; 27.1 ± 4.3, 66.5 ± 9.7, 67.8 ± 9.1; MoCA 27.2 ± 2.6 (n=30), 21.5 ± 4.3 (n=14); age p = 0.57, MoCA p < 0.001 | Methods 3.1, Results 4.1, `demographics_V3/demographics_table.csv` |
| 13 | bad channels 6.3 / 1.4 / 3.4 %; N2 epochs 4.1 / 3.3 / 1.8 % | Methods 3.2 |
| 14 | 1023 bouts, mean 9.8, range 3 to 21 | Methods 3.3 |
| 20, 22 | stage percentages, WASO, sleep efficiency, REM latency, SOL | `demographics_V4/table2_sleep_architecture.csv` |
| 23 | bouts, bout length, analyzed N2 (unchanged), share kept; ANCOVA p = 0.0034 / p = 0.94 | same CSV + `three_groups_V11/three_group_ancova_statistics.txt` |
| 24 | 80.8 % overall; 74.5 / 85.6 / 82.0 % | Results 4.1 |
| 25 | 0.0199 / 0.0226 / 0.0232 Hz; p = 0.0026, η² = 0.111; post-hoc 0.013 / 0.005 / 0.86 | `three_groups_V10/three_group_statistics.txt` |
| 26 | 0.0236 / 0.0281 / 0.0276 Hz; p = 0.061; r = 0.386; ANCOVA p = 0.206 | same + ANCOVA file |
| 27 | AUC 6.46 / 7.31 / 7.48; KW p = 0.38 | `three_group_statistics.txt` |
| 29 | cluster p = 0.023, 9 electrodes; post-hoc 5 / 7 / 1 | `three_groups_V10/three_group_topo_statistics.txt` |
| 30 | 1.099 / 1.040 / 1.012; ANOVA p = 0.143 | `three_group_statistics_extended_ROI_normalized_auc.txt` |
| 32 | n = 44, all abs(r) <= 0.15, all p >= 0.34 | `moca_correlation_V3/correlation_summary.csv` |

Two rules the deck follows deliberately: bandwidth is never called a trend (it tracks analyzed N2 duration),
and the region is only ever "the central-parietal ROI".

## Open items

- **Defense date**: set `DEFENSE_DATE` at the top of `code/defense/build_deck.py`. Examiner names are not on the title slide.
- **No institutional template** was applied; the design is self-contained (navy accent, Segoe UI).
- The Hebrew title line on slide 1 is optional and easy to delete.
- Backup B9 (the naMCI sensitivity run) is written qualitatively on purpose: that run predates the final
  cohort, so its p values are not numbers of record.
