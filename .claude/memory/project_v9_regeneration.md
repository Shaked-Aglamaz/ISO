---
name: project_v9_regeneration
description: "V9 / N=36 regeneration (2026-06-09): sigma_fix data + EL3033 in Young + option-B topo clip; current source-of-truth dirs"
metadata: 
  node_type: memory
  type: project
  originSessionId: ff1fe1b7-71bf-4ef0-b09c-406033c98216
---

**⚠️ SUPERSEDED 2026-06-17 by [[project_v10_regeneration]]** — cohort moved 36/38/31 → final **35/39/30**, all outputs regenerated to `three_groups_V10` / `demographics_V3` / `moca_correlation_V3`. Bandwidth flipped sig→ns. Use V10 dirs + numbers, not V9. The V9 record below is historical.

**2026-06-09 full regeneration on the "V2" sigma_fix results + EL3033 addition.** Current source-of-truth output dirs:
- Group comparison: `results/group_comparison_results/three_groups_V9` (topos, cluster, violins, stats). Supersedes V7/V8B.
- Demographics/sleep: `results/demographics_V2`. Supersedes demographics_V1.
- Per-subject ISFS: `results/sigma_fix_{YA,HE,MCI}` (YA now 36 incl. EL3033, HE 38, MCI 31).
- Standalone figure catalog with before/after stats table: `thesis/figures_V9_sigma_fix.md`.

**What was done:** (1) EL3033 moved `ISO_data/control_clean/a_excluded/`→`control_clean/`, its fixed run copied `new_iso_results/a_excluded_V3/EL3033`→`sigma_fix_YA/EL3033` (20.5% detection); (2) step4/5/6 input dirs repointed `new_*_results`→`sigma_fix_*`, outputs bumped to V9; (3) option-B topo clip implemented (`clip_topo_to_head` in step4, imported into step5; `TOPO_EXTRAPOLATE='head'`); (4) EL3033's 18 derived demographics columns computed (validated vs DG1/EL3002) and written to "subjects" tab row 37 of sheet `1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw`; demographics scripts bumped V1→V2 and re-run.

**Key stat deltas vs N=35/V7:** peak-freq p=0.0020 (still sig), AUC cluster p=0.016 / **10** electrodes (was 9), ROI ns p=0.182, Elderly=MCI. **Bandwidth flipped ns→sig (ANOVA p=0.038, was KW p=0.052)** — test switched because all groups now normal; only Young-vs-Elderly survives; report cautiously.

**F1 composite DONE 2026-06-11:** `code/make_f1_figure.py` → `thesis/figures/hypno_sleep_stages_N36.png` (3469×1835). Horizontal: (a) 3 hypnos (EL3011/MCI27/YS0, subject titles cropped top 105px, group banners added) | (b) table-less pies drawn natively (reuses sleep_stage_pies module loaders/colors) | (c) combined_sleep_n2_table.png image. Panel letters a/b/c via fig.text.

**DONE 2026-06-13 — `thesis/figure_manifest.md` captions updated to V9 + figures recomposed.** All caption bodies on V9/N=36 numbers; paths swapped (three_groups_V9, demographics_V2, hypno_sleep_stages_N36.png). Figure recompositions this session (user-requested, capital panel letters A/B/…):
- **F1** `code/make_f1_figure.py`: pies spaced apart (subgridspec wspace 0.05→0.55, pie radius 0.82) to fix N3/N1 label overlap; panel letters now **A)/B)/C)** (paren after letter, applied in figures + captions everywhere); pies sized to intermediate radius 0.92 / wspace 0.45 (0.82/0.55 was too small, default/0.05 overlapped); caption adds compact "(both p < 0.001; per-stage statistics in panel C)" for N3/REM (user picked compact-note-over-panel-C, not inline p-values; sleep_stage_stats.txt: N3 ANOVA p<0.001, REM p<0.001).
- **F2** new `code/make_f2_figure.py` → `thesis/figures/methods_flow_roi_v2.png`: methods_flow.png + **new ROI panel** `code/debug/AUC ROI 2.png` (user recreated in place, no baked-in title; 239×298) as panel D, capital A)/B)/C)/D). Old `methods_flow_roi.png` preserved.
- **F4 + S1** new `code/make_topo_composites.py`: stacks two V9 topo PNGs each into one with A/B letters → `thesis/figures/f4_auc_composite.png` (A=normalized+cluster, B=raw) and `s1_topo_composite.png` (A=peak-freq, B=bandwidth). Manifest now embeds the single composites, not the pairs.
Caption-number deltas applied: BW borderline ANOVA p=0.038 (Y-vs-E only, no test-switch/normality wording per user), AUC cluster p=0.016/10 ch (E130/142/143/144/153/154/155/184/185/197; posthoc Y>E 5/10, Y>MCI 7/10), peak-freq p=0.0020, ROI p=0.182, AUC KW p=0.348. T1 Young n=36 27.6±4.9. **Still TODO:** re-sync [[project_paper_figure_set]] + [[project_scientific_story]] bandwidth line to V9 (not done this session). Nothing committed (safety rule).

**Original per-figure checklist (for reference; superseded by the DONE block above):** Number source of truth = `thesis/figures_V9_sigma_fix.md`. Per-figure edits:

- **Manifest header note** (currently "Source-of-truth … N=35 → V7"): change to N=36 → V9 / demographics_V2.
- **Path swaps** throughout: `three_groups_V7`→`three_groups_V9`; `demographics_V1`→`demographics_V2`; F1 image `hypno_sleep_stages_N35.png`→`hypno_sleep_stages_N36.png`. S2 (`moca_correlation_V1`) unchanged. Topo figures are now the option-B clipped versions (same filenames, in V9) — no caption-wording change needed for the clip.
- **T1 demographics:** Young **n=36, age 27.6 ± 4.9** (was 35 / 27.6±5.0); Elderly n=38 (66.5±9.9; MoCA 26.9±2.5, n=18); MCI n=31 (67.8±9.0; MoCA 21.0±4.1, n=14).
- **F1:** pies now n=36/38/31; composite is the new `make_f1_figure.py` output (hypnos centered, banners Young/Elderly/MCI).
- **F3 whole-scalp violins** (n 36/38/31): peak freq ANOVA **p=0.0020** (η²=0.114; Tukey Y–E p=0.011, Y–MCI p=0.004, E–MCI ns; means Y 0.0199 / E 0.0227 / MCI 0.0232 Hz). **⚠ BANDWIDTH CLAIM FLIPS:** the current caption says "did not reach significance (Kruskal–Wallis p=0.052)" → now **one-way ANOVA p=0.038** (η²=0.062; Tukey **Y–E p=0.045 sig**, Y–MCI p=0.113 ns, E–MCI ns) because all 3 groups now pass normality (test switched KW→ANOVA). Word it as borderline/Young-vs-Elderly-only, not a robust effect (honest framing per [[project_scientific_story]]). AUC whole-scalp Kruskal–Wallis H=2.11, **p=0.348** ns (was 0.535).
- **F4 AUC topo + cluster:** one significant central-parietal cluster **p=0.016, 10 electrodes** (was 9; add E153) = E130,E142,E143,E144,E153,E154,E155,E184,E185,E197. Post-hoc **Young>Elderly 5/10, Young>MCI 7/10, Elderly–MCI 0** (was 7/9, 6/9, 0).
- **F5 ROI normalized-AUC:** ANOVA **p=0.182** ns (was 0.194); means **Young 1.099 / Elderly 1.042 / MCI 1.021**.
- **S1 peak-freq + bandwidth topos:** neither has a significant cluster (unchanged wording; refresh paths to V9).
- **S2 MoCA:** unchanged (elderly+MCI n=32, all |r|≤0.13).
- Also re-sync [[project_paper_figure_set]] (still describes V7) and the [[project_scientific_story]] bandwidth line after the captions land.

See [[project_negative_sigma_fix]], [[project_topo_projection_fix]], [[project_scientific_story]], [[project_files_cleanup_status]], [[feedback_caption_conventions]].
