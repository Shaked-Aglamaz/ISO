---
name: project_paper_figure_set
description: "ISFS paper figure set — SUPERSEDED for numbering and assets by project_v11_figure_overhaul (2026-08-17); still the record of the V9/N=36 layout decisions (F4 A/B reversal, yellow rings, green ROI dots, panel-letter style)"
metadata: 
  node_type: memory
  type: project
  originSessionId: bc959033-850f-469c-a7e6-b0f2b0a6b6d6
---

> **★★★ SUPERSEDED 2026-08-17 for numbering, assets and cohort — read [[project_v11_figure_overhaul]] first.**
> The figures were renumbered when the sleep overview moved into Results: **methods flow is now Figure 1
> and the sleep overview Figure 2**, the supplementary set shifted (example spectra S1, peak-freq/BW topos
> S2, MoCA S3), Table 2 was added as a native Docs table, and every asset is V11 at N=35/39/30. The "F1
> sleep overview / F2 methods flow" mapping everywhere below is **wrong now**. What is still valid here:
> the layout and styling decisions — F4's A=raw/B=normalized order, yellow significance rings, green ROI
> dots, table-only T1, and the `A)`/`B)` panel-letter convention.

**★ CURRENT STATE = V9 / N=36 (synced 2026-06-13; everything below this block is V5–V7 history).** Source-of-truth dirs: group comparison `three_groups_V9`, demographics/sleep `demographics_V2` (see [[project_v9_regeneration]]). Locked set = **T1 + F1–F5 + S1–S2**, captions in `thesis/figure_manifest.md` already on V9. Cohort Young **36** / Elderly 38 / MCI 31.

**★★ Rotem-review revisions (2026-06-14), all applied + regenerated:** see [[project_two_site_cohort]] for the TASMC/Sydney work. **T1**: removed the % bad-channel / % bad-epoch rows (numbers moved to Methods §3.2 prose), added a TASMC/Sydney site split under N, made it **table-only** (no suptitle, no footnote — `demographics_table.py` `render_table_png`). **F4**: A/B order **REVERSED** → A=raw (top), B=normalized+cluster (bottom); significant-electrode circles changed black→**yellow** (`step5_topo_comparison.py:1291`; pink was tried first but invisible over the red hotspot, yellow reads over both red & blue); the "Normalized/raw AUC" suptitles removed (kept per-group N, raw mean, and the "Significant pairs" line); ROI dots stay green `#00aa00`. **F2 D-panel** ROI dots are green (caption + `roi_128_to_256_mapping.py` updated red→green). Added Sharon-2025 citation (see [[project_two_site_cohort]] / [[project_lit_review_parallel]]).
- **T1** demographics — `results/demographics_V2/demographics_table.png`.
- **F1** sleep overview — `thesis/figures/hypno_sleep_stages_N36.png` (built by `code/make_f1_figure.py`; hypnos EL3011/MCI27/YS0 + group banners, native pies, `combined_sleep_n2_table.png`).
- **F2** methods flow + ROI — `thesis/figures/methods_flow_roi_v3.png` (built by `code/make_f2_figure.py`, **fully rewritten 2026-06-15** — no more screenshot compositing; see [[project_f2_regeneration]]). ROI map panel D still `code/debug/AUC ROI 2.png`. Older composites (`methods_flow*.png`, `_v2`) preserved.
- **F3** whole-scalp violins — `three_groups_V9/group_comparison_violin.png`.
- **F4** AUC topo + cluster — `thesis/figures/f4_auc_composite.png` (built by `code/make_topo_composites.py`; **A=raw (top), B=normalized+cluster (bottom)** after the 2026-06-14 reversal; sig electrodes = yellow circles, ROI = green dots).
- **F5** ROI normalized-AUC violin — `three_groups_V9/group_comparison_violin_extended_ROI_normalized_auc.png`.
- **S1** peak-freq + bandwidth topos — `thesis/figures/s1_topo_composite.png` (same script; A=peak-freq, B=bandwidth). **S2** MoCA grid `moca_correlation_V1/moca_correlations_grid.png` (unchanged).
- Panel letters everywhere are **A)/B)/…** (paren after letter, in figures + captions).
- **V9 stats (N 36/38/31):** peak-freq ANOVA p=0.0020 (Tukey Y-v-E 0.011, Y-v-MCI 0.004, E-v-MCI ns); **BW now borderline-sig** ANOVA p=0.038 (Y-v-E 0.045 only; was ns at N=35); AUC whole-scalp KW p=0.348 ns; AUC cluster p=0.016, **10 electrodes** (Y>E 5/10, Y>MCI 7/10, E-v-MCI 0); ROI normalized-AUC ANOVA p=0.182 ns (Y 1.099 / E 1.042 / MCI 1.021); peak-freq & BW topos no significant cluster; MoCA all |r|≤0.13.

---

**Figure set restructured 2026-06-03 after supervisor talk** (`thesis/figure_manifest.md`); **renumbered later same day** so demographics is a TABLE (T1) and the figures shifted down one. Final = **Table 1 + 5 main figures + 2 supplementary**: **T1** participant demographics table (`results/demographics_V1/demographics_table.png`, N=35); **F1** sleep overview = 3 hypnospectrograms + sleep-stage pies + sleep/N2 comparison table COMBINED into `thesis/figures/hypno_sleep_stages_N35.png` (hypnospectrogram subject-code titles cropped off; panel letters (a)/(b)/(c) drawn on; hand-made N=34 original kept at `hypno_sleep_stages_2.png`); **F2** ISFS-concept + feature-extraction flow + ROI-definition panel = `thesis/figures/methods_flow_roi.png` (= hand-made `methods_flow.png` with ROI head-map `code/debug/AUC ROI.png` appended; panel-lettered **(a)** spindle timeline, **(b)** raw/sigma/envelope, **(c)** FFT+Gaussian, **(d)** ROI map, with a divider line before (d); NO explanatory text baked in — it's all in the caption. ROI-definition map is NO LONGER dropped — it lives here, and the ROI is also green dots in F4); **F3** whole-scalp violins (`three_groups_V7/group_comparison_violin.png`, headline temporal); **F4** AUC topo + cluster (V7; ROI = green dots on normalized map, headline spatial); **F5** ROI AUC violin normalized only (`three_groups_V7/...extended_ROI_normalized_auc.png`). **Supplementary:** S1 peak-freq + bandwidth topos (V7; confirmed BW not AUC), S2 MoCA grid (unaffected by N=35). **Dropped:** within-group spectra, per-channel peak-freq violin, standalone ROI map (ROI now shown as green dots in F4).

**ROI naming:** never call it "extended" in prose/captions; to readers it is just "the ROI" (pre-defined central-parietal set from Dimitriades 2024 young-adult AUC hotspot). See [[feedback_thesis_prose_rules]].

**Non-obvious gotcha — highest-V rule is PER FIGURE, not per folder:** `three_groups_V6` regenerated only the two violins + stats. The cluster-permutation topo figures (`three_group_topo_auc.png`, `three_group_topo_peak_frequency.png`, `three_group_topo_bandwidth.png`, `three_group_fstat_topo_auc.png`) exist **only in `three_groups_V5`**. So violins → V6, topo-comparison figures → V5. Per-group topographies (`new_*_results/*_topographies_avg_*.png`) are newest at top level (04-29).

**Why:** future drafting must not blindly grab everything from the highest folder. **How to apply:** when citing a topo-comparison figure pull from V5; when citing a violin pull from V6. See [[feedback_versioned_output_dirs]] and [[feedback_thesis_prose_rules]].

**SUPERSEDED (V7 → now V9; see the ★ CURRENT block at top) — historical V7 step, 2026-06-03:** EL3029 added to Young (N 34→35; see [[project_files_cleanup_status]]). Violins + stats regenerated in `three_groups_V7/` — that is now the newest violin source (pull F5/F8 from V7, not V6). Topo-comparison figures still only in V5 (unchanged). V7 violin titles now carry small Gaussian-fit-matching color icons next to each parameter name (purple dot=peak `#A23B72`, orange line=BW `#F18F01`, light-purple rect=AUC `#A23B72` α0.2) — a legend linking violins to the Gaussian-fit figure; may warrant a caption note.

**Cohort NOT frozen:** more subjects may still be added to any group, so every stat below is provisional and can shift again (V6→V7 already flipped BW omnibus). Treat exact p-values/N as the latest snapshot, not final — re-run step6 and re-read the highest-V stats before quoting in the paper.

**V7 stats (N: Young 35 / Elderly 38 / MCI 31):** peak freq ANOVA p=0.0034, η²=0.107 (Y<E=MCI; Tukey YvE 0.0184, YvMCI 0.0053, EvMCI 0.834 ns); **BW Kruskal-Wallis p=0.0519 → now NS (was p=0.025 SIG in V6)** — adding EL3029 (BW=0.044, high) lifted Young BW mean to 0.0238 and the omnibus crossed 0.05, so NO BW post-hoc in V7; AUC whole-scalp KW p=0.535 ns; ROI normalized AUC means Y=1.098, E=1.047, MCI=1.019, ANOVA p=0.194 ns, no post-hoc. The BW flip matters for the headline — re-check before asserting BW group differences. **AUC cluster re-verified at N=35 in V7: p=0.016, 9 electrodes (was 0.014); holds — Y vs E sig at 7/9, Y vs MCI at 6/9, E vs MCI 0.** Peak-freq & BW topos: no significant cluster. MoCA all |r|≤0.13.

Prior V6 stats (N Young 34, verified 2026-05-31, now superseded): peak freq ANOVA p=0.002 (Tukey YvE 0.012, YvMCI 0.0034); BW Kruskal-Wallis p=0.025 (YvE 0.044, YvMCI 0.048); AUC whole-scalp KW p=0.474 ns; ROI normalized AUC ANOVA p=0.194 ns. MoCA pooled E+MCI N=32.

**ROI trend caveat:** the "p≈0.074" recalled for ROI is a direct Young-vs-MCI pairwise from the 2026-04-18 ROI-selection step, NOT in the locked V6 stats. Pending user decision whether to report it as a trend (see [[project_roi_choice]]).

Regeneration status N=35 → V7 (2026-06-03): ✅ DONE via step4+step5 — violins (F4/F6), AUC/peak-freq/bandwidth topos (F5/S1), per-group topographies, topo cluster stats, all in `three_groups_V7`. Code edits: step4 line 1043 + step5 line 1566 bumped V5→V7; step5 got a peak-freq/BW mean-topo plot block after the AUC block (the all-metrics loop was commented out). ✅ DEMOGRAPHICS/SLEEP DONE (2026-06-03): user added EL3029 to sheet "subjects" tab (age 25/F); I filled its 18 derived columns (H–Y) computed from cleaned annotations + bad-channels file via `code/utils/_compute_sheet_fields.py` (method validated to reproduce DG1 exactly: recording_sec=span=max(onset+dur); pct_n1..rem = %of TST; pct_wake & sleep_eff = %of recording; n2_bouts_ge_300 = contiguous-NREM2 merge ≥300s; pct_bad_channels = n_bad/176). Ran demographics_table.py / sleep_stage_pies.py / n2_bouts_table.py / combined_demographics_table.py at N=35, then recomposed F2 via `code/utils/_compose_f2.py` → `thesis/figures/hypno_sleep_stages_N35.png` (hand-made N=34 original `hypno_sleep_stages_2.png` preserved). Demographics now Young n=35, age 27.6±5.0. methods_flow.png (F3) + hypnospectrogram examples + MoCA (S2) need no regen.
