---
name: project_v10_regeneration
description: "V10 / final-cohort (YA35 HE39 MCI30) regeneration 2026-06-17: source-of-truth dirs + full stats; bandwidth flipped sig->ns; manuscript text/figures NOT yet updated"
metadata: 
  node_type: memory
  type: project
  originSessionId: 230bceb3-3eff-4a50-945b-1d56191ff97b
  modified: 2026-08-13T15:03:42.728Z
---

**2026-06-17 full figure/stat regeneration at the FINAL cohort YA 35 / HE 39 / MCI 30 (total 104).** Supersedes [[project_v9_regeneration]] (which was 36/38/31). Cohort changes that triggered it: DS6 re-included (HE 38→39), MR5 processed (already counted, MCI=30 after TST exclusions), HG78/MCI13/SM07 TST-excluded ([[project_cohort_change_mr5_tst210]], [[project_clean_bout_criterion]]).

**Source-of-truth output dirs (V9/V2 preserved, nothing overwritten):**
- Group comparison: `results/group_comparison_results/three_groups_V10` (violins, topos, per-group topos, cluster + ROI stats).
- Demographics/sleep: `results/demographics_V3`.
- MoCA: `results/moca_correlation_V3`.
- Per-subject ISFS: `results/sigma_fix_{YA,HE,MCI}` (now include MR5 in MCI, DS6 in HE). The 3 TST-excluded subjects' result dirs were MOVED out 2026-06-17 to `sigma_fix_YA/_excluded_results/HG78` and `sigma_fix_MCI/_excluded_results/{MCI13,SM07}`, so active dirs hold only the active cohort. **Naming gotcha:** the holding folder is `_excluded_results`, NOT `a_excluded`/`excluded` — those names collide with data-folder subdirs (`control_clean/a_excluded`, `MCI_clean/excluded`) and would make step4's `get_all_subjects ∩ result-exists` filter treat the holding folder as a phantom subject (caught+fixed during this session). Any new holding-folder name must not match a data-folder top-level dir.

**How cohort is selected (verified):** step4/5/6 use `get_all_subjects(data_dir)` ∩ has-`sigma_fix`-result-dir (step4 lines ~1095/1101) + `filter_subjects_by_detection_rate` (no-op, all ≥20%). Active = direct-child folders of `control_clean`/`elderly_control_clean`/`MCI_clean`; HG78/MCI13/SM07 sit in `a_excluded`/`excluded` subdirs so they drop out automatically. Demographics/moca scripts pull live from the Google Sheet `subjects` tab (self-correcting). `aggregate_detection_rates.py` was **refactored 2026-06-17 to the SAME data-dir-first pattern** (iterate `os.listdir(data_dir)` subjects, skip `dashboards`/non-dirs → look up each one's `{subj}_all_channels_summary.csv` in `sigma_fix`; `glob` removed). So ALL pipeline scripts now share one rule: **data folder defines the cohort, `sigma_fix` only supplies the numbers** — self-correcting regardless of stray `sigma_fix` dirs. Verified output unchanged (35/39/30; Young 74.5% / Elderly 85.6% / MCI 82.0% detection; bouts 268/430/325).

**FULL V10 STATS (vs V9 in parens):**
- **Peak frequency (whole-scalp): ANOVA p=0.0026, η²=0.111, SIG** (V9 p=0.002). Means Y 0.0199 < E 0.0226 = MCI 0.0232 Hz; Tukey Y–E p=0.013, Y–MCI p=0.005, E–MCI ns. Robust, headline holds.
- **Bandwidth (whole-scalp): ANOVA p=0.0611, η²=0.054, NOT SIG** (V9 was borderline-sig p=0.038). ⚠️ **FLIPPED sig→ns.** Same direction (Y 0.0236 < E 0.0281 ≈ MCI 0.0276). All 3 groups normal → ANOVA. Now a non-significant trend — matches the honest aging framing.
- **AUC whole-scalp: KW H=1.96 p=0.376 ns** (V9 p=0.35). Unchanged.
- **AUC cluster (F4): one central-parietal cluster p=0.0226, 9 electrodes** (E130/143/144/153/154/155/184/185/197) — V9 was p=0.016, 10 ch (E142 dropped). Post-hoc Y>E 5/9, Y>MCI 7/9, E–MCI 1/9 (E197). Holds, slightly weaker.
- **ROI normalized AUC (F5): ANOVA p=0.143 ns** (V9 p=0.182). Means Y 1.099 > E 1.040 > MCI 1.012.
- **Peak-freq & bandwidth topos (S1): no significant cluster** for either (unchanged).
- **Detection (Results §4.2):** Young 74.5±22.6% (range 20.5–100), Elderly 85.6±21.8%, MCI 82.0±22.2%; all 104 ISFS-present ≥20%; bouts Y 7.7 (268) / E 11.0 (430) / MCI 10.8 (325) = 1023 total.
- **Demographics (T1, demographics_V3):** Young n=35 age 27.1±4.3; Elderly n=39 age 66.5±9.7, MoCA 27.2±2.6 (n=30, DS6 added); MCI n=30 age 67.8±9.1, MoCA 21.5±4.3 (n=14, MR5 added). Exclusion counts (from sheet recorded-reason): YA 4 clean-bouts/4 bad-ch/1 bad-ep/1 TST; HE 2/1/1/0; MCI 3/4/2/3.
- **Sleep/N2 (demographics_V3):** N2 bouts YA 7.7/HE 11.0/MCI 10.8 (KW p=.001); bout length YA 629/HE 561/MCI 501 s (KW p=.002); proportion of N2 sampled equal (p=.98).
- **MoCA (S2, moca_correlation_V3):** pooled n=44 (HE 30 + MCI 14); correlations still null (re-verify exact r/p from the file).

**MANUSCRIPT + FIGURE ALIGNMENT DONE 2026-06-17** (this session). Edited `04_results.md`, `05_discussion.md`, `03_methods.md` (intro had no cohort numbers — unchanged), `figure_manifest.md` (both banners + all figure entries/captions), and appended a V10 changelog section to `thesis/figures_V9_sigma_fix.md`. **Bandwidth narrative now reads "not significant (p=0.061), borderline/sensitive-to-subjects, flipped sig→ns" in Results §4.3 + Discussion opening + Discussion limitations + F3 caption + manifest banner.** Composites regenerated to **new V10 names** (user decision; V9 PNGs preserved): `thesis/figures/{hypno_sleep_stages_V10,f4_auc_composite_V10,s1_topo_composite_V10}.png`. Code edits: `make_topo_composites.py` V9→V10 path + `_V10` output names; `make_f1_figure.py` DEMO_DIR V2→V3 + OUT `_V10`. §4.1 age/MoCA tests recomputed at new cohort: **age Welch t=−0.56 p=0.57; MoCA Mann–Whitney U=375.5 p<0.001** (HE n=30 MoCA vs MCI n=14). Detection overall **80.8%** (Y 74.5 / E 85.6 / MCI 82.0), bouts **1023 total, mean 9.8, range 3–21** (from `aggregate_detection_rates.py`). Verified: grep of chapters+manifest for stale V9 tokens (n=36/38/31, 1013, 0.038, borderline, 10 electrodes, E142, 0.182, n=43, three_groups_V9, demographics_V2, etc.) returns **zero hits**. Nothing committed.

**Superseded for FIGURE ASSETS ONLY (2026-08-13):** the V10 *numbers* are still source-of-truth, but every image was restyled and re-emitted to V11 / `demographics_V4` / `moca_correlation_V4`, and Figure 1's panel-C table became a standalone Table 2 — see [[project_v11_figure_overhaul]]. The V10 composite paths named above are the previous generation.

**Remaining (not done):** T2/T3 standalone stats tables (likely not shown per [[feedback_caption_conventions]]) — **note these were renumbered T3/T4 in 2026-08-13 when the new sleep table took the name Table 2**; F2 unchanged (conceptual); Abstract DONE 2026-06-18 (see [[project_scientific_story]]); stale memories listed in [[project_cohort_change_mr5_tst210]] still cite old N.

**Code/sheet state left this session:** `demographics_table.py` fully sheet-driven (reads `excluded` tab "recorded reason" col; TST<210 category added; EXCL_CATEGORY_OVERRIDES + EXCL_SUBJECTS_TO_DROP removed). `step2_auto_bad_channels.py` main() + `new_iso/main_loop.py` main() are currently configured for the DS6/MR5 pair (group-aware) — reconfigure for future runs. Excluded sheet now has `recorded reason` + `clean_bouts` columns. Nothing committed (safety rule).
