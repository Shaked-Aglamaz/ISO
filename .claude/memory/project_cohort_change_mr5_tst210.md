---
name: project_cohort_change_mr5_tst210
description: "2026-06-16 cohort change — MR5 added to MCI (N 31→32), AD+NT3 removed from excluded sheet, 210-min TST criterion introduced"
metadata: 
  node_type: memory
  type: project
  originSessionId: ff261819-123c-41f6-9f4d-47295686fcb2
---

2026-06-16, to make exclusions defensible to a reviewer:

- Introduced **minimum TST ≥ 210 min (3.5 h)** inclusion criterion (standard PSG/spectral cutoff). Added a `tst_sec` column (+`tst_note` for partials) to the `excluded` sheet tab. **DECISION FINAL: rule applied uniformly — all 3 below-210 included subjects EXCLUDED** (reason "TST < 210 min"): HG78 (162.5, YA), MCI13 (96.5, MCI), SM07 (206.1, MCI borderline kept-uniform). Folders moved active→excluded. SM0017 (41) already excluded, NT3 (132) already removed.
- **MR5 moved from `MCI_clean/excluded/` → `MCI_clean/MR5` (active).** Both demo sources confirm MCI (age 70, F, MoCA 26, MMSE 26), resolving the old "not sure if MCI or control". Added to `subjects` tab row 107 with canonical metrics (TST 340.5 min, 11 bouts ≥300 s). **MCI cohort N 31 → 32.**
- MR5/DS6 bad channels RESOLVED 2026-06-16: empty `{sub}_bad_channels.txt` = genuinely no bad channels (user-approved). Both marked n_bad_channels=0 in subjects sheet (W/X/Y = TRUE/0/0). MR5's stale `bad_channels.txt` (13 ch from an earlier clean) intentionally IGNORED per user.
- Removed from `excluded` tab: 4 AD subjects (AB7, AY1, SK6, YG9 — folders stay in `excluded_AD/`; T1's "AD-tagged" line now gone, no record kept per user) and NT3 (truncated FIF/MFF; folder stays in `excluded/`). Excluded tab now 24 = 9 YA + 5 HE + 10 MCI (aligns with T1).

**FINAL COHORT after all 2026-06-16 changes: YA 35, HE 39, MCI 30 (total 104).** Excluded tab = 26 rows (YA 10, HE 4, MCI 12), restructured by user with `recorded reason` + `clean_bouts` cols (my tst_sec/tst_note cols replaced). YA 36 −HG78 = 35; HE 38 +DS6 = 39; MCI 31 +MR5 −MCI13 −SM07 = 30. DS6 RE-INCLUDED (7 clean bouts, 40.5% N2 — "not enough N2" was wrong); folder moved to active, added to subjects row 105, MoCA 28/age 68/M. Excluded reasons standardized (user-added `recorded reason` col) to: Too many bad channels / Not enough clean bouts / Too many bad epochs / TST < 210 min.

**Status of follow-ups — ALL DONE 2026-06-17 (see [[project_v10_regeneration]]):**
- `demographics_table.py`: fully sheet-driven now (reads `recorded reason` col; TST<210 category added; EXCL_CATEGORY_OVERRIDES + EXCL_SUBJECTS_TO_DROP both removed — vestigial cleanup completed).
- MR5 + DS6 ISFS via main_loop: DONE (step2 0 bad ch each + main_loop → sigma_fix_MCI/MR5 100% 11 bouts, sigma_fix_HE/DS6 92.6% 7 bouts). MR5 sheet bad-ch = TRUE/0/0.
- Re-run at new N (35/39/30): DONE → `three_groups_V10`, `demographics_V3`, `moca_correlation_V3`, detection rates. Full V10 stats in [[project_v10_regeneration]]. **NEXT (new session): edit manuscript text + composites to V10; bandwidth now ns.**
- Stale memories/docs to reconcile LATER: [[project_files_cleanup_status]] (34/38/31), [[project_demographics_tables]] (YA36/HE38/MCI31), [[project_scientific_story]] (N=36), [[project_two_site_cohort]] (N splits), [[project_pending_verifications]] (SM07 now excluded). Doc `thesis/low_bout_and_excluded_n2_table.md` is up to date.
