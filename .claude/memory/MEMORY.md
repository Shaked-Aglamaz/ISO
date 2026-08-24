# Memory Index

## Defense (current work)
- [project_defense_deck.md](project_defense_deck.md) — deck = `code/defense/build_deck.py` → `thesis/defense/ISFS_defense_V2.pptx` (54 slides/45 min); few-full-sentences rule, slide figures in `results/defense_slides_V1`, QA via PowerPoint COM, never overwrite while he has it open
- [project_defense_examiners.md](project_defense_examiners.md) — **examiners = Rivi Tauman, Michal Kahn, Yuval Nir**; what each will probe; prep in `thesis/defense/qa_prep.md`
- [reference_champetier_isfs_aging.md](reference_champetier_isfs_aging.md) — **Champetier 2023 already fit the ISFS peak in aging (C3/C4) and found NO group difference**; closest prior work to our headline, have the answer ready

## Thesis status
- [project_final_pass_sent.md](project_final_pass_sent.md) — **V2 SENT TO YUVAL 2026-08-19**; Sydney `[TO SUPPLY]` markers + Dimitriades preprint are intentional, don't re-flag; §5.3 ref error still in `chapters/05_discussion.md:33`
- [project_front_matter_pass.md](project_front_matter_pass.md) — last content pass DONE 2026-08-17; all 9 emails + 17 comments closed; TOC in + sent; only the reply to Yuval is left
- [project_yuval_review.md](project_yuval_review.md) — he reviewed V1; triage `thesis/reviews/yuval_review_triage.md`; **no reply draft exists**; his C395 re-run demand is refuted (MNE excludes bads from avg ref)
- [project_scientific_story.md](project_scientific_story.md) — locked 4-sentence story on V10 stats; AGING-not-MCI framing; never write "ISO" in prose
- [project_thesis.md](project_thesis.md) — TAU Sagol M.Sc. under Yuval Nir; pipeline from Dimitriades 2024; Grollero 2026 concurrent, not motivation
- [project_pre_send_check.md](project_pre_send_check.md) — 2026-08-12 pre-send check; numbers/figures/DOIs verified, 4 fixes; record `thesis/final_check_report.md`
- [project_flavio_review.md](project_flavio_review.md) — 66 comments applied 2026-08-06 to the doc copy only; humanizer done, Bayesian still open

## Revision passes (all closed)
- [project_intro_revision_yuval.md](project_intro_revision_yuval.md) — Intro DONE 2026-08-16; refs 30→49, Intro refs 1–39 permanent; `discussion_edits_before_after.md` §9 VOID
- [project_methods_revision_yuval.md](project_methods_revision_yuval.md) — Methods DONE 2026-08-15; traps: 30% bad-epoch criterion aspirational, SM07 pins the 210-min rule, `n2_bouts_ge_300` doesn't split around BAD
- [project_results_revision_yuval.md](project_results_revision_yuval.md) — Results DONE 2026-08-15 + whole-thesis aMCI relabel; Dimitriades supports §4.5, Grollero cannot; Table 2 rounding bug fixed
- [project_discussion_revision_yuval.md](project_discussion_revision_yuval.md) — Discussion DONE 2026-08-17; refs 49→63; Grollero reports peak **amplitude** not AUC; the LC does not release ACh
- [project_v11_figure_overhaul.md](project_v11_figure_overhaul.md) — figures DONE + in the Doc 2026-08-17: numbering swapped (methods flow = Fig 1), Table 2 native, all figures say aMCI; stats still V10
- [project_c429_c390_results.md](project_c429_c390_results.md) — ANCOVA + sleep stats (three_groups_V11 / demographics_V4): peak-freq survives (p=.0034), **bandwidth is a duration artefact** (r=.386)
- [project_bibliography_expansion.md](project_bibliography_expansion.md) — library.bib 31→84 verified; `thesis/references/new_refs_annotated.md` = ref→section map; the C557 LC review is mined but NOT cited

## Stats & cohort of record
- [project_v10_regeneration.md](project_v10_regeneration.md) — **source of truth: YA35/HE39/MCI30 → three_groups_V10, demographics_V3, moca_correlation_V3**; peak-freq p=0.0026, BW ns p=0.061, AUC cluster p=0.023/9ch, ROI ns p=0.143
- [project_cohort_change_mr5_tst210.md](project_cohort_change_mr5_tst210.md) — 2026-06-16 exclusions overhaul → cohort 104; 210-min TST rule; MR5+DS6 in, AD+NT3 out; excluded=26
- [project_clean_bout_criterion.md](project_clean_bout_criterion.md) — ≥3-clean-bout inclusion rule; BAD-split recount fixed NE32/AH3/MCI11/MCI12; DS6 added (HE 38→39)
- [project_two_site_cohort.md](project_two_site_cohort.md) — TASMC + Sydney; site-from-ID rule; N split 35+0 / 30+9 / 14+16; Sharon-2025 = dataset source paper (no young adults); Sydney recruitment undocumented
- [project_no_namci_sensitivity.md](project_no_namci_sensitivity.md) — aMCI-only sensitivity run REJECTED, full MCI kept; results in `results/no_naMCI`
- [project_dropped_20pct_criterion.md](project_dropped_20pct_criterion.md) — 20%-ISFS exclusion dropped thesis-wide; EL3017/18/21 re-presented as bad channels
- [project_v9_regeneration.md](project_v9_regeneration.md) — SUPERSEDED by V10; historical V9 checklist + option-B topo clip detail
- [project_files_cleanup_status.md](project_files_cleanup_status.md) — 196-ch backup, RS5/RY42 anomalies, annotation dedup; **its per-group counts are superseded**
- [project_pending_verifications.md](project_pending_verifications.md) — resolved 2026-06-17; nothing open

## Pipeline & analysis
- [project_isfs_definition.md](project_isfs_definition.md) — ISFS = Infra-Slow Fluctuations of **Sigma Power**; **the same thing is published as ISO — always search both names and both spellings**
- [project_central_parietal_roi.md](project_central_parietal_roi.md) — core (20-ch) + extended (36-ch) ROI in 256-ch EGI, from the 128-ch YA_AUC hotspot
- [project_roi_choice.md](project_roi_choice.md) — extended ROI chosen 2026-04-18; default to it in new group-comparison code
- [project_group_comparison.md](project_group_comparison.md) — ANOVA/Tukey, cluster permutation, ROI violins; normalized-topo vs raw-violin interpretation pitfall
- [project_isfs_method_vs_dimitriades.md](project_isfs_method_vs_dimitriades.md) — ours is stricter than the reference MATLAB (extra AUC>0 gate, keep it); their rates not directly comparable
- [project_negative_sigma_fix.md](project_negative_sigma_fix.md) — |sigma| fix 2026-06-06; full re-run → `results/sigma_fix_{YA,HE,MCI}`; main_loop is single-subject now
- [project_ry42_projector_fix.md](project_ry42_projector_fix.md) — inactive avg-ref SSP projector in some `_bad-epochs.fif` (plot≠get_data); RY42 fixed, CH53 still has it
- [project_missing_channel_handling.md](project_missing_channel_handling.md) — **3 different NaN policies**; cluster-electrode missingness is balanced, so B2 was fixed by deleting the claim
- [project_raw_topo_aggregation_consistency.md](project_raw_topo_aggregation_consistency.md) — step4/step5 raw topos share `code/utils/topo_aggregation.py` so displayed means match step6
- [project_topo_projection_fix.md](project_topo_projection_fix.md) — EGI-256 topo `sphere='auto'` + option-B head clip (`clip_topo_to_head`, `extrapolate='head'`)
- [project_stage_label_dialects.md](project_stage_label_dialects.md) — `cleaned_annotations.txt` has two dialects (Wake/NREM2 vs WAKE/N2); normalize before counting
- [project_unknown_vs_acq_skip.md](project_unknown_vs_acq_skip.md) — `UNKNOWN`(-1) vs `BAD_ACQ_SKIP` are independent; exclude -1 from numerator AND denominator; rebuild hypnograms from `*_cleaned_annotations.txt`, not `ISO_data/scoring/`
- [project_new_mci_subjects.md](project_new_mci_subjects.md) — new MCI subjects: chunked FIF concat, hypno naming, all processed through main_loop
- [project_hypnospectrogram.md](project_hypnospectrogram.md) — hypnospectrogram.py (sleepeegpy, E101) + inspect_hypnos.py; hypno_freq and letter/int gotchas
- [project_lit_review_parallel.md](project_lit_review_parallel.md) — the 2026-06 lit review (27 entries); superseded in scope by the bibliography expansion

## Figures & tables
- [project_f2_regeneration.md](project_f2_regeneration.md) — methods-flow figure rebuilt 2026-06-15 (`make_f2_figure.py` → `methods_flow_roi_v3.png`); RD43 VREF, panel roles, styling knobs
- [project_demographics_tables.md](project_demographics_tables.md) — sleep pies / n2 bouts / combined table, sheet-driven; latest demographics_V3 (V4 holds Table 2)
- [project_paper_figure_set.md](project_paper_figure_set.md) — superseded for numbering by V11; still the record of V9 layout decisions + the EL3029 sheet-fill method
- [feedback_caption_conventions.md](feedback_caption_conventions.md) — `*Label. Title.* body`, `A)` panel letters, stats-in/claims-out, define-ROI-once
- [feedback_doc_figure_legibility.md](feedback_doc_figure_legibility.md) — Doc figures must be portrait/large-font (page-width shrink); F3/F5 `paper_style`, F4/S1 single colorbar
- [feedback_displayed_mean_independent.md](feedback_displayed_mean_independent.md) — group topo "Mean=X" must be mean-of-subject-means, never `np.nanmean(group_topo)`
- [feedback_versioned_output_dirs.md](feedback_versioned_output_dirs.md) — always write group-comparison outputs to `_V{X}` dirs, never the unversioned base

## Working style (his corrections)
- [feedback_thesis_prose_rules.md](feedback_thesis_prose_rules.md) — prose rules incl. §9 sentence-level style; **§6 em dashes zero document-wide: grep U+2014 only, never U+2013, never Ju et al.'s title**
- [feedback_answer_before_acting.md](feedback_answer_before_acting.md) — diagnostic questions want an answer in chat, not a preemptive edit
- [feedback_diagnose_dont_fix.md](feedback_diagnose_dont_fix.md) — investigating a bug: present findings and stop until he confirms direction
- [feedback_match_effort_to_problem.md](feedback_match_effort_to_problem.md) — config/auth issues: ask what he clicked or what the error said; don't pre-write both branches
- [feedback_doc_edits_review_first.md](feedback_doc_edits_review_first.md) — never edit the manuscript Doc directly: before/after into an .md, wait for approval; `[TO SUPPLY]` for missing facts
- [feedback_stop_if_gdoc_unreadable.md](feedback_stop_if_gdoc_unreadable.md) — docs MCP missing/erroring → say so immediately; never answer from local chapters instead
- [feedback_verify_named_data_source.md](feedback_verify_named_data_source.md) — if you substitute an input file for the one a note names, verify equivalence cohort-wide and report it
- [feedback_no_code_names_in_prose.md](feedback_no_code_names_in_prose.md) — never name his scripts/functions in prose; external packages are fine
- [feedback_normalization_scope.md](feedback_normalization_scope.md) — normalize over ALL channels first, then restrict to ROI, else the ROI mean is ~1 by construction
- [feedback_edit_main_loop_directly.md](feedback_edit_main_loop_directly.md) — ad-hoc ISFS re-runs: edit `main_loop.py` config in place, no separate driver
- [feedback_auto_cleaning.md](feedback_auto_cleaning.md) — step1 auto-cleaning: prefer conservative/narrow epochs; he extends them manually
- [feedback_parallel_session_lanes.md](feedback_parallel_session_lanes.md) — parallel sessions: partition by file ownership; **only ONE session may write to the manuscript Doc**
- [feedback_document_carryover.md](feedback_document_carryover.md) — when a pass ends, write every cross-session finding into the file the next session will open
- [feedback_paper_format_pivot.md](feedback_paper_format_pivot.md) — reverted to FULL THESIS 2026-08-12; the paper-format decision is historical only

## Environment
- [feedback_pythonioencoding.md](feedback_pythonioencoding.md) — always `PYTHONIOENCODING=utf-8` for python on this Windows machine
- [feedback_write_code_with_edit_not_heredoc.md](feedback_write_code_with_edit_not_heredoc.md) — **don't patch code with `python - <<EOF`**: backslash escapes and backticks get mangled; use Write/Edit
- [reference_manuscript_gdoc.md](reference_manuscript_gdoc.md) — **current doc = "Shaked's Thesis V2" id 1YpXrDGFlzRk...**; readDocument hides table cells; 403 = scope not sharing; `invalid_grant` = re-run `npx @a-bonus/google-docs-mcp auth` (tick Drive) then restart Claude Code
- [reference_google_sheets_mcp.md](reference_google_sheets_mcp.md) — sheets MCP gotchas: sharing options, xlsx-in-Drive failure, write workflow, Hebrew sex translations
- [reference_subjects_sheet.md](reference_subjects_sheet.md) — subjects Google Sheet (ID + 3 tabs) + related source sheets + ID naming conventions

## References
- [reference_dimitriades_citation_status.md](reference_dimitriades_citation_status.md) — dev study IS published (Sci Rep 2026) **but he declined the upgrade: the sent thesis cites the 2024 preprint, leave it**; André → Mol Psychiatry 2026
- [reference_sun2026_mecfs_not_miscitation.md](reference_sun2026_mecfs_not_miscitation.md) — `sun2026longcovid` looks mis-cited in §5.4 but is CORRECT (separate ME/CFS arm); check `library_status.md` before flagging by title
- [reference_isfs_frequency_attribution.md](reference_isfs_frequency_attribution.md) — Lázár 2019 = ~0.01 Hz (sigma power) / ~0.02 Hz (spindle events); the 0.02 Hz is Lecci 2017
- [reference_yael_gat_thesis.md](reference_yael_gat_thesis.md) — Yael Gat M.Sc. 2023, same lab: the template Yuval points at for Intro structure; its aMCI-as-prodromal-AD framing is what C571 reverses
- [reference_rd43_alias_and_example_gaussians.md](reference_rd43_alias_and_example_gaussians.md) — "26" = RD43 alias (not EL3026); example_gaussians style/location; mne_qt_browser scale factors
- [reference_project_slide_deck.md](reference_project_slide_deck.md) — project overview deck PDF at `thesis/references/ISFS Project - Shaked Gotfrid.pdf` (Slides URL is auth-walled)
