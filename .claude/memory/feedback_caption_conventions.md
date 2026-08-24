---
name: feedback_caption_conventions
description: "ISFS paper figure/table caption conventions (locked 2026-06-05): format, descriptive titles, stats-in/claims-out, define-ROI-once"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bd4601a8-1b08-4858-8da1-0ce08627cea2
---

Caption conventions for the ISFS paper, locked with the user 2026-06-05 while finalizing `thesis/figure_manifest.md` (T1, F1–F5, S1, S2).

**Format:** `*Label. Title.* body.` — the label+title are italic and inline with the body (e.g. `*Figure 3. Whole-scalp ISFS parameters across groups.* Each dot is one subject's mean…`). Labels: `Table 1.`, `Figure 1.`–`Figure 5.`, supplementary `Figure S1.` / `Figure S2.` (compact form, not "Supplementary Figure 1").

**Panel letters use the `A)` form** (capital letter + closing paren), both drawn on the figure and in the caption body — e.g. "A) Whole-night hypnogram… B) Proportion of each stage…", and cross-references inside the caption match ("the window expanded in B)"). Not `(a)`, not lowercase. (Locked 2026-06-13 while recomposing F1/F2/F4/S1.)

**When a figure already contains a stats panel** (e.g. an embedded comparison table), the caption gives a compact significance phrase and defers to it rather than re-printing every value — e.g. "reduced in older groups (both p < 0.001; per-stage statistics in panel C)". Still self-contained, but no duplication of an on-figure table.

**Titles are descriptive, not declarative** — they name *what the figure shows*, never state the finding (e.g. "Whole-scalp ISFS parameters across groups", NOT "ISFS is faster in aging"). The body must NOT echo the title.

**Stats: self-explanatory but NOT a numeric dump (revised 2026-06-22).** Earlier (2026-06-05) the rule was "all p-values stay in the caption" because there was no stats table to hold them. Revised after checking the scientific-writing skill's figures/tables convention (self-explanatory + avoid-redundancy): a caption keeps the test name, n, dispersion meaning, the significance-marker legend (e.g. "* p<0.05, ** p<0.01"), the qualitative pattern, and **at most the headline p-value** — but the FULL numerics (every post-hoc p, every group mean±SD, per-electrode counts) move to the Results text and are deferred ("full statistics are reported in the Results"). Applied to T1/F2/F3/F4/F5 in `figure_manifest.md`; verified each deferred number was already present in Results §4.1/4.3/4.4/4.5 BEFORE removing it from the caption. Claims still OUT (interpretation lives in Results/Discussion). **Table captions**: describe the table + define abbreviations + state "mean ± SD"; do NOT restate the cell values (T1 trimmed accordingly).
- On the figure itself (not the caption): F3/F5 violins had per-group stat boxes + bracket p-value text drawn on the plot — these were removed via the `paper_style=True` flag in `plot_group_comparison` (see [[feedback_doc_figure_legibility]]); group N moved into the x-tick labels ("Young (N=35)").

**Define the ROI once.** Full provenance ("central-parietal electrodes defined a priori from the young-adult AUC hotspot of Dimitriades et al. 2024") lives in Methods + F2(d) only. F4/F5 refer to it briefly as "the pre-defined central-parietal ROI" — do not repeat the full definition/citation in every legend. (Same logic for any term: define at first mention, reference by name after.)

**Humanizer caveat:** when running the humanizer skill on captions/Methods/Results prose, apply only its AI-*tell* removal (em-dashes, significance inflation, copula avoidance, filler, rule-of-three). Its "add first-person voice / soul" half is for essays, NOT scientific legends — keep the neutral, precise register.

**How to apply:** use these rules for all manuscript figure/table captions in Phase 2, and when revising prose generally. Still obey [[feedback_thesis_prose_rules]] (only "the ROI", never "extended"; pull figures from highest-V dir) and keep framing honest per [[project_scientific_story]].

Related: [[project_paper_figure_set]], [[feedback_thesis_prose_rules]], [[project_scientific_story]], [[feedback_answer_before_acting]]
