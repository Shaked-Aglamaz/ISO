---
name: project_lit_review_parallel
description: "Literature-review pass for the ISFS paper Intro — run in a SEPARATE parallel session while another session drafts Methods; scope, output, citekey convention, no-overlap rule"
metadata: 
  node_type: memory
  type: project
  originSessionId: 10d13e89-eb06-49fb-9325-3482985949df
---

Decided 2026-06-07: while the bug-fix re-run (`results/sigma_fix_*`, see [[project_negative_sigma_fix]]) is in flight and Results is blocked, work is split across two sessions. **This memory governs the parallel literature-review session.** The other session drafts **Methods** ([[project_paper_figure_set]] / writing plan order) — the lit-review session must NOT touch `thesis/chapters/` to avoid edit collisions.

**Goal:** expand `thesis/references/library.bib` so the Introduction isn't reference-bottlenecked. Currently **8 confirmed entries**; target **+20–40**.

**Update 2026-06-14:** +2 more added during Intro drafting (in the main drafting session, not this one): `molle2011fastslow` (slow frontal vs fast centroparietal spindle distinction) and `lazar2019infraslow` (first human ISO-in-spindles demo, cited before Dimitriades). **Bib now 27 entries**, both web-verified, logged in a dated block in `library_status.md`. Intro is now DRAFTED — see [[project_scientific_story]] Intro-decisions block.

**Update 2026-06-14 (later, Rotem review):** +1 → **`sharon2025slowwaves`** (Sharon et al., *Alzheimer's & Dementia* 2025, doi 10.1002/alz.70247) — the **source paper for the TASMC/Ichilov dataset** used here (same lab, Nir; Rotem Falach is a co-author). **Bib now 28 entries**, logged in `library_status.md`, cited in Methods §3.1. See [[project_two_site_cohort]].

**DONE 2026-06-07 (lit-review pass complete per user):** **+17 verified entries appended → 25 total in `library.bib`**, all Crossref-checked, braces balanced (233/233), keys unique. User scoped it gap-list-only (no backfill to the original +20–40 target) + Methods=tools-actually-used. Dropped gap items #6 (not concrete), #19 Cohen textbook, #20 EEGLAB (not used in pipeline). Notable seed fixes logged in `library_status.md` (full per-key table there): Liu was 2021→**2020** (key now `liu2020spindlebiomarkers`, real DOI 10.1007/s11325-019-01970-9; seed vol/pages belonged to a different paper); Osorio-Forero missing Devenoges; Cardis missing 2nd author Lecci; several "et al." expanded to full author lists.

**Intro drafting NOT started** (decided 2026-06-08): held off deliberately — (1) this session must not touch `thesis/chapters/` while the other session drafts Methods; (2) writing order is Intro-after-Methods; (3) Results framing gated on the `sigma_fix_*` full-cohort re-run ([[project_negative_sigma_fix]]) being locked. Green-light the Intro once Methods is drafted/closed AND sigma-fix results are locked. When writing it, apply: AGING-not-MCI honest framing, Grollero=concurrent (never motivation), extended ROI only, no naming user's own scripts ([[feedback_no_code_names_in_prose]]).

**Still open before final submission:** re-verify the 3 preprint entries (Dimitriades, André, Grollero) for peer-reviewed versions.

**Seed already on disk:** `thesis/references/library_status.md` has a curated gap list with best-guess metadata — including the rodent ISO core (Lecci 2017 Sci Adv, Osorio-Forero 2021 Curr Biol, Kjaerby 2022 Nat Neuro, Cardis 2021 eLife, Fernandez & Lüthi 2020 Physiol Rev). Start there.

**Scope (per writing plan):**
- Human infra-slow oscillations in sleep (background)
- **Rodent ISO/ISFS literature — REQUIRED** (supervisor instruction, see [[project_thesis]])
- NREM2 spindles & sigma-envelope dynamics
- MCI/AD sleep-EEG biomarkers (human + rodent models)
- Methods refs: Morlet wavelet, cluster-based permutation tests, MNE-Python

**Conventions:**
- Cite-key: `AuthorYEARkeyword` (e.g. `dimitriades2024isfs`, `grollero2026iso`).
- Verify every DOI/citation before importing — do not import gap-list suggestions blind.
- Output: append to `thesis/references/library.bib`; log status/verification flags in `library_status.md`.
- Framing constraints still apply: Grollero 2026 = concurrent/convergent, never motivation; keep AGING-not-MCI honest ([[project_scientific_story]]).
- Skills: `literature-review` primary; `academic-researcher` / `deep-research` as needed.

**Hand-back:** when done, the bib + status file are ready for the Intro draft (which happens after Methods). Re-verify preprint entries (Dimitriades, André, Grollero) for peer-reviewed versions before final submission.

Related: [[project_thesis]], [[feedback_paper_format_pivot]], [[project_scientific_story]], [[project_negative_sigma_fix]]
