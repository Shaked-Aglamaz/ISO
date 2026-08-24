---
name: project_thesis
description: "Project metadata — TAU Sagol M.Sc. under Yuval Nir, paper-format write-up (pivoted from full thesis 2026-05-28), pipeline lineage Dimitriades 2024, Grollero 2026 timing, lit-review scope"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e0bf61-82d0-4afc-bb1a-ad690a607b82
---

Writing project context started 2026-05-27; format pivoted 2026-05-28.

**Program:** TAU, Sagol School of Neuroscience, M.Sc. Output is **paper-format**, not full thesis — see [[feedback_paper_format_pivot]].
**Supervisor:** Prof. Yuval Nir.
**Topic:** ISFS (see [[project_isfs_definition]]) during NREM2 sleep — 3-group comparison of young controls, elderly controls, MCI patients.
**Language:** English; no Hebrew abstract.

**Pipeline lineage:** The analysis pipeline (`code/new_iso/`) is duplicated and extended from Dimitriades 2024 ("ISFS Development"). That paper must be cited as the methods origin, with our adaptations explicitly noted.

**Grollero 2026 ("ISO sleep Alzheimer"):** Published AFTER the user's analysis was complete. Topically very close (ISO in sleep + Alzheimer's). Cite as concurrent/related work in Discussion (its only home) — **never as motivation**, since it postdates the work. DOI confirmed by user: `https://doi.org/10.64898/2026.04.09.717425` (unusual `10.64898` prefix is correct).
**IMPORTANT — it is an EMPIRICAL study, NOT a hypothesis/proposal paper** (despite the question-style title and the misleading one-line bib note + earlier memory wording). Read the local PDF `thesis/references/ISO_sleep_Alzheimer_Grollero_2026.pdf` (2026-06-14). What they did: RESTED-AD cohort, **10 clinically-diagnosed AD + 20 age-matched healthy controls** (AD, not MCI; small N), home few-channel headband (Dreem-2, frontal-central proxy for centro-parietal — NO topography), same ISO pipeline family (Morlet sigma envelope → infra-slow spectrum → Gaussian fit, peak freq/amplitude/bandwidth), plus plasma AD-biomarker (Aβ42/40, pTau, NfL, GFAP) + word-list memory correlations. **Findings:** ISO peak **amplitude reduced in AD**, **frequency & bandwidth preserved**; amplitude tracked Aβ42/40; bandwidth tracked GFAP/NfL + poorer memory. **Convergence with our work:** their amplitude reduction ≈ our focal central-parietal AUC reduction (both = weakening of strength; ours tracks healthy aging, theirs separates AD from aged controls); frequency separates neither clinical contrast (our faster-freq is an aging axis they can't see, no young group); their AD-positive + our MCI-null ⇒ disease-specific weakening may emerge later than MCI. Discussion §5 paragraph rewritten 2026-06-16 to reflect this (was wrongly framed as "hypothesis not empirical").
**METRIC-MAPPING CAVEAT (user-caught, keep precise in any comparison incl. the Abstract):** our three ISFS params are peak frequency / bandwidth / **AUC** — we have **NO separate "amplitude" parameter**. Grollero reports peak **amplitude**. So never write "we found amplitude reduced"; map their amplitude to our **AUC** under the umbrella term "the rhythm's strength", and keep contrasts apples-to-apples at the disease-vs-age-matched-control level (their AD<controls on amplitude ↔ our MCI=elderly on AUC, i.e. our analogous strength measure did NOT separate the disease contrast). Our AUC reduction is a focal central-parietal CLUSTER and tracks AGING (young>elderly=MCI); whole-scalp AUC preserved.

**Lit-review scope:** Must include rodent ISO/ISFS literature alongside human work.

**Project artifacts:**
- Slide deck (project overview) — **local PDF** at `thesis/references/ISFS Project - Shaked Gotfrid.pdf`. See [[reference_project_slide_deck]]. The Google Slides URL referenced earlier is auth-walled; use the local PDF.
- Cohort source of truth: Google Sheet (see [[reference_subjects_sheet]]) — primary, not the local MDs. Pulled into `thesis/cohort_table.md` on 2026-05-28.
- Lab-style reference: `thesis/references/YaelG_MSc_thesis.pdf` (prior Nir-lab student, mostly English).

**Phase 0 state (2026-05-28):** complete. Skeleton, CLAUDE.md audit, bibliography (8 entries), cohort table all on disk under `thesis/`. CLAUDE.md line 14 ISFS expansion fixed.

**Naming clarifier (do NOT mention in prose):** Several subjects in the elderly-control group (group=HE) have IDs prefixed `MCI` (MCI20, MCI25, etc.) due to recruitment renaming. Label artifact, not a group inconsistency — treat as a non-issue, do not surface in the paper.

**How to apply:** Use as background context whenever drafting paper content, choosing references, or framing the contribution.

Related: [[project_isfs_definition]], [[reference_subjects_sheet]], [[reference_project_slide_deck]], [[feedback_thesis_prose_rules]], [[feedback_paper_format_pivot]], [[project_pending_verifications]], [[project_roi_choice]]
