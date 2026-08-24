---
name: project_two_site_cohort
description: "Cohort is two recording sites (TASMC/Ichilov + Sydney/Woolcock); site-from-ID rule, per-group N split, and Sharon-2025 as the TASMC dataset source paper"
metadata: 
  node_type: memory
  type: project
  originSessionId: b3885f36-2e74-41f2-a89b-5994c3bfa6b1
  modified: 2026-08-15T08:25:05.304Z
---

The ISFS cohort pools **two recording sites**, surfaced in Table 1 and Methods §3.1 as of 2026-06-14 (Rotem review):

- **TASMC** = Tel Aviv Sourasky Medical Center (a.k.a. Ichilov), Tel Aviv, Israel.
- **Sydney** = CIRUS Centre for Sleep and Chronobiology, Woolcock Institute of Medical Research, Macquarie University, Sydney, Australia.

**Site-from-ID rule (the only way to tell them apart in the sheet):** a subject is **Sydney** iff its `subject_id` literally contains "MCI" (e.g. `MCI20`, `MCI43`); **everyone else is TASMC**. This holds across BOTH the HE and MCI analysis groups (the "MCI"-prefixed IDs in the HE group are Sydney; see the cohort_table.md quirk note). Young group = all TASMC. Implemented as `_site()` in `code/demographics_table.py`.

**N SPLIT BELOW IS STALE.** Final cohort (verified from the folders 2026-08-15) is **YA 35 / HE 39 /
MCI 30**, split **Young 35 TASMC + 0 Sydney; Elderly 30 + 9; MCI 14 + 16** — that is what Table 1 and
`results/demographics_V3` carry. See [[project_v10_regeneration]]. The old numbers are kept below
only because the surrounding notes reference them.

**Per-group N split (OLD, N=36/38/31):** Young 36 TASMC + 0 Sydney; Elderly 29 TASMC + 9 Sydney; MCI 14 TASMC + 17 Sydney. Shown in Table 1 as a smaller `(TASMC X + Sydney Y)` sub-line under the group total (overlaid via `fig.text` so the total keeps the normal cell font). MoCA exists only for TASMC subjects, and as of 2026-06-15 is **complete for all of them**: all 29 TASMC elderly + all 14 TASMC-MCI now have MoCA. Sydney has no MoCA, which is why MoCA n (**29/14**) < group N (38/31). This site-completeness is exactly what Methods §3.1 now states ("available for all of the Tel Aviv participants in the two older groups, but not for the Sydney participants") — the 2026-06-15 rewording after 11 elderly scores were filled in. See [[project_demographics_tables]] for the updated stats (elderly MoCA 27.2±2.6 n=29; HE-vs-MCI Mann–Whitney U=367.5).

**Dataset provenance paper:** the TASMC participants are from `@sharon2025slowwaves` — Sharon et al., "Slow wave synchrony during NREM sleep tracks cognitive impairment in prodromal Alzheimer's disease", *Alzheimer's & Dementia* 2025, doi 10.1002/alz.70247 (same lab — Yuval Nir; Rotem Falach is a co-author). Added to `library.bib` + logged in `library_status.md`. Cited in Methods §3.1. **Careful: the paper has NO young adults** (its controls are 52–85), so the
"same cohort" claim covers only the two older Tel Aviv groups — §3.1 was corrected accordingly
2026-08-15. It also supplies the Tel Aviv recruitment route, the diagnosing unit, the AHI ≥ 15 apnea
exclusion, the hardware and the scoring montage; it names **no** MCI diagnostic criterion, so neither
do we.

**Technical exclusion criteria in §3.1 are now numeric** (2026-08-15, Yuval's C345): >20% of
electrodes rejected as bad, >30% of N2 time rejected as artifactual, <3 clean N2 bouts, or TST
<210 min. The older "too many bad channels / artifactual epochs / unreliable sleep scoring" wording
is gone. See [[project_methods_revision_yuval]] — the 30% figure is a forward commitment, not a
measured boundary.

**Site facts settled 2026-08-15:** the user confirmed the two centres used **identical setups** and
that sleep was **scored identically at both sites**, so §3.1–3.2 now assert both and carry the Sharon
2025 hardware/montage across sites. But **Sydney recruitment, inclusion/exclusion and MCI diagnosis
are documented nowhere** in the repo and sit as `[TO SUPPLY]` markers in §3.1; site is still inferred
from the subject-ID string, not a recorded field.

Related: [[project_demographics_tables]], [[project_paper_figure_set]], [[reference_subjects_sheet]], [[project_lit_review_parallel]].
