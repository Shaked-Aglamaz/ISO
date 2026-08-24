# Paper writing — working notes

**Author:** Shaked
**Program:** TAU, Sagol School of Neuroscience, M.Sc.
**Supervisor:** Prof. Yuval Nir
**Working title:** _TBD — decide at end of Phase 1, bring shortlist to supervisor_
**Topic:** Infra-Slow Fluctuations of Sigma Power (ISFS) during NREM2 sleep — young controls vs elderly controls vs MCI patients.
**Pipeline lineage:** Duplicated and extended from Dimitriades 2024.
**Format:** Paper-style (~6000 words, IMRAD, Intro ~1.5 pages). Pivoted from full-thesis format on 2026-05-28.
**Slide deck:** `references/ISFS Project - Shaked Gotfrid.pdf` — figure source + result wording direction.

## Pending verifications (do NOT auto-fix; raise at appropriate moment)
- 4 MCI subjects (SM07, SM13, SM14, VZ9) lack `bad_channels.txt` — Shaked asked to be reminded to verify whether this is intentional before Methods/Results are finalized.

---

## Phase status

- [x] **Phase 0.0** — `CLAUDE.md` audit → `claude_md_audit.md` (92% accurate; parameters table extracted; ISFS expansion error flagged at CLAUDE.md:14; notch-filter timing clarified)
- [x] **Phase 0.1** — folder skeleton + chapter stubs
- [x] **Phase 0.2** — `library.bib` (8 entries from PDFs) + `library_status.md` (verification flags + 20-paper lit-review gap list incl. rodent ISO/ISFS)
- [x] **Phase 0.3** — cohort table from Google Sheet → `cohort_table.md` (3 tabs pulled; 4 quirks flagged for user attention)
- [ ] **Phase 1** — figure selection (with recaps) → `outline.md` → scientific-story paragraph → title shortlist. Slide deck `references/ISFS Project - Shaked Gotfrid.pdf` is the seed.
- [ ] **Phase 2** — paper-format drafts (Methods → Results → Intro → Discussion → Abstract); no separate Conclusion or front matter
- [ ] **Phase 3** — polish + Pandoc export (PDF + .docx) → supervisor handoff. NO TAU template (paper format).

Plan file: `C:\Users\Shaked\.claude\plans\fizzy-inventing-mochi.md`

---

## Glossary

| Term | Expansion |
|------|-----------|
| ISFS | Infra-Slow Fluctuations of **Sigma Power** (NOT "Frequency Shifts") |
| NREM2 | Stage 2 non-rapid-eye-movement sleep |
| ISO | Infra-slow oscillation (general term — superset of ISFS) |
| YC | Young controls |
| EC | Elderly controls |
| MCI | Mild Cognitive Impairment |
| EGI | EEG system (256-channel) |
| GFP | Global Field Power |
| PTP | Peak-to-peak amplitude |
| ROI | Region of interest (here: central-parietal extended 36-channel) |

---

## Open questions for supervisor (Prof. Nir)

_(populate during Phase 1 / 2 — list things to raise at next meeting)_

- Working title: _shortlist to be added_
- Framing of Grollero 2026 in Introduction (concurrent work, postdates this analysis)
- Whether to include MoCA correlation analysis in main text or appendix
- Defense timeline / submission deadline

---

## Reference reading priority

Highest-priority PDFs in `references/`:

1. `ISFS_Development_Dimitriades_2024.pdf` — pipeline source
2. `ISO_sleep_Alzheimer_Grollero_2026.pdf` — concurrent work (cite as related, NOT motivation)
3. `Sleep_Alzheimer_review_Zhang_2022.pdf` — Intro background
4. `REM_slowing_cholinergic_denervation_MCI_André_2025.pdf` (+ supplementary) — MCI biomarker context
5. `Alzheimer_Degeneration_Topography_Cholinergic_BF_Projections_Schmitz_2018.pdf` — AD topography mechanism
6. `Aging_impairs_temporal_spindles_clustering_Niethard_2023.pdf` — aging × spindles
7. `Individualized_temporal_patterns_drive_spindle_timing_Chen_2024.pdf` — spindle dynamics
8. `Age_changes_spindle_memory_consolidation_Champetier_2023.pdf` — aging × cognition

Lab style reference (not for citation): `YaelG_MSc_thesis.pdf` (prior Nir-lab student, mostly English).

Lit-review pass needs to add: **rodent ISO/ISFS work**, NREM2 spindle envelope dynamics, methods refs (Morlet, cluster-based permutation), additional MCI EEG biomarker papers.
