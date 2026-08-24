---
name: project_pre_send_check
description: "Final pre-submission check of the manuscript (2026-08-08) — what was verified clean, the 4 fixes applied, and the ranked list still pending; draft SENT to Yuval 2026-08-12"
metadata: 
  node_type: memory
  type: project
  originSessionId: 43b7471d-917d-4500-8cda-57a99d1429ef
  modified: 2026-08-12T10:13:36.188Z
---

> **Yuval's review came back 2026-08-11 — on V1, not this version.** See [[project_yuval_review]]. Two of
> his comments independently reproduce **S7** and **S11** from the pending list below, so that list is now
> part of the same revision rather than a separate track.

**The draft was SENT to Yuval (PI) on 2026-08-12**, in the state described below. Doc = "Shaked's Thesis V2"
(`1YpXrDGFlzRk…`, see [[reference_manuscript_gdoc]]), last modified 2026-08-08 19:42.

A full pre-submission check ran 2026-08-08 (numbers-vs-source audit, figure/table render check, 29-reference
DOI + claim verification, `scientific-writing` + `scientific-critical-thinking` review, whole-manuscript
`humanizer` pass). **The complete itemised record is `thesis/final_check_report.md`** — read that file rather
than re-deriving; this memory is the index to it.

**Verified clean (do not re-audit):** every number in the doc matches the V10 sources (whole-scalp stats,
the 9 cluster electrodes + post-hoc counts 5/7/1, ROI, demographics, MoCA, 1023 bouts, detection rates, all
wavelet/fit parameters); Table 1 matches `demographics_V3` cell-for-cell; all 7 figures are embedded, in order,
and are the V10 assets; all 29 DOIs resolve; no open comments.

**Applied before sending (2026-08-08):**
1. **B2** — the false claim in Results 4.3 ("interpolated … never for the statistics") was **deleted**, not
   explained; the F5 caption's "the **fitted** electrodes of the ROI" lost "fitted". See
   [[project_missing_channel_handling]].
2. **S3** — Niethard (an **editorial** on Champetier) demoted: out of the Intro bracket (`¹²⁻¹⁵`→`¹²⁻¹⁴`), kept
   in the Discussion as `¹²,²⁸`. Reference list renumbered old 16–28 → 15–27, Niethard → 28, Grollero 29.
   **All superscripts were re-checked against the new list and are consistent.**
3. **S5** — Methods 3.1 exclusion list gained the fourth criterion Table 1 already showed: "or a total sleep
   time under 210 min" (1 young, 3 MCI). Mirrored to `thesis/chapters/03_methods.md` (v5 note).
4. **S8** — Figure 1 panel C caption corrected: "shorter bouts" holds only for MCI (elderly vs young p = 0.154).

**PENDING — ranked, none applied.** (Numbers keyed to `thesis/final_check_report.md`.)
- **B1 + S2 — citations now out of date.** Dimitriades → *Sci Rep* 18 Jun 2026; André → *Mol Psychiatry*
  12 May 2026. Details + the in-text "(2024)"→"(2026)" spots in [[reference_dimitriades_citation_status]].
- **S1 — the two-site confound is never checked.** Young 35 TASMC/0 Sydney, Elderly 30/9, **MCI 14/16**. The
  elderly group is the free test bed (30 vs 9). The young-vs-elderly peak-frequency effect is largely
  within-TASMC and therefore the least site-exposed contrast. See [[project_two_site_cohort]].
- **S4** — no ethics approval, informed consent, funding, COI, or data/code availability statements.
- **S6** — MCI is never defined diagnostically (no criteria, no diagnoser, Sydney path undescribed, AD
  exclusion unstated).
- **S7** — Results 4.1's "N2 remained the largest stage, so the differences don't reflect N2 amount" is a
  non-sequitur and omits that N2 share *does* differ (p = 0.0009). **The S8 caption fix has made the caption
  more accurate than the paragraph it supports.** The supporting facts are proportion-of-N2-analyzed p = 0.98
  and total bout duration p = 0.081.
- **S9** — the Lázár sentence contradicts itself; real values (verified in the PDF) are ~0.01 Hz sigma power,
  ~0.02 Hz spindle events. See [[reference_isfs_frequency_attribution]].
- **S10** — "lower AUC in the older groups" must say **normalized**, else it reads as contradicting 4.2.
- **S11** — reproducibility gaps: who scored sleep, bad-channel/epoch criteria, ROI size (36 ch), topo tests ran
  on 175 not 176 channels, baseline is 0.06–**0.102** Hz.
- **S12/S13** — drop the unquantified "sensitive read-out/marker" and the causal "driven by aging"; add one
  sentence on the three uncorrected whole-scalp omnibus tests (peak freq survives Bonferroni anyway).
- **Humanizer suggestions delivered but NOT applied** (full before→after list in the report): nine
  "rather than" constructions, stacked "Together/Taken together", duplicate NREM expansion + orphan "(LC)",
  the whole-scalp/topographic/ROI triad five times, group-label drift (older adults vs elderly), Abstract at
  282 words, orphan 3.x/4.x section numbering, and the title page still reading **"July, 2026"**.
- Still open from [[project_flavio_review]]: Bayesian/TOST equivalence (#98) and the Results restructuring.

**Unused value-adds found during the check:** Chen 2025 (already ref 25) reports a **shorter spindle refractory
period in older adults**, surviving adjustment for spindle duration — direct mechanistic support for the faster
ISFS. New literature since the bib froze: Jacobsen et al., *eLife* reviewed preprint Mar 2026, "Noradrenergic
infraslow rhythm … heart-rate dynamics and memory consolidation" (mice **and** humans) — it partly answers the
Discussion's "the LC cannot be recorded in a sleeping human". And an open question: our young peak frequency
(0.0199 Hz) is ~double Lázár's human sigma-power estimate.

**Recipe worth reusing:** bulk-verify references by looping DOIs through
`curl -s "https://api.crossref.org/works/<doi>"` and reading `type` / `container-title` / `author` — it caught
both preprint upgrades and the single-author editorial in one pass. Figure verification recipe is in
[[reference_manuscript_gdoc]].
