---
name: reference_dimitriades_citation_status
description: "Dimitriades dev study IS published (Sci Rep, 18 Jun 2026) BUT the thesis deliberately still cites the bioRxiv 2024 preprint — Shaked declined the upgrade 2026-08-19, do not re-flag; André also published (Mol Psychiatry 2026); Grollero still a preprint; Sleep Medicine 2026 DOIs are conference abstracts"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 31e90dee-898a-4bea-9d5d-280af572b055
  modified: 2026-08-14T09:45:40.904Z
---

Citation status of the preprint references in `thesis/references/library.bib`. **Rechecked 2026-08-08 — two upgrades since 2026-06-17.**

**Dimitriades developmental study (`dimitriades2024isfs`, our core pipeline source) IS NOW PUBLISHED:**
*Scientific Reports*, **18 June 2026, doi `10.1038/s41598-026-58423-z`** — "The infraslow fluctuation of sigma
power during sleep and its links to markers of arousal and memory reactivation across development"
(Dimitriades, Osorio-Forero, Fattinger, … Gerstenberg, Huber). Confirmed via Crossref **and** the KNAW Pure record.
Cite this, not the bioRxiv preprint `10.1101/2024.11.06.620875`. In-text mentions change from
"Dimitriades et al. (2024)" to **(2026)** — they appear in the Introduction, Methods 3.4, Methods 3.5 and the
Figure 2D caption.

**André (`andre2025remslowing`) IS ALSO NOW PUBLISHED:** *Molecular Psychiatry*, **12 May 2026,
doi `10.1038/s41380-026-03635-y`** — "Associations between REM sleep EEG slowing and brain cholinergic
denervation in aging and Mild Cognitive Impairment". ⚠️ **Correction (verified via Crossref 2026-08-13):
the author list is NOT the same as the medRxiv version** — the published paper has **14 authors, the bib
entry has 12**; Olga Fliaguine and Serge Gauthier were added. The title also changed substantially. So this
is not a DOI swap: title, journal and authors all need updating together.

**Still a preprint:** `grollero2026iso` — bioRxiv `10.64898/2026.04.09.717425` (Apr 2026). The unusual
`10.64898` prefix is bioRxiv's own and resolves fine — not an error.

**TRAP that still holds — the Sleep Medicine 2026 entries are conference abstracts, not papers:**
Crossref shows `10.1016/j.sleep.2025.107032` ("…Across Development") and `107031` (Schizophrenia) in
*Sleep Medicine* 138 (2026). These are one-page abstracts in a supplement, sequentially numbered under an
"Abstracts <Journal> <vol>" footer. Elsevier's "View PDF" for `107032` serves the whole multi-abstract page and
its embedded metadata title mismatches the on-screen body. Never cite these.

**Different paper, do not confuse:** the Dimitriades *schizophrenia* study is separately published —
*Schizophrenia Research* 2025, doi `10.1016/j.schres.2025.09.029`.

**Earlier resolutions (already in the bib):** `sleepeegpy` → Falach et al. 2025, *Comput Biol Med* 192:110232;
`visbrain` → Combrisson et al. 2019, *Front Neuroinform* 13:14.

**The Sleep Medicine 138 abstract trap is GENERAL, not a Dimitriades quirk (confirmed 2026-08-13).** It also
caught the CAP-and-dementia paper: the real article is Zheng et al., *Alzheimer's & Dementia* 22(4):e71331,
doi `10.1002/alz.71331` (8 authors), while `10.1016/j.sleep.2025.108276` is its *Sleep Medicine* 138 abstract
listing only 2 authors. Same pattern as `107031`/`107032`. **Reliable tell-tales in the Crossref record:**
*Sleep Medicine* volume 138, a 6-digit article number in the 10xxxx range, no issue, and an author list
collapsed to initials or truncated. Lázár's `10.1016/j.sleep.2017.11.529` is the same thing for
`lazar2019infraslow`.

**General lesson:** before upgrading a preprint to a Crossref "journal-article" hit, confirm it is a full article
and not a conference-abstract supplement (tell-tales above). Also check `type`/`subtype` on *every* candidate —
the same check caught `10.1212/WNL.0000000000004826` (Petersen 2018 AAN MCI guideline), whose Crossref title
carries **`[RETIRED]`** and which lists zero authors.

**⚠️ DECISION 2026-08-19 — THE UPGRADE WAS DECLINED. Do not apply it, do not re-flag it.**
In the final pre-send pass I proposed swapping ref 16 to the *Scientific Reports* version and changing
the four in-text "Dimitriades et al. (2024)" mentions to (2026). Shaked's answer was **"dont do"**. The
version sent to Yuval on 2026-08-19 therefore still cites **bioRxiv `10.1101/2024.11.06.620875`, dated
2024**, in both the reference list and the in-text mentions. Everything above about the published record
is still factually correct — it is simply not what the manuscript does, by choice. Re-propose only if
Shaked raises it. See [[project_final_pass_sent]].

Related: [[project_lit_review_parallel]], [[project_thesis]], [[project_scientific_story]], [[reference_manuscript_gdoc]].
