---
name: project_bibliography_expansion
description: "library.bib expanded 31 → 84 verified entries (2026-08-13/14) for Yuval's ≥50-reference demand; thesis/references/new_refs_annotated.md is the ref→section map the prose session writes from; the C557 LC review is used as a SOURCE but deliberately NOT cited"
metadata: 
  node_type: memory
  type: project
  originSessionId: f56c0b50-a5d6-4441-bd6f-4a3b398c1092
  modified: 2026-08-14T10:19:04.535Z
---

Yuval's review asked for **at least 50 references** (email item #9); the manuscript cited 29 and `library.bib` held 31. Expanded to **84 verified entries** across three passes on 2026-08-13/14: the nine-gap sweep (31), the ISO-terminology re-sweep (9, see [[project_isfs_definition]]), and mining the C557 locus coeruleus review's bibliography (13).

## Where things live

- `thesis/references/library.bib` — 71 entries, organised under `% ====` topic banners.
- `thesis/references/library_status.md` — per-entry verification table, drop lists with reasons, flagged-not-changed items.
- **`thesis/references/new_refs_annotated.md`** — the working document for whoever writes the prose: per-gap tables of cite key → the claim it supports → target section → primary/review/preprint, plus a **C571 dossier** with the extracted reversion/conversion/autopsy numbers, an ISFS-vs-ISO **comparability table**, search log, and coverage count. Read this before drafting the Intro or Discussion expansions — it exists so the papers don't have to be re-read.

## What the 40 new refs cover

The nine gaps Yuval named: sleep in general / stages / PSG / EEG signatures; sleep & memory; sleep & aging; sleep & neurodegeneration; AD; MCI & aMCI & sleep in MCI; **aMCI ≠ early AD (C571)**; **CAP and other infra-slow phenomena**; **ISFS/ISO outside aging & MCI**; **locus coeruleus (C557)**.

Two decisions constrain the set: **journal articles only** (no AASM manual or textbook chapters — Shaked's call; scoring rules cited via `silber2007visualscoring`), and **strict ISFS/ISO only** for the "other conditions" gap.

## Facts worth not re-deriving

- **`nasreddine2005moca` filled a real hole** — the MoCA is used throughout Methods and Results 4.6 and was previously uncited.
- **C571 headline numbers:** ~24% of aMCI reverts to normal cognition (14% clinic / 31% community); annual conversion ~5–10%, "most will not progress even after 10 years"; and **29% of aMCI patients who did progress to dementia had non-AD primary pathology at autopsy**, with neither demographics nor cognitive scores predicting which. Defensible framing: *aMCI is enriched for AD without being equivalent to it*.
- **`kjaerby2026neuromodulators` rescues the cholinergic thread** Yuval called "out of context" (C557) — ACh oscillates infra-slowly during NREM under LC control, so `schmitz2018cholinergic` can be re-framed rather than deleted.
- **`parrino2025phasic`** has the CAP authorities themselves linking CAP periodicity to LC infra-slow activity — the CAP↔ISFS bridge Yuval asked for.
- **Verification method that worked:** resolve every DOI through `api.crossref.org`, compare field-by-field, and check `type`/`subtype`. Crossref records **only the first page for AMA journals** (Arch Neurol) — cross-check pagination against PubMed. See [[reference_dimitriades_citation_status]] for the conference-abstract and `[RETIRED]` traps this caught.

## The C557 locus coeruleus review — source, not citation (decided 2026-08-14)

Arrived as `thesis/references/Nir_etAl_LC_NE_Sleep_Review.docx`. **It is Yuval's own lab review** — Nir, Matosevich, Zelinger, Falach, Kimchy, Regev, *"Beyond Arousal: The Locus Coeruleus–Norepinephrine System as a Multimodal Orchestrator of Sleep Physiology"*. "NoaR" in his comment = **Noa Regev**, the last author.

It is **unpublished** — verified against Crossref and PubMed that no journal version and no preprint exists. It was briefly added as `@unpublished{nir2026locuscoeruleus}` and then **removed by Shaked's decision**: rather than put a bare "manuscript" entry for the supervisor's own unpublished work in a thesis reference list, its 227-reference bibliography was mined and those published sources cited directly (13 entries). **Do not re-add it as `@unpublished`.** If it publishes before submission, reconsider — it would then be a normal `@article`.

Two things it settles:
- **It uses "ISO" throughout and defines sigma as 10–16 Hz.** Confirms the naming point in [[project_isfs_definition]] and means the thesis must state the ISFS/ISO equivalence explicitly, or Yuval's own review looks like a different rhythm.
- **The mechanism has a direction:** LC infra-slow activity is *anti-correlated* with spindle power (NE peaks → micro-arousals; NE troughs → spindles, via thalamic NE suppressing spindles). Cite `osorioforero2021noradrenergic`, `kjaerby2022norepinephrine`, `osorioforero2025gatekeeper` for it.

Best two pickups from its bibliography: **`galgani2023locuscoeruleus`** (LC MRI in *amnestic* MCI predicts progression — serves C557 and C571 at once) and **`zarow2003neuronalloss`** (LC neuron loss exceeds nucleus basalis loss in AD — the quantitative licence to lead with LC instead of the cholinergic account Yuval called out).

## Still open

- **Three metadata fixes reported but NOT applied**, as out of scope: `dimitriades2024isfs` → *Sci Rep*; `andre2025remslowing` → *Mol Psychiatry* (needs title + authors changed too, not just the DOI); and `sharon2025slowwaves` is missing volume/issue/pages (Crossref gives **21(5):e70247**). First two detailed in [[reference_dimitriades_citation_status]].
- **`sharon2025slowwaves` is under-used.** C557 also names "Omer's paper on slow wave activity" — confirmed to be the only Omer Sharon slow-wave paper. It is currently cited *only* as dataset provenance in Methods 3.1; Yuval wants it used substantively in the Discussion as the NREM counterpart. Nice angle available: his slow-wave measure separated prodromal AD where this thesis's ISFS did not separate MCI — same lab, same cohort, two NREM measures.
- **Nothing was committed**, and `thesis/chapters/*` was not touched — another session owns the prose.

**Why:** the bibliography is now well ahead of the prose, and the annotated map is the artefact that makes that lead useful. Re-searching this literature would be wasted effort.

**How to apply:** when the Intro/Discussion expansion is written, work from `new_refs_annotated.md` rather than from `library.bib` directly. The count Yuval will check is the **manuscript reference list**, not the .bib file — citing all 40 new entries takes it to 69.

Related: [[project_yuval_review]], [[project_isfs_definition]], [[reference_dimitriades_citation_status]], [[project_scientific_story]], [[feedback_thesis_prose_rules]]
