# Introduction expansion — Yuval's review, writing pass

## Context

Yuval's review (email #3 + two margin notes) says the Introduction is too ISFS-focused and needs
much more on sleep in general, on aMCI/AD, and on sleep and aging — **at thesis scale, not paper
proportions**. His margin note *"for thesis (unnecessary for paper)"* is what settled the
thesis-vs-paper question in the first place.

He did not just complain: **he wrote the section headings he wants directly into his .docx**, as
tracked insertions that the §3 triage summary never listed. Extracted from
`thesis/reviews/Shaked's Thesis_YN.docx` (paragraphs 62–84 of `word/document.xml`):

| His inserted heading | His sub-topics |
|---|---|
| **Human sleep and scalp EEG** | Sleep what is it, how is it defined, sleep stages NREM and REM · What are the function(s) of sleep what is it good for · PSG monitoring, EEG signatures e.g. slow waves, sleep spindles etc |
| **Infra-slow fluctuation of sigma power (ISFS) in NREM sleep** | *(covers the existing ISFS paragraphs)* |
| **Changes in sleep and EEG across aging and MCI** | Sleep, Learning and Memory · Sleep and Aging · Sleep and Neurodegeneration · Alzheimer's Disease (AD) · Sleep in Alzheimer's Disease · Mild Cognitive Impairment (MCI) · Amnestic Mild Cognitive Impairment · Sleep in MCI — what is known |
| **Changes in sleep spindles and ISFS across aging and MCI** | *(covers the existing aging/spindles paragraph)* |

The intended outcome: an Introduction that follows his structure exactly, keeps the existing ISFS
material intact (preceded and followed, not replaced), carries his in-place line-edits, makes the
C571 point (aMCI is not only an early phase of AD), and cites the references
`thesis/references/new_refs_annotated.md` was built to supply.

**Scope rule in force:** only what he explicitly asked for. Items marked ⚑ in the triage are out.

---

## Order of work

**Nothing touches the Doc until you have approved the prose.**

| Phase | What | Gate |
|---|---|---|
| **1** | Write `thesis/reviews/intro_edits_before_after.md` — the **full** new Introduction text, before/after per paragraph, same format as `methods_edits_before_after.md` and `results_edits_before_after.md`. Includes the complete 49-entry reference list as it will be typed, and the 22 anchored superscript replacements as a table. | **STOP — you review and approve.** Anything missing gets a `[TO SUPPLY]` marker rather than an invented fact |
| **2** | Apply to the Google Doc: Introduction body, superscripts, reference list rebuild, Methods 3.1 aMCI de-duplication | — |
| **3** | Mirror into `thesis/chapters/02_introduction.md` and `03_methods.md` | — |
| **4** | Run the verification checks below; update `yuval_review_triage.md` §7/§8 | — |

Phase 1 is the substance of the job — it is where the writing actually happens. Phases 2–4 are
mechanical transcription of an approved artifact.

---

## Target structure

Two heading levels, **in the order he wrote them** — the phenomenon, then the population, then the
intersection. His four headings become `HEADING_3` sections numbered `2.x`, matching the existing
Methods `3.x` / Results `4.x` convention. His eight sub-topics become paragraphs, not headings.

```
Introduction                    (HEADING_2, unchanged)
  ¶ unheaded opener             "Sleep is not a uniform state…"  — kept in place (Flavio #6)
  2.1 Human sleep and scalp EEG                                   NEW  ~4 ¶
  2.2 Infra-slow fluctuation of sigma power (ISFS) in NREM sleep  existing ¶2–¶4 + his edits
  2.3 Changes in sleep and EEG across aging and MCI               NEW  ~8 ¶
  2.4 Changes in sleep spindles and ISFS across aging and MCI     existing ¶5 + ISFS-gap close
  2.5 The present study                                           existing ¶6 + his edits
```

Length is set by what each of his topics needs, at standard M.Sc. thesis scale — not by a word
target.

### 2.1 Human sleep and scalp EEG — new

| ¶ | Content | Cites |
|---|---|---|
| a | What sleep is: behavioural definition, reversibility, homeostatic/circadian regulation; NREM and REM; cycling across the night and how the balance shifts from N3-heavy to REM-heavy | `markov2006normalsleep` |
| b | How sleep is measured: PSG montage (EEG/EOG/EMG), 30 s epochs, visual scoring; **N2 is *defined by* spindles and K-complexes**, N3 by slow waves, REM by desynchronized EEG + REMs + atonia | `silber2007visualscoring` |
| c | Functions: memory consolidation and plasticity; metabolite clearance | `rasch2013memory`, `xie2013clearance` |
| d | The cardinal NREM EEG signatures: the cortical slow oscillation / slow waves, and sleep spindles — handing off to 2.2 | `steriade1993slowoscillation`, `fernandez2020spindles` |

Kept deliberately brief on spindles: the detail belongs to 2.2's first paragraph, which he left in
place.

### 2.2 ISFS in NREM sleep — existing prose + his line-edits

Existing ¶2–¶4 stay. Apply his tracked edits, which survive because V2 left these sentences intact:

| Current V2 | His edit |
|---|---|
| "occupies a large part of the human night" | → **"occupies most of sleep in adults"** (his fix also satisfies Flavio #7, which objected to "most of the *night*") |
| "its second stage (N2) is defined by sleep spindles" | → "**stage 2 (N2) is characterized by the intermittent occurrence of** sleep spindles" |
| "vary systematically in density and topography" | → "vary in density and topography" |
| "closely tied to sleep-dependent memory consolidation¹." | → "closely tied to **plasticity and** sleep-dependent memory consolidation¹, **as well as with increased disconnection from the external sensory environment.**" |
| "In rodents, this infra-slow fluctuation…" | → "**In both rodents and humans**, this infra-slow fluctuation…" |
| "This alternation is clocked by the locus coeruleus (LC) and the noradrenergic system" | → "This alternation is **driven in part by** the locus coeruleus **norepinephrine (LC-NE) system**" |
| "The amplitude of these infra-slow noradrenergic oscillations is in turn coupled" | → "The amplitude of infra-slow noradrenergic oscillations is coupled" |
| "quantifying it per channel as…" | → "quantifying … **for each EEG channel**" |

### 2.3 Changes in sleep and EEG across aging and MCI — new, one ¶ per sub-topic

| ¶ | His sub-topic | Cites |
|---|---|---|
| a | Sleep, learning and memory — systems consolidation, and the SO/spindle/ripple hierarchy that makes spindle **timing** functionally meaningful (sets up why a rhythm organizing spindle *trains* matters) | `rasch2013memory`, `klinzing2019consolidation`, `boutin2020spindleframework`, `champetier2023spindlememory` |
| b | Sleep and aging — architecture and continuity: TST, efficiency, WASO, N3 and REM loss across the lifespan. **Spindles deliberately excluded here**; they are 2.4's job | `ohayon2004metaanalysis`, `li2018normalaging` |
| c | Sleep and neurodegeneration — bidirectional relationship; clearance as the mechanistic reason sleep quality matters; a quantitative NREM EEG feature tracking tau; slow-wave synchrony tracking prodromal AD | `ju2014bidirectional`, `xie2013clearance`, `lucey2019nremtau`, `sharon2025slowwaves` |
| d | Alzheimer's disease — epidemiology, amyloid/tau pathology, clinical course; Braak staging, with tau appearing in the LC before entorhinal cortex | `scheltens2021alzheimer`, `braak2011stages` |
| e | Sleep in AD — PSG meta-analysis; reduced parietal fast-spindle density | `zhang2022alzheimerreview`, `gorgoni2016parietal` |
| f | MCI — the Petersen characterization and the NIA-AA criteria, **including the distinction between the MCI syndrome and MCI *due to* AD**; brief cognitive screening | `petersen1999mci`, `albert2011mcicriteria`, `nasreddine2005moca` |
| g | **Amnestic MCI — C571** (see below) | `ferman2013nonamnestic`, `mitchell2009progression`, `malekahmadi2016reversion`, `jicha2006neuropathologic` |
| h | Sleep in MCI, what is known — objective sleep measurement in MCI; sleep-EEG features proposed as early markers | `drozario2020objectivesleep`, `taillard2019nonrem`, `liu2020spindlebiomarkers` |

**C571, ¶g.** He wrote: *"Not only earlier. Some aMCI will never develop AD… Please read more about
this and describe also in intro."* The position to argue, per the C571 dossier in
`new_refs_annotated.md` §G, is **"aMCI is enriched for AD without being equivalent to it"**:

- subtype does predict outcome — aMCI progresses to probable AD at 17 vs 1.5 events/100 py
  (`ferman2013nonamnestic`), so state the enrichment fairly first;
- but most MCI never converts — 39.2 % (specialist) / 21.9 % (population) cumulative
  (`mitchell2009progression`);
- ≈24 % of aMCI reverts to normal cognition, 14 % in clinic-based samples like ours
  (`malekahmadi2016reversion`);
- and among aMCI patients who *do* convert, 10/34 (29 %) had a non-AD primary pathology, with
  neither demographics nor cognitive measures predicting which (`jicha2006neuropathologic`).

**Abbreviation consequence.** ¶g is where `amnestic MCI (aMCI)` gets its single body definition.
Per triage §8.2.4 this means **Methods 3.1's definition must be de-duplicated** — "patients with
amnestic mild cognitive impairment (aMCI; n = 30…)" → "(n = 30…)". This is a small forced edit to
an already-signed-off Methods section; called out here rather than done silently.

### 2.4 Changes in sleep spindles and ISFS across aging and MCI

Existing ¶5 (he left it unchanged), plus two sentences closing the ISFS half of his heading, which
the current paragraph does not cover: the ISFS has been characterized almost exclusively in young
adults and across development, and the only clinical characterization to date is the concurrent AD
study (`dimitriades2024isfs`, `grollero2026iso`).

The sentence *"Yet the ISFS itself has been characterized almost exclusively in young adults …
remain unknown"* **moves from 2.5 to close 2.4**, where it now belongs, so 2.5 opens on "Here we
quantified…". Without this, 2.4 and 2.5 say the same thing twice.

Judgment call inside ¶5, forced by renumbering: `¹²⁻¹⁴` currently bundles champetier + mander +
helfrich, which after renumbering are non-contiguous (17, 38, 39). Splitting it —
"…breaks down³⁸,³⁹, changes that themselves track cognitive decline¹⁷." — reads better than
"¹⁷,³⁸,³⁹" and is more accurate about which reference carries which claim. Flagged because it
touches a paragraph he left unedited.

### 2.5 The present study

Existing final paragraph with his edits: "during **clean** N2 sleep" → "during N2 sleep";
"at every channel" → "at every **scalp** channel"; "patients with amnestic MCI" → "patients with
aMCI" (now defined in 2.3g).

---

## Reference renumbering

> ### ⚠ SUPERSEDED 2026-08-16 — the map below is no longer the plan
>
> `thesis/reviews/discussion_edits_before_after.md` turned out to be drafted and **unapplied**,
> renumbering the same list on the assumption that "refs 1–26 do not move". Six references are
> claimed by both passes (`braak2011stages`, `ferman2013nonamnestic`, `mitchell2009progression`,
> `malekahmadi2016reversion`, `jicha2006neuropathologic`, `drozario2020objectivesleep` — the C571
> evidence set, which he asked for in both chapters), and it **deletes** `schmitz2018cholinergic`
> and `andre2025remslowing`, which the map below numbers 47 and 48.
>
> **Decision: prose and numbering are split.** The Intro prose ships with `[@citekey]` citations
> and no numbers; a single combined pass afterwards rebuilds the list once and writes every
> superscript once, superseding `discussion_edits_before_after.md` §9.
> Projected combined total: **64** (30 − 2 removed + 36 unique new across both passes).
>
> The map below is kept as the record of the Introduction's citation *order*, which is still
> correct — only the absolute numbers are void.

The list is auto-numbered by order of first appearance, so inserting 19 citations into the
Introduction renumbers **every marker in the document**. Intro-only projection was 30 → 49.

Full old→new map (`—` = new entry):

| New | Key | Old | | New | Key | Old |
|---|---|---|---|---|---|---|
| 1 | markov2006normalsleep | — | | 26 | zhang2022alzheimerreview | 15 |
| 2 | silber2007visualscoring | — | | 27 | gorgoni2016parietal | 16 |
| 3 | rasch2013memory | — | | 28 | petersen1999mci | — |
| 4 | xie2013clearance | — | | 29 | albert2011mcicriteria | — |
| 5 | steriade1993slowoscillation | — | | 30 | nasreddine2005moca | — |
| 6 | fernandez2020spindles | 1 | | 31 | ferman2013nonamnestic | — |
| 7 | andrillon2011intracranial | 2 | | 32 | mitchell2009progression | — |
| 8 | purcell2017characterizing | 3 | | 33 | malekahmadi2016reversion | — |
| 9 | molle2011fastslow | 4 | | 34 | jicha2006neuropathologic | — |
| 10 | boutin2020spindleframework | 5 | | 35 | drozario2020objectivesleep | — |
| 11 | lecci2017infraslow | 6 | | 36 | taillard2019nonrem | 17 |
| 12 | osorioforero2021noradrenergic | 7 | | 37 | liu2020spindlebiomarkers | 18 |
| 13 | kjaerby2022norepinephrine | 8 | | 38 | mander2017aging | 13 |
| 14 | cardis2021corticoautonomic | 9 | | 39 | helfrich2018uncoupled | 14 |
| 15 | lazar2019infraslow | 10 | | 40 | grollero2026iso | 30 |
| 16 | dimitriades2024isfs | 11 | | 41 | visbrain | 20 |
| 17 | champetier2023spindlememory | 12 | | 42 | sleepeegpy | 21 |
| 18 | klinzing2019consolidation | — | | 43 | gramfort2013mnepython | 22 |
| 19 | ohayon2004metaanalysis | 25 | | 44 | maris2007nonparametric | 23 |
| 20 | li2018normalaging | — | | 45 | vallat2021yasa | 24 |
| 21 | ju2014bidirectional | — | | 46 | chen2025spindletiming | 26 |
| 22 | lucey2019nremtau | — | | 47 | schmitz2018cholinergic | 27 |
| 23 | sharon2025slowwaves | 19 | | 48 | andre2025remslowing | 28 |
| 24 | scheltens2021alzheimer | — | | 49 | niethard2023spindleaging | 29 |
| 25 | braak2011stages | — | | | | |

Every planned citation was checked against `library.bib` — all 19 new keys have complete
author/title/journal/year/volume/pages/DOI (`drozario2020objectivesleep` has an article number,
101308, instead of an issue, which is correct for *Sleep Medicine Reviews*).

### Renumbering method — deviation from the descending rule, deliberate

The memory rule is "renumber descending so no two markers collide". That rule assumes numbers only
increase. **Here `ohayon2004metaanalysis` moves 25 → 19, i.e. downward**, because it now first
appears in Intro 2.3b instead of Results 4.1 — so descending order alone is not safe.

Instead: **every replacement is anchored on surrounding words, never on a bare glyph.** Anchored
strings are unique, so order does not matter and collisions are impossible. The 22 sites outside
the Introduction:

| § | Anchor | Old → New |
|---|---|---|
| M 3.1 | "described in our previous work" | ¹⁹ → ²³ |
| M 3.1 | "take part in the sleep study" | ¹⁹ → ²³ |
| M 3.1 | "an AHI of 15 or below" | ¹⁹ → ²³ |
| M 3.2 | "the sleep module of the Visbrain Python package" | ²⁰ → ⁴¹ |
| M 3.2 | "within the SleepEEGpy platform" | ²¹ → ⁴² |
| M 3.4 | "duplicated and extended from Dimitriades et al. (2024)" | ¹¹ → ¹⁶ |
| M 3.5 | "AUC hotspot reported by Dimitriades et al. (2024)" | ¹¹ → ¹⁶ |
| M 3.6 | "cluster-based permutation tests (MNE-Python" / "following Maris and Oostenveld" | ²² → ⁴³ / ²³ → ⁴⁴ |
| M 3.7 | "Analyses used Python with MNE-Python" / "YASA" | ²² → ⁴³ / ²⁴ → ⁴⁵ |
| R 4.1 | "no difference between elderly and aMCI" | ²⁵ → ¹⁹ |
| R 4.3 | "in accordance with previous studies" | ¹⁰,¹¹ → ¹⁵,¹⁶ |
| R 4.5 | "resembled those reported previously" | ¹¹ → ¹⁶ |
| D ¶3 | "over centro-parieto-occipital cortex" | ⁴,¹⁰ → ⁹,¹⁵ |
| D ¶3 | "first described by Dimitriades and colleagues" | ¹¹ → ¹⁶ |
| D ¶3 | "carrying much of the variance" | ²⁶ → ⁴⁶ |
| D ¶4 | "clocks the infra-slow sigma rhythm" | ⁶⁻⁷ → ¹¹⁻¹² |
| D ¶4 | "in early Alzheimer's disease" | ²⁷ → ⁴⁷ |
| D ¶4 | "denervation in aging and MCI" | ²⁸ → ⁴⁸ |
| D ¶4 | "breakdown of fast-spindle timing with age" | ¹²,²⁹ → ¹⁷,⁴⁹ |
| D ¶5 | "markers of incipient cognitive impairment" | ¹⁵⁻¹⁶ → ²⁶⁻²⁷ |
| D ¶6 | "in clinically diagnosed Alzheimer's disease" | ³⁰ → ⁴⁰ |
| D ¶8 | "and that coupling loosens with age" | ¹⁴ → ³⁹ |

Introduction markers are not patched — that whole block is replaced wholesale, already carrying
final numbers.

### Reference list — rebuild in one pass

Because 10 of the 30 existing entries change position (13,14 → 38,39; 15,16 → 26,27; 17,18 → 36,37;
19 → 23; 25 → 19; 30 → 40), a piecewise reorder means deleting and retyping most of them anyway.
Cleaner and less error-prone: **delete the whole list and insert all 49 entries in final order**,
AMA style matching the existing formatting, generated from `library.bib`.

Then restore the DOI hyperlinks: `applyTextStyle` accepts `linkUrl` with a `textToFind` target, and
each DOI URL is unique, so one call per entry re-links all 49. (This is the last step and is purely
cosmetic — it can be dropped if you'd rather not spend the calls. Note it also fixes the existing
plain-text `maris2007nonparametric` DOI noted in triage §8.2.8.)

---

## Files to change

| Phase | File | Change |
|---|---|---|
| 1 | `thesis/reviews/intro_edits_before_after.md` | **New, and written first** — the full new prose, before/after per paragraph, the 49-entry reference list, the 22 superscript replacements. This is what you approve |
| 2 | **Google Doc "Shaked's Thesis V2"** `1YpXrDGFlzRk…` | Introduction body replaced; 22 anchored superscript fixes; reference list rebuilt at 49; Methods 3.1 aMCI de-duplication |
| 3 | `thesis/chapters/02_introduction.md` | Mirror of the new prose, with `[@citekey]` citations (the chapters use citekeys; `07_references.md` is Pandoc-generated, so no numbered list to maintain there) |
| 3 | `thesis/chapters/03_methods.md` | One-clause aMCI de-duplication, mirroring the Doc |
| 4 | `thesis/reviews/yuval_review_triage.md` | §7 and §8 updated: Intro marked done, reference count 30 → 49, carry-over for the Discussion session |

Not touched: `thesis/figure_manifest.md` (captions are unaffected by an Introduction pass),
`library.bib`.

### Doc edit mechanics

- **Insert before delete.** The Introduction body currently spans index 1760 → 6428. Insert the new
  Introduction at the `startIndex` of the *first existing body paragraph* (a `NORMAL_TEXT`
  paragraph), then delete the old block. Inserting there makes every new paragraph inherit
  `NORMAL_TEXT`, sidestepping the "inserting right after a heading inherits the heading style" trap
  entirely — no post-hoc `applyParagraphStyle NORMAL_TEXT` sweep needed.
- Then `applyParagraphStyle HEADING_3` on the five `2.x` heading lines, targeted by `textToFind`.
  `HEADING_3` is what Methods `3.1` and Results `4.1` use — verified via `findSectionsByHeading`.
- Superscript glyphs carry no formatting (verified previously against the Docs API), so typing them
  fresh renders identically. There is no italic text anywhere in the Introduction.
- Re-read indices with `findElement` immediately before each destructive call; do not reuse indices
  across edits.

---

## Verification

1. **Superscript audit (the check you asked for).** Read the finished Doc with `readDocument`, parse
   every Unicode-superscript run into integers, expanding `⁻` ranges and `,` lists. Assert:
   - the set of cited numbers is exactly `{1…49}`, no gaps and no duplicates;
   - **order of first appearance equals list order** — i.e. the *n*-th distinct number encountered
     scanning the body top to bottom is *n*. This is the property Google Docs auto-numbering
     assumes, and it catches any misordering in one pass.
2. **List length.** Count paragraphs in the References section = 49, and confirm the last is
   `niethard2023spindleaging`.
3. **Cross-check against the bib.** Confirm each of the 49 rendered entries matches its
   `library.bib` record (first author surname + year + journal), so nothing was mistyped during the
   rebuild.
4. **Coverage against his asks.** Walk his eight sub-topics plus the two headings and confirm each
   has prose: sleep definition/stages, functions, PSG + EEG signatures, memory, aging,
   neurodegeneration, AD, sleep in AD, MCI, aMCI, sleep in MCI, and C571.
5. **Mirror check.** Diff `02_introduction.md` prose against the Doc's Introduction text so the two
   have not drifted.
6. **Regression.** Confirm the three `[TO SUPPLY]` markers in Methods are still intact and that
   Table 1 / the figure captions were not disturbed.

---

## Flagged — reported, not done (scope rule)

1. **The unheaded opener still says the rhythm "is paced by the noradrenergic locus coeruleus".**
   He softened exactly this claim twice — in the Abstract ("paced by" → "associated with changes in
   LC-NE activity and other arousal systems") and in Intro ¶3 ("clocked by" → "driven in part by").
   The opener did not exist in his copy, so he never saw it. Applying the same softening there is a
   one-clause edit and would make the chapter internally consistent — **say the word and I'll do
   it**; otherwise it stays as is.
2. **Two citation upgrades I will be retyping anyway.** The rebuild means typing
   `dimitriades2024isfs` as "bioRxiv [preprint]" and `andre2025remslowing` as "medRxiv [preprint]"
   when both are published — *Sci Rep* 18 Jun 2026 `10.1038/s41598-026-58423-z` and *Mol Psychiatry*
   12 May 2026 `10.1038/s41380-026-03635-y`. Also `sharon2025slowwaves` is missing its volume/issue/
   pages (Crossref: 21(5):e70247). These are our own B1/S2 items, not his asks, so they are out of
   scope — but the marginal cost of fixing them during a rebuild is zero, and Dimitriades is the
   thesis's central methods reference. Your call.
3. **`galgani2023locuscoeruleus`** (LC MRI abnormality in aMCI predicts progression) is the single
   strongest citation for C571 and would fit 2.3g. Left for the Discussion session, which owns it,
   to avoid citing the same evidence twice.
4. **Reference count lands at 49, one short of his ≥50.** Not worth padding the Introduction for —
   the Discussion pass has ~25 references queued (CAP, ISFS-in-other-conditions, the LC set), so it
   clears comfortably. Noted only so the number is not a surprise.
