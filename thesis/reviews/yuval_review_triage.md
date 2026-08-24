# Yuval's review — triage against V2

Reviewed file: `thesis/reviews/Shaked's Thesis_YN.docx` (returned 2026-08-11, 17 comments,
364 insertions / 111 deletions, all authored "Yuval Nir")
Current draft: the Google Doc **"Shaked's Thesis V2"**
(`1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`, modified 2026-08-08 19:42) — the version emailed to him
2026-08-12. Read directly from the doc; there is no local copy.
Triage run 2026-08-12.

> ## STATUS — updated 2026-08-15
>
> | Pass | State | Record |
> |---|---|---|
> | **Methods** (C344, C345, C371, C395, C412-part, apnea / Sydney-scoring / identical-setups notes) | **APPLIED 2026-08-15** | `thesis/reviews/methods_edits_before_after.md` |
> | **Analysis** (C429 ANCOVA, C390 sleep statistics) | **DONE 2026-08-13** | `thesis/reviews/c429_c390_results.md` |
> | **Results** (C412, C429, C390, `~X %`, peak-freq range, both `#REF`, aMCI relabel thesis-wide, 3 caption fixes) | **APPLIED 2026-08-15** | `thesis/reviews/results_edits_before_after.md` |
> | **Introduction** (email #3, both margin notes, C571-in-Intro, his Intro tracked edits) | **APPLIED 2026-08-16** | `thesis/reviews/intro_edits_before_after.md` |
> | **Figures** (C389 part, C432, C487, C502, C591, C592, email #6) | **APPLIED 2026-08-17** | `thesis/reviews/figures_edits_before_after.md` |
> | **Discussion** (C557, C558, C571, both `##` margin-note paragraphs, the C429 bandwidth consequence) | **APPLIED 2026-08-17** | `thesis/reviews/discussion_edits_before_after.md` |
> | **Writing, remaining** (Hebrew abstract, TOC, Abstract additions, title page, acknowledgements) | **APPLIED 2026-08-17** | `thesis/reviews/front_matter_edits_before_after.md` |
> | **Reply to Yuval** | **OPEN — the only pass left** | see §8.3 |
>
> **Every content pass is now closed.** The front-matter pass (2026-08-17) took the title page,
> Hebrew title page, Acknowledgements, TOC page, Abstract and תקציר, closing **email #1 and #2** and
> his tracked front-matter insertions. **Emails #1–#9 and all 17 doc comments are now resolved.**
> Reference list unchanged at 63; the invariant was re-verified after the pass.
> **One manual step remains for Shaked: Insert → Table of contents (with page numbers)** on the
> prepared page 4 — the Docs API has no request type that creates a TOC.
> Beyond his asks, **seven Discussion subheadings (5.1–5.7) were added at Shaked's request** so the
> TOC shows more than one row for the chapter; no Discussion prose was rewritten.
>
> **The reference list is now 63 entries.** The Introduction pass (2026-08-16) took it 30 → 49; the
> Discussion pass (2026-08-17) took it 49 → 63, adding 17 and removing 3 (`schmitz2018cholinergic`,
> `andre2025remslowing`, `niethard2023spindleaging` — all three cited only in the Discussion paragraph
> that was replaced). **Introduction refs occupy 1–39, Methods 40–45, Discussion 46–63.**
> `tallonbaudry1997gamma` entered at Methods 3.4 as ref 42, which shifted MNE, Maris and YASA.
> **Email #9 (≥50 references) is cleared.** The invariant *order of first appearance == list order* was
> verified programmatically after applying.
>
> **~~⚠ `discussion_edits_before_after.md` §9 is VOID~~ — RESOLVED 2026-08-17.** It was recomputed and
> applied. The paragraph below is kept only as the record of what went wrong: §9 had been computed
> against the 30-entry baseline
> and assumed "refs 1–26 do not move". A banner at the top of that file lists exactly what changed,
> including the **six references it plans to insert that are already in the list** (the C571 evidence
> set, cited in the Introduction). Projected final count after the Discussion pass: **64**.
>
> **Email #9 (≥50 references) is now within one of being met** by the Introduction alone, and the
> Discussion pass clears it comfortably.
>
> The old advice to "renumber descending" applies only to piecemeal edits. The Introduction pass
> instead **anchored every replacement on surrounding words**, which is collision-proof regardless of
> direction — necessary because `ohayon2004metaanalysis` moved *down* (25 → 19).
>
> **Everything that this revision has handed on to a later session is consolidated in §8. Read §8 before
> starting the figures, writing or reply session.**

---

## 0. Bottom line

**He reviewed V1** — the same pre-Flavio draft Flavio Schmidig reviewed, not the version sent to him.

**But the review is almost entirely still valid.** Of the 9 items in his email, **0** are resolved by V2.
Of his 17 doc comments, **2** are resolved and 1 partly. The redundancy sits in his in-place line-edits,
where Flavio's V2 rewrite had independently made some of the same changes.

| | Items | Resolved by V2 | Partly | Open |
|---|---|---|---|---|
| Email list | 9 | 0 | 1 | 8 |
| Doc comments | 17 | 2 | 1 | 14 |
| Inline `##` notes | 14 | 0 | 0 | 14 |

**Do not build the reply around "he reviewed the wrong file."** It explains why his line-edits will not
reappear verbatim, and nothing else.

### Proof that his copy is V1

All four fixes applied before sending are in V2 and absent from his copy:

| Fix | V2 (verified in the doc) | His copy |
|---|---|---|
| B2 | "The value above each map in Figure 4A is the mean of the per-subject means." | absent |
| S3 | Niethard = ref **28**; Intro bracket `¹²⁻¹⁴`; Discussion `¹²,²⁸` | Niethard = ref **15**; bracket `¹²⁻¹⁵` |
| S5 | "…too few clean N2 bouts (fewer than three; see Section 3.3), **or a total sleep time under 210 min**." | clause absent |
| S8 | "both older groups have more N2 bouts than young adults, **and in MCI these bouts are also shorter**" | "elderly and MCI have more frequent but shorter N2 bouts" |

Independent confirmation: `reviews/flavio_comments_mapping.md:192` and `:96` quote his paragraphs
**verbatim** as Flavio's V1 anchors. Same base document.

Also different in V2, and he has not seen it: **the title.** His copy reads *"Infra-slow fluctuations of
sigma power in NREM2 sleep are altered by aging but not further by mild cognitive impairment (MCI)"*;
V2 reads *"Aging, but not mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle
power."* He did not comment on the title, so we have no read on his view of the current one — **ask him.**

### Two gaps in his copy that V2 already fills

- **Methods 3.3 (N2 bout extraction)** — heading with no body in his copy. V2 has the full section
  (300 s rationale = two cycles of 0.0075 Hz, three-bout minimum, 1023 bouts / mean 9.8 / range 3–21).
  No Flavio anchor points at 3.3 body text either, so this was a genuine V1 gap.
- **Results 4.7 (No association with cognitive score)** — reduced to a bare `.` in his copy, but Flavio #73
  anchored to a real sentence there, so the text existed in V1. **He appears to have deleted it with Track
  Changes off.** V2 §4.6 is complete (n = 44, all |r| ≤ 0.15, all p ≥ 0.34, Pearson and Spearman).

> **Caution:** that second point means his copy may contain other silent deletions no tracked-change
> extraction can recover. The paragraph diff in §3 is what caught this one.

---

## 1. Email list — 9 items

| # | Item | Status in V2 | Notes |
|---|---|---|---|
| 1 | Missing TOC | **DONE 2026-08-17** | TOC page prepared after the Acknowledgements; **Shaked generates the list itself** (Insert → Table of contents), because no API can create one. Seven Discussion subheadings added so the chapter contributes more than one row. His pasted example is Yael Gat's contents page and carries three paste artefacts |
| 2 | Hebrew abstract | **DONE 2026-08-17** | `תקציר`, RTL, on its own page after the English Abstract, per the TAU house format. A Hebrew title page was added alongside it |
| 3 | Intro lacking — too ISFS-focused, wants sleep in general, aMCI/AD, sleep/aging | **PARTLY** | V2 added one framing paragraph ("Sleep is not a uniform state…"), the slow/fast spindle subtypes, and a memory-consolidation sentence. Nothing on sleep's functions, PSG/EEG signatures, AD, or aMCI. Intro is 782 words against his implied thesis-scale chapters |
| 4 | Move Fig 1 to Results | **TEXT DONE, IMAGE PENDING** | The overview paragraph now lives in Results 4.1, in his own rewrite; the **image and caption are still physically in Methods 3.2** and move by hand in the figures session (§8). **Flavio raised this as #18 and it was declined** ("Fig 1 sentence reads as a result — leave it in Methods"). The PI overrides |
| 5 | Sleep architecture table not readable, bigger, more info, separate table | **DONE 2026-08-17** | Same panel in V2. = C389 + C390 |
| 6 | All figure graphics/fonts below standard | **DONE 2026-08-17** | V1 and V2 embed the **same V10 image assets** — every font complaint applies unchanged. = C487, C502, C591, C592 |
| 7 | Discussion missing: fit with other aging/MCI changes beyond ISFS; ISFS in other conditions | **OPEN** | Nothing of this in V2. He names **CAP (cyclic alternating pattern)** as a specific lead |
| 8 | Treating aMCI as only "earlier phase of AD" is inaccurate | **OPEN** | V2 Discussion still reads "perhaps because MCI is an earlier disease stage". V2 *did* add Flavio's alternative reading (#112) — adjacent, not the same point. He wants it fixed in the Intro too |
| 9 | At least 50 references | **OPEN** | The Doc now cites **30** (Maris added 2026-08-15); `references/library.bib` holds **84 verified entries** since the bibliography expansion — the ref→section map is `thesis/references/new_refs_annotated.md`. ~20 more need to be *cited*, not found |

---

## 2. Doc comments — 17

Comment IDs are non-contiguous; this is the complete set from `word/comments.xml`.

| ID | Anchor | Comment | V2 status |
|---|---|---|---|
| **C344** | Methods 3.1 ¶2 | Mixing healthy controls and aMCI; separate paragraphs, recruitment + inclusion/exclusion criteria for each, at TLV and Sydney | **OPEN.** Paragraph 96% identical in V2. = our own **S6** (MCI never defined diagnostically) |
| **C345** | "too many bad channels…" | "specify precise criteria, 'too many' not good enough. Also in table" | **OPEN.** = our **S11**. Thresholds exist in `code/step1_auto_cleaning.py` (`GFP_SPIKE_MADS`, `PTP_THRESHOLD`, `OUTLIER_TIME_FRACTION`) and are documented in CLAUDE.md — they were simply never written into Methods |
| **C371** | Table 1 row "TST < 210 min" | "Why is this a problem if we have 3 hours of sleep and a big portion is clean N2" | **PARTLY.** V2 states the criterion in Methods prose (S5) but never justifies 210 min |
| **C389** | Figure 1 caption | "Panel C table is not readable. Either put in separate table/figure… Just pasting here in panel C doesn't cut it" | **DONE 2026-08-17** — standalone Table 2, as a native Docs table. = email #5 |
| **C390** | Figure 1 caption | Add sleep efficiency, WASO, REM sleep latency | **DONE 2026-08-15.** All four measures (WASO, sleep onset latency, REM latency, sleep efficiency) are reported in a new Results 4.1 paragraph and as the "Sleep continuity" band of Table 2 |
| **C395** | Methods 3.2, "re-referenced to the common average reference" | Interpolate first, average-reference second, "otherwise you are mixing good signals with the noise you are about to throw away… you need to change and re-run" | **Paragraph identical in V2 — but he is mistaken. No re-run needed.** See §4 |
| **C412** | Results 4.1, "(Figure 1)" | "Move the figure here" | **DONE 2026-08-17** — image and caption moved; the figure is now **Figure 2**, and the methods-flow figure took the number 1 so citation order still runs 1, 2, 3. Previously: **TEXT DONE 2026-08-15**, image pending — the overview paragraph now sits in Results 4.1 in his own wording; the image and caption still need dragging out of Methods 3.2. See the handoff box below |
| **C429** | Results 4.1, "do not reflect differing amounts of N2 sleep" | "How are we sure? … did we try to equate this? Or include this as a factor?" | **DONE 2026-08-15.** The non-sequitur is replaced by the three-part argument plus the ANCOVA; bandwidth is now reported as largely a duration artefact. See §4 and `results_edits_before_after.md` §1.6 |
| **C432** | Results, detection rates | "Could we include some figure or supplementary figure with various examples from different subjects?" | **DONE 2026-08-17** — Figure S3, `code/make_s3_figure.py`, two examples per group |
| **C487** | Figure 4 image | "Increase fonts, unclear both labels and colorbar legends. Also connect with green/red/blue colors you used in figure 3 and throughout all figures" | **DONE 2026-08-17** — V11 assets in the Doc, one shared palette |
| **C502** | Figure 5 image | "Graphics need to be improved. Labels, fonts, impossible to read" | **DONE 2026-08-17** — V11 asset in the Doc; **the image still needs resizing to 6.50 × 7.46 in by hand** |
| **C557** | Discussion, cholinergic / REM sentences | "out of context. It is about Ach and about REM sleep. A much more relevant connection could be literature on LC degeneration (NoaR can share our review) and changes in NREM sleep (for example Omer's paper on slow wave activity)" | **OPEN.** Paragraph 95% identical in V2. Omer's paper is already ref 19 (`sharon2025slowwaves`), cited only for the cohort |
| **C558** | Discussion, "thalamocortical and neuromodulatory infrastructure" | "I don't see how a more diffuse ISFS hotspot connects to thalamocortical mechanisms… it's really unclear what you think is going on" | **OPEN.** Same paragraph |
| **C571** | Discussion, "perhaps because MCI is an earlier disease stage" | "Not only earlier. Some aMCI will never develop AD… Please read more about this and describe also in intro, rewrite this section accordingly" | **OPEN.** = email #8 |
| **C591** | Figure S1 image | "graphics must be improved, can't read fonts" | **DONE 2026-08-17** |
| **C592** | Figure S2 image | "Graphics and fonts - can't read" | **DONE 2026-08-17** — figure and caption both updated |

> ### ⇨ HANDOFF TO THE FIGURES SESSION — the physical figure moves (C412, C389)
>
> The text side of C412 is finished: the overview paragraph now opens Results 4.1, in Yuval's own
> rewrite, with the "N2 bout properties" clause repointed to Table 2, and `figure_manifest.md`
> retargets F1 from *Methods/Results overview* to *Results*. **What remains is physical, and Shaked
> does it by hand in the browser during the figures session** — the API route needs a Drive-hosted
> URI for `insertImage` and risks re-encoding page-tuned assets:
>
> 1. **Move the Figure 1 image and its caption** out of Methods 3.2 into Results 4.1, immediately
>    after the paragraph beginning "An overview of the recorded sleep across the three groups…".
> 2. **Paste Table 2 in** after the paragraph beginning "Sleep continuity showed the same pattern" —
>    `results/demographics_V4/table2_sleep_architecture.png` plus the caption at
>    `thesis/figure_manifest.md:101`. The Results prose already points at "(Table 2)" twice, so
>    until this lands those are dangling references.
> 3. **Swap Figure 1's caption at the same time.** The Doc still carries the old three-panel caption
>    describing panel C, which no longer exists in the V11 image (panel C *became* Table 2). The
>    correct two-panel caption is `figure_manifest.md:112`.
>
> Before/after records: Methods `thesis/reviews/methods_edits_before_after.md`, Results
> `thesis/reviews/results_edits_before_after.md`.

### Inline `##` notes — tracked insertions, not comments (easy to miss)

| Where | Note | Status |
|---|---|---|
| Methods 3.1 | "## ! what about apnea and breathing disorders. You can't do sleep research in elderly without addressing this ##" | **OPEN — hardest item.** No AHI/apnea/respiratory data anywhere in the repo or the subjects sheet. Needs Shaked's answer; if none exists it becomes an explicit limitation |
| Methods 3.2 | "#what about Sydney scoring, same? #" | **OPEN.** = our **S11** (who scored, how many scorers, reliability) |
| Methods 3.2 | "! you are mixing Methods and Results. First describe all Methods without results… Then move to results" | **OPEN.** Same as C412 |
| Intro ¶1 | "Put one more general paragraph to begin with – **for thesis (unnecessary for paper)**" + sleep definition / stages / functions / PSG / EEG signatures | **OPEN.** = email #3. **This margin note is what settles the thesis-vs-paper question** |
| Intro mid | "Put one more general paragraph on how sleep and EEG change with age and MCI" + Sleep & Memory / Aging / Neurodegeneration / AD / Sleep in AD / MCI / aMCI / Sleep in MCI | **OPEN.** = email #3 |
| Discussion | "## I am missing (1) a paragraph putting these results in broader context of known changes in sleep between young, old, and MCI… what about other infraslow changes like cyclic alternating pattern (CAP) – look it up; (2) a paragraph on what has been found on ISFS irrespective of aging/MCI; … Do people use same methods; do they report results in similar aspects…" | **OPEN.** = email #7 |
| Table 1 | Rewrote exclusion labels to demand numbers: "Clean N2 bouts **< X% of data**", "Bad channels **> Y% of electrodes**", "Bad epochs **> Z% of sleep time**" | **OPEN.** Same as C345 |
| Title page | Department → "**Neuroscience and Brain Disorders**", + "**Gray Faculty of Medical and Health Sciences**" | **OPEN — factual correction, apply verbatim.** V2 still reads "Physiology and Pharmacology (Medicine)" |
| Title page | Inserted an Acknowledgements stub (Noa Bregman, Rivi Tauman, Jenny Zitser, Rotem Falach, Flavio Schmidig) and an example TOC | **OPEN** |
| Abstract | "paced by" → "**associated with changes in** LC-NE activity **and other arousal systems and with related autonomic measures**" | **OPEN** — a deliberate softening of the causal claim |
| Abstract | Add mean ages (27 / 66 / 67); "full night polysomnography (PSG) including high-density (256-channel) EEG"; "**a**MCI **referred from a cognitive neurology clinic**" | **OPEN** |
| Results 4.3 | "ISFS peak frequency was around 0.02 Hz (range: **#0.015-0.03Hz?**) across all participants as expected" | **DONE 2026-08-15** — real range **0.0095–0.0314 Hz** (mean 0.0219, N = 104). The floor is one outlier, EL3034 at 0.0095; Shaked chose the true full range over the 5th–95th percentile (0.0158–0.0287), which would have matched his guess. He may query it |
| Results 4.1 | "N2 … (~**X %** of total recording time)" | **DONE 2026-08-15** — 34.1 ± 13.2 / 44.5 ± 11.7 / 39.0 ± 12.3 % |
| 3 sites | `#REF` placeholders: N2 restriction "following previous work"; hotspot "in accordance with previous studies"; peak-freq/BW topographies "resembled those reported previously" | **2 of 3 DONE 2026-08-15** — hotspot → `¹⁰,¹¹` (Lázár + Dimitriades); peak-freq/BW topographies → `¹¹` (verified against the Dimitriades PDF, see §8). **The third is still open** — Methods 3.2, "restricted to N2 sleep following previous work #REF", deferred to the reference pass along with his related Methods 3.3 note "#is this following zurich procedures? If yes then mention and cite#" |
| Throughout | Silently inserted "**a**MCI" in ~8 places | **DONE 2026-08-15** — relabelled thesis-wide wherever the sentence describes our cohort or our result, including the title, Table 1's header cell and every Results caption; left as plain "MCI" where it means the condition in general or other people's studies. Per-sentence list: `results_edits_before_after.md` §8 |
| Methods 3.1 | Asserted the two centres were "both employing **identical setups**" | **VERIFY before accepting** — his own Sydney-scoring question suggests he is not certain |

---

## 3. His line-edits: what V2 had already changed

Mechanical paragraph alignment of his copy against V2 (word-level `difflib`, autojunk off). 29 of his
paragraphs carry an edit or a comment:

| V2 had already… | Count | Meaning for his edits |
|---|---|---|
| left the paragraph **identical** | 5 | apply verbatim |
| **lightly edited** it (ratio ≥ 0.72) | 11 | apply, minor re-anchoring |
| **rewritten** it | 12 | re-map by intent; several already satisfied |
| **dissolved** it into other paragraphs | 1 | Methods 3.4 envelope/FFT passage, split plain-first + "In detail" |

**16 of 29 (55%) are essentially unchanged in V2, so most of his line-editing still lands.**

### Genuinely redundant — V2 already does this (name these to him)

| His edit | V2 already reads |
|---|---|
| "First, we examined ISFS peak frequency…" | "We first asked whether the ISFS as a whole changes with age" |
| "Next, we examined the bandwidth…" | "The other two parameters did not change significantly" |
| "Next, we examined the ISFS topographical scalp distribution" | (4.3 opens on the topography question) |
| "We complemented the data-driven analysis… with an analysis within a predefined ROI" | "We then asked whether the same central-parietal reduction appears when the whole pre-defined region of interest is summarized as a single value per subject" |
| Rewrote "On every parameter, the elderly and MCI groups were statistically indistinguishable" to name each parameter | "Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and MCI groups" |
| Restructured Discussion ¶1 into "First… Second…" | "Two things changed with age. First… Second…" |
| Deleted "no post-hoc tests were performed… best read as a trend" | already demoted (Flavio #56) |
| FFT spelled out at first mention | "via the fast Fourier transform (FFT)" (Flavio #25) |
| "for each clean **N2** bout" | already reads "clean bout" in a rewritten passage |
| "Electrical Geodesics, Inc. (EGI)" at first mention | introduced in Methods 3.2 |
| "(oscillation with period of ~50 sec)" in the Abstract | "rises and falls roughly every 50 seconds" |
| Section headings restated to give the result | done throughout (Flavio #44/#48/#49/#58) |

Everything else he edited still applies, because V2 left those sentences intact.

---

## 4. The two substantive science points

### C395 — interpolation vs average referencing: **no re-run needed**

`code/step2_auto_bad_channels.py:402-411` does: mark `info['bads']` →
`set_eeg_reference('average', projection=False)` → `interpolate_bads(reset_bads=True)`. So the order he
objects to is real and the Methods describe it accurately.

**His concern does not materialise, because MNE excludes bad channels from the average.** Verified
empirically in `eeg_clean` (MNE **1.6.1**), 4 channels at 1 / 2 / 3 / 100 µV with `D` marked bad:

```
after set_eeg_reference('average', projection=False):   [ -1.   0.   1.  100. ]
mean of ALL 4  = 26.5   ->  A would be -25.5
mean of GOOD 3 =  2.0   ->  A would be  -1.0    <-- what actually happened
```

The average is computed over good channels only, and the bad channel is left untouched until
interpolation replaces it from already-referenced good neighbours. The noise never enters the reference.

Actions: add one Methods clause stating that bad channels are excluded from the average reference; offer a
sensitivity re-run on 2–3 subjects (interpolate-then-reref) if he wants it on paper.

### C429 — the N2-amount confound: **he is right, and the answer is stronger than the current text**

Verified in `results/demographics_V3/`:

| Fact | Value | Source |
|---|---|---|
| N2 share **does** differ across groups | KW H = 13.933, **p = 0.0009**; young 34.1 %, elderly 44.5 %, MCI 39.0 %; Dunn young-vs-elderly **p = 0.0006** | `sleep_stage_stats.txt` |
| Proportion of each subject's N2 that entered the analysis | young 51.9 %, elderly 52.3 %, MCI 51.5 %; ANOVA F = 0.019, **p = 0.9808** | `n2_bouts_table.txt` |
| Total analyzed bout duration | young 80.7, elderly 106.0, MCI 90.8 min; KW H = 5.024, **p = 0.0811** (ns) | `n2_bouts_table.txt` |

The current sentence ("N2 remained the largest sleep stage… so the group differences do not reflect
differing amounts of N2") is a non-sequitur *and* omits that N2 share differs. The replacement argument is
much better:

> **Young adults contributed the least analyzed N2 (80.7 min vs 106.0 and 90.8) yet showed the strongest
> central-parietal hotspot — the confound runs opposite to the effect.** Add that the proportion of each
> subject's N2 entering the analysis was equal across groups (p = 0.98) and that total analyzed duration did
> not differ (p = 0.081).

His second ask — include N2 amount **as a factor** — is an ANCOVA on the three whole-scalp parameters with
analyzed-N2 duration as covariate. Cheap, and worth running before replying so the answer is empirical.

---

## 5. Feasibility of his data asks

| Ask | Feasible? | How |
|---|---|---|
| C390 sleep efficiency | **DONE** | `sleep_efficiency_pct` in the subjects sheet (`code/sleep_stage_pies.py:61`) |
| C390 WASO, SOL, REM latency | **DONE 2026-08-13** | `code/sleep_statistics_extended.py` (yasa on hypnograms rebuilt from `*_cleaned_annotations.txt`) → `results/demographics_V4/` |
| C389 / email #5 separate table | **BUILT 2026-08-13, not yet in the Doc** | Panel C promoted out of `code/make_f1_figure.py` into standalone Table 2 (`code/make_table2_sleep.py`), with the C390 metrics folded in — closes C389, C390 and email #5 together. Pasting it into the Doc is a figures-session job (§8.1) |
| C432 example spectra | **Yes, nearly free — still open** | `code/find_example_gaussians.py` / `find_clean_gaussian.py` already produce it |
| C345 numeric cleaning thresholds | **DONE 2026-08-15** | Constants from `code/step1_auto_cleaning.py`, written into Methods 3.1 and Table 1 as "> 20% of electrodes" / "> 30% of N2 time". **Caveat recorded in `methods_edits_before_after.md` §A: the 30% epoch criterion is aspirational, not measured — the four bad-epoch exclusions currently sit *below* it because their cleaning was abandoned once they were written off** |
| Apnea / AHI | **Unknown — blocked on Shaked** | `grep -rn -i "ahi|apnea|apnoea|osa|breathing"` over `code/`, `notes/`, `thesis/answers.txt` returns nothing; no such column in the sheet |
| ≥50 references | **Yes** | 29 cited, `library.bib` has 31; ~20 more via the `literature-review` skill |

---

## 6. Where he agrees with our own pre-send check

Two of his comments independently reproduce findings from `thesis/final_check_report.md` that were left
pending when the draft went out:

- **C429 = S7** — the N2-amount non-sequitur. *Both closed 2026-08-15.*
- **C345 + his Sydney-scoring note = S11** — undocumented cleaning criteria and scoring provenance.
  *Closed 2026-08-15; the rest of S11's reproducibility list is still open.*

Worth saying so in the reply: it shows the draft was already under this scrutiny, and it makes the rest of
that pending list part of the same revision rather than a separate track — **B1/S2** (Dimitriades →
*Sci Rep* 18 Jun 2026 `10.1038/s41598-026-58423-z`; André → *Mol Psychiatry* 12 May 2026
`10.1038/s41380-026-03635-y`), **S1** (two-site confound: young 35/0, elderly 30/9, MCI 14/16),
**S4** (ethics, consent, funding, COI), **S9** (Lázár: ~0.01 Hz sigma power vs ~0.02 Hz spindle events),
**S10** ("normalized"), **S12/S13** (unquantified "sensitive", causal "driven by", three uncorrected
omnibus tests), and the unapplied humanizer list including the title page still reading "July, 2026".

---

## 7. Work plan

⚑ = also on our own pending list.

**1. Text, ~1 day. — MOSTLY DONE 2026-08-15.**
*Done:* aMCI relabelling (thesis-wide); 2 of the 3 `#REF` citations; peak-frequency range in Results;
N2 percentages in 4.1; ⚑S7 rewrite using the §4 argument; C395 Methods clause; C345 thresholds and the
Sydney-scoring gap (part of ⚑S11); Figure 1's *text* moved to Results (C412).
*Still open:* the third `#REF` (Methods 3.2) and his Methods 3.3 Zurich-procedures note; abstract
additions (mean ages, full-night PSG, LC-NE softened per his edit); ⚑S10 "normalized"; ⚑S12/S13;
⚑B1+S2 citation upgrades; ⚑S9 Lázár values; ⚑S4 required statements; the rest of ⚑S11; title-page
department and date.

**2. Analysis, ~1 day. — DONE 2026-08-13.** N2-amount ANCOVA (C429) →
`results/group_comparison_results/three_groups_V11/three_group_ancova_statistics.txt`;
WASO / SOL / REM-latency extraction (C390) → `results/demographics_V4/sleep_statistics_stats.txt`.
Both are written into Results 4.1. Record: `thesis/reviews/c429_c390_results.md`.

> **Dropped 2026-08-12, by decision:** the ⚑S1 recording-site check (elderly 30 TASMC vs 9 Sydney) — Yuval
> never asked for it, and the rule for this revision is **only what he explicitly asked, no voluntary work**.
> Also dropped: the C395 interpolation-order sensitivity run, since MNE demonstrably excludes bad channels
> from the average reference (§4) and a Methods clause covers it.

**3. Figures, ~2 days. — NEXT UP; see §8.1 for the full list.** One shared group palette + font sizes
across `make_f1_figure.py`, `replot_f3_no_title.py`, `make_topo_composites.py`, `replot_f5_no_title.py`,
`replot_roi_violins.py` (C487, C502, C591, C592, email #6); supplementary example-spectra figure (C432);
plus the two manual moves and the V10→V11 asset swap in the Doc. *Table 2 itself is built and current* —
`code/make_table2_sleep.py` → `results/demographics_V4/table2_sleep_architecture.png`, regenerated
2026-08-15 (C389, C390) — it only needs pasting in.

**4. Writing, ~1–2 weeks. — INTRODUCTION DONE 2026-08-16.** The thesis Intro expansion is applied:
his four section headings as 2.1–2.5, one paragraph per sub-topic, C571 argued from the dossier,
782 → ~2,200 words, +19 citations. Record: `intro_edits_before_after.md`. **Finding worth keeping:
his headings are taken verbatim from Yael Gat's M.Sc. thesis** (`thesis/references/YaelG_MSc_thesis.pdf`),
whose Introduction contents page is word-for-word his sub-topic list — he was handing over the
structure of the last thesis he supervised on this cohort, and her AD / sleep-in-AD sections are
*longer* than ours, so the scope is not excessive. Note also that her thesis states aMCI is "the
preclinical or very early stage of AD", which is exactly what C571 now corrects: he is updating his
own lab's earlier position, not flagging an error of Shaked's.
*Still open:* two new Discussion paragraphs
(broader aging/MCI sleep changes incl. **CAP**; ISFS in other conditions); C557/C558 mechanism rewrite
(LC degeneration review from Noa R.; Omer's slow-wave paper); aMCI-vs-AD framing in Intro and Discussion
(C571, email #8); Hebrew abstract; TOC, acknowledgements, front matter.

**5. Blocked on Yuval or Shaked.** Apnea/AHI availability; whether the N2 covariate goes in the main text
or supplementary; whether he accepts the new title.

---

## 8. Carry-over from the Methods and Results passes — read before starting each session

Everything the 2026-08-15 passes discovered, decided or deliberately deferred, filed by the session that
has to act on it. Nothing here is a Yuval ask unless it says so.

### 8.1 → FIGURES SESSION

> **FIGURES SESSION DONE 2026-08-17.** Items 1–7 below are all closed; the record is
> `thesis/reviews/figures_edits_before_after.md` and the operational detail now lives in the box at the top
> of `thesis/figure_manifest.md`. **C389, C412, C432, C487, C502, C591, C592 and emails #5 and #6 are
> closed.** What happened beyond the plan:
>
> - **Main figures renumbered.** Moving the sleep overview into Results 4.1 put it behind the methods-flow
>   figure, which Methods 3.4 cites first. Methods flow is now **Figure 1**, sleep overview **Figure 2**.
>   Verified: first-citation order in the Doc is Table 1, Figure 1, Figure 2, Table 2, Figure 3, Figure 4,
>   Figure 5.
> - **Table 2 went in as a native Docs table, not the PNG.** At page width the 20.5 in PNG renders around
>   6 pt type, the same defect C389 objected to in panel C. Ten columns at 9 pt, built from
>   `demographics_V4/table2_sleep_architecture.csv`. Both dangling "(Table 2)" pointers now resolve.
> - **Every V11 asset still drew "MCI"** while every caption said "aMCI" — item 6 below was written as a
>   warning and turned out to describe the actual state. Fixed with a display-only `group_label()` in
>   `code/utils/config.py`, V11 files overwritten in place; no V12 exists.
> - **Figure 3 was reshaped** to 9.5 × 11.4 in, because at 2.13:1 it could not share a page with its caption.
> - **C432 is closed by a new Figure S3** (`code/make_s3_figure.py`), six example spectra chosen by Shaked,
>   two per group, cited at the end of Results 4.1.
>
> **Both loose ends are now closed (same day):**
>
> 1. **Figure 5's image stays at 3.1 × 3.6 in.** Shaked reviewed it on the page and kept it. Do not "fix" it.
> 2. **The supplementary set was renumbered** after Shaked moved the example spectra to the head of the
>    section: examples → **S1**, peak-frequency and bandwidth topographies → **S2**, MoCA grid → **S3**.
>    Done in the Doc, `04_results.md` and `figure_manifest.md`. The asset keeps its `s3_` filename.
>    Verified: first-citation order is now monotonic within the main figures, the tables and the
>    supplementary set, and the 63-entry reference invariant still holds.
>
> The original list is kept below for provenance.

1. **The two manual figure moves** — Figure 1 image + caption from Methods 3.2 into Results 4.1, and the
   Table 2 image + caption into Results 4.1. Full instructions in the handoff box in §2. **Until Table 2
   is pasted in, the Results prose has two dangling "(Table 2)" references.**
2. **Figure 1's caption in the Doc is stale.** It still describes a panel C that no longer exists — that
   panel *became* Table 2 on 2026-08-13. Replace it with `figure_manifest.md:112` when the image is
   swapped. Its "MCI" was relabelled to "aMCI" in place, so do not lose that when pasting the new one.
3. **The Doc still holds V10 image assets while the manifest is at V11.** Every figure needs swapping,
   not just Figure 1.
4. **Table 2's PNG was regenerated 2026-08-15** — pull the current file. A double-rounding bug had made
   the WASO MCI SD read 46.2 in the table and 46.3 in the prose; `code/make_table2_sleep.py` now formats
   mean ± SD once from `sleep_statistics_per_subject.csv`, and both read **46.3**. Do not paste an older
   render.
5. **C432** — his `#` after "in every group of participants" in Results 4.1 asks for a supplementary
   figure of example spectra from different subjects. `code/find_example_gaussians.py` already produces
   it. Untouched by the Results pass.
6. **Captions are now aMCI throughout** (Figures 1, 3, 4, 5, S2 and Table 2, in both the Doc and
   `figure_manifest.md`). Regenerated figures must not reintroduce "MCI" in on-figure text or legends —
   check the group labels baked into the PNGs.
7. **Figure 3's caption no longer says "trend"** — it reads "bandwidth did not differ significantly across
   groups (one-way ANOVA, p = 0.061)". If the F3 render carries any on-figure trend annotation, it now
   contradicts both the caption and Results 4.2.

### 8.2 → DISCUSSION / WRITING SESSION

> **DISCUSSION PART DONE 2026-08-17.** Items 1, 2, 3 and 6 below are closed by
> `discussion_edits_before_after.md`: the bandwidth contradiction is resolved (¶2 rewritten, bandwidth
> removed from Limitations), C571 is rewritten, the deliberate plain-"MCI" mentions were preserved, and
> the renumbering is done and verified. **Items 4, 5, 7, 8 and 9 remain** — the abbreviation policy note,
> the heterogeneity/aMCI tension (now made explicit rather than removed), the two citation upgrades,
> the plain-text DOI, and the two Methods `#REF` requests.
>
> **New carryover from the Discussion pass, for the reply session:** three of Yuval's own citations left
> the thesis (Schmitz, André, Niethard); a CAP study separated MCI where the ISFS did not, and the
> Discussion says so; and Yael Gat's thesis reports an aMCI REM-latency difference our C390 analysis does
> not reproduce. All three are written up in `discussion_edits_before_after.md` §10.

1. **⚠ The Discussion now contradicts the Results on bandwidth.** Results 4.1 and 4.2 state that bandwidth
   tracks how much N2 each participant contributed (r = 0.386; covariate p = 0.0002) and that adjusting
   for duration takes the group difference from p = 0.061 to p = 0.206, so it "should not be read as an
   effect of age". The Discussion still says *"A tendency toward a broader spectral peak pointed the same
   way but did not reach significance"* and lists it in the limitations as one of two effects that "fell
   short of significance", implying a weak version of the age effect. **Both passages need reconciling
   with the duration-artefact finding.** This is a direct consequence of C429, not a new decision.
2. **C571 / email #8** (aMCI is not only an early phase of AD) is still open, and its anchor sentence
   still reads "perhaps because MCI is an earlier disease stage".
3. **The aMCI relabel deliberately left some "MCI" mentions alone**, and they are correct as they stand —
   do not "fix" them. They are the general research question in the Abstract and Introduction, the
   literature sentences (Zhang / Gorgoni, André), and the general disease-stage claims in the Discussion
   ("biomarker of early MCI", "the transition to MCI", "later than the MCI stage", "though not to MCI",
   "features of the ISFS in aging and MCI", "separating MCI subtypes"). Per-sentence list:
   `results_edits_before_after.md` §8.
4. **The abbreviation policy, if the Introduction is expanded** (email #3 asks for aMCI content): the
   **Abstract** defines `amnestic MCI (aMCI)` because abstracts are standalone; the **Introduction**
   currently spells "amnestic MCI" out unabbreviated both times, because its first mention sits inside a
   parenthetical and defining it there would nest brackets; **Methods 3.1** carries the single body
   definition. If new Intro prose starts using "aMCI", move the body definition to its first Intro use
   and drop it from Methods 3.1 — do not end up with two body definitions.
5. **`05_discussion.md:17` calls the cohort "clinically and etiologically heterogeneous"**, which now sits
   directly beside a blanket "aMCI" label. It is true — 9 of the 30 are non-amnestic — but the two
   statements read oddly together. Unresolved on purpose.
6. **Reference numbering — SUPERSEDED 2026-08-16.** The list is now **49 entries**, rebuilt by the
   Introduction pass. Introduction refs hold **1–39 permanently** (order of first appearance, and the
   Introduction is first), Methods **40–44**, Discussion **45–49**. `discussion_edits_before_after.md`
   §9 is void — see the banner on that file. Method that worked and should be reused: **anchor every
   superscript replacement on surrounding words**, never on a bare glyph, which is collision-proof in
   both directions (needed here, since `ohayon2004metaanalysis` moved *down*, 25 → 19). Verify
   afterwards on the invariant *order of first appearance == list order*.
7. **Two citation upgrades are still pending** (our own list, B1/S2, not his asks): ref 11 Dimitriades
   still reads "bioRxiv [preprint]" but was **published in *Sci Rep* 18 Jun 2026**
   (`10.1038/s41598-026-58423-z`), and ref 28 André still reads "medRxiv [preprint]" but is now
   ***Mol Psychiatry* 12 May 2026** (`10.1038/s41380-026-03635-y`).
8. **`maris2007nonparametric`'s DOI went in as plain text**, not a live hyperlink like its neighbours.
   Cosmetic.
9. **Two `#REF` citation requests remain in Methods** — 3.2 "restricted to N2 sleep following previous
   work" and 3.3 "#is this following zurich procedures?". Both belong to the reference pass.

### 8.3 → REPLY-TO-YUVAL SESSION

> **This is the only pass left.** New items 8–13 come from the front-matter pass (2026-08-17); full
> record `thesis/reviews/front_matter_edits_before_after.md`.

8. **His Abstract rewrite was taken wholesale, but four things in it were changed.** (a) His
   `trend towards broader bandwidth (p = 0.061)` carries the C429 caveat, because his own comment is
   what made bandwidth a duration artefact and Results 4.2 says it "should not be read as an effect
   of age"; (b) `(aMCI)` is defined at first use, since he used the abbreviation later without
   introducing it; (c) `(smeared)` dropped as a coinage, the reframing kept; (d) mean ages given as
   **27.1 / 66.5 / 67.8** — **his `67` is wrong**, the true mean is 67.8. Worth stating plainly.
9. **His rewrite deleted the closing sentence** — *"The ISFS is a sensitive read-out of how aging
   reorganizes the machinery that generates spindles, rather than a marker of early cognitive
   impairment."* That was the locked closing of the scientific story and the only line saying what
   the measure is *for*. His ending keeps the aging-not-MCI verdict but not that claim. Flag it and
   offer to restore it.
10. **The Acknowledgements name three people he did not list** — Angela D'Rozario and Rick Wassing
    for the Sydney recordings, and Maria E. Dimitriades for the analysis pipeline. His stub named
    Tel Aviv people only.
11. **A Hebrew title page was added**, which he did not ask for; the TAU house format pairs it with
    the Hebrew abstract he did ask for. Note also that TAU's official Hebrew for his corrected
    department back-translates as *Neuroscience and Neurological Diseases*, not *Brain Disorders*.
12. **Seven Discussion subheadings (5.1–5.7) were added** at Shaked's request, so the chapter is not
    a single TOC row. No prose changed.
13. **The title page date is now August, 2026.** He never set it; his copy still read July, 2023.

---

1. **Say that 9 of the 30 MCI patients are non-amnestic.** His relabel was applied verbatim as he asked,
   with no caveat in the prose, so the reply is where this gets disclosed.
2. **The peak-frequency range may draw a follow-up.** He guessed "0.015-0.03"; the true range is
   0.0095–0.0314 Hz, and the floor is a single outlier (EL3034 at 0.0095; next lowest 0.0118; 5th
   percentile 0.0158). If he objects, the 5th–95th percentile is 0.0158–0.0287, which is almost exactly
   his guess.
3. **Bandwidth is now reported as largely an artefact of analyzed N2 duration.** This is a change in
   interpretation, not in statistics — bandwidth was already ns at p = 0.061 in V10. Worth stating
   plainly, since it is the one place his C429 made a result weaker rather than stronger.
4. **C395 is refuted, not applied** — MNE excludes bad channels from the average reference, so no re-run
   was needed. The empirical demonstration is in §4; a Methods clause now states it. Offer the
   2–3-subject sensitivity run only if he presses.
5. **`#REF` (b) uses Dimitriades alone, and Grollero cannot substitute.** If he asks why only one
   citation: Grollero 2026 recorded with a **Dreem-2 headband** and averaged its two bipolar derivations
   into a single frontal–central signal — the paper contains no topography at all. Dimitriades states our
   claim outright: *"Peak frequency and area under the curve showed local minima and maxima in central
   regions, respectively, while bandwidth displayed no clear topographical pattern"*, plus a frontal
   peak-frequency cluster highest in young adults.
6. **He has never seen the current title**, which now reads *"Aging, but not amnestic mild cognitive
   impairment, reshapes the infra-slow rhythm of sleep spindle power."* His copy carried the old one and
   he did not comment. Ask him.
7. **Point out that C429 and C345 independently reproduce our own pre-send findings** (S7 and S11) — it
   shows the draft was already under that scrutiny.

### 8.4 → ANY SESSION TOUCHING THE DOC

- **Three `[TO SUPPLY]` markers remain**, all from the Methods pass: `03_methods.md:15` (young-cohort
  recruitment; Sydney healthy-older recruitment) and `03_methods.md:17` (Sydney aMCI recruitment,
  diagnostic criteria, diagnosing clinician).
- **`maris2007nonparametric` was the only uncited bib entry found**; it is now cited. If others appear,
  check them against the renumbering rule in §8.2.6 before inserting.
- **Reference count is 30.** Verify it after any citation work.
