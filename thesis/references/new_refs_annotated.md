# New references, annotated by section

Working document for the prose session. Built 2026-08-13 in response to Yuval's review
(triage: `thesis/reviews/yuval_review_triage.md`), email item **#9** (≥50 references) and the
Introduction/Discussion gaps he named in his margin notes and comments **C557** and **C571**.

**53 new entries → `library.bib` now holds 84.** Every DOI was resolved through Crossref before
import; verification detail and the drop list are in `library_status.md`.

Scope note: this session owns `library.bib`, `library_status.md` and this file only. Nothing in
`thesis/chapters/` was touched.

> ### Terminology — read this before searching for more
>
> The phenomenon this thesis calls the **infra-slow fluctuation of sigma power (ISFS)** is
> published by other groups as the **infra-slow oscillation (ISO)** of sigma or spindle power.
> Same rhythm, two names. The first pass of this bibliography searched only the ISFS naming and
> concluded that almost nothing existed outside aging and MCI; re-running on the ISO terminology
> found three clinical populations and a substantial mechanistic literature (added 2026-08-14).
>
> This matters beyond the search: **Yuval will know the ISO literature under that name**, so the
> Discussion should note the equivalence once, explicitly, rather than letting the two naming
> conventions look like two phenomena. Note also `grollero2026iso`, already in the library, uses
> "infraslow oscillations" in its title.
>
> The thesis's own prose convention is unchanged — **write ISFS, not ISO**.

---

## How to use this document

Each gap below is one of the sections Yuval asked for. The **Claim it supports** column is what
the reference can carry — it is deliberately narrow, so nothing here gets cited for a claim it
does not make. **Type** distinguishes primary evidence from review, because he objected once
already to a citation that turned out to be an editorial (`niethard2023spindleaging`).

Bracketed markers like `[verify in full text]` mean the metadata is confirmed but that specific
detail was taken from the abstract and should be checked against the PDF before it goes into
prose as a number.

---

## A. Sleep in general: what it is, stages, functions, PSG, EEG signatures

His margin note: *"Put one more general paragraph to begin with – for thesis (unnecessary for
paper)"*, listing sleep definition / stages / functions / PSG / EEG signatures.

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `markov2006normalsleep` | What sleep is; NREM and REM stages and how they cycle across the night; the neurobiology that generates and times them | Intro, new opening §1.1 | Review |
| `silber2007visualscoring` | How sleep is measured: the PSG montage, and the visual criteria that define Wake/N1/N2/N3/REM — including that N2 is *defined by* spindles and K-complexes | Intro §1.1 (PSG); Methods 3.2 (how these recordings were scored) | Consensus criteria, peer-reviewed |
| `steriade1993slowoscillation` | The cortical slow oscillation underlying NREM slow waves — the second cardinal NREM EEG signature alongside spindles | Intro §1.1 | Primary (intracellular) |

Spindles, the other signature he named, are already covered: `fernandez2020spindles`,
`andrillon2011intracranial`, `purcell2017characterizing`, `molle2011fastslow`.

**Note for Methods.** `silber2007visualscoring` also closes part of his inline question
*"#what about Sydney scoring, same? #"* — it gives the standard both sites' scoring can be
stated against.

## B. Sleep, learning and memory

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `rasch2013memory` | Sleep's role in memory consolidation; the active systems-consolidation account | Intro §1.2 | Review |
| `klinzing2019consolidation` | The slow-oscillation / spindle / ripple hierarchy — *why the timing* of spindles should matter functionally, not just their number | Intro §1.2; Discussion, where ISFS timing is argued to be functionally meaningful | Review |

These two set up an argument the thesis needs but has not yet made explicitly: if consolidation
depends on when spindles occur relative to other rhythms, then a rhythm that organises spindle
*trains* is functionally interesting on its own terms. Existing `boutin2020spindleframework` and
`champetier2023spindlememory` then carry it into the infra-slow timescale.

## C. Sleep and aging

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `li2018normalaging` | Sleep in healthy aging at the level of architecture, continuity and EEG microstructure | Intro §1.3; Results 4.1 as expected-aging context | Review |

Deliberately thin — the library already has `mander2017aging`, `ohayon2004metaanalysis`,
`helfrich2018uncoupled` and `champetier2023spindlememory` here. `li2018normalaging` adds the
clinical-architecture view the others do not cover.

## D. Sleep and neurodegeneration

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `ju2014bidirectional` | Sleep disruption is both a consequence *and* a driver of Alzheimer pathology | Intro §1.4 | Review |
| `xie2013clearance` | Why sleep quality is thought to matter mechanistically: interstitial clearance of solutes, including amyloid-β, increases during sleep | Intro §1.4; Discussion, where Grollero's glymphatic argument is discussed | Primary (rodent) |
| `lucey2019nremtau` | A specific NREM EEG feature — reduced slow-wave activity — tracks tau pathology in early AD | Intro §1.4 | Primary (human) |

`lucey2019nremtau` is the strongest single argument in the Introduction for the thesis's whole
premise: that a quantitative NREM EEG measure can index neurodegenerative pathology.

## E. Alzheimer's disease, and sleep in AD

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `scheltens2021alzheimer` | AD as a disease: epidemiology, pathology, diagnosis, clinical course | Intro §1.5 | Review (Lancet Seminar) |

Sleep *in* AD is already carried by `zhang2022alzheimerreview` (PSG meta-analysis) and
`gorgoni2016parietal` (parietal fast-spindle density in AD and aMCI).

## F. MCI and amnestic MCI; sleep in MCI

Answers C344 ("recruitment + inclusion/exclusion criteria") and our own **S6**, where MCI is
never defined diagnostically anywhere in the manuscript.

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `petersen1999mci` | The definition of amnestic MCI and its operational criteria | Intro §1.6; Methods 3.1 | Primary cohort |
| `albert2011mcicriteria` | The current NIA-AA clinical criteria, **and the distinction between the MCI syndrome and MCI *due to AD*** | Intro §1.6; Methods 3.1 | Consensus criteria |
| `nasreddine2005moca` | The MoCA and its validation as an MCI screening instrument | **Methods 3.1 — currently uncited anywhere despite the MoCA being used throughout, including Results 4.6** | Primary validation |
| `drozario2020objectivesleep` | What objective sleep measurement does and does not show in MCI | Intro §1.6; Discussion, placing the null MCI result | Systematic review + meta-analysis |

`albert2011mcicriteria` does double duty: it is the diagnostic citation *and* part of the C571
argument, because the criteria themselves treat AD aetiology as a separate, probabilistic
judgement rather than a property of the MCI label.

## G. Amnestic MCI is not simply an early stage of AD — **C571**

> *"Not only earlier. Some aMCI will never develop AD… Please read more about this and describe
> also in intro, rewrite this section accordingly."*

The sentence he is objecting to is Discussion ¶5: *"perhaps because MCI is an earlier disease
stage"*. The Introduction needs the same correction.

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `mitchell2009progression` | Most people with MCI do **not** progress to dementia, even over long follow-up; and a meaningful share of those who do progress to non-AD dementias | Intro §1.6; Discussion ¶5 | Meta-analysis, 41 inception cohorts |
| `malekahmadi2016reversion` | aMCI frequently **reverts** to normal cognition | Intro §1.6; Discussion ¶5 | Meta-analysis, 25 studies |
| `roberts2014reversion` | MCI status fluctuates within individuals — but reverters stay at elevated risk | Discussion ¶5 | Primary, population-based |
| `jicha2006neuropathologic` | Even among aMCI patients who *do* progress to dementia, the underlying pathology is frequently **not** AD | Intro §1.6; Discussion ¶5 | Primary autopsy series |
| `ferman2013nonamnestic` | MCI subtype predicts which dementia follows — but neither mapping is exclusive | Intro §1.6 (defining aMCI against the other subtypes) | Primary longitudinal |

### C571 dossier — the numbers, ready to write from

All taken from the published abstracts; each is flagged where the full text should be checked
before a specific figure goes into the thesis.

**1. Most MCI does not convert.** `mitchell2009progression`, pooling 41 robust inception
cohorts with Mayo-defined MCI at baseline: the cumulative proportion progressing to dementia was
**39.2 % in specialist settings and 21.9 % in population studies**; the adjusted **annual**
conversion rate was **9.6 % and 4.9 %** respectively. The authors' own conclusion is the
quotable one — *"most people with MCI will not progress to dementia even after 10 years of
follow-up."*

**2. Conversion is not only to AD.** In the same meta-analysis, annual conversion to vascular
dementia was **1.9 % (specialist) / 1.6 % (community)** against **8.1 % / 6.8 %** to AD — so a
non-trivial minority of converters go somewhere other than AD, before other dementias are even
counted.

**3. aMCI often reverts.** `malekahmadi2016reversion`, 25 studies, overall reversion rate to
normal cognition **≈ 24 %**, and strongly setting-dependent: **14 % in clinic-based samples
versus 31 % in community samples**. This paper is specifically about *amnestic* MCI, which is
exactly Yuval's wording. The clinic/community split is directly relevant here, since this
cohort was recruited through memory clinics — i.e. the *lower* reversion figure applies.

**4. Reversion does not mean the risk is gone.** `roberts2014reversion` (Mayo Clinic Study of
Aging, n = 534, median 5.1 y): **38 % reverted to normal cognition**, but **65 % of those
subsequently developed MCI or dementia again** (HR 6.6 versus consistently normal). Useful for
keeping the claim honest — the point is that MCI is a fluctuating state, not that it is benign.

**5. Even among converters, the pathology is heterogeneous.** `jicha2006neuropathologic`,
34 community-based aMCI patients who progressed to dementia and came to autopsy: **10 of 34
(29 %) had a non-AD primary pathological diagnosis** — hippocampal sclerosis, argyrophilic grain
disease, Lewy body disease and vascular pathology. All had enough mesial-temporal pathology to
explain the amnestic presentation *regardless of cause*, and **neither demographics nor cognitive
measures predicted which patients would turn out to have AD pathology**. This is the single
strongest citation for his point, and the last clause is worth quoting: an aMCI diagnosis, and
the cognitive testing behind it, does not identify the disease.

**6. State the counter-evidence fairly.** `ferman2013nonamnestic` (337 MCI patients, 2–12 y)
shows subtype *does* predict outcome: amnestic MCI progressed to probable AD at **17 events per
100 person-years** versus **1.5** to DLB, while non-amnestic MCI reversed this (**20** to DLB
versus **1.6** to AD). So aMCI genuinely is enriched for AD. The defensible position is not
"aMCI is unrelated to AD" but **"aMCI is enriched for AD without being equivalent to it"** —
enriched by subtype, yet ~24 % revert, most never convert, and ~29 % of those who do have
something else in the brain.

**Suggested framing for the Discussion ¶5 rewrite.** The current clause explains the null MCI
result by MCI being "an earlier disease stage". The above supports a better explanation, and one
that does not depend on the AD-continuum assumption he rejects: an MCI group recruited by
cognitive criteria is *aetiologically mixed*, so any AD-specific ISFS effect would be diluted by
the substantial fraction of participants who do not have, and will not develop, AD pathology.
That reading also sits naturally beside the existing limitation about clinical and etiological
heterogeneity in Discussion ¶6, which currently asserts heterogeneity without citing anything.

## H. Cyclic alternating pattern (CAP) and other infra-slow phenomena

> *"what about other infraslow changes like cyclic alternating pattern (CAP) – look it up"*

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `terzano2001cap` | What CAP is, and the consensus rules for scoring it: a periodic alternation of phase A and phase B within NREM | Discussion, new "broader context" ¶ | Consensus scoring rules |
| `parrino2012cap` | CAP as *the* established marker of NREM sleep instability; the A1/A2/A3 subtypes; how CAP changes with age and disease | Discussion, same ¶ | Review |
| `maestri2015capmci` | CAP measured in MCI: CAP rate and the slow A1 component are reduced in MCI, and further in AD | Discussion, same ¶ | Primary (11 MCI / 11 AD / 11 controls) |
| `zheng2026capdementia` | CAP features prospectively predict incident dementia, and do so **better than conventional sleep parameters** | Discussion, same ¶ | Primary prospective cohort (MrOS) |
| `silvani2026cycles` | CAP and the infra-slow sigma fluctuation belong to one framework of nested NREM cycles | Discussion, same ¶ — **the bridge sentence** | Review (2026) |
| `vanhatalo2004infraslow` | An independent, non-sigma infra-slow (<0.1 Hz) cortical rhythm in human sleep that modulates cortical excitability and epileptiform activity | Discussion, same ¶ | Primary (human full-band EEG) |
| `fultz2019coupled` | NREM sleep carries coupled neural, haemodynamic and CSF oscillations on an infra-slow timescale | Discussion ¶8 (future work) | Primary (EEG-fMRI) |
| `parrino2025phasic` | CAP across sleep disorders — **and, in the authors' own words, that LC infra-slow oscillatory activity during NREM partially overlaps CAP in periodicity** | Discussion, same ¶ — the CAP↔ISO↔LC link, stated by the CAP authorities themselves | Review (2025) |
| `picchioni2011infraslow` | Infra-slow EEG oscillations organise large-scale cortical/subcortical activity during sleep | Discussion, same ¶ | Primary (EEG/fMRI) |
| `dash2019infraslow` | **Slow-wave activity also fluctuates infra-slowly** (~40–120 s), coordinated across cortical sites | Discussion, scope caveat | Primary (rodent) |
| `bergel2026conserved` | Sleep-dependent infra-slow rhythms are **evolutionarily conserved across reptiles and mammals** | Intro §1.2 or Discussion — establishes the rhythm as a deeply conserved feature of sleep, not a rodent curiosity | Primary (comparative) |

**Two things to flag for the writer.**

1. `silvani2026cycles` is the reference that makes this paragraph writable in one move — it is a
   2026 *Sleep Medicine Reviews* piece that already places CAP and the infra-slow sigma
   fluctuation in a single framework, so the paragraph does not have to construct the link from
   scratch.
2. `maestri2015capmci` is the closest published analogue to this thesis's question — the same
   contrast (MCI vs healthy elderly), the same night, an infra-slow NREM-instability measure —
   but using CAP instead of the sigma envelope, and it **did** find a group difference where the
   ISFS did not. Worth engaging with directly rather than only citing: n = 11 per group, so it is
   small, and CAP rate is a different construct from sigma-envelope spectral power. This is a
   genuine point of contrast, not a contradiction, and handling it explicitly will read well.
3. `fultz2019coupled` fills an existing hole: Discussion ¶8 currently refers to *"the infra-slow
   hemodynamic fluctuations that share its timescale"* with no citation at all.

## I. ISFS outside aging and MCI — **and a finding about the field**

> *"a paragraph on what has been found on ISFS irrespective of aging/MCI; … Do people use same
> methods; do they report results in similar aspects"*

**Answer: three clinical populations, two research communities, two different methods, and
effects that point in opposite directions.** That is a far better paragraph than the one this
section originally contained — see the terminology box at the top for why the first version got
it wrong.

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `dimitriades2025schizophrenia` | ISFS **strength reduced** in childhood- and early-onset schizophrenia, specifically over central-parietal electrodes; no correlation with clinical characteristics or spindle density | Discussion, new "ISFS elsewhere" ¶ | Primary |
| `liu2026autism` | ISO measurable in early childhood (peak just below 0.02 Hz in both groups); **no surviving group difference** in autism, but ISO power correlated with autism severity in males over posterior/temporal regions | Discussion, same ¶ | Primary |
| `sun2026longcovid` | **ISO power in the slow sigma band elevated** in ME/CFS; long COVID showed other microstructural changes | Discussion, same ¶ | Primary |
| `osorioforero2025gatekeeper` | Current rodent statement of the mechanism: infra-slow LC activity fluctuations gate NREM substates and the NREM→REM transition | Discussion, mechanism ¶ (with C557) | Primary (rodent) |

### Comparability table — the direct answer to "same methods, same aspects?"

Two communities measure this rhythm, and they do not do it the same way. **Group A** (Zurich —
Dimitriades/Huber; and Bristol — Grollero) fits a Gaussian to the sigma-envelope spectrum and
reports peak frequency, bandwidth and strength; this thesis's pipeline is Group A. **Group B**
(Boston — Sun/Westover; shared authorship across `liu2026autism` and `sun2026longcovid`)
computes ISO **relative band power** in a fixed 0.005–0.03 Hz window, with no peak fit and hence
no frequency or bandwidth parameter at all.

| Study | Population | Band | Parameters reported | Comparable to ours? |
|---|---|---|---|---|
| `dimitriades2024isfs` (now *Sci Rep*, see flags) | Development, childhood → young adult | fast spindle | peak frequency, bandwidth, AUC | **Yes — identical.** This thesis's pipeline is ported from it |
| `dimitriades2025schizophrenia` | Childhood- / early-onset schizophrenia (17 + 11), age 9–21, vs 56 matched controls | **10–16 Hz** | ISFS **strength**, per electrode | **Mostly.** Same group, same pipeline family. Band is wider than our 13–16 Hz — *note the difference rather than glossing it*. Whether frequency and bandwidth were also tested is not stated in the abstract `[verify in full text]` |
| `grollero2026iso` | Clinical AD (10) vs 20 age-matched controls | fast spindle | peak frequency, **amplitude**, bandwidth | **Nearly.** Strength is peak amplitude, not AUC — the manuscript already says this |
| `liu2026autism` | Autistic (26) vs typically developing (27) children, 1.1–5.1 y | ISO 0.005–0.03 Hz, relative power | ISO **relative band power** only | **Partly.** Confirms a peak just below 0.02 Hz, but no Gaussian fit, so no frequency/bandwidth to compare |
| `sun2026longcovid` | Long COVID (28), ME/CFS (19), controls (28) | ISO 0.005–0.03 Hz, in the **slow sigma band 11–13 Hz** | ISO **relative band power** only | **Partly**, same as above — and note it is *slow* sigma, i.e. the frontal slow-spindle band, not our fast-spindle 13–16 Hz |
| `lazar2019infraslow` | Healthy adults | high sigma | ISO frequency; differs for sigma power vs spindle events | Partly — no bandwidth/AUC parameterisation |
| `lecci2017infraslow` | Mouse + human validation | sigma | ~0.02 Hz periodicity | Foundational, not parameterised the same way |
| `chen2025spindletiming` | Large-N humans | — | point-process spindle timing, not a spectral fit | No — different formalism |

### What to say in the Discussion

**1. The naming.** State once that ISFS and ISO refer to the same rhythm. Without it, half this
literature looks unrelated to the thesis.

**2. The method split is the real answer to his question.** He asked "do people use same
methods; do they report results in similar aspects". They do not, and the split is clean: only
the Zurich/Bristol lineage parameterises the spectral peak, so **peak frequency and bandwidth —
two of this thesis's three measures — have essentially no comparison literature outside it.**
Strength is the only parameter with any cross-study currency, and even there the operational
definition varies (AUC here, peak amplitude in Grollero, relative band power in the Boston
studies). That is a genuine methodological observation and it strengthens rather than weakens
the thesis.

**3. The schizophrenia parallel is the strongest single point.** `dimitriades2025schizophrenia`
reports reduced ISFS strength **specifically over central-parietal electrodes**, with **no
correlation to clinical characteristics** and none to spindle density. That is structurally the
same result as this thesis: a weakened central-parietal hotspot that does not track clinical
severity (our null MoCA). Different condition, same regional signature, same failure to track
symptom scores — supporting the reading of central-parietal ISFS as a non-specific marker of
thalamocortical/neuromodulatory integrity rather than a disease-specific one.

**4. Direction is not consistent across conditions, and that is worth saying plainly.** Strength
is *reduced* in schizophrenia and in aging (here), but *elevated* in ME/CFS
(`sun2026longcovid`). If the rhythm indexes arousability, deviation in either direction may be
what matters — which is exactly the inverted-U that `luthi2025microarousals` proposes for
noradrenergic fluctuation amplitude. The two ideas support each other and give the paragraph a
conclusion rather than a list.

**5. Do not overstate `liu2026autism`.** Its group difference in ISO power **did not survive
correction for multiple comparisons**; only a within-sex severity correlation did. Cite it for
"the rhythm is present and measurable in early childhood, and has been looked for in autism",
not as evidence of an autism effect.

## J. Locus coeruleus degeneration in aging and neurodegeneration — **C557**

> *"out of context. It is about Ach and about REM sleep. A much more relevant connection could be
> literature on LC degeneration (NoaR can share our review) and changes in NREM sleep (for
> example Omer's paper on slow wave activity)"*

The passage to replace is Discussion ¶4, which currently runs through `schmitz2018cholinergic`
(cholinergic basal forebrain) and `andre2025remslowing` (REM slowing) — exactly the ACh/REM
material he called out.

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `mather2016locuscoeruleus` | LC function, its vulnerability in aging, and why LC integrity matters for cognition | Discussion ¶4, opening the replacement argument | Review |
| `braak2011stages` | Abnormal tau appears **in the locus coeruleus before the entorhinal cortex**, often in young adulthood — LC degeneration is the earliest event, not a late one | Discussion ¶4; also Intro §1.4 | Primary neuropathology (2332 brains) |
| `dahl2019rostrallc` | LC integrity measured in vivo is associated with memory in older adults | Discussion ¶4, landing the argument in humans rather than rodents | Primary (human MRI) |
| `osorioforero2022locuscoeruleus` | What the LC does specifically *during sleep*, including its infra-slow activity patterns | Discussion ¶4 | Review |
| `luthi2025microarousals` | **Weakened noradrenergic infra-slow fluctuations occur in neurodegenerative disease**; benefits follow an inverted-U in fluctuation amplitude | Discussion ¶4 — the load-bearing citation | Review |
| `hauglund2025vasomotion` | Infra-slow NE oscillations drive glymphatic clearance during NREM; LC optogenetics and zolpidem establish causality | Discussion ¶4, and where Grollero's glymphatic argument is engaged | Primary (rodent) |
| `kjaerby2026neuromodulators` | ACh, 5-HT, DA, histamine and NE **all** oscillate infra-slowly and synchronously in NREM; silencing LC abolishes the NE oscillation and reduces the cholinergic one | Discussion ¶4 | Primary (rodent) |

**`kjaerby2026neuromodulators` rescues the cholinergic thread.** Yuval's objection to the current
¶4 was that it is "about Ach and about REM sleep" and therefore out of context. This paper puts
acetylcholine back in context: it fluctuates infra-slowly *during NREM*, under locus-coeruleus
control. So `schmitz2018cholinergic` need not be deleted — it can be re-framed as one arm of a
neuromodulatory system that oscillates on the thesis's own timescale, with the REM material
(`andre2025remslowing`) demoted to a brief parallel rather than carrying the argument.

**The argument this makes available.** The rodent work already cited
(`lecci2017infraslow`, `osorioforero2021noradrenergic`, now `osorioforero2025gatekeeper`) says
the LC clocks the infra-slow sigma rhythm. `braak2011stages` says the LC is the *first* site of
AD-type tau pathology. `dahl2019rostrallc` says LC integrity tracks cognition in older humans.
Together these make a much tighter chain than the cholinergic/REM passage: the structure that
paces the rhythm is the structure that degenerates earliest — which also predicts, correctly,
that the effect should be visible in *healthy aging* rather than appearing only at the MCI stage.
That is the thesis's actual result, so C557 and the AGING-not-MCI framing reinforce each other.

**The NREM half of his comment** is already in the library: `sharon2025slowwaves` (Omer Sharon's
slow-wave paper) is currently cited only for cohort provenance in Methods 3.1. He is asking for
it to be used *substantively* in the Discussion as the NREM-sleep counterpart.

### The review he offered — used as a source, **not cited**

`thesis/references/Nir_etAl_LC_NE_Sleep_Review.docx`, Yuval's own lab review ("NoaR" is **Noa
Regev**, last author). It is unpublished — no journal version and no preprint exists — so by
decision (Shaked, 2026-08-14) **the review itself is not in the bibliography**. Its 227-reference
list was mined instead, and those published sources are cited directly. Everything below is
therefore background for the writer, to be attributed to the primary references named, not to the
review.

| Cite key | Claim it supports | Target | Type |
|---|---|---|---|
| `poe2020locuscoeruleus` | The LC as a structure: organization, function, modularity | Discussion ¶4, first mention of the LC | Review |
| `matchett2021vulnerability` | **Why** the LC is selectively vulnerable in AD — mechanism, not just observation | Discussion ¶4 | Review |
| `theofilas2017stereology` | Stereological LC neuron loss across Braak stages in postmortem human brain | Discussion ¶4; Intro §1.4 | Primary |
| `zarow2003neuronalloss` | LC neuron loss **exceeds** that of the cholinergic nucleus basalis and the substantia nigra in AD and PD | Discussion ¶4 — the quantitative reason to lead with LC over the cholinergic account | Primary |
| `weinshenker2018noradrenergic` | Noradrenergic dysfunction is early and causally relevant across neurodegenerative disease | Discussion ¶4 | Review |
| `galgani2023locuscoeruleus` | LC MRI abnormality **in amnestic MCI** predicts future progression to dementia | Discussion ¶4 **and** the C571 argument | Primary |
| `vanegroo2022sleepwake` | LC-NE in sleep-wake regulation, and what its degeneration implies for aging and AD | Discussion ¶4 — best single citation for the replacement paragraph | Review |
| `vanegroo2021awakenings` | LC integrity (7T MRI) relates to disrupted sleep continuity, alongside AD plasma markers | Discussion ¶4 — lands the LC-and-sleep argument in humans | Primary |
| `teng2025synchrony` | Multiple neuromodulatory systems fluctuate in synchrony during NREM | Discussion ¶4, with `kjaerby2026neuromodulators` | Primary |

**`galgani2023locuscoeruleus` is the pick of the batch** — it answers C557 and C571 in one
citation. LC imaging measured specifically in *amnestic* MCI, where the abnormality separates
those who later progress from those who don't. For C557 it puts LC degeneration in exactly this
thesis's population; for C571 it is biological evidence that aMCI is a heterogeneous group rather
than a uniform pre-AD stage.

**`zarow2003neuronalloss` is what licenses the rewrite.** Yuval's objection was that the passage
leads with acetylcholine; this paper shows LC loss is quantitatively *greater* than nucleus
basalis loss in AD. That is the reason to reorder the paragraph, and it can be stated in a clause.

What the review establishes as background (cite the primaries, not the review):

- **The mechanism, with its direction.** During NREM, LC activity and NE levels across thalamus,
  basal forebrain and brainstem oscillate at ~0.02 Hz / ~50 s cycles. NE **peaks** coincide with
  micro-arousals; NE **troughs** coincide with spindles, because thalamic NE release suppresses
  spindles. So **LC infra-slow activity is anti-correlated with spindle power** — a directional
  detail the current Discussion does not state and should.
  → cite `osorioforero2021noradrenergic`, `kjaerby2022norepinephrine`, `osorioforero2025gatekeeper`.
- **The disconnection framing.** High LC-NE + low spindles = windows of sensitivity to
  awakening; low LC-NE + high spindles = moments of disconnection — the fragile/offline
  alternation the thesis already invokes.
  → cite `lecci2017infraslow`, `osorioforero2021noradrenergic`, `cardis2021corticoautonomic`.
- **The neurodegeneration link for C557.** Excessive LC-NE signalling intruding into sleep as a
  common mechanism linking disrupted sleep to neurodegeneration, in a vicious cycle spanning
  stress disorders, Alzheimer's and Parkinson's. This is the replacement argument for the ACh/REM
  passage he objected to.
  → cite `matchett2021vulnerability`, `weinshenker2018noradrenergic`, `vanegroo2022sleepwake`,
  `luthi2025microarousals`, and `slutsky2024dyshomeostasis` for the circuit-level version.
- **The rhythm is not noradrenaline-only, and not mammal-only.**
  → cite `kjaerby2026neuromodulators` and `teng2025synchrony` (multiple neuromodulators
  synchronized), `bergel2026conserved` (conserved across reptiles and mammals).
- **It uses the ISO naming throughout, and defines sigma as 10–16 Hz.** Two consequences: the
  ISFS/ISO equivalence must be stated explicitly in the thesis — Yuval's own review would
  otherwise look like it is about a different rhythm — and his band definition differs from this
  thesis's 13–16 Hz, which is worth a clause.

**Independent validation of the ISO sweep:** before it arrived, the library already contained
five of the papers the review cites for these points — `osorioforero2025gatekeeper`,
`kjaerby2026neuromodulators`, `hauglund2025vasomotion`, `osorioforero2022locuscoeruleus` and
`lecci2017infraslow`.

**If the review is published before submission**, reconsider citing it directly — it would then
be a straightforward `@article`. Until then it stays out, and the primaries above carry the
argument.

---

## Coverage

| | Count |
|---|---|
| Entries in `library.bib` before | 31 |
| New entries, 2026-08-13 (gap sweep) | 31 |
| New entries, 2026-08-14 (ISO-term re-sweep) | 9 |
| New entries, 2026-08-14 (mined from the C557 review's bibliography) | 13 |
| **Entries now** | **84** |
| Cite keys actually used in `thesis/chapters/*.md` | 29 |
| Available but not yet cited | 55 |

Yuval's threshold is ≥50. The bibliography clears it comfortably, but **the count that will be
checked is the reference list in the manuscript, not the .bib file** — so the Intro and
Discussion expansions have to actually use these. Citing all 53 new entries takes the manuscript
to 82; there is now enough slack to leave some uncited and still clear 50 well.

Two of the 55 uncited keys are *not* new: `maris2007nonparametric` and `tallonbaudry1997gamma`
appear in no chapter `.md` file. They may be cited in the Google Doc Methods only; worth a check
during the citation pass.

---

## Search log

Databases: Crossref REST API (`api.crossref.org`), PubMed E-utilities, web search. Date searched:
2026-08-13. No date restriction; English only; journal articles only, per the agreed constraint.

| Source | Query / operation | Results | Kept |
|---|---|---|---|
| Crossref | Direct DOI resolution of 17 seed candidates, gaps A–E | 17/17 resolved | 11 |
| Crossref | Direct DOI resolution of 20 seed candidates, gaps F–J | 20/20 resolved | 16 |
| Web | `infraslow oscillation sigma power sleep spindle insomnia OR "sleep apnea" OR depression human EEG 0.02 Hz` | 8 | 0 direct; led to Watson 2018 |
| Web | `cyclic alternating pattern CAP "mild cognitive impairment" OR "Alzheimer" NREM sleep instability study` | 9 | 2 (Zheng 2026, Maestri 2015) |
| Web | `"infraslow" sigma power fluctuation NREM sleep patients "0.02 Hz" clinical population Luthi human study 2023 2024` | 8 | 2 (schizophrenia ISFS, Silvani 2026) |
| Crossref | `query.bibliographic` title searches for the 4 papers above | 3 rows each | 4 |
| Crossref | `infraslow fluctuation of sigma power sleep` | 12 | 1 |
| Crossref | `infraslow oscillation sigma power NREM sleep insomnia patients` | 12 | 0 |
| Crossref | `0.02 Hz infraslow sigma spindle sleep apnea arousal patients` | 12 | 0 |
| Crossref | `infraslow sigma power fluctuation Parkinson REM sleep behavior disorder epilepsy` | 12 | 0 |
| PubMed | `esummary` pagination check, 5 AMA/Neurology entries | 5 | — |
| PubMed | `efetch` abstracts for the C571 evidence set + schizophrenia ISFS + Maestri | 8 | — |
| Crossref | Full-file re-resolution of all 62 entries post-import | 62/62 resolved | — |

**ISO-term re-sweep, 2026-08-14.** Prompted by Shaked: the phenomenon is also published as
"infra-slow oscillations (ISO)" of sigma/spindle power, which the 2026-08-13 queries missed.
PubMed, seven formulations, no date restriction:

| Source | Query | Results | Kept |
|---|---|---|---|
| PubMed | `("infraslow"[tiab] OR "infra-slow"[tiab]) AND ("sigma"[tiab] OR "spindle"[tiab] OR "spindles"[tiab])` | 29 | 3 |
| PubMed | `("infraslow oscillation"[tiab] OR "infra-slow oscillation"[tiab] OR …"oscillations"…) AND sleep[tiab]` | 33 | 4 |
| PubMed | `("0.02 Hz"[tiab] OR "0.02-Hz"[tiab]) AND sleep[tiab]` | 18 | 0 new |
| PubMed | `("infraslow"[tiab] OR "infra-slow"[tiab]) AND ("NREM"[tiab] OR "non-REM"[tiab] OR "NREMS"[tiab])` | 37 | 2 |
| PubMed | `("sigma power"[tiab] OR "spindle activity"[tiab]) AND (fluctuation…) AND ("50 s"[tiab] OR infraslow…)` | 11 | 0 new |
| PubMed | `"fragility"[tiab] AND "NREM"[tiab] AND sleep[tiab]` | 2 | 0 new |
| PubMed | `"NREM substates"[tiab] OR "NREM sleep substates"[tiab]` | 1 | 0 new |
| — | **Unique PMIDs across all seven** | **74** | **9** |
| Crossref | Verification of the 9 kept DOIs | 9/9 resolved | 9 |
| Crossref | Full-file re-resolution of all 71 entries post-import | 71/71 resolved | — |

Deduplication was by DOI, then by normalised title + first author + year. Three works appeared
under more than one DOI; in every case the journal article was kept and the conference abstract
or preprint dropped. See `library_status.md` for both drop lists and reasons.

**Review-bibliography mining, 2026-08-14 (third pass).** Source: `Nir_etAl_LC_NE_Sleep_Review.docx`.

| Operation | Results | Kept |
|---|---|---|
| Extracted the numbered reference list from `word/document.xml` | 227 of 230 parsed | — |
| Filtered on LC / noradrenergic / neurodegeneration / aging / AD / PD / dementia / glymphatic | 127 | — |
| Crossref `query.bibliographic` verification of the shortlist | 13 resolved, 1 was a conference abstract → traced to the real paper | 12 |
| Plus `bergel2026conserved`, taken from the same list earlier the same day | 1 | 1 |

**Lessons worth recording.**

1. The first sweep's conclusion that "ISFS outside aging/MCI is a near-empty literature" was an
   artefact of searching one of the two names in use. Search **both** ISFS and ISO, and both
   "infra-slow"/"infraslow" spellings — PubMed treats them as distinct `[tiab]` tokens.
2. **A relevant review's reference list is a better search tool than any query.** Three passes
   were run here; the third — reading one review's bibliography — produced the single most useful
   reference of the whole exercise (`galgani2023locuscoeruleus`), which no keyword sweep had
   surfaced because it sits at the intersection of two topics rather than inside either.
3. The conference-abstract trap appeared **three times** across the three passes, each time as a
   Crossref `journal-article` record. Always check `type`, page count, issue (`S3`-style
   supplements) and whether the author list has been collapsed to initials.

---

## Raised, not acted on

Per the scope rule for this session, these are reported rather than fixed.

1. **`dimitriades2024isfs` is now published.** The entry is the bioRxiv preprint; the peer-reviewed
   version is *Scientific Reports* (2026), `10.1038/s41598-026-58423-z`. Verified resolving, 14
   authors matching. Title changed slightly. This is the thesis's central methods reference, so it
   matters more than the others.
2. **`andre2025remslowing` is now published.** *Molecular Psychiatry* (2026),
   `10.1038/s41380-026-03635-y`. Verified resolving. Both title and author list changed — now 14
   authors (Fliaguine and Gauthier added versus the 12 currently in the entry).
3. **`sharon2025slowwaves` is missing volume, issue and pages.** Crossref gives
   **21(5):e70247**; the entry was created when the paper was early-access. Worth fixing in the
   same pass as items 1–2. Separately — and more importantly — **this reference is under-used**:
   C557 names "Omer's paper on slow wave activity" alongside the LC literature, and it is
   currently cited *only* as dataset provenance in Methods 3.1. He wants it working in the
   Discussion as the NREM counterpart to the LC argument. See gap J.
4. **`grollero2026iso`** re-checked — still a preprint, no change needed.
5. **`visbrain` and `sleepeegpy`** are the only two cite keys that break the `authorYEARkeyword`
   convention. Pre-existing; left alone.
6. **`niethard2023spindleaging`** remains an editorial. Already known and already demoted to a
   secondary cite — flagged here only because it is the precedent for checking publication type,
   which caught the retired AAN guideline and **three** conference abstracts across the three
   passes.
7. **The Nir-lab LC review is deliberately not cited** (unpublished; decided 2026-08-14). If it
   is published before submission, revisit — it would then be a normal `@article`.
