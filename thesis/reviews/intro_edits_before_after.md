# Introduction edits — before / after, for review before anything touches the Doc

**Target:** Google Doc "Shaked's Thesis V2" (`1YpXrDGFlzRk…`), the Introduction chapter.
**Mirror:** `thesis/chapters/02_introduction.md`.
Drafted 2026-08-16, revised 2026-08-16 after your read-through.
**APPLIED to the Doc and mirrored to `02_introduction.md` / `03_methods.md` on 2026-08-16.**

> ### ✅ What was applied, and the numbering decision that changed
>
> The Introduction, the two Methods de-duplications, and **the full reference list** all went in.
>
> **The reference list was rebuilt to 49 entries, not left unnumbered.** The split-prose-from-numbering
> plan assumed the Introduction's numbers could not be fixed until the Discussion pass landed. That
> turned out to be wrong, and in our favour: the list is auto-numbered by **order of first
> appearance**, and the Introduction is the first chapter in the document, so **every reference that
> first appears in the Introduction occupies 1–39 permanently**, whatever the Discussion later adds.
> Methods takes 40–44. Only 45–49 (Chen, Schmitz, André, Niethard, Grollero) can still move.
>
> Applying prose without numbers would have left placeholder tokens in a document you might open or
> share, so the Doc was brought to a fully self-consistent 49-entry state instead.
>
> **Verified after applying** (Docs API, programmatic):
> | Check | Result |
> |---|---|
> | Reference list entries | 49 |
> | Distinct numbers cited | 49, max 49, no gaps, none cited without an entry |
> | **Order of first appearance == list order** | **YES** (1…49 exactly) |
> | Malformed or reversed ranges | none |
> | Reference paragraphs keeping auto-numbered list formatting | 49 / 49, one listId |
> | DOI hyperlinks live | 49 / 49, link target == link text |
> | Table 1 | intact |
> | Heading outline | 2.1–2.5 as HEADING_3, matching Methods 3.x / Results 4.x |
> | Doc Introduction vs `02_introduction.md` | identical, 24 / 24 paragraphs |
>
> **Consequence for the Discussion session: `discussion_edits_before_after.md` §9 is void.** It was
> computed against a 30-entry baseline and assumed "refs 1–26 do not move". See the banner added to
> the top of that file.

Scope = only the Introduction items Yuval explicitly raised: **email #3** (too ISFS-focused; wants
sleep in general, aMCI/AD, sleep and aging), his **two margin notes with their section headings and
sub-topics**, **C571** (aMCI is not only an early phase of AD, "describe also in intro"), and his
**in-place tracked edits** inside the Introduction.

> ### ⚠ Citations carry no numbers in this document
>
> `discussion_edits_before_after.md` is drafted and unapplied, and it renumbers the same list — six
> references are claimed by both passes, and it deletes two entries this pass would otherwise have
> numbered. Per your call, **prose and numbering are split**: citations appear here as `[@citekey]`,
> and a single combined pass afterwards rebuilds the reference list once and writes every
> superscript once. That pass **supersedes `discussion_edits_before_after.md` §9**.

---

## §0 — YOUR QUESTIONS, ANSWERED

### Q1. The definition of "sleep" — take it from Yael's intro ✅ done

Yael's opening sentence (`YaelG_MSc_thesis.pdf` p. 8), attributed to Markov & Goldman 2006 —
**the same reference already in our library as `markov2006normalsleep`**:

> *"Sleep is a state of reduced consciousness and immobility, defined by a reversible disconnection
> from the environment that is homeostatically regulated."*

Adopted, near-verbatim, as the first sentence of 2.1. Four elements, all kept: reduced
consciousness, immobility, reversible disconnection, homeostatic regulation.

> **One thing to decide:** it now sits almost word-for-word in two theses from the same lab under
> the same supervisor. That is defensible — it is a textbook definition from a cited source, and it
> is the phrasing he likes — but say so if you would rather I reword it while keeping all four
> elements.

### Q2. Do we need the per-stage EEG scoring criteria? ❌ **No — removed**

**Checked: Yael does not have them.** Her intro never lists what defines each stage epoch by epoch,
never mentions K-complexes, and never mentions 30 s epochs or scoring rules. What she does instead
is describe the signatures narratively — the waking EEG slows, high-amplitude (>75 µV) slow waves
at 0.5–4 Hz emerge, N3 is >20% slow-wave activity, REM has wake-like cortical activity with rapid
eye movements and suppressed muscle tone — and then names *"sleep spindles in N2, slow-wave
activity of SWS"* as the stage-characteristic patterns.

So the four-item criteria list I had drafted is more granular than the reference thesis, and it
duplicated 2.1 ¶4, which already describes slow waves and spindles properly. **Deleted.** What
survives in the PSG paragraph is the montage, the 30 s epoching, and the single criterion that is
load-bearing for this thesis — that **N2 is *defined by* spindles**. The slow-wave/N3 point moved
down into ¶4 where the slow oscillation is described.

### Q3. Is the AD + sleep-in-AD material too much, or off topic? — **No, and here is why**

**Yael's Introduction headings are, word for word, the sub-topic list Yuval wrote in your margin.**
Her table of contents (p. 4):

| Yael's Introduction | Yuval's margin note to you |
|---|---|
| Sleep | *"Sleep what is it, how is it defined, sleep stages NREM and REM"* |
| Sleep, Learning and Memory | *"Sleep, Learning and Memory"* |
| Sleep and Aging | *"Sleep and Aging"* |
| Sleep and Neurodegeneration | *"Sleep and Neurodegeneration"* |
| Alzheimer's Disease (AD) | *"Alzheimer's Disease (AD)"* |
| Sleep in Alzheimer's Disease | *"Sleep in Alzheimer's Disease"* |
| Mild Cognitive Impairment (MCI) | *"Mild Cognitive Impairment (MCI)"* |
| Amnestic Mild Cognitive Impairment | *"Amnestic Mild Cognitive Impairment"* |
| Sleep in MCI | *"Sleep in MCI — what is known"* |

He was not inventing a structure. **He was handing you the one from the last thesis he supervised
on this cohort**, in her order, with her headings.

On proportion: her AD section runs about two pages (amyloid cascade vs tau, clinical course and
BPSD, the cholinergic basal forebrain in detail, and current drug approaches), and sleep-in-AD
about a page and a half, inside a nine-page introduction. **Ours is one paragraph each — leaner
than the precedent, not heavier.** So the material is neither too much nor off topic by his
standard. If anything, the honest note is that our thesis is about the ISFS with aMCI as one of
three groups and a *null* result there, whereas hers was about aMCI/AD sleep as its whole subject —
which is a fair reason to stay at one paragraph each rather than expanding to match her.

> **Worth knowing, and a little awkward.** Yael's thesis asserts repeatedly that *"aMCI is
> considered the preclinical or very early stage of AD"* and *"widely considered as a prodromal
> phase of AD"*, with conversion rates of 30–60% at three years and up to 80% at six (Morris 2001).
> **That is precisely the framing C571 now tells us to correct.** His comment is not aimed at a
> mistake of yours — it is him updating a position his own lab published in 2023. Two consequences:
> the C571 paragraph should read as current evidence rather than as a correction of anyone, which
> is how it is written below; and the conversion figures in the two theses will not match
> (`mitchell2009progression` gives 39.2% specialist / 21.9% population) because hers come from a
> single clinic-based cohort and ours from a 41-cohort meta-analysis. No action needed, but do not
> be surprised by it.

### Q4. Grollero in 2.4 — **you are right, and I was wrong. Removed.**

The memories are explicit and I contradicted them:

- `project_thesis.md`: *"Published AFTER the user's analysis was complete… Cite as concurrent/
  related work in Discussion (its only home) — **never as motivation**, since it postdates the
  work."*
- `project_scientific_story.md`: *"**Grollero moved OUT of the Intro → Discussion §5.3 is its ONLY
  home** (postdates analysis, never motivation)."*

My draft had put it in 2.4 as background, which is exactly the framing both memories forbid. **The
Grollero sentence is deleted**; 2.4 now closes on the young-adult/development gap alone, and
Grollero stays where it belongs — the Discussion, as convergent concurrent work.

### Q5. `galgani2023locuscoeruleus` — Intro or Discussion? → **Discussion**

Three reasons:

1. **It does double duty there and only single duty here.** In the Discussion it answers C557 (LC
   degeneration) and C571 (aMCI heterogeneity) with one citation, inside a paragraph being built
   around the LC. In the Intro it would serve C571 only.
2. **It would arrive before its own argument.** The Intro mentions the LC twice — as the pacemaker
   of the ISFS (2.2) and as the first site of tau pathology (2.3, Braak) — but never builds the LC
   degeneration case. An LC-imaging biomarker result would land without that scaffolding.
3. **You asked me to shorten the aMCI paragraph**, and C571 is already carried there by four
   converging references. A fifth dilutes rather than strengthens it.

`discussion_edits_before_after.md` already claims it as its ref 33. Leaving it there.

> **If you ever do want it in the Intro**, the place is the *AD* paragraph attached to the Braak
> sentence — "tau appears in the LC first, and LC abnormality measured in vivo in aMCI predicts who
> later progresses" — **not** the aMCI paragraph. Noted so the option is not lost.

---

## §1 — DECISIONS TAKEN (both settled by you, 2026-08-16)

| | Decision |
|---|---|
| **A1** | **Option A** — refocus the existing aging/spindles paragraph on what its heading says, rather than repeating §2.3 two paragraphs later. *"There is no reason to repeat ourselves."* |
| **A2** | **Split the `¹²⁻¹⁴` bundle** — Mander and Helfrich carry the changes, Champetier carries the link to cognitive decline |
| **Abbreviations** | Define **`Alzheimer's disease (AD)`** and **`amnestic MCI (aMCI)`** once each, at first use, then use the short forms throughout |

**Where the abbreviations are defined** (each at its genuine first use, so nothing is used before
it is introduced):

| Term | Defined in | Consequence |
|---|---|---|
| NREM | opening paragraph (unchanged) | 2.1 and 2.2 no longer re-expand it |
| REM, PSG, EEG | 2.1 | Methods 3.2 re-expands PSG — see §6.2 |
| MCI | 2.3, the MCI paragraph | 2.4 no longer re-expands it |
| **AD** | 2.3, the neurodegeneration paragraph | every later mention is "AD" |
| **aMCI** | 2.3, the amnestic-MCI paragraph | **Methods 3.1 must drop its definition** — see §6.2 |

One knock-on: the sleep-in-AD paragraph originally cited `gorgoni2016parietal` for "AD **and
amnestic MCI**", which would have used aMCI a paragraph before it is defined. It now cites Gorgoni
for the AD finding only, and **2.4 carries the extension to aMCI** — where the term is already
defined, and where the finding does more work anyway.

---

## §2 — the structure, and where it comes from

Extracted from `Shaked's Thesis_YN.docx` (`word/document.xml`, paragraphs 62–84, walking `w:ins`) —
the §3 triage summary never listed these. **He wrote the section headings himself**, and per §0.Q3
he took them from Yael's thesis. Layout, in his order, numbered to match Methods `3.x` /
Results `4.x`:

```
Introduction
  ¶ unheaded opener                                               kept, one clause softened (§7.1)
  2.1 Human sleep and scalp EEG                                   NEW,  4 ¶
  2.2 Infra-slow fluctuation of sigma power (ISFS) in NREM sleep  existing 3 ¶ + his edits
  2.3 Changes in sleep and EEG across aging and MCI               NEW,  8 ¶ (one per sub-topic)
  2.4 Changes in sleep spindles and ISFS across aging and MCI     existing 1 ¶, refocused + close
  2.5 The present study                                           existing 1 ¶ + his edits
```

782 words → ≈2,200 (down from ≈2,550 in the version you read, after the trims in §6).

---

## §3 — the opening paragraph (§7.1 applied)

**BEFORE:**

> Sleep is not a uniform state, and it is not uniformly protected. Within a single stage of
> non-rapid eye movement (NREM) sleep the brain alternates every few tens of seconds between
> periods in which it is easily woken and periods in which it is relatively sealed off from the
> outside world. In rodents this alternation is a rhythm with a clock: **it is paced by** the
> noradrenergic locus coeruleus, and in the electroencephalogram it appears as a slow waxing and
> waning of sleep spindle activity. Because both the spindles and the neuromodulatory systems that
> **pace** them are known to deteriorate with age, this rhythm is a natural place to look for the
> sleep signature of an aging brain.

**AFTER:**

> Sleep is not a uniform state, and it is not uniformly protected. Within a single stage of
> non-rapid eye movement (NREM) sleep the brain alternates every few tens of seconds between
> periods in which it is easily woken and periods in which it is relatively sealed off from the
> outside world. In rodents this alternation is rhythmic and **closely tied to** the noradrenergic
> locus coeruleus, and in the electroencephalogram it appears as a slow waxing and waning of sleep
> spindle activity. Because both the spindles and the neuromodulatory systems that **shape** them
> are known to deteriorate with age, this rhythm is a natural place to look for the sleep signature
> of an aging brain.

Two clauses. This is the causal softening he applied twice elsewhere and could not apply here
because the paragraph did not exist in his copy — Abstract ("paced by" → "associated with changes
in LC-NE activity and other arousal systems") and 2.2 ¶2 ("clocked by" → "driven in part by").
"a rhythm with a clock" also had to go, since it asserts the same thing the softened clause
withdraws.

---

## §4 — NEW SECTION 2.1: Human sleep and scalp EEG

**BEFORE:** does not exist.

**AFTER:**

> ### 2.1 Human sleep and scalp EEG
>
> Sleep is a state of reduced consciousness and immobility, defined by a reversible disconnection
> from the environment and subject to homeostatic regulation [@markov2006normalsleep]. Its
> reversibility separates it from coma and anaesthesia; its homeostatic control separates it from
> simple rest, in that pressure to sleep accumulates with time spent awake and dissipates during
> sleep, while a circadian process independently sets the times of day at which sleep is most
> likely to occur. In humans sleep recurs daily and is built from cycles of roughly 90 minutes,
> each containing both rapid eye movement (REM) sleep and NREM sleep, with NREM divided in turn
> into stages N1, N2 and N3, through which sleep progressively deepens
> [@markov2006normalsleep]. The balance between the two states shifts across the night: N3
> dominates the early cycles, whereas REM periods lengthen towards morning.
>
> These states are distinguished by polysomnography (PSG), the simultaneous recording of the
> electroencephalogram (EEG) together with the electrooculogram and the submental electromyogram.
> The record is divided into consecutive 30 s epochs, and each epoch is assigned to a single stage
> on the basis of the waveforms it contains rather than on behaviour [@silber2007visualscoring].
> One of those criteria matters directly here: N2 is *defined by* the presence of sleep spindles,
> so any question asked about spindles is also a question about the stage in which they are scored.
>
> Why sleep is necessary is not fully settled, but several functions are well supported. The best
> characterised is its role in memory: material encoded during the day is reactivated during sleep
> and progressively integrated into long-term storage, and sleep after learning improves later
> retention across a wide range of tasks and species [@rasch2013memory]. Sleep also changes the
> physical state of the brain. The interstitial space expands during sleep and the exchange between
> cerebrospinal and interstitial fluid increases, so that metabolic waste, including amyloid-β, is
> cleared substantially faster than during wakefulness [@xie2013clearance]. These two functions
> matter for what follows: the first ties sleep to cognition, the second to the proteins that
> accumulate in neurodegenerative disease.
>
> Two EEG signatures dominate NREM sleep. The first is the slow oscillation, a rhythm below 1 Hz in
> which cortical neurons alternate between a depolarised up state of sustained firing and a
> hyperpolarised down state of near-total silence; this alternation is what appears on the scalp as
> the high-amplitude slow waves that give N3 its name [@steriade1993slowoscillation]. The second is
> the sleep spindle, a brief waxing-and-waning burst of activity in the sigma frequency range,
> generated by the interaction between thalamic reticular and thalamocortical neurons
> [@fernandez2020spindles]. Spindles are the signature this thesis is concerned with, and the
> following section takes them up in detail.

**Covers his sub-topics:** what sleep is / how defined / NREM and REM stages (¶1) · PSG monitoring
(¶2) · functions of sleep (¶3) · EEG signatures, slow waves and spindles (¶4).

---

## §5 — SECTION 2.2: existing ISFS paragraphs, with his edits applied

**You approved all three paragraphs.** Reproduced here as the record of what goes into the Doc. His
edits are **bold**; forced de-duplications are ⟨…⟩ and listed again in §6.2.

### 2.2 ¶1 — *approved*

> ⟨NREM⟩ sleep **occupies most of sleep in adults**, and **stage 2 (N2) is characterized by the
> intermittent occurrence of** sleep spindles: brief oscillations in the sigma band (11–16 Hz)
> generated by thalamocortical circuits [@fernandez2020spindles; @andrillon2011intracranial].
> Spindles are among the most reliable electrophysiological signatures of N2 and vary in density
> and topography across individuals [@purcell2017characterizing]. Two subtypes are commonly
> distinguished by frequency and scalp topography: slow spindles (≈11–13 Hz), maximal over frontal
> cortex, and fast spindles (≈13–16 Hz), maximal over centroparietal cortex [@molle2011fastslow].
> Spindles are also closely tied to **plasticity and** sleep-dependent memory consolidation
> [@fernandez2020spindles], **as well as with increased disconnection from the external sensory
> environment.**

### 2.2 ¶2 — *approved*

> Spindles are not distributed evenly through NREM sleep. They come in trains, and those trains
> recur in a slow rhythm of roughly one train every 50 seconds [@boutin2020spindleframework]. What
> fluctuates at that rate is the sigma envelope, the amplitude of sigma-band activity over time,
> which is modulated at a frequency of about 0.02 Hz. **In both rodents and humans**, this
> infra-slow fluctuation of sigma power (ISFS) organizes NREM sleep into alternating fragile and
> offline substates: during fragile periods sigma power is low and ⟨arousal is easily triggered⟩,
> whereas during offline periods sigma power is high and ⟨the sleeper is⟩ relatively disconnected
> from sensory input and protected from awakening. This alternation is **driven in part by** the
> locus coeruleus **norepinephrine (LC-NE) system** [@lecci2017infraslow;
> @osorioforero2021noradrenergic]. The amplitude of infra-slow noradrenergic oscillations is
> coupled to spindle dynamics and to the restorative and mnemonic functions of sleep
> [@kjaerby2022norepinephrine; @cardis2021corticoautonomic].

⟨…⟩ is forced by his own edit: "In both rodents and humans" makes the two "the animal" clauses
wrong, so they are made species-neutral.

### 2.2 ¶3 — *approved*

Unchanged from the Doc apart from his one surviving preference:

> …following the ISFS across development into young adulthood and quantifying the peak frequency,
> spectral width, and area under the spectral peak of the sigma envelope **for each EEG channel**;
> they linked these three measures to markers of arousal and of memory reactivation
> [@dimitriades2024isfs]. …

---

## §6 — NEW SECTION 2.3, and the reworked 2.4 / 2.5

### 2.3 Changes in sleep and EEG across aging and MCI

**BEFORE:** does not exist.

**AFTER:**

> ### 2.3 Changes in sleep and EEG across aging and MCI
>
> **[Sleep, learning and memory]** The dominant account of what sleep does for memory is systems
> consolidation: representations that initially depend on the hippocampus are reactivated during
> NREM sleep and gradually integrated into neocortical networks, where they become independent of
> the hippocampus and resistant to interference [@rasch2013memory]. What makes this more than a
> transfer of stored material is its timing. Reactivation is carried by a nested hierarchy of
> rhythms in which the slow oscillation groups spindles, and spindles in turn group hippocampal
> sharp-wave ripples, so that the information a ripple carries reaches cortex during a window in
> which cortical plasticity is favoured [@klinzing2019consolidation]. On this account *when*
> spindles occur matters as much as how many of them there are. That is why a rhythm which
> organises spindles into trains is of interest [@boutin2020spindleframework], and the expectation
> is borne out: how tightly spindles cluster on the infra-slow timescale predicts overnight
> retention [@champetier2023spindlememory].
>
> **[Sleep and aging]** Sleep changes across the lifespan in a stereotyped way. Meta-analysis of
> quantitative polysomnographic parameters in healthy sleepers shows total sleep time, sleep
> efficiency, N3 and REM sleep all declining with age, while time in the lighter stages N1 and N2
> and wake after sleep onset increase; most of the change occurs between young adulthood and
> middle age, after which the parameters largely plateau [@ohayon2004metaanalysis]. Clinically the
> picture is of sleep that is shorter, lighter and more fragmented, together with an advance of
> circadian phase and a blunted homeostatic response to sleep loss [@li2018normalaging]. None of
> this is in itself pathological, and that is exactly why it matters here: it is the baseline
> against which any additional effect of cognitive impairment has to be judged.
>
> **[Sleep and neurodegeneration]** The relationship between sleep and neurodegeneration runs in
> both directions. Disrupted sleep is a consequence of Alzheimer's disease (AD) pathology, but it
> is also a contributor to it: interstitial amyloid-β rises during wakefulness and falls during
> sleep, and experimental sleep restriction raises it further, so that chronically poor sleep
> plausibly accelerates deposition, which in turn disrupts sleep [@ju2014bidirectional]. The
> clearance mechanism described in Section 2.1 gives that loop a physical basis [@xie2013clearance].
> What makes it tractable with EEG is that the pathology tracks specific NREM features rather than
> sleep duration: reduced NREM slow-wave activity is associated with tau pathology in early
> symptomatic AD independently of how long patients sleep [@lucey2019nremtau], and the synchrony of
> slow waves across the scalp tracks cognitive impairment in prodromal AD [@sharon2025slowwaves]. A
> quantitative NREM EEG measure can therefore index neurodegenerative pathology, which is the
> premise on which this thesis rests.
>
> **[Alzheimer's disease]** AD is the most common cause of dementia. It is defined
> neuropathologically by extracellular amyloid-β plaques and intracellular neurofibrillary tangles
> of hyperphosphorylated tau, accompanied by synaptic loss and neurodegeneration, and clinically by
> an insidious amnestic decline that later spreads to other cognitive domains
> [@scheltens2021alzheimer]. Its course is long, and it begins far earlier than the clinical
> diagnosis. Tau pathology follows an orderly anatomical progression, and in large autopsy series
> the earliest abnormal tau is found not in cortex but in subcortical nuclei, the locus coeruleus
> in particular, frequently in people in their twenties and thirties, decades before any symptom
> appears [@braak2011stages]. The nucleus in which that pathology appears first is also the nucleus
> whose activity has been linked to the infra-slow sigma rhythm in rodents, a convergence the
> Discussion returns to.
>
> **[Sleep in Alzheimer's disease]** Sleep in AD is disturbed beyond what age alone would predict. A
> systematic review and meta-analysis of polysomnographic studies found reduced total sleep time and
> sleep efficiency, more wake after sleep onset, and less N3 and REM sleep than in healthy
> older controls, with the differences graded by disease severity [@zhang2022alzheimerreview]. The
> microstructure is affected as well, and selectively: fast spindle density is reduced over parietal
> cortex in AD, while frontal slow spindles are comparatively preserved [@gorgoni2016parietal].
>
> **[Mild cognitive impairment]** Mild cognitive impairment (MCI) names the ground between normal
> aging and dementia. As originally characterised it requires a subjective cognitive complaint,
> objective impairment on formal testing relative to age and education, largely preserved general
> cognition, essentially intact activities of daily living, and the absence of dementia
> [@petersen1999mci]. Current criteria retain that clinical core and add an explicit second step:
> once the syndrome is established, the likelihood that it is *due to* AD is judged separately, on
> aetiological rather than syndromic grounds [@albert2011mcicriteria]. The separation of the
> syndrome from its cause is built into the diagnosis itself. In practice identification usually
> begins with a brief cognitive screen, most commonly the Montreal Cognitive Assessment, developed
> specifically to detect the milder deficits that longer-established instruments miss
> [@nasreddine2005moca].
>
> **[Amnestic MCI / C571]** MCI is subdivided by the cognitive domains affected. Amnestic MCI
> (aMCI), in which memory is impaired, is the subtype most closely associated with AD: in a
> longitudinal clinical series it progressed to probable AD at 17 events per 100 person-years
> against 1.5 to dementia with Lewy bodies, while non-amnestic MCI showed the reverse pattern
> [@ferman2013nonamnestic]. It does not follow that aMCI is simply an earlier phase of AD. Most
> people diagnosed with MCI never develop dementia at all: pooling 41 robust inception cohorts,
> 39.2% progressed in specialist settings and 21.9% in population samples, leading the authors to
> conclude that most will not progress even after ten years of follow-up
> [@mitchell2009progression]. Roughly a quarter of aMCI reverts to normal cognition, 14% in
> clinic-based samples such as the present one [@malekahmadi2016reversion]. Among those who do
> convert, the pathology is not
> uniformly Alzheimer's: in an autopsy series of aMCI patients who had progressed to dementia, 10
> of 34 carried a non-AD primary diagnosis, and neither demographic nor cognitive measures
> predicted which [@jicha2006neuropathologic]. aMCI is therefore enriched for AD without being
> equivalent to it, and a group recruited on cognitive criteria will be aetiologically mixed enough
> to dilute any disease-specific effect measured in it.
>
> **[Sleep in MCI: what is known]** What is known about sleep in MCI is less settled than for AD. A
> systematic review and meta-analysis of objectively measured sleep found that people with MCI
> differ from healthy older adults on polysomnographic and actigraphic measures, but with
> substantial heterogeneity between studies and effects generally smaller than those reported in
> dementia [@drozario2020objectivesleep]. Against that background, individual NREM features have
> been put forward as early markers: NREM sleep characteristics predict subsequent cognitive
> impairment in aging populations [@taillard2019nonrem], and spindles, K-complexes and sleep-stage
> proportions have each been proposed as candidate biomarkers of aMCI and AD
> [@liu2020spindlebiomarkers]. Whether any of them separates MCI from healthy aging reliably enough
> to be useful remains open. That is the question this thesis puts to one particular feature of N2
> sleep.

Bracketed labels are for your reading only and do not go into the Doc.

> **What changed since your read-through.** Memory paragraph: *"in its own right rather than as a
> statistical curiosity"* deleted. aMCI paragraph: *"and that association is real"* deleted; the
> MCI-is-not-AD point now stated **once**, in the closing sentence, where the original said it three
> times ("it would be wrong to treat…", "enriched without being equivalent…", "aetiologically
> mixed…"); the list of non-AD pathologies and the 31%-community reversion figure dropped as detail
> the argument does not need. The paragraph is down from ~230 to ~185 words. Sleep-in-AD paragraph:
> `gorgoni2016parietal` now cited for AD only, with the aMCI extension moved to 2.4 (see §1).
>
> **Provenance of the C571 figures.** All are the headline numbers from the published abstracts, as
> recorded in `new_refs_annotated.md` §G. Say the word if you want any checked against full text —
> the `jicha2006neuropathologic` 10-of-34 and the `ferman2013nonamnestic` event rates are the most
> specific.

### 2.4 Changes in sleep spindles and ISFS across aging and MCI

**BEFORE:**

> Healthy aging reshapes NREM sleep and its spindles. Spindle density falls, slow-wave–spindle
> coupling loosens, and the temporal clustering of spindles breaks down, changes that themselves
> track cognitive decline¹²⁻¹⁴. In mild cognitive impairment (MCI) and Alzheimer's disease,
> polysomnographic studies report further sleep disruption and, in particular, reduced parietal
> fast-spindle density¹⁵⁻¹⁶. Because these N2 changes emerge early and carry prognostic
> information, sleep-EEG features have been proposed as candidate markers of incipient cognitive
> impairment¹⁷⁻¹⁸.

**AFTER** (Option A, per your A1):

> ### 2.4 Changes in sleep spindles and ISFS across aging and MCI
>
> Healthy aging reshapes NREM sleep and its spindles. Spindle density falls, slow-wave–spindle
> coupling loosens, and the temporal clustering of spindles breaks down [@mander2017aging;
> @helfrich2018uncoupled], changes that themselves track cognitive decline
> [@champetier2023spindlememory]. The parietal fast-spindle deficit described above extends to aMCI
> as well as AD [@gorgoni2016parietal], and it is focal rather than global: it falls on the same
> spindle subtype and the same region in which the ISFS is most pronounced in young adults
> [@lazar2019infraslow]. Because these N2 changes emerge early and carry prognostic information,
> sleep-EEG features have been proposed as candidate markers of incipient cognitive impairment
> [@taillard2019nonrem; @liu2020spindlebiomarkers].
>
> The ISFS itself, however, has been characterized almost exclusively in young adults and across
> development [@dimitriades2024isfs]. Whether and how it changes with healthy aging, and whether
> aMCI adds a signal beyond that of aging, therefore remain unknown.

Changes from the version you read: **Grollero removed** (§0.Q4); the middle sentence no longer
restates §2.3 but instead carries the aMCI extension of Gorgoni and connects it to the ISFS
hotspot; `¹²⁻¹⁴` split per A2; `(MCI)` no longer expanded.

The second paragraph is the sentence *"Yet the ISFS itself has been characterized almost
exclusively in young adults…"* **moved here from the start of 2.5**. It is the "and ISFS" half of
his heading, which the existing paragraph did not address, and moving it stops 2.4 and 2.5 saying
the same thing twice.

### 2.5 The present study

**BEFORE:**

> Yet the ISFS itself has been characterized almost exclusively in young adults; whether and how it
> changes with healthy aging, and whether MCI adds a signal beyond that of aging, remain unknown.
> Here we quantified the ISFS during clean N2 sleep in three groups (young controls, healthy older
> adults, and patients with amnestic MCI), applying the analysis pipeline of Dimitriades et al.
> (2024) to high-density 256-channel EEG. For each participant we measured the peak frequency,
> bandwidth, and AUC of the fast-spindle (13–16 Hz) sigma-envelope spectrum at every channel, and
> compared the groups across the whole scalp, topographically, and within a pre-defined
> central-parietal region of interest. We asked whether these ISFS parameters and their scalp
> topography differ with age, and whether the ISFS of patients with amnestic MCI is comparable to,
> or different from, that of healthy older adults.

**AFTER** — *approved*:

> ### 2.5 The present study
>
> Here we quantified the ISFS during N2 sleep in three groups (young controls, healthy older
> adults, and patients with **aMCI**), applying the analysis pipeline of Dimitriades et al. (2024)
> to high-density 256-channel EEG. For each participant we measured the peak frequency, bandwidth,
> and AUC of the fast-spindle (13–16 Hz) sigma-envelope spectrum at every **scalp** channel, and
> compared the groups across the whole scalp, topographically, and within a pre-defined
> central-parietal region of interest. We asked whether these ISFS parameters and their scalp
> topography differ with age, and whether the ISFS of patients with **aMCI** is comparable to, or
> different from, that of healthy older adults.

---

## §7 — his tracked edits, and where each one landed

### 7.1 Applied

| # | His edit | Where |
|---|---|---|
| 1 | "occupies most of the human night" → **"occupies most of sleep in adults"** | 2.2 ¶1 |
| 2 | "its second stage (N2) is defined by" → **"stage 2 (N2) is characterized by the intermittent occurrence of"** | 2.2 ¶1 |
| 3 | "vary **systematically** in density" → "vary in density" | 2.2 ¶1 |
| 4 | + **"plasticity and"** sleep-dependent memory consolidation | 2.2 ¶1 |
| 5 | + **"as well as with increased disconnection from the external sensory environment"** | 2.2 ¶1 |
| 6 | "In rodents" → **"In both rodents and humans"** | 2.2 ¶2 |
| 7 | "clocked by the locus coeruleus (LC) and the noradrenergic system" → **"driven in part by the locus coeruleus norepinephrine (LC-NE) system"** | 2.2 ¶2 |
| 8 | "The amplitude of **these** infra-slow noradrenergic oscillations is **in turn** coupled" → "The amplitude of infra-slow noradrenergic oscillations is coupled" | 2.2 ¶2 |
| 9 | "quantifying it per channel" → **"for each EEG channel"** | 2.2 ¶3 |
| 10 | "during **clean** N2 sleep" → "during N2 sleep" | 2.5 |
| 11 | "at every **scalp** channel" | 2.5 |
| 12 | "patients with MCI" → "patients with **aMCI**" | 2.5 |
| 13 | Four section headings + their sub-topic lists | the structure of §4 and §6 |
| 14 | His LC softening, extended to the opening paragraph | §3 |

His #1 also settles **Flavio #7**, which had objected to "most of the human *night*" — night
includes wake, so "most" was wrong there and V2 had softened it to "a large part". NREM genuinely is
most of *sleep*, so his wording is better than both. Applied verbatim.

### 7.2 Forced by his edits or by the new sections — flagged, not silent

| Change | Why |
|---|---|
| "the animal is readily aroused / the animal is relatively disconnected" → species-neutral | His "In both rodents and humans" makes "the animal" wrong (2.2 ¶2) |
| 2.2 ¶1 no longer expands "non-rapid eye movement (NREM)" | Defined in the opener and again in 2.1 — three definitions of one abbreviation |
| 2.4 no longer expands "mild cognitive impairment (MCI)" | Defined in 2.3 |
| `Alzheimer's disease (AD)` and `amnestic MCI (aMCI)` defined once, then abbreviated | Your instruction, 2026-08-16 |
| **Methods 3.1 must drop its aMCI definition** — "patients with amnestic mild cognitive impairment (aMCI; n = 30…)" → "…impairment (n = 30…)" | Otherwise two body definitions. **A one-clause edit to already-approved Methods text** |
| **Methods 3.2 drops its PSG expansion** — "High-density polysomnography (PSG) was recorded" → "High-density PSG was recorded" | Same rule; PSG is now defined in 2.1. **Confirmed by you 2026-08-16** |

### 7.3 Superseded — V2 already does this, no action

| His edit | V2 already reads |
|---|---|
| **Splits 2.2 ¶3's first sentence in two** — "…at approximately 0.02 Hz. **Such modulations are** most prominent in fast spindles…" | **Declined, and this one matters — see the box below** |
| Deletes "subsequently" from "Dimitriades et al. (2024) subsequently narrowed" | Kept — V2's sentence needs it to mark the sequence after Lázár |
| "whether MCI departs from healthy aging or instead mirrors it" | V2 says the same thing, already relabelled aMCI |

> ### ⚠ Do not apply his 2.2 ¶3 restructure — it would reinstate a corrected mis-citation
>
> His edit preserves the V1 sentence *"the ISFS appears during N2 as a modulation of the sigma
> amplitude envelope at approximately **0.02 Hz**"* with `lazar2019infraslow` attached, and merely
> splits it. **That attribution is wrong, and it was fixed on 2026-08-06** — Lázár, Dijk & Lázár
> 2019 report roughly **0.01 Hz** for the sigma-*power* fluctuation and ~0.02 Hz only when tracking
> individual spindle *events*. The 0.02 Hz figure belongs to Lecci 2017.
> (Record: `reference_isfs_frequency_attribution` memory.)
>
> V2's replacement sentence exists precisely to carry that distinction, which is why his version
> cannot be adopted even though the paragraph is otherwise 95% his. **Only "for each EEG channel" is
> taken.** This also pre-empts the obvious examiner question — why our peaks sit near 0.02 Hz when
> the paper cited for the human phenomenon reports 0.01 Hz for the same quantity — so it is worth a
> line in the reply to him.

---

## §8 — spotted, NOT done

1. ~~The opening paragraph still says "paced by"~~ — **done, §3.**
2. `galgani2023locuscoeruleus` → **Discussion**, per §0.Q5. Left there.
3. **Declined by you:** `dimitriades2024isfs` still listed as a bioRxiv preprint though published in
   *Sci Rep* 18 Jun 2026 (`10.1038/s41598-026-58423-z`), and `sharon2025slowwaves` missing
   volume/issue/pages (Crossref: 21(5):e70247). Recorded here so the combined numbering pass does
   not silently "fix" them, and so they stay on the pre-submission list.
4. **Declined by you:** `bergel2026conserved`. Not needed for any claim he asked for.

---

## §9 — audit against the recorded prose rules

Run before applying, against the memories and the returned .docx. All clear except where noted.

| Rule | Source | Result |
|---|---|---|
| No comments anchored in the Introduction | `Shaked's Thesis_YN.docx` `word/comments.xml` | ✅ verified — all 16 comments sit in Methods, Results, Discussion or figures. His Intro input is the margin notes only |
| No tracked edits missed | full `w:ins`/`w:del` walk, paragraphs 50–95 | ✅ verified — P70, P71, P72, P87 and the 13 heading insertions are the complete set |
| Never write "ISO" in prose — always ISFS | `project_isfs_definition` | ✅ 0 occurrences |
| Author names dropped from the Intro; Dimitriades the only one | Flavio #9, `project_flavio_review` | ✅ only "Dimitriades et al. (2024)" |
| `niethard2023spindleaging` is an **editorial**, demoted out of the Intro | S3, `project_pre_send_check` | ✅ absent. **Flagged for the numbering pass so it is not reintroduced** |
| `chen2025spindletiming` reserved for the Discussion | `project_scientific_story` | ✅ absent |
| ROI is "pre-defined central-parietal region of interest" only, never "extended" | `feedback_thesis_prose_rules` | ✅ |
| Aims posed as open questions, no directional hypothesis | `project_scientific_story` | ✅ 2.5 unchanged in this respect |
| No names of our own scripts or files | `feedback_no_code_names_in_prose` | ✅ |
| **Stage labels N1/N2/N3 + REM; "NREM" only for the general category** | `feedback_thesis_prose_rules` (2026-06-13) | ⚠️ **was violated — now fixed.** "slow-wave sleep" → **N3** in the aging paragraph, and "less slow-wave and REM sleep" → "less N3 and REM sleep" in the sleep-in-AD paragraph. "Lighter NREM stages" → "lighter stages N1 and N2" |
| **Em dashes: the Abstract and Discussion were humanized to zero, and the current Introduction has none** | `project_scientific_story` | ⚠️ **was violated — now fixed.** 7 em dashes in the new prose rewritten to colons, commas or sentence breaks. New prose now has **0** |
| Lázár attribution must stay corrected | `reference_isfs_frequency_attribution` | ⚠️ **his own edit would have broken it** — see the box in §7.3 |

**One consequence to carry forward, not to act on now.** If `dimitriades2024isfs` is ever upgraded
to the published *Sci Rep* version (declined by you, §8.3), the in-text year changes too:
"Dimitriades et al. (**2024**)" → "(**2026**)" in **2.2 ¶3 and 2.5**, plus Methods 3.4 and 3.5.
Four sites, not just the list entry.
