# Discussion edits — before / after, for review before anything touches the Doc

**Target:** Google Doc "Shaked's Thesis V2" (`1YpXrDGFlzRk…`), the Discussion chapter, one sentence of the
Introduction (§7), one citation in Methods 3.4 (§10.6), and the reference list.
**Mirror:** `thesis/chapters/05_discussion.md`, `02_introduction.md`, `03_methods.md`.
Drafted 2026-08-15 · rebuilt 2026-08-16 against the applied Introduction · revised through 2026-08-17 on
your line-by-line comments.

> ## ✅ APPLIED TO THE DOC AND MIRRORED — 2026-08-17
>
> Everything below is in the Google Doc and in `thesis/chapters/05_discussion.md`,
> `02_introduction.md` and `03_methods.md`. Verified after applying:
>
> | Check | Result |
> |---|---|
> | Reference list length | **63 entries**, in exactly the planned order |
> | Every ref cited | **63 of 63**; none orphaned, none dangling above 63 |
> | Order of first appearance == list order | **true**, verified programmatically over the whole body |
> | Hyperlink bleed onto the appended entries | **none** — all 17 new entries were plain, and all 17 DOIs were then linked in the house style (#1155CC, underlined) |
> | Em dashes in the Discussion | **0** |
> | "ISO" | **2 occurrences**, both describing other groups' terminology and results |
> | "REM" as a stage word in the Discussion | **1**, in the sleep-architecture sentence; all other hits are "NREM" |
> | "paces" | **2**, one hedged ("whatever paces the cycle") and one explicitly rodent |
> | Bandwidth presented as an age effect | **nowhere** |
> | Discussion paragraph count | 10 → **17** |
>
> Two small wording calls made during application, both flagged rather than silent:
> **(1)** `locus coeruleus (LC)` is defined at its first Discussion use. The Introduction defines
> "LC-NE system" but never bare "LC", so using it undefined would have been wrong.
> **(2)** "a coordinated neuromodulatory rhythm **that the LC coordinates**" became "**under LC
> control**" — the approved wording repeated "coordinated…coordinates". Same meaning, same softening.

> **Every drafted block is followed by a `▸ Plain version` box.** That is for you only, not thesis text.

---

## §0 — WHERE THIS STANDS

### 0.1 What this round changed

| Your point | What I did |
|---|---|
| §4.1 "coinciding" | Replaced. Now "its peaks **marking** micro-arousals and its troughs the spindle trains…" |
| §4.3 rewritten elsewhere | **Taken verbatim.** Not touched, not re-edited |
| §5.2 name the parameter | Done, **and it corrected a real error** — Grollero reports peak **amplitude**, not AUC. See §5.2 |
| §5.2 "cognitive testing" → clinical evaluation; drop "recruited on cognitive criteria" | Both done. The subtype sentence (Ferman) is **also dropped**, since it only made sense when subtypes were in play |
| §5.2 the long sentence | Split into three |
| §5.2 "cognitive label" → mild cognitive impairment | Done |
| §5.2 stop at "is established" | Done. The plasma/memory comparison is gone |
| §5.2 "what relation to AD??" | **Whole closing sentence removed.** You were right, it asserted something the paragraph never showed |
| §6.4 dot → comma | Done, and made grammatical without a splice |
| §6.4 too many "CAP", point repeated | "CAP" **7 → 3**; the conclusion now appears **once**, at the end |
| §8 limitations | **Rebuilt to your three.** Heterogeneity is now about symptom severity and clinic presentation; bandwidth removed entirely |
| §10.1 soften humans not rodents | Applied |
| §10.5 drop Niethard | Applied |
| §10.6 add Tallon-Baudry now | Applied — **and it renumbers Methods and everything after it.** See §9 |

### 0.2 Changed in the last two rounds

| Your point | What I did |
|---|---|
| §5.2 — don't tie their result to any of our three parameters | Rewritten. Their effect was on **peak amplitude**, we did not measure peak amplitude, so it has no direct counterpart here. Our side stated separately: none of our three parameters separated aMCI |
| §5.2 — drop "established" | Done: "reduced in patients with Alzheimer's disease relative to age-matched controls" |
| §5.2 — "most patients who meet the **aMCI** criteria" | **Partly, and you accepted the split.** Ref 32 is about MCI generally, so "aMCI" there would misreport it; the next two clauses are now explicitly amnestic. Reasoning in §5.2 |
| §5.2 — "once **AD** pathology is established" | Done: "once **Alzheimer's** pathology is established" |
| §5.3 — the "parallels" contradiction | **Approved and folded in** |
| `mather2016locuscoeruleus` | **Dropped.** Renumbers 48 onward; list is now **63** |
| Serotonin and dopamine in §4.2 | **Kept**, per your ruling |
| Refs 32 / 34 in §8 | **Kept**, per your ruling |

### 0.3 ✅ ALL DECISIONS CLOSED — nothing is waiting on you

Closed 2026-08-17: **§5.3 approved** · **`mather2016locuscoeruleus` dropped** · **refs 32 and 34 kept** in
§8 · **the MCI/aMCI attribution split accepted** as drafted in §5.2.

Closed earlier: C558 Option A · §4.3 taken verbatim · placement before Limitations · "REM" kept in §6.3 ·
basal-forebrain wording · §4.5 · serotonin and dopamine kept · `niethard2023spindleaging` dropped ·
`tallonbaudry1997gamma` added at Methods 3.4 · LC softening split rodent/human · the second dropped V1
limitation (detection-rate criterion) stays out.

**This file is ready to apply.** Execute §9.4 in order, then §11.

### 0.4 Reference notes, kept for the record

**`mather2016locuscoeruleus` — DROPPED.** Mather M & Harley CW, *"The Locus Coeruleus: Essential for
Maintaining Cognitive Function and the Aging Brain"*, Trends in Cognitive Sciences 2016. A **review**, not
a study: the LC matters unusually much for cognitive reserve and is unusually vulnerable in aging. It had
sat on one topic sentence, *"This is a structure that ages badly"*, which the **next two sentences prove
outright** — tau appears in the LC before anywhere else (ref 25), and the LC loses more neurons than the
nucleus basalis (ref 48). That sentence now carries no citation, deliberately. **List 64 → 63.**

**Refs 32 and 34 on the heterogeneity limitation in §8 — KEPT.**

- **32 = Mitchell & Shiri-Feshki 2009** — meta-analysis of 41 inception cohorts. Most people meeting MCI
  criteria never progress to dementia (39.2% specialist, 21.9% population), and converters do not all
  convert to Alzheimer's.
- **34 = Jicha et al. 2006** — autopsy series of 34 amnestic MCI patients who had progressed to dementia.
  **10 of 34 (29%) had a primary pathology other than Alzheimer's** (hippocampal sclerosis, argyrophilic
  grain disease, Lewy body, vascular), and neither demographics nor cognitive test scores predicted which.

They sit on the clause *"as well as in what underlies them"*, which is exactly what they show. They do
**not** support the symptom-severity half of that limitation, which argues from how the cohort was
recruited rather than from any paper, and which is left uncited for that reason.

---

## §1 — verification

| Checked | Result |
|---|---|
| Doc Discussion + reference list | re-read in full **2026-08-16**, after the Intro pass. 49 entries; 45 Chen, 46 Schmitz, 47 André, 48 Niethard, 49 Grollero |
| Refs 1–41 | frozen (Intro 1–39, Methods 3.2 = 40, 41). **42 onward now moves**, because Tallon-Baudry enters at Methods 3.4 |
| Schmitz (46), André (47) | each cited **only** in Discussion ¶4 → both leave with the ¶4 replacement |
| Niethard (48) | *"Editorial / Commentary on Champetier et al. 2023"*, same journal and issue; cited only in ¶4 → leaves |
| The six already-cited refs | Braak **25**, Ferman **31**, Mitchell **32**, Malek-Ahmadi **33**, Jicha **34**, D'Rozario **35** |
| **Grollero's strength measure** | **peak amplitude, not AUC** — the Doc already says so earlier in ¶6. §5.2 corrected accordingly |
| Bandwidth numbers | `c429_c390_results.md` §1.2–1.3 |
| Em dashes | all rewritten; re-check mechanically before applying |

---

## §2 — ¶2, the bandwidth contradiction (his C429) — settled

**BEFORE**

> A tendency toward a broader spectral peak pointed the same way but did not reach significance.

**AFTER**

> The spectral peak was also numerically broader in both older groups, but that difference did not reach
> significance and largely disappeared once the amount of analyzed N2 sleep was taken into account, so it
> is not read here as an effect of age.

---

## §3 — ¶3, his surviving tracked edits — settled

**BEFORE:** …thalamocortical **machinery that organizes** fast spindles. **In young adults the** ISFS…
**AFTER:** …thalamocortical **mechanisms that organize** fast spindles. **In young adults, the** ISFS…

Plus the softening H2 from §10.1: *"the **clock that paces** spindle trains speeds up"* → *"the **rhythm
that organizes** spindle trains speeds up"*.

---

## §4 — ¶4, the mechanism paragraph (C557 + C558)

**BEFORE** (whole paragraph, replaced)

> The neuromodulatory systems that pace and generate spindles are a likely place for these age effects to
> arise. The same rodent locus-coeruleus and noradrenergic machinery that clocks the infra-slow sigma
> rhythm¹¹⁻¹² is eroded by aging and early neurodegeneration. Cholinergic basal-forebrain projections
> degenerate along a defined cortical topography in early Alzheimer's disease⁴⁶, and REM-sleep EEG slowing
> has been shown to track cortical cholinergic denervation in aging and MCI⁴⁷, a parallel sleep-EEG
> signature of the same decline in a different sleep state. Against this background of waning
> neuromodulation, and the documented breakdown of fast-spindle timing with age¹⁷,⁴⁸, a faster and
> spatially more diffuse ISFS points to a thalamocortical and neuromodulatory infrastructure whose
> temporal and spatial organization coarsens with age, a coarsening that in these data is already fully
> expressed in healthy older adults.

### §4.1 — AFTER, first new paragraph *("coinciding" replaced; numbering updated)*

> The most direct place for these age effects to arise is the noradrenergic system tied to the rhythm
> itself. In rodents the LC does not merely accompany the infra-slow sigma fluctuation but sets its phase.
> Noradrenaline levels across thalamus, basal forebrain and brainstem rise and fall on the same roughly
> 50 s cycle, and because thalamic noradrenaline suppresses spindle generation, LC activity runs
> anti-correlated with sigma power, its peaks marking micro-arousals and its troughs the spindle trains of
> the offline substate¹¹⁻¹³,⁴⁷. This is a structure that ages badly. As noted earlier, the earliest
> abnormal tau in the brain appears in the LC²⁵, and the LC also loses proportionally more neurons in
> Alzheimer's disease than either the cholinergic nucleus basalis or the substantia nigra⁴⁸. The same
> relationship holds in living humans. LC integrity measured in vivo tracks memory performance in older
> adults⁴⁹ and the regulation of their sleep⁵⁰, and in patients with aMCI an LC abnormality distinguishes
> those who go on to develop dementia from those who do not⁵¹. Where the infra-slow noradrenergic
> fluctuation itself has been examined, it is weakened in neurodegenerative disease⁵².

> *Changed earlier:* "its peaks **coinciding with** micro-arousals and its troughs with the spindle
> trains" → "its peaks **marking** micro-arousals and its troughs the spindle trains". Also drops the
> second "with", which was doing nothing. All superscripts shifted by the Methods insertion (§9).
>
> **▸ `mather2016locuscoeruleus` DROPPED (settled).** *"This is a structure that ages badly"* now carries
> no citation, and the two sentences after it establish the claim outright. The reference does not enter
> the list at all. **Final count 63.**

### §4.2 — AFTER, second new paragraph *(numbering updated)*

> Acetylcholine, released from the basal forebrain, runs on the same clock: acetylcholine, serotonin,
> dopamine and noradrenaline all oscillate infra-slowly and in synchrony during NREM sleep, and silencing
> the LC abolishes the noradrenergic oscillation and blunts the cholinergic one⁵³. What changes with age is
> therefore not noradrenaline alone but a coordinated neuromodulatory rhythm that the LC coordinates. A
> second NREM measure points the same way. Slow-wave synchrony, measured in a cohort overlapping the one
> studied here, tracks cognitive impairment in prodromal Alzheimer's disease²³, where the ISFS did not
> separate aMCI from healthy older adults. Two measures of the same nights therefore behave differently,
> and that difference is informative about what each one indexes.

> **▸ Serotonin and dopamine — SETTLED, kept.** They are what make "a coordinated neuromodulatory rhythm"
> true rather than an ACh-plus-NE pair, which is the claim the next sentence rests on.
>
> *Changed earlier:* "a coordinated neuromodulatory rhythm that the LC **paces**" → "**coordinates**"
> (softening H4, §10.1).

### §4.3 — AFTER, third paragraph — **OPTION A, SETTLED (taken verbatim from your rewrite)**

Three short paragraphs. No opening question, no throat-clearing openers, one claim per sentence.

> The ISFS is a modulation of fast-spindle amplitude, so its strength (AUC) at an electrode reflects how
> deeply spindle amplitude rises and falls at that site, not how many spindles occur there. In young
> adults the AUC hotspot was central-parietal, over the region where fast spindles are largest⁹. In
> both older groups that hotspot was gone, while average AUC across the scalp was unchanged, so what was
> lost was regional focus rather than overall strength. That pattern points to a drive that has grown
> less precise rather than weaker.
>
> The peak-frequency effect appeared across the whole scalp, with no region standing out. A change
> present everywhere is more easily placed in whatever paces the cycle than in the cortex where the
> spindles appear, and in rodents the pacing comes from the LC. On that reading the noradrenergic cycle
> itself runs faster in older adults: fragile and offline substates alternate more often, and the
> noradrenergic peak each cycle carries arrives more often with them.
>
> This chain is not shown in these data. Neither noradrenergic nor thalamic activity was measured here,
> so the link from LC decline to a flattened scalp hotspot rests on rodent, neuropathological and imaging
> work. What these data can check is the prediction it makes. A mechanism anchored in a structure that
> degenerates early, and in the general population rather than only in patients, predicts a change that
> is complete in healthy older adults and no further advanced in aMCI, which is the pattern found here.

**⁹** = `molle2011fastslow`, already cited in Introduction 2.2 for "fast spindles (≈13–16 Hz), maximal over
centroparietal cortex", which is exactly the claim being leaned on here. **Existing reference, re-cited —
nothing added to the list.** *(The placeholder in the previous draft is now the real number.)*

> **▸ Plain version — six beats.**
> 1. **AUC = how deeply spindle amplitude rises and falls at that electrode.** Not a spindle count. This
>    is the step he says is missing, and everything else follows from it.
> 2. **The young-adult hotspot is central-parietal** — where fast spindles are largest, now cited.
> 3. **Hotspot gone, scalp average unchanged** → what was lost is regional focus, not strength.
> 4. **A less precise drive fits that** — the mechanistic conclusion, stated once, not re-described.
> 5. **Peak frequency changed everywhere, with no cluster** → it is a statement about the pacing of the
>    cycle, not about the cortex. In rodents that pacing is the LC, so the noradrenergic cycle itself
>    runs faster: both substates alternate more often, and the NE peak comes round with them.
> 6. **Then the honesty paragraph:** we measured neither NE nor thalamus, but the prediction the chain
>    makes — change complete in healthy elderly, no further in aMCI — is the pattern we found.
>
> **SETTLED: no micro-arousal sentence.** ¶2 ends at the noradrenergic peaks. Do not reintroduce it in a
> later pass.
>
> **This section is closed.** Taken exactly as you rewrote it; the only change is the citation placeholder
> resolving to **9**.

### §4.4 — **OPTION B** — superseded, kept only as a record

Option A is chosen. Option B's text is no longer maintained.

### §4.5 — knock-on *(settled)*

Future-work paragraph: *"The **locus coeruleus** that paces the rhythm in rodents…"* → *"The **LC** that
paces the rhythm in rodents…"*. "Paces" **stays** here: it is an explicitly rodent claim (§10.1, R2).

---

## §5 — ¶6, the Grollero paragraph (C571)

### §5.1 — his tracked edit *(settled)*

**BEFORE:** …the same **family** of ISFS features…⁴⁹  **AFTER:** …the same **collection** of ISFS features…⁵⁴

### §5.2 — the C571 rewrite *(rebuilt on your seven points)*

**BEFORE** (from mid-paragraph to the end)

> They diverge on whether cognitive impairment leaves a mark beyond age: the strength of the rhythm was
> reduced in their patients with established AD relative to age-matched controls, whereas our analogous
> strength measure did not separate aMCI from healthy older adults, perhaps because MCI is an earlier
> disease stage. Together, these findings suggest that a disease-specific weakening of the rhythm may
> emerge later than the MCI stage, and that plasma and memory measures can detect associations a brief
> cognitive screen cannot. Taken together, the ISFS is altered by age, and a relation to Alzheimer's
> disease, though not to MCI, has now been reported.

**AFTER**

> They diverge on whether cognitive impairment leaves a mark beyond age. Their effect was on the amplitude
> of the spectral peak, which was reduced in patients with Alzheimer's disease relative to age-matched
> controls. Peak amplitude was not among the parameters measured here, so that result has no
> direct counterpart in our data; what can be said is that none of the three parameters we did measure
> separated aMCI from healthy older adults. The two patient groups are also not the same kind of group.
> Theirs was defined by a confirmed diagnosis of Alzheimer's disease; ours by clinical evaluation of
> cognitive complaints. Amnestic MCI is enriched for Alzheimer's disease but is not equivalent to it. Most
> patients who meet criteria for MCI never progress to dementia³², and roughly a quarter of amnestic cases
> return to normal cognition³³. Among amnestic patients who do progress, a substantial minority are found
> to have a primary pathology other than Alzheimer's³⁴. An Alzheimer's-specific weakening of the rhythm
> would therefore be diluted by the participants who do not have, and will not develop, Alzheimer's
> pathology. The present null is not evidence that the rhythm is preserved in early Alzheimer's disease. It
> is evidence that the rhythm does not track a diagnosis of mild cognitive impairment. A disease-specific
> weakening may become detectable only once Alzheimer's pathology is established.

> **▸ Where this landed.**
> 1. **Not tied to any of our three parameters, as you asked.** The comparison now states the honest
>    position: their effect was on **peak amplitude**, we did not measure peak amplitude, so that result
>    has no direct counterpart here. Our side is then stated separately and without forcing an equivalence:
>    none of our three parameters separated aMCI. *(Naming the parameter also caught a real error — the
>    previous draft called their measure AUC, which the Doc itself contradicts two sentences earlier.)*
> 2. "cognitive testing" → **"clinical evaluation of cognitive complaints"**. "recruited on cognitive
>    criteria is aetiologically mixed" → **deleted**. The mixture idea survives only in the dilution
>    sentence you liked.
> 3. The long sentence → **three short ones**, one fact each: never convert / revert / non-AD pathology.
>    **Now specified**: "roughly a quarter of **amnestic cases**", and "Among **amnestic patients** who do
>    progress" (Jicha's series is amnestic MCI).
> 4. The dilution sentence kept verbatim.
> 5. "a cognitive label" → **"a diagnosis of mild cognitive impairment"**.
> 6. **"once Alzheimer's pathology is established"**, as you asked. The plasma-and-memory comparison is gone.
> 7. **Closing sentence removed.**
>
> **⚠ One thing I did NOT write "aMCI" on, and why.** You asked for *"Most patients who meet the **aMCI**
> criteria…"*. Ref 32 (Mitchell & Shiri-Feshki) pooled cohorts with **Mayo-defined MCI**, not amnestic MCI
> specifically, so attaching "aMCI" to it would misreport the source. It now reads *"Most patients who meet
> criteria for **MCI** never progress to dementia³²"*, and the **next two clauses are both explicitly
> amnestic**, which is where refs 33 and 34 genuinely are. This matches how Introduction 2.3 already splits
> them. Say the word if you'd rather have "aMCI" throughout and accept the looser attribution.
>
> **Also dropped: the Ferman subtype sentence** ("Subtype does carry information³¹"). It only earned its
> place while the paragraph was arguing about MCI subtypes. Ferman stays in the reference list — it is
> cited in Introduction 2.3.

### §5.3 — ⚠ NEW: your change contradicts a sentence earlier in the same paragraph

Saying "peak amplitude has no direct counterpart in our data" is the honest position, but **four sentences
earlier the same paragraph claims a parallel**, and that claim is now too strong:

> **Live text:** "Their result **parallels** our central-parietal cluster, since both implicate a weakening
> of ISFS strength over central regions, although in our data this weakening accompanies healthy aging
> whereas in theirs it further separated AD from age-matched controls."

One paragraph cannot both draw a parallel between the two strength measures and say they cannot be
compared. **Recommended minimal fix:**

> **AFTER:** "Their result is **broadly consistent with** our central-parietal cluster, since both point to
> a weakening of the rhythm over central regions, **though the two strength measures are not the same**;
> in our data this weakening accompanies healthy aging, whereas in theirs it further separated AD from
> age-matched controls."

Two changes: "parallels" → "is broadly consistent with", and one added clause. That keeps the comparison
the paragraph is built on while removing the implication that amplitude and AUC are interchangeable.

**APPROVED 2026-08-17 — applies with the rest of §5.**

---

## §6 — NEW paragraphs (his `##` note) — after ¶6, before Limitations

### §6.1 — ISFS outside aging and MCI, first paragraph *(settled)*

> Beyond aging and cognitive impairment the rhythm has now been measured in a handful of other
> populations. The same phenomenon appears in the literature under two names: what is called here the
> infra-slow fluctuation of sigma power is described by several groups as an infra-slow oscillation (ISO)
> of sigma or spindle power. In young people with childhood- and early-onset schizophrenia, ISFS strength
> is reduced specifically over central-parietal electrodes, and correlates neither with clinical
> characteristics nor with spindle density⁵⁵. That is structurally the same result reported here, a
> weakened central-parietal focus that does not track symptom severity, in a different condition and a far
> younger sample. In the opposite direction, ISO power in the slow-sigma band is elevated in myalgic
> encephalomyelitis and chronic fatigue syndrome⁵⁶. The rhythm is measurable much earlier in life as well:
> in children aged one to five it is already present, with a peak just below 0.02 Hz in autistic and
> typically developing children alike, and no difference between the two groups survived correction for
> multiple comparisons⁵⁷.

### §6.2 — ISFS outside aging and MCI, second paragraph *(numbering updated)*

> The rhythm is not measured the same way across these studies. Two approaches are in use. One fits a
> Gaussian to the sigma-envelope spectrum and reports peak frequency, bandwidth and strength¹⁶,⁵⁴,⁵⁵; this
> is the approach used here. The other computes relative band power in a fixed 0.005–0.03 Hz window, with
> no peak fit and therefore no frequency or bandwidth parameter at all⁵⁶,⁵⁷. Even strength, the one
> parameter with any cross-study currency, is operationalized differently in each: the area under the
> fitted peak here, the peak amplitude in Grollero and colleagues, relative band power in the latter
> approach. The sigma bands differ as well. Two of the three measures reported in this thesis, peak
> frequency and bandwidth, therefore have essentially no comparison literature outside the approach they
> come from. It also means the inconsistent direction of the strength effects, reduced in schizophrenia and
> in aging but elevated in chronic fatigue, cannot yet be read as a contradiction. If the rhythm indexes
> how deeply arousal is modulated across the night, deviation in either direction may be what matters, as
> the inverted-U proposed for the amplitude of noradrenergic fluctuations would predict⁵².

### §6.3 — broader context beyond ISFS *(settled)*

> The ISFS is one measure among many that change across these groups, and it behaves like the others. This
> cohort showed the expected architectural changes with age: less N3 and REM sleep, more wake after sleep
> onset, and lower sleep efficiency. On none of those measures, and on no ISFS parameter, did the aMCI
> group differ from healthy older adults. The absence of an aMCI effect is therefore not specific to the
> ISFS; it is what every measure taken from these nights showed. That is worth setting beside a literature
> in which individual NREM features have been proposed as early markers of cognitive impairment³⁶,³⁷. One
> of those features bears directly on the present result: the parietal fast-spindle deficit reported in
> aMCI²⁷ falls on the same spindle subtype and roughly the same region as the hotspot that flattened here,
> yet in this cohort that regional effect tracked age and not diagnosis. Objectively measured sleep in MCI
> is in general less consistent, and its effects smaller, than in Alzheimer's disease³⁵.

### §6.4 — CAP *(rebuilt on both of your points)*

> The infra-slow organization of NREM sleep also has a much older description in the human literature: the
> cyclic alternating pattern (CAP), a periodic alternation between phases of transient EEG activation and
> relative quiescence, scored by consensus rules and long treated as the marker of NREM instability⁵⁸,⁵⁹.
> The two are measured very differently, CAP by eye as a sequence of discrete events and the ISFS as a
> spectral property of a continuous amplitude envelope. They nonetheless describe the same nested
> organization of NREM sleep on overlapping timescales, and have recently been placed in a single
> framework⁶⁰; that literature also notes that infra-slow LC activity during NREM sleep partially overlaps
> the pattern in periodicity⁶¹. The comparison is therefore informative. CAP is reduced in MCI and reduced
> further in Alzheimer's disease⁶², and its features predict incident dementia better than conventional
> sleep parameters do⁶³. The infra-slow domain evidently carries information about cognitive status that
> the sigma-envelope parameterization, at least as applied here, does not capture.

> **▸ Both points applied.**
> 1. **Dot → comma, without a splice:** *"The two are measured very differently, CAP by eye as a sequence
>    of discrete events and the ISFS as a spectral property of a continuous amplitude envelope."* One
>    sentence, one comma, grammatical.
> 2. **"CAP" 7 → 3 uses.** "the CAP literature itself" → "that literature also"; "overlaps CAP in
>    periodicity" → "overlaps the pattern in periodicity"; "The CAP findings are therefore worth comparing
>    directly" → "The comparison is therefore informative"; "CAP features predict" → "its features
>    predict".
> 3. **The conclusion now appears once, at the end.** Deleted: "An infra-slow measure of NREM instability
>    does therefore separate MCI from healthy aging where ours did not" and "The two results need not
>    conflict, since CAP counts discrete events whereas the ISFS quantifies…". The construct difference is
>    already stated in sentence two, so the defence was redundant as well as repetitive.

---

## §7 — one Introduction sentence *(settled)*

**Where:** Introduction 2.4, final paragraph.

**BEFORE**

> The ISFS itself, however, has been characterized almost exclusively in young adults and across
> development¹⁶.

**AFTER**

> The ISFS itself, however, has not been characterized in older adults or in aMCI¹⁶.

---

## §8 — ¶7, Limitations *(rebuilt to your three)*

**BEFORE** (the live paragraph)

> Several limitations qualify these conclusions. The groups were of modest size and drawn from two
> recording sites, and the older and aMCI cohorts were clinically and etiologically heterogeneous, which
> would obscure any subtle aMCI-specific effect; both cohorts were recruited through memory clinics rather
> than assembled to be etiologically uniform, and separating MCI subtypes would require a substantially
> larger sample than either site could provide. Two effects fell short of significance and should not be
> treated as robust: the broader spectral peak in the older groups (p = 0.061), and the reduction in ISFS
> strength when averaged over the region of interest (p = 0.143), the latter because averaging across the
> whole region dilutes an effect that the cluster-based analysis showed to be focal; a larger cohort may
> resolve whether either is genuine. The design was cross-sectional, and therefore cannot distinguish
> individual aging trajectories or identify the subset of aMCI patients who will progress; only a
> within-subject longitudinal design can do that, which for a rhythm measured across a whole night means
> repeated overnight recordings years apart.

**AFTER**

> Several limitations qualify these conclusions. The groups were of modest size and drawn from two
> recording sites, and the patients were recruited through memory clinics rather than assembled as a
> uniform research cohort. Patients reach a memory clinic at different points in their illness, some
> seeking help earlier than others, so the aMCI group varies in how far cognitive symptoms have progressed
> as well as in what underlies them³²,³⁴. An effect confined to more advanced patients could therefore have
> been missed. Two results would benefit from a larger and more complete sample. The reduction in ISFS
> strength (AUC) when averaged over the region of interest fell short of significance (p = 0.143), because
> averaging across the whole region dilutes an effect that the cluster-based analysis showed to be focal;
> a larger cohort may resolve whether it is genuine. Cognitive scores were available for only part of the
> older sample, which limits the power of the analysis relating the ISFS to MoCA. The design was
> cross-sectional, and therefore cannot distinguish individual aging trajectories or identify the subset
> of aMCI patients who will progress; only a within-subject longitudinal design can do that, which for a
> rhythm measured across a whole night means repeated overnight recordings years apart.

> **▸ Three limitations, as you specified.**
> 1. **Heterogeneity of symptoms, from clinic recruitment.** Patients arrive at a memory clinic at
>    different stages, so the aMCI group is uneven in symptom severity. Then the one honest consequence:
>    an effect confined to more advanced patients could have been missed. **No subtype language, and the
>    sentence you called wrong ("which would obscure any subtle aMCI-specific effect") is gone.**
> 2. **A larger and more complete sample would help**, on two counts: the ROI AUC null (p = 0.143) and the
>    MoCA analysis, which had scores for only part of the older sample.
> 3. **Cross-sectional design** — unchanged.
>
> *Also:* "and should not be treated as robust" **deleted** as you asked; the rest of that clause is
> verbatim. **Bandwidth removed entirely** from Limitations — Results 4.1 and 4.2 already explain the
> duration dependence twice, and §2 above stops the Discussion contradicting them, so nothing is lost.
>
> **▸ Open item 3.** The citations ³²,³⁴ sit on "as well as in what underlies them", which is the clause
> they actually support (Mitchell = conversion rates, Jicha = non-AD pathology at autopsy). They do **not**
> support the symptom-severity claim, which is an argument from how the cohort was recruited rather than
> from a paper. Keep them there, or drop them and leave the whole limitation uncited?

---

## §9 — reference changes **(RECOMPUTED — two structural changes since the last version)**

Three decisions move the numbering:

- **`niethard2023spindleaging` dropped** (§10.5) → one fewer.
- **`mather2016locuscoeruleus` dropped** (§0.4, settled 2026-08-17) → never enters the list.
- **`tallonbaudry1997gamma` added at Methods 3.4** (§10.6) → one more, **and it enters before the existing
  Methods citations**, so 42 onward all shift.

Net: **49 + 17 new − 3 removed = 63.**

### 9.1 — removed (3)

| Old # | Key | Why |
|---|---|---|
| 46 | `schmitz2018cholinergic` | ACh kept only as the Kjaerby point |
| 47 | `andre2025remslowing` | REM removed entirely |
| 48 | `niethard2023spindleaging` | editorial on Champetier; adds nothing over ref 17 |

All three are cited only in Discussion ¶4, so all three markers vanish with the replacement.

### 9.2 — existing entries that shift

| Key | Old | New | Where the marker lives |
|---|---|---|---|
| `mnepython` | 42 | **43** | Methods 3.6 and 3.7 (**two sites**) |
| `maris2007nonparametric` | 43 | **44** | Methods 3.6 |
| `vallat2021yasa` | 44 | **45** | Methods 3.7 |
| `chen2025spindletiming` | 45 | **46** | Discussion ¶3 |
| `grollero2026iso` | 49 | **54** | Discussion ¶6 |

Refs **1–41 do not move.** Use the Intro pass's method: **anchor every replacement on surrounding words,
never a bare glyph.** Numbers move in one direction here, but anchoring is collision-proof either way.

### 9.3 — the 17 new refs

**New in Methods 3.4**, inserted after `sleepeegpy` (41):

| # | Key | Entry |
|---|---|---|
| 42 | `tallonbaudry1997gamma` | Tallon-Baudry C, Bertrand O, Delpuech C, Pernier J. Oscillatory gamma-band (30–70 Hz) activity induced by a visual search task in humans. Journal of Neuroscience. 1997;17(2):722–734. https://doi.org/10.1523/JNEUROSCI.17-02-00722.1997 |

Marker placement: Methods 3.4, after *"with a fixed width of 4 cycles"*.

**New in Discussion ¶4**, inserted after Chen (46):

| # | Key | Entry |
|---|---|---|
| 47 | `osorioforero2025gatekeeper` | Osorio-Forero A, Foustoukos G, Cardis R, Cherrad N, Devenoges C, Fernandez LMJ, et al. Infraslow noradrenergic locus coeruleus activity fluctuations are gatekeepers of the NREM–REM sleep cycle. Nature Neuroscience. 2025;28(1):84–96. https://doi.org/10.1038/s41593-024-01822-0 |
| 48 | `zarow2003neuronalloss` | Zarow C, Lyness SA, Mortimer JA, Chui HC. Neuronal Loss Is Greater in the Locus Coeruleus Than Nucleus Basalis and Substantia Nigra in Alzheimer and Parkinson Diseases. Archives of Neurology. 2003;60(3):337–341. https://doi.org/10.1001/archneur.60.3.337 |
| 49 | `dahl2019rostrallc` | Dahl MJ, Mather M, Düzel S, Bodammer NC, Lindenberger U, Kühn S, et al. Rostral locus coeruleus integrity is associated with better memory performance in older adults. Nature Human Behaviour. 2019;3(11):1203–1214. https://doi.org/10.1038/s41562-019-0715-2 |
| 50 | `vanegroo2022sleepwake` | Van Egroo M, Koshmanova E, Vandewalle G, Jacobs HIL. Importance of the locus coeruleus-norepinephrine system in sleep-wake regulation: Implications for aging and Alzheimer's disease. Sleep Medicine Reviews. 2022;62:101592. https://doi.org/10.1016/j.smrv.2022.101592 |
| 51 | `galgani2023locuscoeruleus` | Galgani A, Lombardo F, Martini N, Vergallo A, Bastiani L, Hampel H, et al. Magnetic resonance imaging locus coeruleus abnormality in amnestic Mild Cognitive Impairment is associated with future progression to dementia. European Journal of Neurology. 2023;30(1):32–46. https://doi.org/10.1111/ene.15556 |
| 52 | `luthi2025microarousals` | Lüthi A, Nedergaard M. Anything but small: Microarousals stand at the crossroad between noradrenaline signaling and key sleep functions. Neuron. 2025;113(4):509–523. https://doi.org/10.1016/j.neuron.2024.12.009 |
| 53 | `kjaerby2026neuromodulators` | Kjaerby C, Radovanovic T, Kovács ER, Bojarowska Z, Tokarska KA, Tsopanidou A, et al. Coordinated infraslow cortical oscillations of neuromodulators during NREM sleep. iScience. 2026;29(2):114554. https://doi.org/10.1016/j.isci.2025.114554 |

**54 = Grollero** (existing, auto-renumbered).

*`mather2016locuscoeruleus` is **not** in this list — dropped 2026-08-17, see §0.4.*

**Appended after Grollero:**

| # | Key | Entry |
|---|---|---|
| 55 | `dimitriades2025schizophrenia` | Dimitriades ME, Schumacher E, Arudchelvam J, Fattinger S, Kurth S, Pugin F, et al. The infraslow fluctuation of sigma power during sleep in young individuals with schizophrenia. Schizophrenia Research. 2025;285:295–303. https://doi.org/10.1016/j.schres.2025.09.029 |
| 56 | `sun2026longcovid` | Sun H, Dang R, Li P, Xiao W, Scott-Sutherland J, Sassower KC, et al. Facility-measured sleep electroencephalographic microstructures in long COVID. SLEEP. 2026;49(8):zsag090. https://doi.org/10.1093/sleep/zsag090 |
| 57 | `liu2026autism` | Liu K, Sun B, Wang BK, Chen J, Westover MB, Tian FY, et al. An Electroencephalographic Study of Sleep Spindle and Infraslow Oscillation in Children With Autism Spectrum Disorder. Journal of Sleep Research. 2026;35(4):e70309. https://doi.org/10.1111/jsr.70309 |
| 58 | `terzano2001cap` | Terzano MG, Parrino L, Sherieri A, Chervin R, Chokroverty S, Guilleminault C, et al. Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep. Sleep Medicine. 2001;2(6):537–553. https://doi.org/10.1016/S1389-9457(01)00149-6 |
| 59 | `parrino2012cap` | Parrino L, Ferri R, Bruni O, Terzano MG. Cyclic alternating pattern (CAP): The marker of sleep instability. Sleep Medicine Reviews. 2012;16(1):27–45. https://doi.org/10.1016/j.smrv.2011.02.003 |
| 60 | `silvani2026cycles` | Silvani A, Sicbaldi M, Mogavero MP, Lanza G, Quercia A, Bruni O, et al. Sleeping in cycles within cycles: A nonlinear framework for NREM sleep microstructure. Sleep Medicine Reviews. 2026;90:102351. https://doi.org/10.1016/j.smrv.2026.102351 |
| 61 | `parrino2025phasic` | Parrino L, Balella G, Bottignole D, Melpignano A, Rosenzweig I, Ughetti G, et al. Phasic events, cyclic alternating pattern (CAP) and sleep disorders. Clinical Neurophysiology. 2025;179:2111004. https://doi.org/10.1016/j.clinph.2025.2111004 |
| 62 | `maestri2015capmci` | Maestri M, Carnicelli L, Tognoni G, Di Coscio E, Giorgi FS, Volpi L, et al. Non-rapid eye movement sleep instability in mild cognitive impairment: a pilot study. Sleep Medicine. 2015;16(9):1139–1145. https://doi.org/10.1016/j.sleep.2015.04.027 |
| 63 | `zheng2026capdementia` | Zheng Y, Wu X, Chen H, Li F, Yaffe K, Stone K, et al. Cyclic alternating pattern in sleep electroencephalography as a novel predictor of dementia: A prospective study. Alzheimer's & Dementia. 2026;22(4):e71331. https://doi.org/10.1002/alz.71331 |

### 9.4 — execution order

1. **Methods first.** Insert the Tallon-Baudry marker at 3.4 and its list entry after `sleepeegpy`. Then
   fix the four shifted Methods markers (MNE ×2, Maris, YASA), anchored on surrounding words.
2. Renumber Chen in Discussion ¶3 (`variance⁴⁵` → `variance⁴⁶`) and Grollero in ¶6
   (`Alzheimer's disease⁴⁹` → `Alzheimer's disease⁵⁴`).
3. Delete the Schmitz, André and Niethard list entries.
4. Replace ¶4 with §4.1 + §4.2 + §4.3.
5. Insert entries 47–53 after Chen. Grollero auto-renumbers to 54.
6. Apply §2, §3, §5 (including §5.3), §6, §7, §8.
7. Append 55–63 and **re-link every new DOI**.

**Final count: 63.** Email #9 (≥50) cleared.

---

## §10 — spotted, with your rulings applied

### 10.1 — LC softening *(your ruling: humans only, not rodents — applied)*

**Rodent claims kept.** `osorioforero2025gatekeeper` is optogenetic; the LC really does gate these
substates in mice.

| # | Where | Status |
|---|---|---|
| R1 | §4.1 "In rodents the LC… **sets its phase**" | kept |
| R2 | ¶8 "The LC **that paces the rhythm in rodents**" | kept ("locus coeruleus" → "LC" only) |
| R3 | §4.3 ¶2 "in rodents **the pacing comes from** the LC" | kept |

**Human or unmarked claims softened.**

| # | Where | BEFORE | AFTER | Status |
|---|---|---|---|---|
| H1 | ¶1 opening | "the infra-slow rhythm that **paces** spindle trains" | "the infra-slow rhythm that **organizes** spindle trains" | **applied** |
| H2 | ¶3 | "the **clock that paces** spindle trains speeds up" | "the **rhythm that organizes** spindle trains speeds up" | **applied** (§3) |
| H3 | §4.1 | "the noradrenergic system **that paces the rhythm**" | "the noradrenergic system **tied to the rhythm**" | applied |
| H4 | §4.2 | "a coordinated neuromodulatory rhythm **that the LC paces**" | "…**that the LC coordinates**" | **applied** |
| H5 | §4.3 | resolved in your rewrite: "a **drive**", "**whatever paces the cycle**" (hedged) | — | no change needed |
| H6 | Closing ¶ | "thalamocortical and spindle infrastructure" | unchanged — §4.3 now earns it | no change |

**Note for the Abstract session:** its "in rodents is paced by the locus-coeruleus noradrenergic system" is
itself a **rodent** claim, so by this split it may not need changing, even though his tracked edit asked
for it.

### 10.2–10.4, 10.8, 10.9 — no action, as instructed

Recorded so a later session does not reopen them: the Yael Gat REM-latency discrepancy; `fultz2019coupled`
for the uncited hemodynamic clause; the declined metadata fixes on refs 16 and 23; the CAP concession for
the reply; and the three of his citations that now leave the thesis.

### 10.5 — Niethard: **dropped**

Editorial on Champetier, same journal and issue, no data, and always cited beside ref 17 which makes the
claim itself. Removes an editorial from a list Yuval will count. Applied in §9.

### 10.6 — Tallon-Baudry: **added in Methods 3.4**

Methods 3.4 states *"a fixed width of 4 cycles"* with no source; this is the conventional citation for that
parameter. Added now rather than later because it renumbers everything from 42 onward, and doing it while
the list is already being rebuilt costs nothing. Becomes ref **42**.

### 10.7 — typo: **fixed** in `thesis/figure_manifest.md:60`

---

## §11 — after you approve

1. Execute §9.4 in order.
2. Mirror to `05_discussion.md`, `02_introduction.md` and `03_methods.md` in `[@key]` form, with changelog
   notes.
3. Verify: highest superscript = list length = **63**; no number used twice; order of first appearance ==
   list order; all new DOIs hyperlinked.
4. Prose audit: **zero em dashes**; stage labels N1/N2/N3 + REM; no script or file names; "ISO" only ever
   describing other people's studies; no sentence presenting bandwidth as an age effect; "paces" only in
   rodent contexts.
5. REM check: survives only in §6.3, describing our own architecture result.
6. Update `yuval_review_triage.md` STATUS table and §8.2.
