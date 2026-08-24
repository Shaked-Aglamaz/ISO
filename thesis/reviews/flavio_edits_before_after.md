# Flavio's review → proposed thesis edits (before → after)

Companion to `flavio_comments_review.md` (categories) and `flavio_comments_mapping.md` (comment IDs → anchors). Decisions taken from `answers.txt`. Comment IDs unchanged.

**Target: the NEW Doc copy only** — `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`. The original (`1miqNSnMpbX0…`) is **not** touched. Also mirrored into `thesis/chapters/*.md` and `thesis/figure_manifest.md`.

**Prose rules honoured:** extended ROI only (reader sees just "the ROI"), never "ISO", ISFS = infra-slow fluctuations of *sigma power*, no script/file/function names in prose.

- **[PART A](#part-a--reviewed-and-approved)** — the new material and the items you flagged. **All approved 2026-08-06**, with your revisions folded in.
- **[PART B](#part-b--already-approved-no-need-to-re-read)** — everything you approved earlier. Skim or skip.

---

# PART A — reviewed and approved

Status: **A1, A2, A3, A5, A6, A7, A8, A9, A10, A11 approved.** **A4 resolved** — names dropped, sample size dropped, Variant 2 adopted. Your revisions are marked ✎ where they changed my draft.

## A1. Category 5 (NEW) — Flavio's tracked in-place edits

Extracted from the tracked changes in `Shaked's Thesis_FS.docx`: **38 insertions, 27 deletions across 8 paragraphs**, none attached to a comment, so none appeared in the original review. Six distinct edits:

| # | Where | What he did | My verdict |
|---|-------|-------------|------------|
| **C5-1** | Methods 3.4 | "wavelet transform" → "transform**ation**"; "via the FFT" → "via the **fast Fourier transformation** (FFT)" | **Accept the FFT spell-out; decline "transformation."** "Wavelet transform" is the standard English term; "transformation" reads as a non-native correction. Folded into **#25** (A5). |
| **C5-2** | Results 4.3 | Deleted the whole opener: "For each subject the three ISFS parameters were summarized as the mean across all fitted electrodes and compared across groups (Figure 3)." | **Accept** — consistent with #50/#51. Replaced by a question-first lead-in that keeps the Figure 3 reference. See B10. |
| **C5-3** | Figure 4 caption | "Raw **(unnormalized)** per-group AUC maps" → "Raw per-group AUC maps" | **Accept.** Folded into the #62 caption reorder, B15. |
| **C5-4** | Discussion ¶1 | Restructured to "First… Second…", **removed all in-text p-values**, and replaced the closing sentence with "Overall, changes in ISFS characteristics are primarily driven by ageing rather than cognitive impairment." | **Accept**, with #81/#94/#96 layered on. See A11 — note it drops p = 0.0026 / 0.023 / 0.061 from the Discussion opening (they stay in Results). |
| **C5-5** | Discussion ¶4 | Deleted "within the pooled older sample" | **Accept** — redundant, the sample is named earlier in the sentence. See B20. |
| **C5-6** | Discussion ¶7 | **Split** the future-work paragraph in two, and rewrote the closing sentences | **Accept the split and the intent, fix the grammar** — his "distinguish between healthy and cognitive impairment" needs an object. See A10. |

---

## A2. #2 — Title

Settled with you: "spindle power". Flag for the record — Flavio's own phrasing ("Healthy, but not pathological, aging alters…") misstates the result: it reads as *pathological aging does not alter ISFS*, whereas you found MCI shows **the same** changes as healthy aging. The wording below keeps his readability and fixes the logic.

**Before**
> Infra-slow fluctuations of sigma power in NREM2 sleep are altered by aging but not further by mild cognitive impairment (MCI)

**After**
> Aging, but not mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle power

---

## A3. #6 — new Introduction opener

Flavio: the spindle paragraph is good but not the best *opener*; he wants curiosity, closer to the question. The existing first paragraph stays exactly as it is, one paragraph lower.

**Insert before the current first paragraph**
> Sleep is not a uniform state, and it is not uniformly protected. Within a single stage of non-rapid eye movement (NREM) sleep the brain alternates every few tens of seconds between periods in which it is easily woken and periods in which it is relatively sealed off from the outside world. In rodents this alternation is a rhythm with a clock: it is paced by the noradrenergic locus coeruleus, and in the electroencephalogram it appears as a slow waxing and waning of sleep spindle activity. Because both the spindles and the neuromodulatory systems that pace them are known to deteriorate with age, this rhythm is a natural place to look for the sleep signature of an aging brain.

---

## A4. #9 — differentiating the two prior characterizations — **RESOLVED**

Your read was right: the two sentences are near-redundant because both say "someone characterized ISFS". The fix gives them different jobs — the first *establishes the phenomenon in humans*, the second is *the narrower developmental follow-up* that defined the three per-channel parameters we use — and folds the third sentence in as support for the memory link.

### ✎ Your two questions on naming — answered

**"Maybe drop the name and just write it as a statement with the citation number? Is this the usual approach to an intro?"** Yes, and it is the better choice here. In a numbered (Vancouver) citation style, naming authors in running text is normally reserved for cases where *the study itself* is the subject of the sentence — you are comparing it, adopting it, or arguing with it — rather than for every factual claim. Findings you are simply reporting carry a bare superscript.

**"Dimitriades' name is later mentioned because I used their pipeline, maybe that's the only name worth mentioning in the intro?"** Correct, and I checked: after dropping the two names below, **Dimitriades is the only name left in the Introduction**, appearing twice — once when the parameters are defined and once when the pipeline is adopted. That is exactly the case the convention reserves in-text naming for, so the Introduction ends up internally consistent: named = we build on it; unnamed = we cite it.

### What I took from `references/ISFS_humans_Lazar_2019.pdf`

| Detail | Value | Used? |
|---|---|---|
| Sample | 34 healthy young adults, baseline sleep | ✎ **No** — dropped per your instruction |
| Sleep stages | NREM stage 2 **and slow-wave sleep** | Yes — our restriction to N2 is a choice, not a given |
| Band | fast sigma **13–15 Hz**, effect strongest in high sigma | Yes — supports our 13–16 Hz choice |
| Topography | **centro-parieto-occipital**, left hemisphere, stronger in the second half of the night | The first part was already in our text |
| Two measures | ISO of **sigma power ≈ 0.01 Hz**; ISO of **spindle events ≈ 0.02 Hz** | Yes — this is the one that matters (below) |

### ⚠ The accuracy problem this fixes

Our Introduction currently attributes "**approximately 0.02 Hz**" to reference 10. That is **not what that paper reports.** Its headline result is the opposite: *"We confirm the existence of ISO in sigma activity albeit with a frequency **below** the previously reported 0.02 Hz"* — the sigma-power estimate is **≈ 0.01 Hz**, and the 0.02 Hz figure is attributed to our reference 6. They do recover ≈ 0.02 Hz, but from a different measure: spindle *events* reduced to an on/off signal. So the figure is currently mis-cited, and the fix below corrects it.

### ✎ Final text — Variant 2, names and sample size dropped

**Before**
> In the human EEG, the ISFS appears during N2 as a modulation of the sigma amplitude envelope at approximately 0.02 Hz, most prominent in fast spindles and over centro-parieto-occipital regions¹⁰. Dimitriades et al. (2024) subsequently characterized the ISFS across development and in young adults, quantifying per channel the peak frequency, spectral width, and area under the spectral peak of the sigma envelope, and linking these to markers of arousal and memory reactivation¹¹. The same infra-slow grouping of spindles tracks memory consolidation in humans¹².

**After**
> The rhythm is also present in the human EEG, where fast-sigma activity during N2 and slow-wave sleep is modulated on an infra-slow timescale, most prominently in the high sigma band and over centro-parieto-occipital regions¹⁰. How fast that modulation runs depends on what is measured: the dominant rate of the sigma-power fluctuation has been placed somewhat below the 0.02 Hz reported previously⁶, and closer to 0.02 Hz when the occurrence of individual spindles is tracked instead¹⁰. Dimitriades et al. (2024) subsequently narrowed the question, following the ISFS across development into young adulthood and quantifying it per channel as the peak frequency, spectral width, and area under the spectral peak of the sigma envelope; they linked these three measures to markers of arousal and of memory reactivation¹¹. Consistent with that link, how tightly spindles cluster on this infra-slow timescale is related to how well memories are consolidated overnight¹².

**No knock-on elsewhere:** the "roughly 0.02 Hz" statements in the Abstract, the #8 paragraph, and the Discussion are all fine as written — they describe *our* measurements and the rodent literature, both of which genuinely sit at ≈ 0.02 Hz. Only that one attribution was wrong.

---

## A5. #25 — the confusing envelope / FFT / normalization passage

Your draft, lightly fixed, then the technical elaboration underneath — plus C5-1 (FFT spelled out). This also removes the "for each channel" / "for each clean bout" double opening you and Flavio both flagged.

**What I changed in your wording, and why:** "the actual power of each frequency to measure the intensity of its fluctuations" → "the power at each frequency, which measures how strongly the envelope fluctuates at that rate" (the thing fluctuating is the envelope, not the frequency); "baseline differences in signal strength" → "bouts and channels differing in overall signal strength" (names what is being equalized). Everything else is your sequence, verbatim in order.

### ✎ You asked me to check the FFT first-mention. Two findings.

**1. It is the first mention — but the expansion is duplicated later.** "FFT" appears exactly twice in the manuscript: bare in Methods 3.4 (the passage below, the first occurrence in reading order), and **already spelled out** in the Figure 2 caption — "C) The fast Fourier transform (FFT) of the envelope…". So spelling it out in 3.4 as Flavio asks creates a duplicate definition downstream. Fix: expand at first mention in 3.4, and reduce the caption to the bare abbreviation. Caption change added to **B19**.

**2. "transform", not "transformation".** Flavio wrote "fast Fourier transform**ation** (FFT)", but the standard term is "fast Fourier transform" — and your own Figure 2 caption already uses that correct form. Same reasoning as declining his "wavelet transformation" in C5-1, so I have used "fast Fourier transform (FFT)" below. This keeps 3.4 and the caption consistent with each other.

**Before (two paragraphs)**
> For each channel, the fast-spindle sigma-band (13–16 Hz) amplitude envelope was obtained with a Gabor–Morlet wavelet transform, implemented as frequency-domain convolution via the FFT. Wavelets spanned 13.0–16.0 Hz in 0.2 Hz steps with a fixed width of 4 cycles, and the resulting complex coefficients gave the time-varying sigma-power envelope.
>
> For each clean bout, the envelope was mean-subtracted to remove the DC component and Fourier-transformed; the squared-magnitude spectrum was interpolated onto a common frequency axis and normalized by its own mean. If the spectral maximum fell within the lowest frequency bins (0–0.006 Hz), those bins were discarded to suppress residual low-frequency drift. The normalized bout spectra were averaged within a channel, and a baseline estimated from 0.06–0.10 Hz, above the ISFS band, was subtracted to flatten the spectral floor.

**After (plain first, then technical; the baseline step moves out into #26, B7)**
> For each clean bout in each channel we extracted the fast-spindle (13–16 Hz) amplitude envelope, subtracted its mean, and applied a Fourier transform to move it into the frequency domain. We then took the power at each frequency, which measures how strongly the envelope fluctuates at that rate, and scaled these values by their own average, so that bouts and channels differing in overall signal strength could be compared on a common scale.
>
> In detail, the envelope was obtained with a Gabor–Morlet wavelet transform, implemented as a frequency-domain convolution via the fast Fourier transform (FFT); wavelets spanned 13.0–16.0 Hz in 0.2 Hz steps with a fixed width of 4 cycles, and the magnitude of the resulting complex coefficients gave the time-varying sigma-power envelope. Subtracting the mean of the envelope removes its constant offset, which would otherwise place the envelope's entire average level in the 0 Hz bin and dominate the low-frequency end of the spectrum. The spectrum itself is the squared magnitude of the Fourier coefficients, resampled onto a frequency axis common to all bouts so that bouts of different length could be averaged, and each bout spectrum was then divided by its own mean power. If the spectral maximum fell within the lowest frequency bins (0–0.006 Hz), those bins were discarded to suppress residual low-frequency drift.

---

## A6. #40 — you asked WDYT. **I partly disagree with Flavio.**

He wants the *result* stated here in Methods. But those results are already in Results 4.1 with full statistics, so following him duplicates them and blurs the Methods/Results boundary the rest of the chapter keeps clean. What is actually wrong with the sentence is that it is over-worded ("under the same normality-gated scheme", "the same three-group omnibus and post-hoc scheme as the whole-scalp ISFS parameters").

**My recommendation: shorten it, keep it in Methods, leave the numbers in Results.**

**Before**
> **Demographics and sleep architecture.** Demographic variables were compared between the two older groups (age and MoCA) under the same normality-gated scheme: a Welch t-test when both groups passed Shapiro–Wilk and a Mann–Whitney U test otherwise. Group differences in sleep-stage proportions were tested with the same three-group omnibus and post-hoc scheme as the whole-scalp ISFS parameters.

**After (proposed)**
> **Demographics and sleep architecture.** Age and MoCA were compared between the two older groups with a Welch t-test, or a Mann–Whitney U test where normality failed. Sleep-stage proportions were compared across the three groups with the same omnibus and post-hoc scheme as the ISFS parameters.

If you'd rather follow him literally, say so and I'll move the numbers up into Methods instead.

---

## A7. #46 — merge Results 4.1 + 4.2 — **needs your sign-off**

The only structural change in this pass. Rationale check first: of the material in 4.1–4.2, the demographic means and the data-quality percentages are **already stated in Methods 3.1 and 3.2 verbatim**, so removing them from Results is de-duplication, not relocation. The bout count moves to Methods 3.3. What stays in Results is the sleep-architecture comparison (a real result, and Figure 1's payload) and the detection rate.

**Before** — `4.1 Cohort and sleep architecture` + `4.2 ISFS detection and bout extraction`, two sections, ~5 paragraphs, as they stand today.

**After — one section, two paragraphs**
> **4.1 The groups are age-matched and N2 sleep is plentiful in all of them**
>
> The two older groups were closely matched in age and differed in cognitive score as expected, and both were older than the young group by design (Welch t = −0.56, p = 0.57 for age; Mann–Whitney U = 375.5, p < 0.001 for MoCA; Table 1). Sleep architecture followed the normative age-related pattern (Figure 1): deep (N3) and REM sleep were reduced in the older and MCI groups relative to young adults (both one-way ANOVA p < 0.001), with the effect carried by young versus each older group and no difference between elderly and MCI²⁵. N2 nonetheless remained the largest sleep stage in all three groups, so the group differences in ISFS reported below do not reflect differing amounts of N2 sleep.
>
> The ISFS was present in the vast majority of the data: across the 1023 clean N2 bouts analyzed, a valid Gaussian fit was obtained in a mean of 80.8% of channels overall, and in 74.5% of channels in young adults, 85.6% in older adults, and 82.0% in MCI.

✎ Your revision: "A detectable ISFS was the rule rather than the exception" is gone. I also collapsed the following sentence into the same one, because keeping both left "vast majority" and "large majority" two clauses apart.

**Plus one added sentence in Methods 3.3**
> Across the cohort this yielded 1023 clean bouts, a mean of 9.8 per subject (range 3–21).

*If you'd rather leave 4.1/4.2 alone, everything else in this document still applies unchanged — this is the only item with structural reach.*

---

## A8. #77 — Discussion paragraph surgery — **proposal, then applied**

Flavio's ask is structural, not a wording fix: shorter, simpler paragraphs, one conclusion each, more descriptive and less interpretive (Yuval's style). Rather than rewrite the chapter, the 7 current paragraphs split into **11 blocks, one conclusion each**. The individual items do the actual simplifying; this is the map they slot into.

| ¶ | The one conclusion it carries | Source | Item |
|---|---|---|---|
| 1 | What we set out to ask, and how | new | #81 |
| 2 | Two things changed with age; MCI added nothing | current ¶1 | C5-4, #94, #96 |
| 3 | Both changes point to a loss of temporal and spatial precision | ¶2 start | #104 |
| 4 | The clock speeds up and loses its central-parietal focus | ¶2 middle | #105 |
| 5 | A weaker rhythm is not the same as impaired spindles | ¶2 end | #106 |
| 6 | Neuromodulatory decline is a likely origin | ¶3 | #108 |
| 7 | MCI added nothing beyond age, against expectation from the literature | ¶4 start | #109 |
| 8 | Two readings: aging already complete, or ISFS changes precede detectable MCI | ¶4 end | #112, #113 |
| 9 | How this sits against the concurrent AD study | ¶5 | #114, #115 |
| 10 | Limitations (three) | ¶6 | #116 |
| 11 | Future directions, and what we still cannot reach | ¶7, split in two | C5-6, #119, #120 |

---

## A9. #116 — limitations cut from five to three

Your instructions: keep 1 as is; **shorten 2** (this is the bit you asked about); keep 3 as is; delete 4 (cognitive scores partial); delete 5 (detection criterion). Each survivor now runs describe → why it was unavoidable → what could fix it, as Flavio asked.

**On your point 2** — the old version repeated the bandwidth caveat that already appears in the Abstract, Results 4.3, and the Discussion opening, *and* re-argued the ROI dilution point. The version below states both sub-threshold effects once, with their p-values, in a single sentence.

**Before**
> Several limitations qualify these conclusions. The groups were of modest size and drawn from two recording sites, and the older and MCI cohorts were clinically and etiologically heterogeneous, which would obscure any subtle MCI-specific effect. Two of the parameters showed age-related tendencies that did not reach significance and should not be treated as robust. The bandwidth difference was not significant (p = 0.061) and, as noted, sensitive to which subjects were included, and the region-of-interest comparison likewise did not reach significance because averaging across the region dilutes a spatially focal effect that the cluster-based analysis captured directly; a larger cohort may yet resolve whether these sub-threshold effects are genuine. The design was cross-sectional, and therefore cannot distinguish individual aging trajectories or identify the subset of MCI patients who will progress; only a within-subject longitudinal design can do that. Cognitive scores were available for only part of the older sample, limiting the power of the brain–behavior analysis. Finally, the criterion we used to count a channel as showing a detectable rhythm was stricter than in earlier work, because we additionally rejected implausibly narrow spectral peaks that a looser criterion would have retained. The detection rates reported here are therefore conservative, and their lower values relative to prior reports reflect this stricter threshold rather than a real scarcity of the rhythm.

**After**
> Several limitations qualify these conclusions. The groups were of modest size and drawn from two recording sites, and the older and MCI cohorts were clinically and etiologically heterogeneous, which would obscure any subtle MCI-specific effect; both cohorts were recruited through memory clinics rather than assembled to be etiologically uniform, and separating MCI subtypes would require a substantially larger sample than either site could provide. Two effects fell short of significance and should not be treated as robust: the broader spectral peak in the older groups (p = 0.061), and the reduction in ISFS strength when averaged over the region of interest (p = 0.143), the latter because averaging across the whole region dilutes an effect that the cluster-based analysis showed to be focal; a larger cohort may resolve whether either is genuine. The design was cross-sectional, and therefore cannot distinguish individual aging trajectories or identify the subset of MCI patients who will progress; only a within-subject longitudinal design can do that, which for a rhythm measured across a whole night means repeated overnight recordings years apart.

**Your point 5, "make sure this doesn't appear in any of the thesis parts" — checked.** The stricter-detection-criterion claim appears **only** in this limitations sentence. Methods 3.4 keeps the positive-area validation as a method step (correct, and not a comparative claim), and Results 4.2 reports the detection rates without comparing them to prior work. Nothing else to remove.

**Per your instruction on #109:** nothing about equivalence or Bayesian statistics is added here.

---

## A10. #119 + #120 + C5-6 — future work, and the closing paragraph

### #119 — you asked whether we can point at a comparable coupling. Yes — slow-oscillation–spindle coupling, which is already in the bibliography (reference 14, Helfrich et al. 2018) and carries exactly the right precedent: a nested coupling with functional meaning that *loosens with age*.

**Before**
> It would also be informative to examine how the ISFS couples to other infra-slow rhythms of sleep, including the slow oscillation and infra-slow hemodynamic fluctuations, and to pursue the rodent locus-coeruleus account translationally, given that the mechanistic origin of the human rhythm remains inferred rather than measured.

**After (first half; the second half becomes #120)**
> It would also be informative to ask what the ISFS is coupled to. Sleep is built from nested rhythms whose coupling carries functional meaning: the timing of spindles relative to the slow oscillation predicts overnight memory retention, and that coupling loosens with age¹⁴. If the infra-slow rhythm sets when trains of spindles occur, then its relation to the slow oscillations occurring inside those windows, and to the infra-slow hemodynamic fluctuations that share its timescale, would show whether the changes reported here disturb that nesting or leave it intact.

### #120 — close the paragraph on the LC limit, in your framing ("as long as we can't measure the LC, this is the best we can do")

**Append to the future-work paragraph**
> None of this reaches the mechanism directly. The locus coeruleus that paces the rhythm in rodents cannot be recorded in a sleeping human, so the origin of the human rhythm will stay inferred rather than measured. As long as that is the case, converging indirect evidence of the kind outlined here is the closest we can come to testing the noradrenergic account.

### C5-6 — then, as its own final paragraph (Flavio's split and rewrite, grammar fixed)

**Before**
> The infra-slow sigma rhythm of N2 sleep is, on this evidence, a sensitive marker of how healthy aging reshapes the thalamocortical and spindle infrastructure of sleep, becoming faster and losing its central-parietal focus, but in this cohort it carried no signal specific to mild cognitive impairment. Beyond these findings, the study contributes a whole-scalp, topographic, and region-based characterization of the ISFS in aging and MCI that extends a measure previously described only in the young.

**After**
> The infra-slow sigma rhythm of N2 sleep is, on this evidence, a sensitive marker of how healthy aging reshapes the thalamocortical and spindle infrastructure of sleep, becoming faster and losing its central-parietal focus. It did not, in contrast, distinguish patients with mild cognitive impairment from healthy older adults of the same age. Beyond these findings, the study characterizes the whole-scalp, topographic, and regional features of the ISFS in aging and MCI by comparing them with those of young, healthy adults.

---

## A11. Discussion ¶1 — C5-4 + #94 + #96 (contains a Category 5 rewrite, so flagged here)

Flavio's own in-place restructure, with #94's plain language and #96's removal of the "earlier subject set" result, and the trend no longer over-weighted. **His removal of the in-text p-values is retained** — they remain in Results.

**Before**
> Applying an established young-adult analysis pipeline to high-density EEG in three groups, we found that the infra-slow fluctuation of sigma power during N2 sleep is reshaped by age along two axes. Its peak frequency was faster in both older groups than in young adults (p = 0.0026), and its overall strength, although preserved when averaged across the whole scalp, was reduced focally over central-parietal cortex, where young adults showed a pronounced spectral-area hotspot that flattened with age (cluster p = 0.023). A parallel tendency toward broader spectral width was present in the same direction but did not reach significance (p = 0.061); this effect was borderline and sensitive to which subjects were included, having reached significance in an earlier subject set, and is best read as a trend rather than an established effect. Most consequentially, patients with MCI were statistically indistinguishable from healthy older adults on every ISFS measure, and no ISFS parameter tracked the MoCA. The dominant signal in these data is therefore one of healthy aging rather than of cognitive impairment.

**After**
> Two things changed with age. First, the rhythm ran faster in both older groups than in young adults. Second, it lost its focus: young adults showed a pronounced central-parietal hotspot of ISFS strength that flattened in both older groups, even though strength averaged over the whole scalp was unchanged. A tendency toward a broader spectral peak pointed the same way but did not reach significance. Patients with MCI were indistinguishable from healthy older adults on every ISFS measure, and no ISFS measure tracked the MoCA. Overall, changes in ISFS are driven by aging rather than by cognitive impairment.

---
---

# PART B — already approved, no need to re-read

Drafted per your instructions in `answers.txt`. Listed in manuscript order.

## B1. #4 — Abstract rewrite

Sentences 1–2 merged and shortened; "characterized almost exclusively in young adults" cut in favour of stating the gap; the per-channel parameter list dropped; results plainer; ending moved off the tempering note onto what the measure *is* good for. ~250 → ~200 words.

**Before**
> During the second stage of non-rapid eye movement (NREM) sleep (N2), sleep spindles recur in trains rather than at random, following a slow rhythm. The amplitude of activity in the fast-spindle sigma band (13–16 Hz) rises and falls on an infra-slow timescale of roughly 0.02 Hz, an infra-slow fluctuation of sigma power (ISFS) that in rodents is paced by the locus-coeruleus noradrenergic system. In humans, the ISFS has been characterized almost exclusively in young adults, leaving open whether it changes with healthy aging and whether mild cognitive impairment (MCI) adds a signal beyond that of age. We quantified the ISFS during clean N2 sleep in 35 young adults, 39 healthy older adults, and 30 patients with MCI, applying an established young-adult analysis pipeline to high-density 256-channel EEG and measuring the peak frequency, bandwidth, and area under the spectral peak (AUC) of the sigma-envelope spectrum at every channel. Peak frequency was faster in both older groups than in young adults (p = 0.0026), with a parallel, non-significant tendency toward broader bandwidth (p = 0.061). Whole-scalp AUC was preserved, but its central-parietal hotspot, prominent in young adults, was focally reduced with age (cluster p = 0.023). On every measure, patients with MCI were statistically indistinguishable from healthy older adults, and no ISFS parameter correlated with cognitive score. The ISFS is therefore reshaped by healthy aging, becoming faster and losing its central-parietal focus, rather than by cognitive impairment. In this cohort it indexes chronological age, not cognitive status, which tempers its use as a standalone marker of early MCI.

**After**
> During the second stage of non-rapid eye movement (NREM) sleep (N2), sleep spindles arrive in trains rather than at random: power in the fast-spindle sigma band (13–16 Hz) rises and falls roughly every 50 seconds, an infra-slow fluctuation of sigma power (ISFS) that in rodents is paced by the locus-coeruleus noradrenergic system. Whether the human ISFS changes with healthy aging, and whether mild cognitive impairment (MCI) adds a signal beyond that of age, is unknown. We quantified the ISFS during clean N2 sleep in 35 young adults, 39 healthy older adults, and 30 patients with MCI, applying an established young-adult analysis of the sigma-envelope spectrum to high-density 256-channel EEG. The rhythm ran faster in both older groups than in young adults (p = 0.0026), with a non-significant tendency toward a broader spectral peak. Its overall strength was unchanged, but the central-parietal focus prominent in young adults was flattened in both older groups (cluster p = 0.023). Patients with MCI were indistinguishable from healthy older adults on every measure, and no measure tracked cognitive score. The rhythm that paces spindle trains therefore becomes faster and less spatially focused with age, and does so whether or not cognition has begun to decline. The ISFS is a sensitive read-out of how aging reorganizes the machinery that generates spindles, rather than a marker of early cognitive impairment.

## B2. #7 — soften "most"

**Before:** Non-rapid eye movement (NREM) sleep occupies **most of** the human night…
**After:** Non-rapid eye movement (NREM) sleep occupies **a large part of** the human night…

## B3. #8 — plain statement first, then technical, then why

**Before**
> Spindles are not distributed evenly through NREM sleep. The amplitude of sigma-band activity over time forms the sigma envelope, and this envelope fluctuates on an infra-slow timescale of roughly 0.02 Hz, so that spindles occur in trains separated by intervals of tens of seconds⁵.

**After**
> Spindles are not distributed evenly through NREM sleep. They come in trains, and those trains recur in a slow rhythm of roughly one train every 50 seconds⁵. What fluctuates at that rate is the sigma envelope, the amplitude of sigma-band activity over time, which is modulated at a frequency of about 0.02 Hz.

*(Rest of the paragraph — fragile/offline substates, then the LC and noradrenergic clock — unchanged, so the "why" now follows the plain statement.)*

## B4. #12 — recast the MCI question

**Before:** …and whether MCI **departs from healthy aging or instead mirrors it**.
**After:** …and whether **the ISFS of patients with MCI is comparable to, or different from, that of healthy older adults**.

## B5. #17 — Methods roadmap paragraph

Inserted under the `Methods` heading, before 3.1. (Night duration left out — we have no group mean TST in the manuscript, only the ≥ 210 min inclusion rule, and I won't state a number we don't report.)

**Insert**
> Participants spent a full night in the sleep laboratory with a high-density EEG net applied, and slept undisturbed while the EEG was recorded. The goal of the analysis was to extract each participant's ISFS, and the sections below follow the steps that led to it: who took part (3.1); how the night was recorded, filtered, and cleaned of bad channels and artifactual epochs (3.2); how the continuous artifact-free stretches of N2 sleep were isolated (3.3); how the ISFS was measured within them (3.4–3.5); and how the resulting measures were compared between groups (3.6).

**Plus, at the start of 3.3, to make "keep only N2" explicit as he asked**

**Before:** Clean N2 bouts were extracted by splitting each scored N2 period around any overlapping artifact annotation…
**After:** All epochs not scored as N2 were discarded. Clean N2 bouts were then extracted by splitting each remaining N2 period around any overlapping artifact annotation…

## B6. #18 — no change (your call). The Figure 1 sentence stays in Methods 3.2.

## B7. #26 — baseline subtraction: problem first, then fix

**Before** *(currently the tail of the paragraph rewritten in A5)*
> The normalized bout spectra were averaged within a channel, and a baseline estimated from 0.06–0.10 Hz, above the ISFS band, was subtracted to flatten the spectral floor.

**After** *(its own paragraph)*
> The normalized bout spectra were then averaged within each channel. Such an averaged spectrum does not sit on a flat floor: broadband noise adds an offset that varies from channel to channel and that would inflate any peak measured on top of it. We therefore estimated that offset from the 0.06–0.10 Hz range, which lies above the ISFS band and so contains no ISFS signal, and subtracted it from the whole spectrum, leaving a flat floor against which a peak can be measured.

## B8. #27 + #28 — Gaussian fit: goal first, then the equation, and point at the figure

**Before**
> A Gaussian, *a*·exp(−((*f*−*b*)/*c*)²), was fitted to the baseline-corrected mean spectrum by nonlinear least squares. A fit was accepted only if its peak amplitude exceeded the detection threshold (1.5 times the standard deviation of the spectrum), its peak frequency fell within the ISFS band 0.0075–0.04 Hz, and the numerically integrated area under the fitted peak was positive, which rejects degenerate fits. From each accepted fit we extracted three per-channel ISFS parameters: **peak frequency** (the frequency of the fitted maximum), **bandwidth** (the width of the *b* ± |*c*| band), and **AUC** (the integral of the fitted Gaussian over *b* ± |*c*|). Channels whose fit failed validation yielded no ISFS values and were treated as missing.

**After**
> To characterize the ISFS we needed three numbers per channel: how fast the rhythm runs (**peak frequency**), how sharply that rate is defined (**bandwidth**), and how strong the rhythm is (**AUC**). We obtained all three by fitting a bell-shaped curve to the peak of the baseline-corrected mean spectrum and reading the values off the fit, as illustrated in Figure 2C. The curve was a Gaussian, *a*·exp(−((*f*−*b*)/*c*)²), fitted by nonlinear least squares: peak frequency is the frequency of the fitted maximum (*b*), bandwidth is the width of the *b* ± |*c*| band, and AUC is the integral of the fitted Gaussian over that band. A fit was accepted only if its peak rose above a detection threshold set at 1.5 times the standard deviation of the spectrum, its peak frequency fell inside the ISFS band (0.0075–0.04 Hz), and the numerically integrated area under the peak was positive, the last condition rejecting degenerate fits. Channels whose fit failed any of these checks were treated as having no measurable ISFS and left missing.

## B9. #35 — gloss the three levels of analysis (Methods 3.6 opening)

**Before:** Group comparisons were performed at three levels: whole-scalp, topographic, and within the ROI.
**After:** Group comparisons were performed at three levels of spatial detail: averaged across the whole scalp, mapped channel by channel as topographies, and summarized within the pre-defined region of interest.

## B10. #37 — state what actually happened, not the hypothetical

**Before**
> Per-group normality and variance homogeneity were checked with Shapiro–Wilk and Levene's tests. The three-group omnibus test was a one-way ANOVA when all groups were normal and a Kruskal–Wallis test otherwise, with effect size reported as η². When the omnibus was significant, post-hoc pairwise comparisons used Tukey's HSD (after ANOVA) or Dunn's test with Holm correction (after Kruskal–Wallis).

**After**
> Per-group normality and variance homogeneity were checked with Shapiro–Wilk and Levene's tests. Peak frequency, bandwidth, and the normalized ROI value met these assumptions and were compared with one-way ANOVAs, with effect size reported as η²; whole-scalp AUC did not and was compared with a Kruskal–Wallis test. Where the omnibus test was significant, post-hoc pairwise comparisons used Tukey's HSD (Dunn's test with Holm correction was the non-parametric counterpart, which was not required here).

## B11. #39 — break up the "kitbag" paragraph

Per your instruction: **delete from Methods**, move the topography note to Results 4.4 where the maps are described, drop the violin-plot clause entirely, drop the MoCA sentence.

**Before (delete)**
> Throughout, the group mean displayed on a topography is the mean of the per-subject means, computed independently of the plotted image and matching the violin-plot statistic; missing channels in a topographic image were filled by spatial neighbor imputation for display only, never for the group statistics. Correlations between ISFS parameters and MoCA were assessed within the pooled older-adult and MCI sample.

**After:** *(paragraph removed)*

**You asked me to check whether anything needs adding for the MoCA correlation — yes, one thing.** That Methods sentence was the only place the sample was defined, and nowhere in the manuscript states *which* correlation was computed; only the Figure S2 caption mentions Pearson and Spearman. So Results 4.7 gains the test names (it already has the sample size), and no Methods sentence is needed. Handled in B18.

## B12. #64 — nothing to add to Methods

The sentence Flavio wants moved out of Results 4.5 already exists verbatim in Methods 3.6 under "ROI and normalization". So it is a **deletion from Results only** — handled in B16.

## B13. #44 / #48 / #49 / #58 — headings state the result and name the parameter

| Section | Before | After |
|---|---|---|
| 4.1 | Cohort and sleep architecture | The groups are age-matched and N2 sleep is plentiful in all of them |
| 4.3 | Whole-scalp ISFS parameters differ with age | ISFS peak frequency is higher in both older groups |
| 4.4 | ISFS strength is focally reduced over central-parietal cortex | …over central-parietal cortex **in both older groups** |
| 4.5 | ISFS strength within the central-parietal ROI | Averaging over the region of interest does not resolve the central-parietal reduction |
| 4.6 | Peak-frequency and bandwidth topographies show no significant cluster | *(unchanged — already states the result)* |
| 4.7 | No association with cognitive score | *(unchanged)* |

**#36 — no change** (you disagree with Flavio): "whole-scalp" stays as the term throughout.

## B14. #50 / #51 / #54 / #56 + C5-2 — Results 4.3 restructured

Opens with the question rather than the summarization method (absorbing Flavio's deletion of that opener); leads with the significant effect; non-significant ones demoted to one sentence; elderly-vs-MCI named parameter by parameter.

**Before**
> For each subject the three ISFS parameters were summarized as the mean across all fitted electrodes and compared across groups (Figure 3).
>
> Peak frequency was higher in both older groups than in young adults (young 0.0199 ± 0.0041 Hz, elderly 0.0226 ± 0.0041 Hz, MCI 0.0232 ± 0.0040 Hz; one-way ANOVA F = 6.32, p = 0.0026, η² = 0.111). Post-hoc comparisons confirmed that young adults differed from both older groups (young vs elderly p = 0.013, young vs MCI p = 0.005), whereas the two older groups did not differ from each other (p = 0.86).
>
> Bandwidth followed the same direction but did not reach significance (young 0.0236 ± 0.0089 Hz, elderly 0.0281 ± 0.0085 Hz, MCI 0.0276 ± 0.0088 Hz; one-way ANOVA F = 2.87, p = 0.061, η² = 0.054); no post-hoc tests were performed, and this difference is best read as a trend rather than an established effect.
>
> Overall ISFS strength, measured as the area under the spectral peak (AUC), did not differ across groups at the whole-scalp level (young 6.46 ± 3.28, elderly 7.31 ± 2.88, MCI 7.48 ± 3.49; Kruskal–Wallis H = 1.96, p = 0.38).
>
> On every parameter, the elderly and MCI groups were statistically indistinguishable.

**After**
> We first asked whether the ISFS as a whole changes with age. For this we compared its peak frequency, bandwidth, and strength (AUC) between young adults, healthy older adults, and patients with MCI, each summarized per subject as the mean across the fitted electrodes (Figure 3).
>
> The rhythm ran faster in both older groups than in young adults (young 0.0199 ± 0.0041 Hz, elderly 0.0226 ± 0.0041 Hz, MCI 0.0232 ± 0.0040 Hz; one-way ANOVA F = 6.32, p = 0.0026, η² = 0.111). Post-hoc comparisons placed the difference between young adults and each older group (young vs elderly p = 0.013, young vs MCI p = 0.005), not between the two older groups (p = 0.86).
>
> The other two parameters did not change significantly. The spectral peak was broader in both older groups, in the same direction as the frequency effect, but the comparison fell short of significance and is best read as a trend (young 0.0236 ± 0.0089 Hz, elderly 0.0281 ± 0.0085 Hz, MCI 0.0276 ± 0.0088 Hz; one-way ANOVA F = 2.87, p = 0.061, η² = 0.054; no post-hoc tests). Overall ISFS strength did not differ at the whole-scalp level (young 6.46 ± 3.28, elderly 7.31 ± 2.88, MCI 7.48 ± 3.49; Kruskal–Wallis H = 1.96, p = 0.38).
>
> Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and MCI groups.

## B15. #39 (relocated note) — Results 4.4

The topography convention Flavio found dense in Methods lands here, in his own simpler wording. Appended to the end of the 4.4 paragraph:

**Append**
> The group maps show the mean of the per-subject means, and channels without a valid fit were interpolated from their neighbours for visualization purposes only, never for the statistics.

## B16. #64 / #65 / #66 / #67 — Results 4.5 rewritten

Deletes the normalization sentence (it lives in Methods), opens with the question, makes the descriptive-decline-but-not-significant reading explicit in Flavio's words.

**Before**
> To quantify the central-parietal effect within the pre-defined region of interest, each subject's per-channel AUC was first normalized by that subject's whole-scalp mean and then averaged over the ROI electrodes (Figure 5). The group means followed the same order as the topographic result (young 1.099, elderly 1.040, MCI 1.012), but the three-group comparison was not significant (one-way ANOVA p = 0.143).

**After**
> We then asked whether the same central-parietal reduction appears when the whole pre-defined region of interest is summarized as a single value per subject (Figure 5). ISFS strength declined descriptively from young to older participants (young 1.099, elderly 1.040, MCI 1.012), but the group comparison was not significant (one-way ANOVA p = 0.143).

## B17. #70 / #71 — Results 4.6 rewritten

What and why in plain terms; the cluster-statistics machinery is already fully described in Methods 3.6, so it is not repeated.

**Before**
> The same cluster-based permutation procedure was applied to the peak-frequency and bandwidth maps (Figure S1). Peak frequency showed a frontal emphasis in young adults that flattened in the older groups, and bandwidth showed no consistent spatial pattern, but neither parameter produced a significant spatial cluster.

**After**
> We also asked whether the faster rhythm and the broader spectral peak were confined to particular regions, as the strength effect was, or spread evenly across the scalp. They were not regionally confined: young adults, healthy older adults, and patients with MCI did not differ significantly at any location in either peak frequency or bandwidth (Figure S1). Peak frequency showed a frontal emphasis in young adults that flattened in the older groups, and bandwidth showed no consistent spatial pattern, but neither produced a significant cluster.

## B18. #73 — Results 4.7 rewritten (picks up the test names dropped from Methods by #39)

**Before**
> Within the pooled older-adult and MCI sample (n = 44), none of the ISFS scalars (peak frequency, bandwidth, whole-scalp AUC, or ROI AUC) correlated with MoCA score (all |r| ≤ 0.15, all p ≥ 0.34; Figure S2).

**After**
> Finally, we found no relationship between the ISFS and cognitive capacity. Neither the frequency, the bandwidth, nor the strength of the rhythm correlated with MoCA score in the pooled older-adult and MCI sample, whether strength was taken over the whole scalp or within the region of interest (n = 44; all |r| ≤ 0.15, all p ≥ 0.34, Pearson and Spearman; Figure S2).

## B19. Figure captions — #32, #55, #56, #62 + C5-3

Applied to the Doc **and** to the `> *Figure N. Title.* …` blocks in `thesis/figure_manifest.md`, so the manifest stays the caption source of truth. Figures 1, 5, S1, S2 unchanged — no comments landed on them.

**#32 — Figure 2 title**
- **Before:** *Figure 2. ISFS concept, feature extraction, and the central-parietal ROI.*
- **After:** *Figure 2. Characterization of infra-slow fluctuations of sigma power.*
- Caption title only; the rest of the caption body is unchanged apart from the FFT fix below. #33, his reaction to "expanded in B)", he marked non-actionable, so the panel cross-reference stays.

**✎ FFT de-duplication in the Figure 2 caption** (from the A5 check — the abbreviation is now expanded at its first mention in Methods 3.4, so the caption should not redefine it)
- **Before:** C) The **fast Fourier transform (FFT)** of the envelope, averaged across the subject's clean N2 bouts, fitted with a Gaussian…
- **After:** C) The **FFT** of the envelope, averaged across the subject's clean N2 bouts, fitted with a Gaussian…

**#55 + #56 — Figure 3: lead with the finding, drop "kernel density"**

**Before**
> *Figure 3. Whole-scalp ISFS parameters across groups.* Each dot is one subject's mean across the fitted electrodes; the box shows the median and interquartile range, and the half-violin shows the kernel density. Peak frequency was higher in both older groups than in young adults (young < elderly = MCI; one-way ANOVA, p = 0.0026), whereas bandwidth showed a non-significant trend in the same direction and overall strength (AUC) did not differ across groups; elderly and MCI did not differ on any parameter. Peak frequency and bandwidth were compared by one-way ANOVA and AUC by Kruskal–Wallis, each with pairwise post-hoc tests; full statistics are reported in the Results. Significant pairwise post-hoc differences are marked with asterisks (* p < 0.05, ** p < 0.01).

**After**
> *Figure 3. Whole-scalp ISFS parameters across groups.* Peak frequency was higher in both older groups than in young adults (young < elderly = MCI; one-way ANOVA, p = 0.0026), whereas bandwidth showed a non-significant trend in the same direction and overall strength (AUC) did not differ across groups; elderly and MCI did not differ on any parameter. Each dot is one subject's mean across the fitted electrodes; the box shows the median and interquartile range, and the half-violin estimates the distribution. Peak frequency and bandwidth were compared by one-way ANOVA and AUC by Kruskal–Wallis, each with pairwise post-hoc tests; full statistics are reported in the Results. Significant pairwise post-hoc differences are marked with asterisks (* p < 0.05, ** p < 0.01).

**#62 + C5-3 — Figure 4: lead with the finding, drop "(unnormalized)"**

**Before**
> *Figure 4. Topography of ISFS strength (AUC) and the central-parietal cluster.* A) Raw (unnormalized) per-group AUC scalp maps. B) The corresponding per-subject normalized AUC scalp maps, with the cluster-based permutation result overlaid: the pre-defined central-parietal ROI is marked with green dots, and electrodes with a significant post-hoc group difference are circled in yellow. Young adults show higher AUC over the central-parietal electrodes, and the hotspot flattens and spreads in aging and MCI. A cluster-based permutation test identified a single significant central-parietal cluster (p = 0.023, 9 electrodes), driven by lower AUC in elderly and MCI than in young adults; per-electrode post-hoc counts are reported in the Results.

**After**
> *Figure 4. Topography of ISFS strength (AUC) and the central-parietal cluster.* Young adults show higher AUC over the central-parietal electrodes, and the hotspot flattens and spreads in aging and MCI. A) Raw per-group AUC scalp maps. B) The corresponding per-subject normalized AUC scalp maps, with the cluster-based permutation result overlaid: the pre-defined central-parietal ROI is marked with green dots, and electrodes with a significant post-hoc group difference are circled in yellow. A cluster-based permutation test identified a single significant central-parietal cluster (p = 0.023, 9 electrodes), driven by lower AUC in elderly and MCI than in young adults; per-electrode post-hoc counts are reported in the Results.

## B20. #81 — Discussion opening recap

**Insert as the first paragraph of the Discussion** (the rewritten ¶1 in A11 follows it)
> We set out to ask whether the infra-slow rhythm that paces spindle trains changes as the brain ages, and whether mild cognitive impairment adds anything beyond age. For this we compared the ISFS across three groups: young adults, healthy older adults, and patients with MCI.

## B21. #104 — name the two changes

**Before:** **Both directional changes** are consistent with a loss of spatial and temporal precision in the thalamocortical machinery that organizes fast spindles.
**After:** **Both changes, the faster rhythm and the flattened central-parietal hotspot,** are consistent with a loss of temporal and spatial precision in the thalamocortical machinery that organizes fast spindles.

## B22. #105 — simplify the 0.02 Hz "clock" passage

**Before**
> The flattening of that hotspot together with an acceleration of the rhythm suggests that the roughly 0.02 Hz clock pacing spindle trains becomes both faster and less spatially concentrated, so that its energy is no longer focused where fast-spindle generators are densest. A similar loss of spatial structure was visible, though it did not reach significance, in the peak-frequency map: the frontal emphasis seen in young adults became more uniform across the scalp in the older groups, consistent with a general blurring of regional differentiation rather than a change confined to one parameter.

**After**
> With age, then, the clock that paces spindle trains speeds up, and its energy is no longer concentrated where fast-spindle generators are densest. The peak-frequency map showed the same blurring without reaching significance: the frontal emphasis of young adults became more uniform across the scalp in the older groups.

## B23. #106 — clarify what the AUC effect does and does not mean

**Before**
> A focal reduction in ISFS strength therefore need not translate proportionally into disorganized spindle output, and the AUC effect is more safely interpreted as a sensitive index of how the rhythm is distributed than as direct evidence of degraded spindle function.

**After**
> A weaker rhythm over central-parietal cortex therefore does not necessarily mean that spindles themselves are impaired. It is safer to read the effect as a change in where the rhythm is expressed than as direct evidence that spindle generation has degraded.

## B24. #108 — "plausible substrate"

**Before:** The neuromodulatory systems that pace and generate spindles offer **a plausible substrate for** these age effects.
**After:** The neuromodulatory systems that pace and generate spindles are **a likely place for** these age effects **to arise**.

*(Full humanizer pass on the Discussion stays parked until you trigger it.)*

## B25. #109 / #98 — "negative result" → "no evidence" (per your instruction: wording only, nothing added to the limitations)

Flavio's point is that frequentist non-significance does not license the word "negative result". Since the Bayesian analysis is parked, this is the wording-only fix.

**Before**
> The clearest **negative result** is that MCI contributed nothing detectable beyond age.

**After**
> The clearest finding in the other direction is that we found **no evidence** that MCI contributes anything detectable beyond age.

*Nothing about equivalence, power, or Bayes factors is added anywhere, per your instruction.*

## B26. #112 — add his alternative reading

**Before** *(end of the MCI paragraph)*
> One reading is that the changes captured here reflect neuromodulatory and thalamocortical aging that is already well advanced in healthy older adults and that the transition to MCI does not measurably accelerate, at least not at the resolution of these scalp-level summaries.

**After**
> One reading is that the changes captured here reflect neuromodulatory and thalamocortical aging that is already well advanced in healthy older adults and that the transition to MCI does not measurably accelerate, at least not at the resolution of these scalp-level summaries. A second reading points the other way in time. If the ISFS tracks the process that eventually damages cognition, it may change before cognitive impairment becomes detectable, in which case a group defined by cognitive score would not be expected to separate from healthy older adults of the same age.

## B27. #113 — cut the unclear clause

**Before:** On this view the ISFS tracks how far the brain has aged rather than whether cognition has begun to decline**, and its insensitivity to MCI is itself informative about what the rhythm indexes**.
**After:** On this view the ISFS tracks how far the brain has aged rather than whether cognition has begun to decline.

## B28. C5-5 — his in-place deletion in the same paragraph

**Before:** …and no ISFS measure correlated with the MoCA **within the pooled older sample**.
**After:** …and no ISFS measure correlated with the MoCA.

## B29. #114 — call both effects "ISFS strength", explain amplitude once

**Before**
> They found the amplitude of the rhythm selectively reduced in AD, with its frequency and bandwidth preserved; the amplitude reduction tracked the plasma amyloid ratio, and bandwidth tracked markers of neurodegeneration and poorer memory retention. Their amplitude result parallels our central-parietal AUC cluster, since both implicate a weakening of the rhythm's strength over central regions, although…

**After**
> They found ISFS strength (in their case the amplitude of the spectral peak rather than its area) selectively reduced in AD, with frequency and bandwidth preserved; the reduction in strength tracked the plasma amyloid ratio, and bandwidth tracked markers of neurodegeneration and poorer memory retention. Their result parallels our central-parietal cluster, since both implicate a weakening of ISFS strength over central regions, although…

## B30. #115 — replace the overclaiming ending

**Before**
> Because their study appeared after the present analysis was complete, the two are best read as independent and largely convergent evidence that the infra-slow sigma rhythm is sensitive to both aging and neurodegeneration.

**After**
> Taken together, the ISFS is altered by age, and a relation to Alzheimer's disease, though not to MCI, has now been reported.

---

## Not changed, on your instruction

| # | Item | Reason |
|---|------|--------|
| A (email), 98 | Bayesian equivalence test for elderly = MCI | Category 1, paper-level, deferred (the #109 wording fix is applied — B25) |
| B (email) | Whole Results restructuring | Category 1, deferred (#46 carved out — A7) |
| 18 | Figure 1 sentence reads as a result | You said don't change |
| 36 | "whole scalp" → "main/overall ISFS" | You disagree with Flavio |
| 33 | "referring to B triggered me" | He marked it non-actionable |
| 10, 11, 15, 30, 38, 59, 60, 78, 107, 110, 118 | Compliments | No action |
| 108 (full pass) | Discussion humanizer sweep | Parked until you trigger it |
