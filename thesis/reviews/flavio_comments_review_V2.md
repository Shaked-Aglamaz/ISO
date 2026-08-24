# Flavio's Comments — Review V2 (status after applying)

Companion to `flavio_comments_mapping.md`, which is **unchanged**: comment IDs and anchors are the same as in V1, so both documents still line up.

**Applied:** 2026-08-06, to a **copy** of the manuscript Google Doc (`1YpXrDGFlzRk…`). The original Doc is untouched. Local mirrors updated: `thesis/chapters/01_abstract.md` … `05_discussion.md` and the three affected caption blocks in `thesis/figure_manifest.md`. Full before → after text for every change: `thesis/flavio_edits_before_after.md`.

**What changed since V1 of this review:** a **fifth category** was added. Flavio also edited the manuscript in place, as tracked changes rather than comments, and those never appeared in V1. Extracting them from `Shaked's Thesis_FS.docx` turned up **38 insertions and 27 deletions across 8 paragraphs**, which reduce to six distinct edits (C5-1 … C5-6 below). Comment **46** was also moved out of Category 1 and applied.

---

## Summary

| Status | Count | IDs |
|---|---|---|
| **Applied** | 43 | 2, 4, 6, 7, 8, 9, 12, 17, 25, 26, 27, 28, 30\*, 32, 35, 37, 39, 40, 44, 46, 48, 49, 50, 51, 54, 55, 56, 58, 62, 64, 65, 66, 67, 70, 71, 73, 77, 81, 94, 96, 104, 105, 106, 108, 109, 112, 113, 114, 115, 116, 119, 120 |
| **Applied — Flavio's in-place edits** | 6 | C5-1 … C5-6 (new category) |
| **Declined by you** | 2 | 18, 36 |
| **Deferred (paper-level)** | 3 | A (email), B (email), 98 |
| **No action (compliments / non-actionable)** | 12 | 10, 11, 15, 30, 33, 38, 59, 60, 78, 107, 110, 118 |
| **Still open** | 1 | 108 — the phrase is fixed; the full humanizer pass awaits your trigger |

\* 30 was a compliment on the ROI section; the caption-title change he asked for in the same breath is #32.

---

## Category 5 (NEW) — Flavio's tracked in-place edits

None of these carried a comment, so none were in V1.

| # | Where | What he did | Outcome |
|---|-------|-------------|---------|
| **C5-1** | Methods 3.4 | "wavelet transform" → "transform**ation**"; "via the FFT" → spell out "fast Fourier transformation (FFT)" | **Partly applied.** FFT is now spelled out at its first mention. "Transform**ation**" declined both times — "transform" is the standard term and matches the Figure 2 caption. |
| **C5-2** | Results 4.3 | Deleted the "For each subject the three ISFS parameters were summarized…" opener | **Applied** — replaced by a question-first lead-in that keeps the Figure 3 reference (#50/#51). |
| **C5-3** | Figure 4 caption | "Raw **(unnormalized)** per-group AUC maps" → "Raw…" | **Applied**, together with the #62 caption reorder. |
| **C5-4** | Discussion ¶1 | Restructured to "First… Second…", removed all in-text p-values, new closing sentence | **Applied**, with #81/#94/#96 layered on. The p-values are gone from the Discussion opening and remain in the Results. |
| **C5-5** | Discussion ¶4 | Deleted "within the pooled older sample" | **Applied.** |
| **C5-6** | Discussion ¶7 | Split the future-work paragraph in two and rewrote the closing | **Applied**, with the grammar of his closing sentence fixed ("distinguish between healthy and cognitive impairment" needed an object). |

---

## Two corrections the review process turned up

These are not Flavio's comments; they are errors found while acting on them.

**1. A mis-citation in the Introduction (found while doing #9).** The text attributed "approximately 0.02 Hz" to reference 10. That paper reports the opposite: *"we confirm the existence of ISO in sigma activity albeit with a frequency **below** the previously reported 0.02 Hz"* — about 0.01 Hz for sigma power — and attributes the 0.02 Hz figure to reference 6. It does recover ≈ 0.02 Hz, but from a different measure, the timing of individual spindle events. The rewritten passage now states both rates and which measure each belongs to. The "roughly 0.02 Hz" statements elsewhere (Abstract, Introduction ¶3, Discussion) were correct and are unchanged: they describe our own measurements and the rodent work.

**2. A section-numbering gap (created by #46).** Merging Results 4.1 and 4.2 left the chapter running 4.1, 4.3, 4.4… All Results sections below the merge were renumbered down by one; the chapter now runs 4.1 → 4.6. No cross-reference in the manuscript pointed at a Results section number, so nothing else needed updating.

---

## Category 1 — deferred to a paper

| # | What he wants | Why deferred |
|---|---|---|
| **A** (email) + **98** | Bayesian analysis to support "elderly and MCI are indistinguishable", since frequentist non-significance cannot | Real new analysis; we are submitting to the school now. **Partly mitigated:** #109 below fixes the overclaiming *wording* without new statistics. |
| **B** (email) | Reorganize Results — young-vs-old then old-vs-MCI, or a consistent whole-scalp → topo → ROI sweep | Structural rewrite of the chapter. **#46 carved out and applied.** |

---

## Category 2 — applied

**Title & Abstract**
- **2** — Less technical title: *"Aging, but not mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle power."* Note his own suggested phrasing ("Healthy, but not pathological, aging alters…") misstated the result — it reads as *pathological aging does not alter ISFS*, when MCI in fact shows the same changes as healthy aging. Fixed.
- **4** — Abstract rewritten: first two sentences merged, the "characterized almost exclusively in young adults" framing replaced by a plain statement of the gap, the per-channel parameter list dropped, results in plainer language, and the closing moved onto what the measure is good for rather than the tempering note. ~250 → ~200 words.

**Introduction**
- **6** — New opening paragraph, leading with what the rhythm does (alternating easily-woken and sealed-off periods) and landing on why aging makes it interesting. The old opening paragraph is unchanged, one paragraph lower.
- **7** — "most of the human night" → "a large part of the human night".
- **8** — The spindle-train rhythm is now stated plainly first ("trains recur in a slow rhythm of roughly one train every 50 seconds"), then the technical meaning (the sigma envelope at about 0.02 Hz), then the why (the LC and noradrenaline), which already followed.
- **9** — The two near-redundant sentences given distinct jobs, and the redundant memory sentence folded in as support for the memory link. Both author names dropped in favour of bare citations: in a numbered citation style, naming a study in running text is reserved for one you adopt or argue with, which after this change leaves **Dimitriades as the only name in the Introduction** — exactly where the convention wants it.
- **12** — MCI question recast as "whether the ISFS of patients with MCI is comparable to, or different from, that of healthy older adults".

**Methods**
- **17** — Narrative roadmap paragraph added under the Methods heading, tracing participants → recording and cleaning → N2 bouts → ISFS extraction → group comparison. "All epochs not scored as N2 were discarded" added to 3.3 to make that step explicit. Night duration deliberately not stated, since the manuscript reports no group mean total sleep time.
- **25** — The confusing passage rewritten as a plain paragraph followed by a technical one, removing the "for each channel" / "for each clean bout" double opening. Your wording was kept in sequence with two fixes: the thing that fluctuates is the envelope, not the frequency; and the normalization equalizes bouts and channels of differing signal strength.
- **26** — Baseline subtraction now states the problem (a spectrum does not sit on a flat floor, and broadband noise would inflate any peak on top of it) before the fix.
- **27, 28** — The Gaussian fit now opens with what the three numbers are for, points at Figure 2C, and gives the equation afterwards. The acceptance criteria follow the parameter definitions rather than interrupting them.
- **35** — The three levels of analysis glossed inline: "averaged across the whole scalp, mapped channel by channel as topographies, and summarized within the pre-defined region of interest".
- **37** — Normality is now reported as what happened rather than hypothetically: peak frequency, bandwidth and the ROI value met the assumptions and took ANOVAs; whole-scalp AUC did not and took a Kruskal–Wallis test.
- **39** — The dense display/interpolation/MoCA paragraph deleted. The topography convention moved to Results 4.3 in your simpler wording; the violin-plot clause dropped entirely; the MoCA sentence dropped. **One thing did need adding:** that sentence was the only place the correlation sample was defined, and nowhere stated which correlation was computed, so Results 4.6 now names Pearson and Spearman.
- **40** — Shortened and kept in Methods. Flavio wanted the *result* stated here, but it is already in Results 4.1 with full statistics, so following him would have duplicated it and blurred the Methods/Results boundary the rest of the chapter keeps.
- **64** — Deletion from Results only; the sentence he wanted "in Methods" was already there verbatim under "ROI and normalization".

**Results**
- **44, 48, 49, 58** — Every heading now states the result and names the parameter: *"ISFS peak frequency is higher in both older groups"*, *"…focally reduced over central-parietal cortex **in both older groups**"*, *"Averaging over the region of interest does not resolve the central-parietal reduction"*.
- **46** — Old 4.1 and 4.2 merged into one short section. The demographic means, the data-quality percentages and the bout count were duplicates of Methods 3.1/3.2/3.3, so removing them from Results is de-duplication rather than relocation; the bout count moved to Methods 3.3. What stays is the sleep-architecture comparison and the detection rate.
- **50, 51, 65, 66, 70, 73** — Each result now opens with the question: "We first asked whether…", "We then asked whether…", "We also asked whether…", "Finally, we found no relationship between…".
- **54** — "On every parameter… statistically indistinguishable" → "Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and MCI groups."
- **55** — "the half-violin shows the kernel density" → "the half-violin estimates the distribution".
- **56, 62** — The significant effect now leads both the text and the figure captions; the non-significant parameters are demoted to a single sentence.
- **67** — Made explicit: "ISFS strength declined descriptively from young to older participants… but the group comparison was not significant".
- **70, 71** — 4.5 now states what was compared and why in plain terms; the cluster-statistics machinery is left in Methods 3.6 rather than repeated.
- **30, 32** — Figure 2 title → *"Characterization of infra-slow fluctuations of sigma power."* The caption's second expansion of "fast Fourier transform (FFT)" was reduced to "FFT", since the abbreviation is now defined at its first mention in Methods 3.4.

**Discussion**
- **77** — The chapter was split from 7 paragraphs into 10, each carrying a single conclusion, rather than rewritten. The individual items below do the simplifying.
- **81** — Opening recap added: "We set out to ask… For this we compared the ISFS across three groups…".
- **94** — Opening summary in plain language, with the cluster-p phrasing gone.
- **96** — The "earlier subject set" result removed from the Discussion, and the trend no longer over-weighted. It is not reported anywhere else, so no new result now appears in the Discussion.
- **104** — "Both directional changes" → "Both changes, the faster rhythm and the flattened central-parietal hotspot".
- **105** — The 0.02 Hz clock passage cut roughly in half.
- **106** — Rewritten: a weaker rhythm over central-parietal cortex does not necessarily mean spindles themselves are impaired; the effect is safer read as a change in where the rhythm is expressed.
- **108** — "offer a plausible substrate for" → "are a likely place for these age effects to arise". **The full humanizer pass on the Discussion is still pending your trigger.**
- **109** — "The clearest **negative result**" → "The clearest finding in the other direction is that we found **no evidence** that MCI contributes anything detectable beyond age." Wording only; nothing about equivalence, power or Bayes factors was added anywhere, per your instruction.
- **112** — His alternative reading added: if the ISFS tracks the process that eventually damages cognition, it may change before impairment is detectable, in which case a group defined by cognitive score would not be expected to separate.
- **113** — The "informative about what the rhythm indexes" clause cut.
- **114** — Both effects now called ISFS strength, with "the amplitude of the spectral peak rather than its area" explained once in brackets.
- **115** — Replaced with "Taken together, the ISFS is altered by age, and a relation to Alzheimer's disease, though not to MCI, has now been reported."
- **116** — Cut from five limitations to three: cohort size/sites/heterogeneity (kept, with why it was unavoidable and what would fix it), the two sub-threshold effects (shortened into one sentence, since the bandwidth caveat already appears in the Abstract, Results and Discussion opening), and the cross-sectional design (kept). Dropped: the partial cognitive scores, and the stricter-detection-criterion paragraph. **Checked as you asked:** that detection-criterion claim appeared *only* there. Methods 3.4 keeps the positive-area validation as a method step, which is correct and not a comparative claim, and the Results report detection rates without comparing them to prior work.
- **119** — The coupling idea now carries a motivation and a precedent: spindle timing relative to the slow oscillation predicts overnight memory retention and loosens with age (reference 14), so the ISFS's relation to slow oscillations and to infra-slow hemodynamic fluctuations would show whether aging disturbs that nesting.
- **120** — The paragraph now closes on the limit, in your framing: the locus coeruleus cannot be recorded in a sleeping human, so the origin of the human rhythm stays inferred, and as long as that holds, converging indirect evidence is the closest we can come.

---

## Category 3 — compliments, no action

**10** "Nice!" · **11** "great" · **15** "Clear and good" (Participants) · **30** "Nice. clear, easy to understand and still precise" · **38** "Love it and well described" (cluster-based permutation) · **59** "GREAT" · **60** "most result paragraphs should be written like that" — the principle behind it was applied throughout · **78** "Already great" · **107** "Great paragraph" · **110** "Very good paragraph" · **118** "great" (Future directions).

**33** "referring to B triggered me 😃, probably a me problem" — he marked it non-actionable, so the panel cross-reference stays.

---

## Not changed, on your instruction

| # | Item | Reason |
|---|------|--------|
| **18** | The Figure 1 sentence in Methods 3.2 reads as a result | You said don't change |
| **36** | "whole scalp" → "main / overall ISFS" | You disagree with Flavio; "whole-scalp" retained throughout |
