# Flavio's Comments — Categorized Review

Flavio Schmidig reviewed `Shaked's Thesis_FS.docx` and left **66 comments**. His cover email headline:

> "Looks very good already… I only focused on the things I would change, not the things I liked."

Two **paper-level** asks from the email (both → Category 1, defer):
- **(A)** Add **Bayesian statistics** to back the claim that MCI and elderly ISFS are *not* different (currently framed as a "negative result", which frequentist non-significance can't support).
- **(B)** Reconsider **Results structure** — either *young vs old* then *old vs MCI*, **or** stay consistent: whole-scalp 3-group (freq/bandwidth/strength) → topographic 3×3 → ROI 3×3. He'd group topographic + whole-scalp together.

Cross-cutting writing theme (repeated ~15×): **lead each Results/Discussion paragraph with the plain-language finding first, then the technical/statistical detail.** And **say things in general language, not jargon.**

We're submitting to the school now, so paper-only asks are noted but parked. No thesis text changes until you mark which items to apply.

---

## Category 1 — Real work for a *paper*, defer for now (not just rephrasing)

| # | Where | What Flavio wants |
|---|-------|-------------------|
| A (email) + 98 + 109 | Discussion (MCI=elderly claim) | **Bayesian analysis** to actually support "elderly and MCI are indistinguishable." Right now it's framed as a negative result, which non-significance doesn't license. |
| B (email) | Results structure | Reorganize Results: either young-vs-old → old-vs-MCI, or a consistent whole-scalp → topo → ROI sweep across all 3 parameters; merge topographic + whole-scalp sections. |
| 46 | Results 4.1/4.2 | Move the first descriptive sub-sections (4.1/4.2) into Methods, summarize in ~2 sentences in Results, so "interesting" results arrive sooner. (Part of B.) |

**Note:** 116 (trim limitations to 3) and 96 (remove "earlier subject set" mention) are substantial but doable for the thesis — listed under Category 2.

---

## Category 2 — Suggested rephrasings (do now, in his spirit)

He often wrote the *intended* phrasing; we just formalize it.

**Title & Abstract**
- **2 — Title:** less technical for a paper, e.g. *"Healthy, but not pathological, aging alters infra-slow fluctuations of sleep sigma/spindle power."* *(Paper-framed — for thesis a technical title may be fine; your call.)*
- **4 — Abstract** (several): first two sentences too long/repetitive; don't say "almost none has looked at ISFS in old" — just state what's unknown; the "peak frequency, bandwidth, AUC of the sigma-envelope spectrum at every channel" clause is too specific; results too technical (recast "peak frequency faster… broader bandwidth" as plain "spindle rhythm becomes faster / less focal with age"); **end on impact/outlook, not on the trend that didn't reach significance.**

**Introduction**
- **6** — Opening spindle paragraph is good but maybe not the best *opener*; start with something that makes the reader curious / closer to the question.
- **7** — "most" → "a large part?" (soften/qualify).
- **8** — Spindle-train rhythm: state plainly first ("trains recur in a slow ~0.02 Hz rhythm — about every 50 s"), then the technical meaning (envelope), then the *why* (LC/noradrenaline).
- **12** — Recast the MCI question as: *"whether ISFS in MCI is comparable to, or different from, ISFS in healthy elderly."* *(He offered the phrasing.)*

**Methods**
- **17** — Give Methods more narrative structure: participants came to the lab, EEG applied, slept X h; goal = extract ISFS; preprocessing → channel rejection; N2 bouts (cite AASM, ≥300 s clean); then ISFS extraction.
- **18** — The Figure-1 sleep-overview sentence reads as a *result*; relocate/reframe.
- **25** — "mean-subtracted to remove the DC component and Fourier-transformed… normalized by its own mean" — confusing wording (see Cat 4 for proposed plain rewrite).
- **26** — Baseline subtraction "nice hack": first state the *problem*, then how the higher-band baseline subtraction fixes it.
- **27** — Gaussian fit: first state the goal ("to capture ISFS we estimated peak freq, bandwidth, AUC… rejected bouts with peak outside X… fitted a Gaussian"), then the equation.
- **28** — Refer to the figure here ("the figure helps a lot").
- **39** — Display-mean / interpolation sentence is dense: split into (1) an uncoupled statement about correlations, (2) simpler "group mean = mean of per-subject means," (3) "missing electrode values were interpolated for visualization only."
- **40** — Demographics/sleep-stage stats: just state the result ("Age and MoCA did not differ between elderly and MCI… sleep-staging (did/did not) differ, omnibus ANOVA + post-hoc").
- **64** — The ROI-normalization sentence belongs in **Methods**, not Results.

**Results**
- **30, 32** — Figure title "ISFS concept, feature extraction, central-parietal ROI" → simpler, e.g. *"Characterization of infra-slow fluctuations of sigma power."*
- **36** — "whole scalp" → "main ISFS" / "overall ISFS."
- **37** — Don't write normality in the hypothetical ("when all groups were normal"); state what actually happened.
- **44, 49** — Section/figure titles should state the *main result*, not the category — e.g. *"ISFS frequency changes with ageing."*
- **48** — Name the parameter explicitly (don't just say "parameters").
- **50, 51, 65** — Open each result with what you were interested in: *"We wondered…; for this we compared peak frequency, bandwidth and AUC between groups X, Y, Z."*
- **54** — "On every parameter… statistically indistinguishable" → *"Neither peak frequency, bandwidth, nor strength differed significantly between elderly and MCI."*
- **55** — "kernel density" too technical → "the half-violin estimates the distribution."
- **56, 62, 66** — Lead with the headline finding (use it as the figure description); focus on the significant effect, not the non-significant ones.
- **58** — (after "ISFS strength is focally reduced over central-parietal cortex") add *which group*.
- **67** — Be explicit: *"ISFS strength descriptively declined from young to older participants, but the group comparison was not significant (one-way ANOVA…)."*
- **70, 71** — Peak-freq/bandwidth maps: state *what and why* you compared in plain terms; defer which-cluster-stats detail to Methods.
- **73** — MoCA correlation: *"We found no relationship between ISFS and cognition (MoCA); neither frequency, bandwidth, nor strength correlated (all r < …)."*

**Discussion**
- **77** — Shorter, simpler paragraphs; aim for one conclusion each (Yuval's style: more descriptive, less interpretive).
- **81** — Add a recap sentence: *"We set out to… therefore we compared ISFS across three groups…"*
- **94** — Plain language: e.g. *"Compared to young adults, ISFS is less focal and weakens with age"* instead of the cluster-p phrasing.
- **96** — **Don't introduce new results in the Discussion** (the "earlier subject set" bandwidth point). If it matters, report it in Results; otherwise just state the trend as-is, don't over-weight it.
- **104** — Name the two directional changes explicitly ("repetition is good").
- **105** — Simplify the 0.02 Hz "clock" paragraph — mostly repeats the findings.
- **108** — "plausible substrate" reads "very LLM" → reword. *(AI-tell; consider a humanizer pass.)*
- **112** — Consider adding his alternative reading: ISFS changes may *precede* detectable MCI (relates to the upstream factor before cognition is impaired).
- **114** — Call both effects "ISFS strength," explain "amplitude" once in brackets, rather than switching terms.
- **115** — Ending reads "too much LLM" and overclaims: drop "convergent evidence of sensitivity to neurodegeneration"; end simply: *"both studies show an age effect, while a relation of ISFS to AD (but not MCI) has been found."*
- **116** — Too many limitations: pick **3**, discuss each properly (describe → why unavoidable → future fix).
- **119** — Coupling-to-other-rhythms idea needs motivation: explain *why* coupling ISFS to slow oscillations / hemodynamics could be interesting or expected.

---

## Category 3 — Compliments (no action)

- **10** "Nice!" · **11** "great" · **15** "Clear and good" (Participants) · **30** "Nice. clear, easy to understand and still precise" · **38** "Love it and well described" (cluster-based permutation) · **59** "GREAT" · **78** "Already great" (Discussion opening) · **107** "Great paragraph… generally good" · **110** "Very good paragraph" · **118** "great" (Future directions).
- **58** "GREAT!" (+ minor add — see Cat 2) · **60** "most result paragraphs should be written like that" (praise + the lead-with-finding principle).
- **33** "referring to B triggered me 😃, probably a me problem" — non-actionable; leave as is.

---

## Category 4 — "I don't understand" with no fix offered → proposed clarifications

Flavio flagged confusion but didn't supply wording. Proposed plain rewrites for your review:

- **9** — *(Intro: Dimitriades characterization sentence + "the same infra-slow grouping tracks memory consolidation")* — he doesn't see the difference between the two sentences. They're near-redundant. **Proposal:** fold into one — keep Dimitriades' characterization + memory-reactivation link, and either cut the second sentence or make it a *distinct* behavioral-consolidation point with its own citation.

- **25** — *(Methods: "mean-subtracted to remove the DC component… normalized by its own mean")* — "subtract the mean then normalize by the mean?"; "DC"/"transformed" unclear. **Proposal:** *"From each bout we removed its average level (which otherwise dominates the spectrum at 0 Hz) and computed its power spectrum; each spectrum was then scaled by its own average power so bouts could be compared on a common scale."* (Clarify: the first mean is in time, the normalization is of the spectrum — two different operations.)

- **35** — *("whole-scalp, topographic")* — distinction unclear without reading on. **Proposal:** gloss inline: *"…at three levels of spatial detail: averaged across the whole scalp, mapped channel-by-channel as topographies, and summarized within an a-priori region of interest."*

- **106** — *("A focal reduction in ISFS strength therefore need not translate proportionally into disorganized spindle output…")* **Proposal:** *"A weaker AUC over central-parietal cortex does not necessarily mean spindles themselves are impaired; it more likely reflects how the rhythm is spatially distributed than a direct loss of spindle function."*

- **113** — *("its insensitivity to MCI is itself informative about what the rhythm indexes")* — "informative about what index?" **Proposal:** *"that ISFS does not change in MCI tells us something about what the rhythm reflects: it appears to track aging-related processes rather than the additional pathology of MCI."*

- **120** — *("pursue the rodent locus-coeruleus account translationally…")* **Proposal:** *"test the locus-coeruleus / noradrenaline mechanism in humans — in human sleep this mechanism has so far only been inferred indirectly, never measured directly."*

---

## Notes / cross-cutting
- **AI-tells flagged** (108 "plausible substrate", 115 ending) — a humanizer pass on the Discussion is worth it before finalizing.
- The single biggest writing lever (cited ~15× across Results/Discussion) is **lead with the plain finding, then the stats** — applying that consistently resolves 56, 62, 65, 66, 67, 70, 73, 81, 94, 50, 51 in one pass.

---

## Next step
Tell me which items (by #) to apply and any wording preferences, and I'll make the edits in the thesis text.
