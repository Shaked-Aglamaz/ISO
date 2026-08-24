# Final pre-submission check — "Shaked's Thesis V2"

Document: `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY` ("Shaked's Thesis V2", last modified 2026-08-06 15:08)
Check run: 2026-08-08. **Read-only — nothing in the document was edited.**

Passes run: numbers-vs-source audit · figure/table render check · 29-reference DOI + claim verification (web) ·
academic writing review (`scientific-writing`, `scientific-critical-thinking`) · humanizer (whole manuscript).

---

## 0. What came back clean

| Check | Result |
|---|---|
| Whole-scalp stats (peak freq, BW, AUC), post-hocs, η² | Match `three_groups_V10/three_group_statistics.txt` exactly |
| Cluster p, 9 electrode labels, post-hoc counts (5 / 7 / 1) | Match `three_group_topo_statistics.txt` exactly |
| ROI value (1.099 / 1.040 / 1.012, p = 0.143) | Matches the **extended 36-ch** ROI file (locked choice) |
| Table 1 (N, site split, age, MoCA, exclusions) | Matches `demographics_V3/demographics_table.txt` cell for cell |
| Demographic means/SDs, bad-channel %, bad-epoch % | Match `demographics_V3` |
| MoCA correlations (n = 44, all \|r\| ≤ 0.15, all p ≥ 0.34) | Match `moca_correlation_V3/correlation_summary.csv` (max r = 0.147, min p = 0.342) |
| Bouts: 1023, mean 9.8, range 3–21 | Matches the ISFS pipeline (268 + 430 + 325 = 1023) |
| Detection rates 74.5 / 85.6 / 82.0 %, overall 80.8 % | Reproduced by re-running the aggregation script |
| Sleep-architecture claims (N3, REM both p < 0.001; no elderly-vs-MCI difference) | Match `sleep_stage_stats.txt` |
| Methods parameters: 13.0–16.0 Hz / 0.2 Hz steps / 4 cycles / threshold 1.5 SD / band 0.0075–0.04 Hz / 0–0.006 Hz discard / 300 s / 176 channels | Match the code |
| All 7 figures embedded, correct order, **all are the V10 assets** | Verified by image matching against the source PNGs |
| All 29 DOIs | Resolve; metadata matches (Grollero's odd `10.64898` prefix is legitimate) |
| Terminology | "ISO" never appears in prose; ISFS expanded correctly throughout |
| Open comments in the doc | None |

The prose is also largely free of the classic AI tells: no em-dash overuse, no boldface, no
"delve/tapestry/underscore/landscape", no chatbot artifacts, no promotional vocabulary.

---

## 1. Blockers — fix before sending

### B1. The Dimitriades paper is no longer a preprint

Reference 11 — the study the whole pipeline is adopted from — was **published in *Scientific Reports*,
18 June 2026, DOI `10.1038/s41598-026-58423-z`** (confirmed via Crossref and the KNAW repository record).
The manuscript still cites the 2024 bioRxiv preprint.

Affects: reference 11, and the four in-text "Dimitriades et al. (2024)" mentions (Introduction, Methods 3.4,
Methods 3.5, Figure 2D caption) which become (2026).
Do **not** use the *Sleep Medicine* 2026 DOIs (`10.1016/j.sleep.2025.107031/107032`) — those are conference abstracts.

### B2. Results 4.3 misstates how missing channels were handled

The manuscript says: *"channels without a valid fit were interpolated from their neighbours for visualization
purposes only, never for the statistics."*

The code does the opposite. In `step5_topo_comparison.py:74` every subject's per-channel vector passes through
`normalize_subject_channels` (`step4_distribution_analysis.py:83-87`) **before** the EvokedArrays that feed the
cluster-based permutation test are built, and that function imputes every NaN with the subject's mean over
valid channels. After normalization those channels sit at exactly 1.0. The ROI analysis
(`step6_groups_comparison.py:990-991`) uses the same path. Neighbour interpolation is a *different* routine
(`utils/topo_aggregation.py`) and is indeed display-only.

Why it matters beyond wording: the imputed fraction is large and **differs by group** — young 25.5 %,
elderly 14.4 %, MCI 18.0 % of channels. Young adults, who carry the central-parietal hotspot, have the most
channels pulled to the neutral value. The direction of that bias is arguable, but a PI will ask, and the
current sentence forecloses the question with a claim that is not true.

Minimum fix: describe the imputation accurately in Methods 3.6 and correct the Results sentence.
Stronger fix: a sensitivity run of the AUC cluster test restricted to channels with a valid fit in most
subjects, to show the cluster is not an imputation artefact.

> **Status 2026-08-08 — addressed in the chapters, pending in the doc.** Resolved by removing the claim
> rather than explaining the mechanism: the sentence is now *"The value above each map in Figure 4A is the
> mean of the per-subject means."* No Methods paragraph was added and no sensitivity run was made. Backup
> for a PI question: the imputed fraction at the nine cluster electrodes is comparable across groups
> (11.7 % young / 10.8 % elderly / 14.1 % MCI, vs the whole-scalp 25.5 / 14.4 / 18.0 %), and dropping
> imputed ROI channels moves the F5 ANOVA from p = 0.143 to p = 0.126 — the fill is conservative.
> A second error found in the same check: the F5 caption said "the **fitted** electrodes of the ROI",
> which the normalized path does not do; the word was removed.

---

## 2. Should fix before sending

### S1. The two-site design is confounded with group and is never checked
Site composition is Young 35 TASMC / 0 Sydney, Elderly 30 / 9, MCI **14 / 16**. Group and recording site are
therefore entangled, most severely for MCI (53 % Sydney) against Young (0 %). The Limitations mention two sites
only as a heterogeneity caveat; no site covariate, no site comparison, no sensitivity analysis is reported.
The elderly group is the natural test bed — 30 TASMC vs 9 Sydney within one diagnostic group. Worth stating
explicitly that the young-vs-elderly peak-frequency effect (p = 0.013) is largely within-TASMC and therefore
the least site-exposed of the contrasts.

### S2. André et al. is also published
Reference 28 (medRxiv 2025) appeared as **Molecular Psychiatry, 12 May 2026, DOI `10.1038/s41380-026-03635-y`**,
"Associations between REM sleep EEG slowing and brain cholinergic denervation in aging and Mild Cognitive
Impairment" (same author list). Grollero (ref 29) is still a preprint — correctly cited as one.

### S3. Reference 15 is an editorial, not primary evidence
Niethard, SLEEP 2023;46(5):zsad011 is a single-author **Editorial** commenting on Champetier et al. — which is
already reference 12. It is currently cited for "the documented breakdown of fast-spindle timing with age" and
inside the range 12–15. Cite Champetier (12) for the finding; keep 15 only if the editorial's own argument is
what is being invoked.

> **APPLIED 2026-08-08.** Kept, demoted to a secondary cite: dropped from the Introduction bracket (now 12–14),
> retained in the Discussion paired with Champetier (now 12,28). This moved its first appearance to the
> Discussion, so the list was renumbered: old 16–28 → 15–27, Niethard 15 → 28, Grollero 29 unchanged.
> **Reference numbers quoted elsewhere in this report (e.g. André "28" in S2) predate that renumber — André is
> now 27.**

### S4. Required statements are missing
No ethics-committee approval, no informed-consent statement (Methods 3.1), no funding statement, no
conflict-of-interest statement, no data- or code-availability statement. Ethics/consent is the one a supervisor
will notice immediately for a two-site human study.

### S5. Methods omit an exclusion criterion that Table 1 reports
Methods 3.1 lists three technical exclusion grounds (bad channels, artifactual epochs, too few clean bouts);
Table 1 has a fourth row, **TST < 210 min**, accounting for 1 young and 3 MCI exclusions. Add it to the prose.

> **APPLIED 2026-08-08** (doc + `thesis/chapters/03_methods.md`). The Methods 3.1 list now reads
> "...too many bad channels, too many artifactual epochs, too few clean N2 bouts (fewer than three; see
> Section 3.3), **or a total sleep time under 210 min**." The four criteria now match Table 1's four rows.
> The 210 min threshold itself is stated but not justified in the text — add a clause if the PI asks why 210.

### S6. MCI is never defined diagnostically
The manuscript says Tel Aviv patients "had been referred for subjective cognitive complaints" but never states
the diagnostic criteria for MCI, who made the diagnosis, how the Sydney MCI patients were diagnosed, or that
patients with dementia/AD were excluded (they were — the demographics script tracks an AD-flag exclusion).
This is a STROBE eligibility-criteria item and the most likely first question about the cohort.

### S7. Results 4.1 draws a conclusion its evidence doesn't support
*"N2 nonetheless remained the largest sleep stage in all three groups, so the group differences in ISFS reported
below do not reflect differing amounts of N2 sleep."* Being the largest stage does not establish that. And N2
proportion **does** differ across groups (KW p = 0.0009; young 34.1 % vs elderly 44.5 %, p = 0.0006), which the
text never mentions. The two facts that actually support the claim are in the same source files: the proportion
of each subject's N2 retained as clean bouts is equal across groups (ANOVA p = 0.98), and total analyzed bout
duration does not differ (KW p = 0.081).

### S8. Figure 1 caption overstates one comparison
*"elderly and MCI have more frequent but shorter N2 bouts than young adults."* More frequent holds for both
(p = 0.0021, p = 0.0091). **Shorter holds only for MCI** (p = 0.0015); elderly vs young is p = 0.154 (ns).

### S9. The Lázár sentence contradicts itself
*"...placed the dominant rate of the sigma-power fluctuation somewhat below the 0.02 Hz described in rodents,
and closer to 0.02 Hz when it tracked the occurrence of individual spindles instead."* Both halves point at
0.02 Hz. The paper's actual values (verified in the PDF): sigma-power ISO **≈ 0.01 Hz**, spindle-event ISO
**≈ 0.02 Hz** ("our estimate of its dominant frequency is at ∼0.01 Hz"; "Peak frequencies of the ISO for the
fast sleep spindles (∼0.02 Hz) appear to be somewhat higher"). State the two numbers.

### S10. "Lower AUC in the older groups" reads as contradicting 4.2
Results 4.3 and the Figure 4 caption say older groups have lower AUC centrally, while 4.2 reports raw
whole-scalp AUC nominally **higher** in the older groups (6.46 vs 7.31 / 7.48). Both are true — 4.3 is about
per-subject **normalized** maps — but the word "normalized" only appears in the preceding sentence and never in
the caption. Say it explicitly in both places; Figure 4A's shared colour scale makes the raw/normalized
distinction visually confusing on its own.

### S11. Reproducibility gaps in Methods
- Who scored the sleep, how many scorers, any reliability check — not stated.
- "Semi-automatic" bad channel/epoch rejection is named but no criteria or thresholds are given.
- The ROI is never given a size: it is **36 electrodes** (extended ROI; the figures show 36 green dots — verified).
- Topographic tests ran on **175** channels, not 176 (VREF has no scalp position).
- The baseline window is **0.06–0.102 Hz** in the code; the text rounds it to "0.06–0.10 Hz".

### S12. Two over-claims
- *"The ISFS is a sensitive read-out..."* (Abstract) and *"a sensitive marker..."* (Discussion). Sensitivity was
  never quantified — no classification or effect-size-based sensitivity analysis. η² = 0.11 is a medium effect.
- *"changes in ISFS are driven by aging rather than by cognitive impairment"* — causal language from a
  cross-sectional design. "Associated with age" costs nothing and is defensible.

### S13. No statement about testing three parameters
Three whole-scalp omnibus tests were run with no correction and no stated primary outcome. Peak frequency
(p = 0.0026) survives Bonferroni (0.05/3 = 0.0167) anyway, so one sentence — that peak frequency was the
pre-specified primary parameter, or that the effect survives correction — closes it.

---

## 3. Humanizer pass — whole manuscript

The manuscript reads as human-written. One construction does most of the damage.

**H1. "X rather than Y" appears ~12 times** and is the manuscript's signature move: *"read-out ... rather than a
marker"*, *"driven by aging rather than by cognitive impairment"*, *"marker of chronological aging rather than of
cognitive status"*, *"tracks how far the brain has aged rather than whether cognition has begun to decline"*,
*"recruited through memory clinics rather than assembled to be etiologically uniform"*, *"stay inferred rather
than measured"*, *"the amplitude of the spectral peak rather than its area"*, *"temper, rather than support"*,
*"arrive in trains rather than at random"*, plus the parallel *"safer to read the effect as ... than as ..."*.
One sentence carries it twice: *"the ISFS behaves as a marker of chronological aging rather than of cognitive
status, and the data temper, rather than support, the use of ISFS as a standalone biomarker."* Keep it where the
contrast is the point (title, abstract closer); vary the rest.

**H2. Stacked summary connectives.** The Grollero paragraph ends *"Together, these findings suggest that..."*
and then immediately *"Taken together, the ISFS is altered by age..."*. Cut one.

**H3. Introduction redundancy.** "Non-rapid eye movement (NREM) sleep" is spelled out in both ¶1 and ¶2. The
locus-coeruleus pacing claim is made in ¶1 and again in ¶3. "(LC)" is defined and then never used again.

**H4. The triad repeats.** "Whole-scalp / topographic / ROI" (and its variants) appears five times in near-identical
form — Introduction, Methods 3.6 opener, Results, Discussion closer. Legitimate as a description of the analysis;
monotonous at that frequency.

**H5.** *"The clearest finding in the other direction is that we found no evidence..."* — "finding ... we found",
and "in the other direction" is doing unclear work.

**H6.** *"The ISFS was present in the vast majority of the data"* immediately followed by exact percentages —
drop the vague quantifier.

**H7. Spelling drift.** "neighbours" (Results 4.3) vs "neighboring channels" (Methods 3.2). Pick one.
> **Status 2026-08-08 — addressed in the chapters, pending in the doc.** Closed by the B2 rewrite, which
> deleted the sentence containing "neighbours"; US spelling is now consistent throughout.

**H8. Orphan section numbering.** Methods subsections are numbered 3.1–3.7 and Results 4.1–4.6, but "Methods"
and "Results" themselves are unnumbered and Introduction/Discussion have no numbers at all.

**H9. Abstract is 282 words** — over the usual 250 ceiling if this is heading toward a journal.

**H10. Title page reads "July, 2026."**

---

## 4. Optional — worth considering, not blocking

- **A citation you already have supports your main result and isn't being used.** Chen et al. 2025 (ref 26)
  reports that the spindle **refractory period is significantly shorter in older age groups**, and that this
  survives adjustment for spindle duration. That is a direct mechanistic parallel to a faster ISFS with age.
- **New literature since the reference list was frozen:** Jacobsen et al., eLife reviewed preprint, March 2026 —
  *"Noradrenergic infraslow rhythm during sleep is the critical link between heart-rate dynamics and memory
  consolidation"* (mice **and** humans). It partly qualifies the Discussion's claim that the LC "cannot be
  recorded in a sleeping human, so the origin of the human rhythm will stay inferred" — heart-rate coupling is
  exactly the non-invasive proxy. Also Dimitriades et al., ISFS in young people with schizophrenia (2025/2026).
- **An open question the Discussion doesn't address:** our young-adult peak frequency (0.0199 Hz) agrees with the
  rodent/Dimitriades ~0.02 Hz but is roughly **double** Lázár's human sigma-power estimate (~0.01 Hz) — the one
  prior human characterization. Worth a sentence, since it is the same measurement in the same band.
- **Figure 5 is displayed 3.15 in wide** while every other figure is at the full 6.5 in text width. Fonts are
  large enough to survive, but it looks like an oversight next to the others.
- **Figure 4 caption doesn't explain what yellow means per panel.** Verified from the image: young shows 8
  circles (all electrodes significant in any contrast involving young), elderly 6, MCI 7 — i.e. each panel marks
  the contrasts that group takes part in. As written, a reader expects the same 9 in all three.
- **Stray legend swatches** sit next to the panel titles in Figures 3 and 5 (a dot, a dash, a pink box).
- **1022 vs 1023 bouts.** The ISFS pipeline totals 1023 (the number in the text, correct); the demographics
  recount totals 1022 (one fewer in the elderly group). Harmless, but the two scripts disagree by one.
- **Sex counts (20/15, 23/16, 20/10) could not be verified** against any results file — they come from the
  subjects Google Sheet. The group totals are right; worth one look at the sheet since the cohort changed at V10.
- **Bayesian / TOST equivalence for "MCI indistinguishable from elderly"** remains open from Flavio's review
  (#98). Flagged only, per your instruction not to draft it.

---

## 5. Recommended verification before you send

1. **Run the site check** (S1). Elderly TASMC (n = 30) vs Sydney (n = 9) on the three ISFS parameters. Even a null
   result is worth one sentence in the Limitations, and it is the question most likely to come back at you.
2. **Sensitivity run for the imputation** (B2) — the AUC cluster test on channels with a valid fit in a
   sufficient fraction of subjects.
3. **Fix the two citations** (B1, S2) — both are one-line reference swaps plus the "(2024)" → "(2026)" mentions.
4. **Decide how the PI receives it** — the doc is already shared; confirm comment rights, or export a PDF so
   pagination is fixed. Either way, page through the exported PDF once: this check verified figure content and
   order, not where page breaks land relative to captions.
5. **Ask about the equivalence analysis** — whether it is wanted now or deferred to the paper stage.
