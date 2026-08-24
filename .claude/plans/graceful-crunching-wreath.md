# Results pass — Yuval's review (C412, C429, C390, ~X %, peak-freq range, 2 × #REF, aMCI)

**This file is the handoff.** The google-docs MCP never registered this session, so nothing was
read from or written to the Doc. Everything below is drafted from `thesis/chapters/04_results.md`
(the repo mirror), Yuval's tracked changes extracted from `thesis/reviews/Shaked's Thesis_YN.docx`,
and the stats files. Re-prompt with this path after restarting Claude Code.

---

## Context

Yuval returned a reviewed copy on 2026-08-11. The Methods items were applied 2026-08-15
(`thesis/reviews/methods_edits_before_after.md`). This pass is the **Results** items only:
doc sections 4.1–4.6 and `thesis/chapters/04_results.md`. Scope rule in force — only what he
explicitly asked for; ⚑ items from our own pending list stay out (S7 counts, it is his C429).

**Target:** Google Doc "Shaked's Thesis V2", `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`.
**Mirror:** `thesis/chapters/04_results.md`. **Captions:** `thesis/figure_manifest.md`.
Never export a local copy. Do not commit. Do not delete files.

### Section numbering — his copy vs V2

He reviewed V1, which had seven Results subsections. V2 merged the first two. Map his paragraph
numbers to V2 sections before anchoring anything:

| His V1 | V1 heading | V2 |
|---|---|---|
| ¶173, ¶174, ¶176, ¶177 | 4.1 Cohort + 4.2 ISFS detection | **4.1** |
| ¶179–183 | 4.3 Whole-scalp | **4.2** |
| ¶188 | 4.4 Topography | **4.3** |
| ¶192 | 4.5 ROI | **4.4** |
| ¶198 | 4.6 Peak-freq/BW topos | **4.5** |
| ¶200 | 4.7 MoCA | **4.6** |

### Blocker resolved before drafting

**google-docs MCP.** Not registered at session start. Launched manually with the client
id/secret from `~/.claude.json`: it printed `Using saved credentials` / `authorized successfully`
and exited cleanly. **Auth is fine — it just missed the connect window on a cold `npx` resolve**
(the documented failure mode; the npx cache is now warm). Fix = restart Claude Code. Do **not**
send the user to re-auth.

### Decisions taken by the user this session

1. **aMCI relabel: apply verbatim as he asked.** Noted and overridden: 9 of the 30 subjects in the
   current MCI group are non-amnestic (MCI01/02/07/14/16/17/22/28/39 — the naMCI list from the
   2026-06-15 sensitivity run, minus MCI13 which has since been excluded). No caveat sentence goes
   in the prose. Worth one line in the eventual reply to him.
2. **Relabel scope: whole thesis, but only where the text refers to our own cohort/work.** Sentences
   about MCI in general or in other people's studies (Intro, parts of Discussion) keep plain "MCI".
   Per-sentence classification in §7.
3. **Sequencing:** draft file first, restart, then apply — matching the standing review-first rule.

---

## 1. §4.1 — sleep architecture, Figure 1, C390 continuity, C429

This section absorbs the most change: the Figure 1 move (C412), his `~X %` fill-in, the new sleep
continuity paragraph (C390), and the C429 replacement argument.

### 1a. Proposed new heading

**BEFORE:** `4.1 The groups are age-matched and N2 sleep is plentiful in all of them`

**AFTER:** `4.1 Sleep differs with age, but the N2 sleep entering the analysis does not`

*Reason:* "N2 sleep is plentiful" restates the exact non-sequitur C429 attacks. The new heading
states what the section now actually shows.

### 1b. Age / MoCA sentence — his "significantly"

**BEFORE**
> The two older groups were closely matched in age and differed in cognitive score as expected, and
> both were older than the young group by design (Welch t = −0.56, p = 0.57 for age; Mann–Whitney
> U = 375.5, p < 0.001 for MoCA; Table 1).

**AFTER**
> The two older groups were closely matched in age and differed significantly in cognitive score as
> expected, and both were older than the young group by design (Welch t = −0.56, p = 0.57 for age;
> Mann–Whitney U = 375.5, p < 0.001 for MoCA; Table 1).

*Reason:* his ¶173 insertion of `significantly `. V2 rewrote the sentence around it; this carries
the edit across.

### 1c. Sleep architecture paragraph — Figure 1 lands here, `~X %` filled

**BEFORE** (one paragraph; the second sentence is the C429 non-sequitur and is removed)
> Sleep architecture followed the normative age-related pattern (Figure 1): deep (N3) and REM sleep
> were reduced in the older and MCI groups relative to young adults (both one-way ANOVA p < 0.001),
> with the effect carried by young versus each older group and no difference between elderly and MCI
> [ref 24]. N2 nonetheless remained the largest sleep stage in all three groups, so the group
> differences in ISFS reported below do not reflect differing amounts of N2 sleep.

**AFTER**
> An overview of the recorded sleep across the three groups, including whole-night hypnograms
> (time-course of sleep stage dynamics) superimposed with EEG spectrograms (time-frequency
> dynamics) and the distribution of sleep stages, is shown in Figure 1; group statistics for every
> measure below are reported in Table 2.
>
> Sleep architecture followed the normative age-related pattern. Time spent in N3 and in REM sleep
> was reduced in the older and aMCI groups relative to young adults (N3 one-way ANOVA F = 10.85,
> p = 0.0001; REM sleep F = 36.70, p < 0.0001), the effect evident when comparing young versus each
> older group and with no difference between elderly and aMCI [ref 24]. Time spent in N2 sleep
> nonetheless remained the highest among all sleep stages in all three groups (34.1 ± 13.2%,
> 44.5 ± 11.7% and 39.0 ± 12.3% of total recording time in young, elderly and aMCI participants
> respectively).

*Reason:* C412 (Figure 1 moves to Results, and his own rewrite of the overview sentence is
reinstated); his ¶174 wording ("Time spent in N3 and in REM sleep", "evident when comparing",
"highest among all sleep stages"); his `(~X % of total recording time)` placeholder filled with the
real numbers; the trailing non-sequitur deleted, replaced by §1e.

*Note:* his overview sentence also listed "N2 bout properties". Those moved to **Table 2** on
2026-08-13, so that clause is dropped and the sentence points at Table 2 instead.

*Numbers:* `results/demographics_V3/sleep_stage_stats.txt` (F, p, post-hoc) and
`results/demographics_V4/table2_sleep_architecture.csv` (the SDs — the V3 txt has none).

### 1d. NEW paragraph — sleep continuity (C390)

Insert after §1c. He asked for WASO, sleep onset latency, REM latency, sleep efficiency.

**AFTER** (new text)
> Sleep continuity showed the same pattern (Table 2). Wake after sleep onset was longer in both
> older groups than in young adults (23.2 ± 19.6, 51.9 ± 30.0 and 71.0 ± 46.3 min;
> Kruskal–Wallis H = 31.44, p < 0.0001, η² = 0.291; young vs elderly p = 0.0001, young vs aMCI
> p < 0.0001), REM sleep latency was longer (92.8 ± 45.9, 125.4 ± 63.1 and 130.3 ± 55.9 min;
> H = 10.53, p = 0.005, η² = 0.084; young vs elderly p = 0.020, young vs aMCI p = 0.008), and sleep
> efficiency was lower (89.2 ± 6.7%, 83.6 ± 8.5% and 78.6 ± 11.7%; H = 19.19, p = 0.0001,
> η² = 0.170; young vs elderly p = 0.008, young vs aMCI p = 0.0001). On each of the three the
> elderly and aMCI groups were statistically indistinguishable (all p ≥ 0.11). Sleep onset latency
> did not differ across groups (17.4 ± 17.1, 15.6 ± 15.1 and 19.9 ± 18.4 min; H = 1.93, p = 0.38).

*Reason:* C390, straight. He asked for all four, so the null one is reported too.
*Numbers:* `results/demographics_V4/sleep_statistics_stats.txt`. Values are mean ± SD; all four
omnibus tests are Kruskal–Wallis (all four failed Shapiro–Wilk); REM latency is measured from sleep
onset (AASM), i.e. yasa `Lat_REM − SOL`.

⚠ **Rounding conflict to settle once:** WASO MCI SD is `46.3` in `sleep_statistics_stats.txt` and
`46.2` in `table2_sleep_architecture.csv`. The draft uses **46.3** (the value the analysis script
prints). Table 2's PNG shows 46.2 — either accept the 0.1 discrepancy or regenerate the table.

### 1e. NEW paragraph — the C429 answer, replacing the non-sequitur

**AFTER** (new text; replaces the deleted "N2 nonetheless remained the largest sleep stage…" clause)
> Because the amount of N2 sleep itself differed across groups (Kruskal–Wallis H = 13.93,
> p = 0.0009, η² = 0.118; young vs elderly p = 0.0006), we asked directly whether the group
> differences in ISFS reported below could follow from that imbalance. Three observations argue that
> they do not. First, the proportion of each participant's N2 sleep that survived artifact rejection
> and entered the analysis was the same in all three groups (51.9 ± 15.9%, 52.3 ± 17.4% and
> 51.5 ± 15.5%; one-way ANOVA F = 0.019, p = 0.98), and the total duration of analyzed N2 did not
> differ significantly (80.7 ± 44.2, 106.0 ± 50.1 and 90.8 ± 39.9 min; Kruskal–Wallis H = 5.02,
> p = 0.081). Second, the imbalance that does exist runs opposite to the effects: young adults
> contributed the least analyzed N2 sleep of the three groups, yet showed the strongest
> central-parietal hotspot (Section 4.3). Third, entering each participant's analyzed N2 duration as
> a covariate leaves the peak-frequency effect intact (ANCOVA F = 6.03, p = 0.0034, partial
> η² = 0.108), with covariate-adjusted group means identical to the raw means and both pairwise
> contrasts surviving Holm correction (young vs elderly p = 0.013, young vs aMCI p = 0.006); the
> covariate itself explained no peak-frequency variance at all (F = 0.006, p = 0.94). Whole-scalp
> ISFS strength was likewise unaffected (group F = 0.82, p = 0.45; covariate p = 0.49). Bandwidth is
> the one exception. The amount of analyzed N2 sleep is itself strongly related to ISFS bandwidth
> (r = 0.386; covariate F = 14.73, p = 0.0002, partial η² = 0.128, a larger effect than that of
> group), and once it is included the group difference in bandwidth weakens further, from p = 0.061
> to p = 0.206. The bandwidth difference is therefore not interpretable as an effect of age.

*Reason:* C429, both halves — "how are we sure?" and "include this as a factor".
*Numbers:* `results/demographics_V3/{sleep_stage_stats,n2_bouts_table}.txt` and
`results/group_comparison_results/three_groups_V11/three_group_ancova_statistics.txt`. Full
analysis record: `thesis/reviews/c429_c390_results.md`.

*Honesty note, per your instruction:* the bandwidth sentence is written so it reads as "really
non-significant", not as a suppressed trend. Nothing published changes — bandwidth was already ns
at p = 0.061 in V10.

*Judgement call flagged:* the AUC ANCOVA residuals fail Shapiro–Wilk (W = 0.962, p = 0.0048), so
strictly that line is a covariate-adjusted supplement and the Kruskal–Wallis stays primary. The
draft states the result without the caveat to keep the paragraph readable. Say the word and I add
"(reported as a covariate-adjusted supplement; the unadjusted Kruskal–Wallis test remains primary
for this parameter)".

### 1f. Detection-rate paragraph — relabel only

**BEFORE**
> …and in 74.5% of channels in young adults, 85.6% in older adults, and 82.0% in MCI.

**AFTER**
> …and in 74.5% of channels in young adults, 85.6% in older adults, and 82.0% in aMCI.

*Not done here:* his `#` after "in every group of participants" is the anchor for **C432** (example
spectra supplementary figure). That is a figures item, out of scope for this pass.

---

## 2. §4.2 — whole-scalp parameters

### 2a. Peak frequency — his range sentence, with the real numbers

**BEFORE**
> The rhythm ran faster in both older groups than in young adults (young 0.0199 ± 0.0041 Hz,
> elderly 0.0226 ± 0.0041 Hz, MCI 0.0232 ± 0.0040 Hz; one-way ANOVA F = 6.32, p = 0.0026,
> η² = 0.111). Post-hoc comparisons placed the difference between young adults and each older group
> (young vs elderly p = 0.013, young vs MCI p = 0.005), not between the two older groups (p = 0.86).

**AFTER**
> Across all participants the ISFS peak frequency lay around 0.02 Hz, as expected (mean 0.0219 Hz,
> range 0.0095–0.0314 Hz across the 104 participants). The rhythm ran faster in both older groups
> than in young adults (young 0.0199 ± 0.0041 Hz, elderly 0.0226 ± 0.0041 Hz, aMCI
> 0.0232 ± 0.0040 Hz; one-way ANOVA F = 6.32, p = 0.0026, η² = 0.111). Post-hoc comparisons placed
> the difference between young adults and each older group (young vs elderly p = 0.013, young vs
> aMCI p = 0.005), not between the two older groups (p = 0.86).

*Reason:* his ¶180 insertion "ISFS peak frequency was around 0.02 Hz (range: #0.015-0.03Hz?) across
all participants as expected". His `#0.015-0.03Hz?` was a highlighted placeholder — computed, not
guessed.

*The number:* computed from
`results/group_comparison_results/three_groups_V11/three_group_ancova_per_subject.csv`
(whole-scalp per-subject means, N = 104): **min 0.00953, max 0.03138, mean 0.02188.**
Per group — Young 0.0095–0.0284, Elderly 0.0159–0.0314, aMCI 0.0129–0.0295.

⚠ **Decision you may want to make:** the floor is a single low outlier (subject EL3034 at
0.0095 Hz; the next lowest is 0.0118, and the 5th percentile is 0.0158). Yuval guessed
"0.015-0.03". Options: (a) report the true full range 0.0095–0.0314 — the draft's choice, honest and
what he asked for; (b) report the 5th–95th percentile 0.0158–0.0287, which lands almost exactly on
his guess but needs saying so explicitly. Recommend (a).

*Also dropped as redundant:* his "First, we examined ISFS peak frequency across all scalp
electrodes" and "When comparing across groups, we found that". V2's section opener already does
both jobs (triage §3 lists these as superseded).

### 2b. Bandwidth — the hedge removed, the duration artefact stated

**BEFORE**
> The other two parameters did not change significantly. The spectral peak was broader in both older
> groups, in the same direction as the frequency effect, but the comparison fell short of
> significance and is best read as a trend (young 0.0236 ± 0.0089 Hz, elderly 0.0281 ± 0.0085 Hz,
> MCI 0.0276 ± 0.0088 Hz; one-way ANOVA F = 2.87, p = 0.061, η² = 0.054; no post-hoc tests).

**AFTER**
> The other two parameters did not change significantly. Bandwidth, the extent to which the ISFS was
> tightly or loosely locked around its peak frequency, was numerically higher in both older groups
> but did not reach statistical significance (young 0.0236 ± 0.0089 Hz, elderly 0.0281 ± 0.0085 Hz,
> aMCI 0.0276 ± 0.0088 Hz; one-way ANOVA F = 2.87, p = 0.061, η² = 0.054). As noted in Section 4.1,
> bandwidth was the one parameter that tracked how much N2 sleep each participant contributed, and
> adjusting for that duration removed what remained of the group difference (p = 0.206), so it
> should not be read as an effect of age.

*Reason:* his ¶181 rewrite — he deleted "no post-hoc tests were performed, and this difference is
best read as a trend rather than an established effect" and added the gloss on what bandwidth means.
Combined with the C429 ANCOVA finding, per your instruction that it should now read as really ns
rather than a might-be-trend.

### 2c. AUC — his sentence split

**BEFORE**
> Overall ISFS strength did not differ at the whole-scalp level (young 6.46 ± 3.28, elderly
> 7.31 ± 2.88, MCI 7.48 ± 3.49; Kruskal–Wallis H = 1.96, p = 0.38).

**AFTER**
> ISFS strength, measured as the area under the spectral peak (AUC), did not differ significantly
> across groups at the whole-scalp level (young 6.46 ± 3.28, elderly 7.31 ± 2.88, aMCI 7.48 ± 3.49;
> Kruskal–Wallis H = 1.96, p = 0.38).

*Reason:* his ¶182 edits — names AUC at first use in the section and adds "significantly".

### 2d. Elderly-vs-aMCI summary — relabel only

**BEFORE**
> Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and MCI
> groups.

**AFTER**
> Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and
> aMCI groups.

*Reason:* his ¶183 rewrite was "The three examined ISFS parameters (peak frequency, bandwidth, and
AUC) did not differ significantly between healthy elderly and aMCI groups" — the triage lists this
as already satisfied by V2's phrasing, which names all three parameters more compactly. Only the
`a` is carried across. Tell me if you'd rather take his sentence wholesale.

---

## 3. §4.3 — topography, and #REF (a)

**BEFORE**
> Although whole-scalp AUC did not differ across groups, its scalp topography did (Figure 4). In
> young adults the AUC maps showed a central-parietal hotspot; in the older and MCI groups this
> hotspot was flatter and more diffuse (Figure 4A). A cluster-based permutation test … Post-hoc
> tests at the cluster electrodes localized the effect to lower AUC in the older and MCI groups than
> in young adults: the young-versus-elderly contrast … and the young-versus-MCI contrast at 7, while
> the elderly-versus-MCI contrast was significant at only one (E197).

**AFTER**
> Although whole-scalp AUC did not differ across groups when averaged across all electrodes, its
> scalp topography did (Figure 4). In young adults, the AUC maps showed a central-parietal hotspot,
> in accordance with previous studies¹⁰,¹¹. By contrast, in the older and aMCI groups this hotspot
> was less focal and more diffuse (Figure 4A). A cluster-based permutation test on the per-subject
> normalized maps identified a single significant cluster over central-parietal electrodes
> (Figure 4B; p = 0.023, 9 electrodes: E130, E143, E144, E153, E154, E155, E184, E185, and E197).
> Post-hoc tests performed on these electrodes revealed lower AUC in the older and aMCI groups than
> in young adults: the young-versus-elderly contrast was significant at 5 of the 9 electrodes and
> the young-versus-aMCI contrast at 7, while the elderly-versus-aMCI contrast was significant at
> only one (E197). The value above each map in Figure 4A is the mean of the per-subject means.

*Reason:* his ¶188 edits — "when averaged across all electrodes", "in accordance with previous
studies #REF", "By contrast", "less focal", "performed on these electrodes revealed", and two `a`
insertions.

### The citation for #REF (a)

**Use `lazar2019infraslow` + `dimitriades2024isfs` → superscript ¹⁰,¹¹.** Both are already in the
manuscript's numbered list, so **nothing renumbers**.

- **Lázár 2019** (Lázár, Dijk & Lázár, *J Neurosci Methods* 316:22–34) — first detailed human
  demonstration that sigma activity carries an infra-slow oscillation, most prominent in fast
  spindles **over centro-parieto-occipital regions**. An independent, non-Zurich topographic
  corroboration, which is what "previous stud**ies**" (plural) needs.
- **Dimitriades 2024** (*Sci Rep*, published 18 Jun 2026 — see below) — the young-adult AUC hotspot
  this thesis's ROI is literally built from.

The Discussion already makes exactly this pairing (`05_discussion.md:9`), so the Results citation is
consistent with what the thesis already claims.

⚠ **Check before inserting:** the reference-list numbers 10 and 11 are reconstructed from
order-of-first-appearance, cross-checked against five attested superscript anchors in
`thesis/reviews/flavio_edits_before_after.md:81,84`. The authoritative numbered list lives **only in
the Google Doc** and is hand-maintained. Read it and confirm 10 = Lázár, 11 = Dimitriades before
typing the superscripts.

⚠ **Separate issue, out of scope, flagging it:** `reference_dimitriades_citation_status` records
that Dimitriades was **published in Scientific Reports on 18 Jun 2026** but `library.bib` and the
Doc may still carry the bioRxiv preprint. That upgrade is item B1/S2 on the pending list, not a
Yuval ask — say if you want it folded in.

---

## 4. §4.4 — ROI

**BEFORE**
> We then asked whether the same central-parietal reduction appears when the whole pre-defined
> region of interest is summarized as a single value per subject (Figure 5). ISFS strength declined
> descriptively from young to older participants (young 1.099, elderly 1.040, MCI 1.012), but the
> group comparison was not significant (one-way ANOVA p = 0.143).

**AFTER**
> We then asked whether the same central-parietal reduction appears when the whole pre-defined
> region of interest is summarized as a single value per subject, that is, as the proportion of a
> participant's overall ISFS strength that falls within the region (Figure 5). ISFS strength over
> central-parietal electrodes followed the same order as the topographic result, highest in young
> adults, lower in healthy elderly, and lowest in individuals with aMCI (young 1.099, elderly 1.040,
> aMCI 1.012), but the three-group comparison was not significant (one-way ANOVA p = 0.143).

*Reason:* his ¶192 rewrite — it adds the motivation ("go beyond absolute AUC to characterize the
proportion of overall AUC contained within this predefined ROI") and spells out the ordering. His
own typo `central-parietel` is corrected. His "The results revealed a trend" is **not** used: at
p = 0.143 "trend" overstates it, and V2's descriptive framing is safer. Tell me if you want his word.

---

## 5. §4.5 — peak-freq / bandwidth topographies, and #REF (b)

**BEFORE**
> They were not regionally confined: young adults, healthy older adults, and patients with MCI did
> not differ significantly at any location in either peak frequency or bandwidth (Figure S1). Peak
> frequency showed a frontal emphasis in young adults that flattened in the older groups, and
> bandwidth showed no consistent spatial pattern, but neither produced a significant cluster.

**AFTER**
> They were not regionally confined: young adults, healthy older adults, and patients with aMCI did
> not differ significantly at any location in either peak frequency or bandwidth (Figure S1). In
> young adults, both topographies resembled those reported previously¹¹. Peak frequency showed a
> frontal emphasis in young adults that flattened in the older groups, and bandwidth showed no
> consistent spatial pattern, but neither produced a significant cluster.

*Reason:* his ¶198 insertion "In young adults, topographies resembled those reported previously
#REF".

### The citation for #REF (b) — and a real problem with the claim

**Use `dimitriades2024isfs` → superscript ¹¹.** It is effectively the only defensible option: only
the Zurich/Bristol lineage parameterises the spectral peak, so peak frequency and bandwidth have
essentially no comparison literature outside it (`thesis/references/new_refs_annotated.md:265`).
Backup is `grollero2026iso`, the only other study reporting bandwidth — but citing it in Results
moves its first appearance ahead of the whole Discussion block and **renumbers five references**, so
avoid it.

🚩 **`[VERIFY]` — the claim may not be supportable as written.** Nothing in the repo states that
prior work reports a *frontal* peak-frequency emphasis in young adults, which is what §4.5 and
Figure S1 claim. Every prior-topography statement in the repo is about ISFS **strength**, not
frequency or bandwidth. Before this sentence goes in, open
`thesis/references/ISFS_Development_Dimitriades_2024.pdf` and check whether it reports young-adult
peak-frequency and bandwidth topographies at all. (It could not be text-extracted this session — no
poppler.) If it does not, the honest fallback is to drop the sentence and tell Yuval the
comparison literature does not exist for these two parameters — which is itself a defensible answer
to his #REF.

---

## 6. §4.6 — MoCA

**BEFORE:** `…in the pooled older-adult and MCI sample…`
**AFTER:** `…in the pooled older-adult and aMCI sample…`

*Reason:* relabel only. His ¶200 edit ("scalars" → "metrics") has no anchor — V2 does not use the
word "scalars" in this sentence.

---

## 7. aMCI relabel — whole-thesis sweep

Per your rule: **relabel where the sentence describes our cohort or our result; leave plain "MCI"
where it describes MCI in general or other people's studies.**

### Relabel → aMCI

| File | Sentence (anchor) |
|---|---|
| `01_abstract.md:5` | "30 patients with MCI" (our sample) |
| `01_abstract.md:5` | "Patients with MCI were indistinguishable from healthy older adults" |
| `02_introduction.md:15` | "three groups (young controls, healthy older adults, and patients with MCI)" |
| `02_introduction.md:15` | "whether the ISFS of patients with MCI is comparable to…" |
| `03_methods.md:13` | "patients with mild cognitive impairment (MCI; n = 30; …)" → define **aMCI** here, at first use |
| `03_methods.md:17` | "Patients with MCI recorded in Tel Aviv…" and "At the Sydney site, patients with MCI…" |
| `03_methods.md:21` | "MCI 21.5 ± 4.3, n = 14" (MoCA, our sample) |
| `03_methods.md:31` | "3.4% in MCI patients" (our data quality) |
| `05_discussion.md:5` | "we compared the ISFS across three groups: … and patients with MCI" |
| `05_discussion.md:7` | "Patients with MCI were indistinguishable from healthy older adults on every ISFS measure" |
| `05_discussion.md:13` | "The MCI and elderly groups were closely matched in age…" |
| `05_discussion.md:13` | "we found no evidence that MCI contributes anything detectable beyond age" |
| `05_discussion.md:15` | "our analogous strength measure did not separate MCI from healthy older adults" |
| `05_discussion.md:17` | "the older and MCI cohorts were clinically and etiologically heterogeneous… any subtle MCI-specific effect" |
| `05_discussion.md:17` | "the subset of MCI patients who will progress" (our design) |
| `05_discussion.md:19` | "following older and MCI participants over time" (our future work) |
| all Results figure captions | Figures 3, 4, 5, S2 in `thesis/figure_manifest.md` |

### Leave as plain MCI

| File | Sentence | Why |
|---|---|---|
| `01_abstract.md:5` | "whether mild cognitive impairment (MCI) adds a signal beyond that of age, is unknown" | the general research question |
| `02_introduction.md:13` | "In mild cognitive impairment (MCI) and Alzheimer's disease, polysomnographic studies report…" | other people's literature |
| `02_introduction.md:15` | "whether MCI adds a signal beyond that of aging, remain unknown" | general question |
| `05_discussion.md:11` | "REM-sleep EEG slowing … in aging and MCI [@andre2025remslowing]" | describes André's study |
| `05_discussion.md:13` | "biomarker of early MCI"; "the transition to MCI does not measurably accelerate" | general claims about the condition |
| `05_discussion.md:15` | "later than the MCI stage"; "though not to MCI, has now been reported" | general disease-stage claims |
| `05_discussion.md:21` | "features of the ISFS in aging and MCI" | general framing |

🚩 **Terminology conflict to be aware of when the Discussion session runs:** `05_discussion.md:17`
currently says the MCI cohort was "clinically and etiologically heterogeneous". That is true (9/30
non-amnestic) and now sits next to a blanket "aMCI" label. Not resolved here — flagging it so it is
not lost.

---

## 8. Figure 1 move (C412) — mechanics

The 2026-08-15 Methods pass deleted the overview *sentence* from 3.2 but deliberately left the
**image and caption** sitting there. Three steps remain:

1. **Move the image + caption** out of Methods 3.2 into Results 4.1, immediately after the paragraph
   in §1c. **Recommend you do this by hand in the browser** (select the image and its caption
   paragraph, cut, paste). The Docs API route means `deleteRange` on the inline image plus
   `insertImage`, and the MCP's `insertImage` needs a Drive-hosted or public URI — re-uploading
   risks a re-encode and a size change on a figure that is already page-tuned. Thirty seconds by
   hand, zero risk.
2. **Reinstate the overview sentence** in Results — done as the first paragraph of §1c above, using
   his own improved wording, with the "N2 bout properties" clause repointed to Table 2.
3. **`thesis/figure_manifest.md:105`** — retarget the F1 heading from `*Methods/Results overview*`
   to `*Results*`. One-word edit; Methods no longer references Figure 1.

*Not touched:* the Doc still holds the **V10** F1 image while the manifest points at
`hypno_sleep_stages_V11.png`. Swapping every figure to V11 is the figures pass (C487/C502/C591/C592,
email #6) — out of scope here.

---

## 9. Caption inconsistencies his rewrites create

He left all three Results captions untouched, so two now contradict the revised body text. Both are
in `thesis/figure_manifest.md` and in the Doc:

- **Figure 3 caption** still reads "Bandwidth trended in the same direction but did not reach
  significance (one-way ANOVA p = 0.061; **no post-hoc tests**)" and "Elderly and MCI did not differ
  on any parameter" — the exact wording he deleted in ¶181 and rewrote in ¶183.
  **Proposed:** "Bandwidth did not differ significantly across groups (one-way ANOVA p = 0.061)."
  plus the aMCI relabel.
- **Figure 5 caption** still reads "The group means followed the expected order" — reworded in ¶192.
  Minor; the aMCI relabel is the only mandatory change.

Both are consequences of edits he asked for, so they are in scope. Confirm and I apply them.

---

## 10. Execution order

1. Restart Claude Code so `google-docs` registers. Verify with a `readDocument` on the Results range
   before touching anything.
2. Read the Doc's Results text and confirm each **BEFORE** block above matches it verbatim; the
   drafts are built from the repo mirror, which should be identical but is not authoritative.
3. Read the Doc's numbered reference list; confirm 10 = Lázár, 11 = Dimitriades.
4. Resolve the `[VERIFY]` in §5 (Dimitriades PDF, peak-freq/bandwidth topography).
5. Apply §1–§6 to the Doc, then mirror into `thesis/chapters/04_results.md` with a `v3` note at the
   top in the style of the existing version notes.
6. Apply §7 across the other chapters and their Doc sections.
7. Apply §8 (Figure 1, by hand) and §9 (captions), and update `thesis/figure_manifest.md:105`.
8. Mark this file APPLIED with the options chosen, as
   `thesis/reviews/methods_edits_before_after.md` does.

**Docs API reminders:** `readDocument` silently omits table cells — use `listDocumentTables` +
`getTableStructure`. `findAndReplace` cannot match across a paragraph mark, so deleting a paragraph
means `findElement` → `deleteRange(start, textEnd + 1)`. Inserting right after a heading inherits
the heading style — follow with `applyParagraphStyle NORMAL_TEXT`. Citation superscripts are plain
Unicode glyphs with no formatting, so replacements can span them; the Gaussian formula variables in
Methods are genuinely italic, but nothing in Results is.

---

## 11. Open questions for you

1. **Peak-frequency range** (§2a) — full range 0.0095–0.0314, or 5th–95th percentile 0.0158–0.0287?
   Recommend the full range.
2. **AUC ANCOVA caveat** (§1e) — add the non-normal-residuals qualifier, or keep the paragraph lean?
3. **§2d and §4** — carry his sentences wholesale, or keep V2's phrasing with only the `a` inserted?
   Recommend the latter for both.
4. **Figure 3 / Figure 5 captions** (§9) — apply the rewrites, or leave captions to the figures pass?
5. **Dimitriades preprint → *Sci Rep*** (§3) — fold the citation upgrade in, or leave it on the
   pending list?

## 12. Out of scope, noticed, not acted on

- **C432** (his `#` in §4.1) — supplementary figure of example spectra from different subjects.
  Figures pass.
- The Doc's figures are still V10 assets while the manifest is V11. Figures pass.
- `maris2007nonparametric` appears to be cited nowhere, including the Doc, despite a cluster-based
  permutation Methods section. Not a Yuval item.
- He inserted a stray empty `Heading3` paragraph after the Figure 5 caption block (his ¶196). Cosmetic.
