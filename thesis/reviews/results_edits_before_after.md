# Results edits — before / after, for review before anything touches the Doc

**Target:** Google Doc "Shaked's Thesis V2" (`1YpXrDGFlzRk…`), sections 4.1–4.6, the Figure 3 / Figure 5 /
Figure S2 captions, and the aMCI relabel across the whole document.
**Mirror:** `thesis/chapters/04_results.md` (+ `01_abstract.md`, `02_introduction.md`, `03_methods.md`,
`05_discussion.md` for the relabel). **Captions:** `thesis/figure_manifest.md`.
Drafted 2026-08-15. **APPLIED to the Doc and mirrored to the chapters on 2026-08-15**, after your
approval of every entry. Decisions taken are recorded inline; the before/after text below is what
actually went in.

**Changes made after approval, all agreed in the same round:**

- **§0 rounding conflict — resolved, not accepted.** `code/make_table2_sleep.py` now formats the
  continuity mean ± SD once, from `sleep_statistics_per_subject.csv`, instead of re-rounding the
  two-decimal values in `sleep_statistics_table.csv` (46.25 → `.1f` → 46.2 under banker's rounding).
  Table 2 regenerated; the WASO MCI SD now reads **46.3** in the table, the stats file and the prose
  alike. No other cell changed.
- **§5.2 Grollero — checked and rejected.** You asked whether `grollero2026iso` also supports the
  topographic claim. It cannot: they recorded with a **Dreem-2 headband** and *"a frontal–central
  signal of interest was obtained by averaging the two available bipolar derivations"* — a single
  channel, and the word "topograph" does not appear in the paper. It reports bandwidth, but never
  spatially. `dimitriades2024isfs` stays the sole citation and **nothing renumbers.**
- **Three anchors the draft missed, relabelled under the same §8 rule:** the §4.2 opening sentence
  ("…and patients with MCI, each summarized per subject"), the Figure S2 caption's "pooled elderly and
  MCI group", and the Table 2 caption's "Elderly and MCI did not differ significantly on any measure".

Scope = only the Results items Yuval explicitly raised: **C412, C429, C390**, his `~X %` and
peak-frequency-range placeholders, his two `#REF` placeholders, and his aMCI relabel, plus his stylistic
tracked edits inside Results. Everything else is listed in §11 and left alone.

**Decisions you took before drafting** (recorded here so the file stands on its own):

1. Peak-frequency range → the **full range 0.0095–0.0314 Hz**, not the 5th–95th percentile.
2. Prose keeps **"(Table 2)"** pointers; you paste the Table 2 image in by hand (§10).
3. **All three captions fixed now** (Figure 3, Figure 5, Figure S2).
4. aMCI relabel = **whole thesis**, but only where the sentence means *our cohort* (§8).

Two smaller calls I made without asking, flagged so you can overturn them:

- The AUC ANCOVA in §1.6 is reported without a "residuals are non-normal" qualifier. It is null before
  and after adjustment, and Kruskal–Wallis remains the stated primary test for AUC, so the caveat would
  add a sentence without changing anything. Say the word and it goes in.
- In §2.4 and §4.1 I kept V2's phrasing and folded in only his additions, rather than swapping in his
  sentences wholesale — the triage lists both of those rewrites as already satisfied by V2.

---

## §0 — verification done before drafting

Nothing below is taken on trust from the earlier draft:

| Checked | Result |
|---|---|
| Doc §4.1–4.6 vs `04_results.md` | **identical** — every BEFORE block below is the Doc's live text |
| Doc reference list | **10 = Lázár 2019, 11 = Dimitriades 2024** — so both `#REF` fills renumber nothing |
| His tracked changes ¶172–200 | re-extracted from `Shaked's Thesis_YN.docx` (`w:ins` / `w:del`) |
| Every number below | re-read from `demographics_V3/`, `demographics_V4/`, `three_groups_V11/` |
| Peak-frequency range | recomputed: min 0.00953, max 0.03138, mean 0.02188 (N = 104) |
| **The `#REF` (b) claim** | **verified against the Dimitriades PDF — it holds.** See §5.2. |

**One rounding conflict, not silently reconciled:** the WASO SD for the MCI group is **46.3** in
`demographics_V4/sleep_statistics_stats.txt` and **46.2** in `table2_sleep_architecture.csv` (and so in
Table 2's PNG). The prose below uses **46.3**, the value the analysis script prints. Either accept the
0.1 mismatch between text and table, or regenerate Table 2.

---

## §1 — Results 4.1

This section takes the most change: the Figure 1 move (C412), his `~X %` fill-in, the new sleep-continuity
paragraph (C390), and the C429 replacement argument.

### 1.1 — Section heading

**Context:** the first Results heading.

**BEFORE**
> 4.1 The groups are age-matched and N2 sleep is plentiful in all of them

**AFTER**
> 4.1 Sleep differs with age, but the N2 sleep entering the analysis does not

*Reason:* "N2 sleep is plentiful" is the exact non-sequitur C429 attacks. The new heading states what the
section now actually shows.

### 1.2 — Age and MoCA sentence

**Context:** first paragraph of 4.1, unchanged apart from one word.

**BEFORE**
> The two older groups were closely matched in age and differed in cognitive score as expected, and both
> were older than the young group by design (Welch t = −0.56, p = 0.57 for age; Mann–Whitney U = 375.5,
> p < 0.001 for MoCA; Table 1).

**AFTER**
> The two older groups were closely matched in age and differed significantly in cognitive score as
> expected, and both were older than the young group by design (Welch t = −0.56, p = 0.57 for age;
> Mann–Whitney U = 375.5, p < 0.001 for MoCA; Table 1).

*Reason:* his ¶173 insertion of `significantly`. V2 rewrote the sentence around it; this carries the edit
across.

### 1.3 — NEW paragraph: the Figure 1 overview, reinstated in his own words

**Context:** inserted immediately after 1.2, as the second paragraph of 4.1. This is the sentence the
Methods pass deleted from 3.2; C412 asks for the figure to live here.

**AFTER** (new text)
> An overview of the recorded sleep across the three groups, including whole-night hypnograms
> (time-course of sleep stage dynamics) superimposed with EEG spectrograms (time-frequency dynamics),
> and the distribution of sleep stages, is shown in Figure 1; group statistics for every measure below
> are reported in Table 2.

*Reason:* C412. The wording is **his own tracked rewrite** of the overview sentence, which improved on
ours and should not be lost. One change to it: he also listed "N2 bout properties", which have been in
**Table 2** since 2026-08-13 rather than in Figure 1, so that clause is dropped and the sentence points
at Table 2 instead.

### 1.4 — Sleep architecture paragraph

**Context:** the paragraph that currently opens "Sleep architecture followed the normative age-related
pattern (Figure 1)". Its second sentence is the C429 non-sequitur and is removed here; §1.6 replaces it.

**BEFORE**
> Sleep architecture followed the normative age-related pattern (Figure 1): deep (N3) and REM sleep were
> reduced in the older and MCI groups relative to young adults (both one-way ANOVA p < 0.001), with the
> effect carried by young versus each older group and no difference between elderly and MCI²⁴. N2
> nonetheless remained the largest sleep stage in all three groups, so the group differences in ISFS
> reported below do not reflect differing amounts of N2 sleep.

**AFTER**
> Sleep architecture followed the normative age-related pattern. Time spent in N3 and in REM sleep was
> reduced in the older and aMCI groups relative to young adults (N3 one-way ANOVA F = 10.85, p = 0.0001;
> REM sleep F = 36.70, p < 0.0001), the effect evident when comparing young adults versus each older
> group and with no difference between elderly and aMCI²⁴. Time spent in N2 sleep nonetheless remained
> the highest among all sleep stages across all three groups (34.1 ± 13.2%, 44.5 ± 11.7% and
> 39.0 ± 12.3% of total recording time in young, elderly and aMCI participants respectively).

*Reason:* his ¶174 wording throughout ("Time spent in N3 and in REM sleep", "evident when comparing",
"highest among all sleep stages"); his `(~X % of total recording time)` placeholder filled with the real
numbers; the "(Figure 1)" pointer moves up into §1.3; the trailing non-sequitur deleted and replaced by
§1.6.
*Numbers:* `demographics_V3/sleep_stage_stats.txt` (F and p) and
`demographics_V4/table2_sleep_architecture.csv` (the SDs — the V3 text file carries none).

### 1.5 — NEW paragraph: sleep continuity (C390)

**Context:** inserted after §1.4. He asked for WASO, sleep onset latency, REM latency and sleep
efficiency; all four are reported, including the null one.

**AFTER** (new text)
> Sleep continuity showed the same pattern (Table 2). Wake after sleep onset was longer in both older
> groups than in young adults (23.2 ± 19.6, 51.9 ± 30.0 and 71.0 ± 46.3 min; Kruskal–Wallis H = 31.44,
> p < 0.0001, η² = 0.291; young vs elderly p = 0.0001, young vs aMCI p < 0.0001), REM sleep latency was
> longer (92.8 ± 45.9, 125.4 ± 63.1 and 130.3 ± 55.9 min; H = 10.53, p = 0.005, η² = 0.084; young vs
> elderly p = 0.020, young vs aMCI p = 0.008), and sleep efficiency was lower (89.2 ± 6.7%, 83.6 ± 8.5%
> and 78.6 ± 11.7%; H = 19.19, p = 0.0001, η² = 0.170; young vs elderly p = 0.008, young vs aMCI
> p = 0.0001). On each of these three measures the elderly and aMCI groups were statistically
> indistinguishable (all p ≥ 0.11). Sleep onset latency did not differ across groups (17.4 ± 17.1,
> 15.6 ± 15.1 and 19.9 ± 18.4 min; H = 1.93, p = 0.38).

*Reason:* C390, straight.
*Numbers:* `demographics_V4/sleep_statistics_stats.txt`. Values are mean ± SD; all four omnibus tests are
Kruskal–Wallis because all four failed Shapiro–Wilk; REM latency is measured from sleep onset (AASM),
i.e. yasa `Lat_REM − SOL`; sleep efficiency is the subjects-sheet column, not recomputed.

### 1.6 — NEW paragraph: the C429 answer

**Context:** inserted after §1.5, closing 4.1. This is what replaces the deleted "N2 nonetheless remained
the largest sleep stage… so the group differences do not reflect differing amounts of N2 sleep".

**AFTER** (new text)
> Because the amount of N2 sleep itself differed across groups (Kruskal–Wallis H = 13.93, p = 0.0009,
> η² = 0.118; young vs elderly p = 0.0006), we asked directly whether the group differences in ISFS
> reported below could follow from that imbalance. Three observations argue that they do not. First, the
> proportion of each participant's N2 sleep that survived artifact rejection and entered the analysis was
> the same in all three groups (51.9 ± 15.9%, 52.3 ± 17.4% and 51.5 ± 15.5%; one-way ANOVA F = 0.019,
> p = 0.98), and the total duration of analyzed N2 did not differ significantly (80.7 ± 44.2,
> 106.0 ± 50.1 and 90.8 ± 39.9 min; Kruskal–Wallis H = 5.02, p = 0.081). Second, the imbalance that does
> exist runs opposite to the effects: young adults contributed the least analyzed N2 sleep of the three
> groups, yet showed the strongest central-parietal hotspot (Section 4.3). Third, entering each
> participant's analyzed N2 duration as a covariate leaves the peak-frequency effect intact (ANCOVA
> F = 6.03, p = 0.0034, partial η² = 0.108), with covariate-adjusted group means identical to the raw
> means and both pairwise contrasts surviving Holm correction (young vs elderly p = 0.013, young vs aMCI
> p = 0.006); the covariate itself explained no peak-frequency variance at all (F = 0.006, p = 0.94).
> Whole-scalp ISFS strength was likewise unaffected (group F = 0.82, p = 0.45; covariate p = 0.49).
> Bandwidth is the one exception. The amount of analyzed N2 sleep is itself strongly related to ISFS
> bandwidth (r = 0.386; covariate F = 14.73, p = 0.0002, partial η² = 0.128, a larger effect than that of
> group), and once it is included the group difference in bandwidth weakens further, from p = 0.061 to
> p = 0.206. The bandwidth difference is therefore not interpretable as an effect of age.

*Reason:* C429, both halves — "how are we sure?" and "did we try to equate this? Or include this as a
factor?".
*Numbers:* `demographics_V3/{sleep_stage_stats,n2_bouts_table}.txt` and
`three_groups_V11/three_group_ancova_statistics.txt`. Full analysis record:
`thesis/reviews/c429_c390_results.md`.

**Honesty note, per your instruction:** the bandwidth sentences are written so the parameter reads as
genuinely non-significant rather than as a suppressed trend. Nothing published changes — bandwidth was
already ns at p = 0.061 in V10.

### 1.7 — Detection-rate paragraph

**BEFORE**
> …and in 74.5% of channels in young adults, 85.6% in older adults, and 82.0% in MCI.

**AFTER**
> …and in 74.5% of channels in young adults, 85.6% in older adults, and 82.0% in aMCI.

*Reason:* relabel only. *Not done here:* his `#` mark after "in every group of participants" is the anchor
for **C432** (a supplementary figure of example spectra from different subjects) — a figures item.

---

## §2 — Results 4.2

### 2.1 — Peak frequency: his range sentence, with the real numbers

**BEFORE**
> The rhythm ran faster in both older groups than in young adults (young 0.0199 ± 0.0041 Hz, elderly
> 0.0226 ± 0.0041 Hz, MCI 0.0232 ± 0.0040 Hz; one-way ANOVA F = 6.32, p = 0.0026, η² = 0.111). Post-hoc
> comparisons placed the difference between young adults and each older group (young vs elderly p = 0.013,
> young vs MCI p = 0.005), not between the two older groups (p = 0.86).

**AFTER**
> Across all participants the ISFS peak frequency lay around 0.02 Hz, as expected (mean 0.0219 Hz, range
> 0.0095–0.0314 Hz across the 104 participants). The rhythm ran faster in both older groups than in young
> adults (young 0.0199 ± 0.0041 Hz, elderly 0.0226 ± 0.0041 Hz, aMCI 0.0232 ± 0.0040 Hz; one-way ANOVA
> F = 6.32, p = 0.0026, η² = 0.111). Post-hoc comparisons placed the difference between young adults and
> each older group (young vs elderly p = 0.013, young vs aMCI p = 0.005), not between the two older
> groups (p = 0.86).

*Reason:* his ¶180 insertion, "ISFS peak frequency was around 0.02 Hz (range: #0.015-0.03Hz?) across all
participants as expected". The `#0.015-0.03Hz?` was a highlighted guess; this is the computed value.
*The number:* `three_groups_V11/three_group_ancova_per_subject.csv` (whole-scalp per-subject means,
N = 104): min 0.00953, max 0.03138, mean 0.02188. Per group — young 0.0095–0.0284, elderly 0.0159–0.0314,
aMCI 0.0129–0.0295. Note the floor is one low outlier (EL3034 at 0.0095; the next lowest is 0.0118), which
is why he may query it; you chose the true full range over the 5th–95th percentile (0.0158–0.0287).
*Also dropped as redundant:* his "First, we examined ISFS peak frequency across all scalp electrodes" and
"When comparing across groups, we found that" — V2's section opener already does both jobs (triage §3).

### 2.2 — Bandwidth: the hedge removed, the duration artefact stated

**BEFORE**
> The other two parameters did not change significantly. The spectral peak was broader in both older
> groups, in the same direction as the frequency effect, but the comparison fell short of significance
> and is best read as a trend (young 0.0236 ± 0.0089 Hz, elderly 0.0281 ± 0.0085 Hz, MCI
> 0.0276 ± 0.0088 Hz; one-way ANOVA F = 2.87, p = 0.061, η² = 0.054; no post-hoc tests).

**AFTER**
> The other two parameters did not change significantly. Bandwidth, the extent to which the ISFS was
> tightly or loosely locked around its peak frequency, was numerically higher in both older groups but
> did not reach statistical significance (young 0.0236 ± 0.0089 Hz, elderly 0.0281 ± 0.0085 Hz, aMCI
> 0.0276 ± 0.0088 Hz; one-way ANOVA F = 2.87, p = 0.061, η² = 0.054). As noted in Section 4.1, bandwidth
> was the one parameter that tracked how much N2 sleep each participant contributed, and adjusting for
> that duration removed what remained of the group difference (p = 0.206), so it should not be read as an
> effect of age.

*Reason:* his ¶181 rewrite — he deleted "no post-hoc tests were performed, and this difference is best
read as a trend rather than an established effect" and added the gloss on what bandwidth means. Combined
with the C429 ANCOVA result, per your instruction that it should now read as really ns rather than a
might-be-trend.

### 2.3 — AUC: his sentence split

**BEFORE**
> Overall ISFS strength did not differ at the whole-scalp level (young 6.46 ± 3.28, elderly 7.31 ± 2.88,
> MCI 7.48 ± 3.49; Kruskal–Wallis H = 1.96, p = 0.38).

**AFTER**
> ISFS strength, measured as the area under the spectral peak (AUC), did not differ significantly across
> groups at the whole-scalp level (young 6.46 ± 3.28, elderly 7.31 ± 2.88, aMCI 7.48 ± 3.49;
> Kruskal–Wallis H = 1.96, p = 0.38).

*Reason:* his ¶182 edits — names AUC at first use in the section, adds "significantly".

### 2.4 — Elderly-vs-aMCI summary

**BEFORE**
> Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and MCI
> groups.

**AFTER**
> Neither peak frequency, bandwidth, nor strength differed significantly between the healthy elderly and
> aMCI groups.

*Reason:* his ¶183 rewrite was "The three examined ISFS parameters (peak frequency, bandwidth, and AUC)
did not differ significantly between healthy elderly and aMCI groups". The triage lists this as already
satisfied by V2's phrasing, which names all three parameters more compactly, so only his `healthy` and
`a` are carried across. Tell me if you would rather take his sentence wholesale.

---

## §3 — Results 4.3, and `#REF` (a)

**BEFORE**
> Although whole-scalp AUC did not differ across groups, its scalp topography did (Figure 4). In young
> adults the AUC maps showed a central-parietal hotspot; in the older and MCI groups this hotspot was
> flatter and more diffuse (Figure 4A). A cluster-based permutation test on the per-subject normalized
> maps identified a single significant cluster over central-parietal electrodes (Figure 4B; p = 0.023,
> 9 electrodes: E130, E143, E144, E153, E154, E155, E184, E185, and E197). Post-hoc tests at the cluster
> electrodes localized the effect to lower AUC in the older and MCI groups than in young adults: the
> young-versus-elderly contrast was significant at 5 of the 9 electrodes and the young-versus-MCI contrast
> at 7, while the elderly-versus-MCI contrast was significant at only one (E197). The value above each map
> in Figure 4A is the mean of the per-subject means.

**AFTER**
> Although whole-scalp AUC did not differ across groups when averaged across all electrodes, its scalp
> topography did (Figure 4). In young adults, the AUC maps showed a central-parietal hotspot, in
> accordance with previous studies¹⁰,¹¹. By contrast, in the older and aMCI groups this hotspot was less
> focal and more diffuse (Figure 4A). A cluster-based permutation test on the per-subject normalized maps
> identified a single significant cluster over central-parietal electrodes (Figure 4B; p = 0.023,
> 9 electrodes: E130, E143, E144, E153, E154, E155, E184, E185, and E197). Post-hoc tests performed on
> these electrodes revealed lower AUC in the older and aMCI groups than in young adults: the
> young-versus-elderly contrast was significant at 5 of the 9 electrodes and the young-versus-aMCI
> contrast at 7, while the elderly-versus-aMCI contrast was significant at only one (E197). The value
> above each map in Figure 4A is the mean of the per-subject means.

*Reason:* his ¶188 edits — "when averaged across all electrodes", "in accordance with previous studies
#REF", "By contrast", "less focal", "performed on these electrodes revealed", and the `a` insertions.

**The citation:** `lazar2019infraslow` + `dimitriades2024isfs` → **¹⁰,¹¹**. Confirmed against the Doc's
own numbered list, so nothing renumbers. Lázár 2019 is an independent, non-Zurich demonstration that the
sigma infra-slow oscillation is most prominent over centro-parieto-occipital regions — which is what
"previous stud**ies**" (plural) needs; Dimitriades 2024 is the young-adult AUC hotspot this thesis's ROI
is built from. The Discussion already makes exactly this pairing.

---

## §4 — Results 4.4

**BEFORE**
> We then asked whether the same central-parietal reduction appears when the whole pre-defined region of
> interest is summarized as a single value per subject (Figure 5). ISFS strength declined descriptively
> from young to older participants (young 1.099, elderly 1.040, MCI 1.012), but the group comparison was
> not significant (one-way ANOVA p = 0.143).

**AFTER**
> We then asked whether the same central-parietal reduction appears when the whole pre-defined region of
> interest is summarized as a single value per subject, that is, as the proportion of a participant's
> overall ISFS strength that falls within the region (Figure 5). ISFS strength over central-parietal
> electrodes followed the same order as the topographic result, highest in young adults, lower in healthy
> elderly, and lowest in individuals with aMCI (young 1.099, elderly 1.040, aMCI 1.012), but the
> three-group comparison was not significant (one-way ANOVA p = 0.143).

*Reason:* his ¶192 rewrite adds the motivation ("go beyond absolute AUC to characterize the proportion of
overall AUC contained within this predefined ROI") and spells out the ordering. His own typo
`central-parietel` is corrected. His phrase "The results revealed a trend" is **not** used: at p = 0.143
"trend" overstates it, and V2's descriptive framing is safer. Tell me if you want his word.

---

## §5 — Results 4.5, and `#REF` (b)

### 5.1 — The text

**BEFORE**
> …They were not regionally confined: young adults, healthy older adults, and patients with MCI did not
> differ significantly at any location in either peak frequency or bandwidth (Figure S1). Peak frequency
> showed a frontal emphasis in young adults that flattened in the older groups, and bandwidth showed no
> consistent spatial pattern, but neither produced a significant cluster.

**AFTER**
> …They were not regionally confined: young adults, healthy older adults, and patients with aMCI did not
> differ significantly at any location in either peak frequency or bandwidth (Figure S1). In young adults,
> both topographies resembled those reported previously¹¹: peak frequency showed a frontal emphasis that
> flattened in the older groups, and bandwidth showed no consistent spatial pattern, but neither produced
> a significant cluster.

*Reason:* his ¶198 insertion, "In young adults, topographies resembled those reported previously #REF".
Merged with the following sentence rather than added in front of it, to avoid saying "in young adults"
twice.

### 5.2 — The citation, and the `[VERIFY]` the earlier draft left open

**Use `dimitriades2024isfs` → ¹¹.** The earlier draft flagged that nothing in the repo supported the claim
that prior work reports these two topographies, and that the sentence might have to be dropped. **That is
now resolved: the claim holds.** From the Dimitriades 2024 Results (`thesis/references/ISFS_Development_Dimitriades_2024.pdf`,
extracted with pypdf this session):

> "…a frontal cluster of seven electrodes revealed significantly higher values in the young adult group
> compared to the child and early adolescent groups … indicating higher peak frequencies frontally with
> age. **Peak frequency and area under the curve showed local minima and maxima in central regions,
> respectively, while bandwidth displayed no clear topographical pattern.**"

Both halves of our sentence match: a frontal emphasis of peak frequency in young adults (their oldest
group), and no consistent bandwidth topography. Backup citation `grollero2026iso` is deliberately avoided
— it is the only other study reporting bandwidth, but citing it here moves its first appearance ahead of
the whole Discussion block and **renumbers five references**.

---

## §6 — Results 4.6

**BEFORE**
> …in the pooled older-adult and MCI sample…

**AFTER**
> …in the pooled older-adult and aMCI sample…

*Reason:* relabel only. His ¶200 edit was "scalars" → "metrics"; V2's body sentence no longer uses the
word, but the **Figure S2 caption still does** — handled in §7.3.

---

## §7 — Captions (your decision 3)

Applied both in the Doc and in `thesis/figure_manifest.md`.

### 7.1 — Figure 3

**BEFORE**
> …whereas bandwidth showed a non-significant trend in the same direction and overall strength (AUC) did
> not differ across groups; elderly and MCI did not differ on any parameter.

**AFTER**
> …whereas bandwidth did not differ significantly across groups (one-way ANOVA, p = 0.061) and overall
> strength (AUC) did not differ across groups; elderly and aMCI did not differ on any parameter.

*Reason:* the caption still carries the "trend" reading he deleted in ¶181, which now contradicts §2.2.

### 7.2 — Figure 5

**BEFORE**
> Although the group means followed the expected order (young > elderly > MCI), the three-group comparison
> was not significant (one-way ANOVA, p = 0.143).

**AFTER**
> Although the group means followed the same order as the topographic result (young > elderly > aMCI), the
> three-group comparison was not significant (one-way ANOVA, p = 0.143).

*Reason:* "the expected order" was reworded in his ¶192; relabel.

### 7.3 — Figure S2

**BEFORE**
> *Figure S2. ISFS scalars versus MoCA.* … Each panel plots one scalar (peak frequency, bandwidth,
> whole-scalp AUC, or ROI AUC) against MoCA score. No scalar correlated with MoCA…

**AFTER**
> *Figure S2. ISFS metrics versus MoCA.* … Each panel plots one metric (peak frequency, bandwidth,
> whole-scalp AUC, or ROI AUC) against MoCA score. No metric correlated with MoCA…

*Reason:* his ¶200 edit, "scalars" → "metrics". The earlier draft recorded this as having no anchor in V2;
it does — the anchor is this caption, not the §4.6 body.

### 7.4 — Figures 1 and 4 — relabel only

Figure 1's caption ("reduced in elderly and MCI") and Figure 4's ("flattens and spreads in aging and MCI",
"lower AUC in elderly and MCI") get **MCI → aMCI and nothing else**. Figure 1's caption in the Doc still
describes the removed panel C and still refers to the V10 image; rewriting it belongs to the figures pass.

---

## §8 — The aMCI relabel, whole thesis (your decision 4)

Rule: relabel where the sentence describes **our cohort or our result**; leave plain "MCI" where it
describes MCI in general or other people's studies.

### Relabel → aMCI

| File / place | Anchor |
|---|---|
| Doc **Table 1**, header cell row 0 col 3 | `MCI` → `aMCI` |
| `01_abstract.md` | "30 patients with MCI" (our sample) |
| `01_abstract.md` | "Patients with MCI were indistinguishable from healthy older adults" |
| `02_introduction.md:15` | "three groups (young controls, healthy older adults, and patients with MCI)" |
| `02_introduction.md:15` | "whether the ISFS of patients with MCI is comparable to…" |
| `03_methods.md:13` | "patients with mild cognitive impairment (MCI; n = 30…)" → **define aMCI here, at first use**: "patients with amnestic mild cognitive impairment (aMCI; n = 30…)" |
| `03_methods.md:17` | "Patients with MCI recorded in Tel Aviv…" and "At the Sydney site, patients with MCI…" |
| `03_methods.md:21` | "MCI 21.5 ± 4.3, n = 14" (MoCA, our sample) |
| `03_methods.md:31` | "3.4 ± 4.4% in MCI patients" (our data quality) |
| `04_results.md` | §§4.1–4.6 as above |
| `05_discussion.md:5` | "we compared the ISFS across three groups: … and patients with MCI" |
| `05_discussion.md:7` | "Patients with MCI were indistinguishable from healthy older adults on every ISFS measure" |
| `05_discussion.md:13` | "The MCI and elderly groups were closely matched in age…" |
| `05_discussion.md:13` | "we found no evidence that MCI contributes anything detectable beyond age" |
| `05_discussion.md:15` | "our analogous strength measure did not separate MCI from healthy older adults" |
| `05_discussion.md:17` | "the older and MCI cohorts were clinically and etiologically heterogeneous… any subtle MCI-specific effect" |
| `05_discussion.md:17` | "the subset of MCI patients who will progress" (our design) |
| `05_discussion.md:19` | "following older and MCI participants over time" (our future work) |
| `figure_manifest.md` | captions for Figures 1, 3, 4, 5, S2 |

### Leave as plain MCI

| File | Sentence | Why |
|---|---|---|
| `01_abstract.md` | "whether mild cognitive impairment (MCI) adds a signal beyond that of age, is unknown" | the general research question |
| `02_introduction.md:13` | "In mild cognitive impairment (MCI) and Alzheimer's disease, polysomnographic studies report…" | other people's literature |
| `02_introduction.md:15` | "whether MCI adds a signal beyond that of aging, remain unknown" | general question |
| `05_discussion.md:11` | "in aging and MCI [André]" | describes André's study |
| `05_discussion.md:13` | "biomarker of early MCI"; "the transition to MCI does not measurably accelerate" | general claims about the condition |
| `05_discussion.md:15` | "later than the MCI stage"; "though not to MCI, has now been reported" | general disease-stage claims |
| `05_discussion.md:21` | "features of the ISFS in aging and MCI" | general framing |

**Two things to be aware of, not resolved here:**

- 9 of our 30 MCI subjects are **non-amnestic** (the naMCI list from the 2026-06-15 sensitivity run, minus
  the since-excluded MCI13). You decided to apply his relabel verbatim with no caveat sentence in the
  prose — worth one line in the eventual reply to him.
- `03_methods.md:17` says the Sydney patients "were recruited and diagnosed [TO SUPPLY]". Calling them
  aMCI asserts a diagnosis we have not documented for that site. The relabel is applied as you decided;
  flagging it because it is the one place the label outruns the evidence.
- `05_discussion.md:17` calls the cohort "clinically and etiologically heterogeneous", which now sits next
  to a blanket "aMCI" label. That tension belongs to the Discussion pass.

---

## §9 — `thesis/figure_manifest.md`

Beyond the three caption rewrites: **line 105** currently labels F1 as *Methods/Results overview*. Methods
no longer references Figure 1, so it becomes ***Results***.

---

## §10 — For you to do by hand in the browser

Two 30-second jobs the API should not do:

1. **Move the Figure 1 image and its caption** out of Methods 3.2 and into Results 4.1, immediately after
   the new overview paragraph (§1.3). Doing it through the API means `deleteRange` on the inline image
   plus `insertImage`, which needs a Drive-hosted URI and risks a re-encode and size change on a figure
   that is already page-tuned.
2. **Paste in Table 2** at the same time — `results/demographics_V4/table2_sleep_architecture.png` plus its
   caption from `figure_manifest.md:101`, placed after §1.5. The new prose in §1.3 and §1.5 points at it.

---

## §11 — Out of scope: noticed, not acted on

- **C432** — his `#` in §4.1 asks for a supplementary figure of example spectra from different subjects.
  Figures pass; `code/find_example_gaussians.py` already produces it.
- The Doc still holds **V10** figure assets while the manifest is at V11, and the Doc's Figure 1 caption
  still describes the panel C that became Table 2. Figures pass (C487, C502, C591, C592, email #6).
- **Dimitriades is now published** (*Sci Rep*, 18 Jun 2026, `10.1038/s41598-026-58423-z`) but the Doc's
  reference 11 still reads "bioRxiv [preprint]". Pending item B1/S2, not one of his asks — say the word
  and it folds in.
- `maris2007nonparametric` appears cited nowhere in the Doc despite the cluster-permutation Methods
  section.
- He inserted a stray empty Heading 3 paragraph after the Figure 5 caption block (his ¶196). Cosmetic.
- **`maris2007nonparametric` is in `library.bib` but cited nowhere** — Maris & Oostenveld (2007),
  *Nonparametric statistical testing of EEG- and MEG-data*, `10.1016/j.jneumeth.2007.03.024`. It is the
  canonical methods paper for cluster-based permutation testing, and the bib entry's own note says to
  cite it in the Methods for exactly this. Methods 3.6 currently credits only MNE-Python²². Adding it
  would append a **reference 30** and renumber nothing. Not one of Yuval's asks, so not done.
- **Two "MCI" mentions that arguably describe our own result but were left plain**, because they fall
  outside §8's approved list and are spelled out rather than abbreviated: the Discussion's closing
  sentence ("It did not, in contrast, distinguish patients with mild cognitive impairment from healthy
  older adults of the same age") and the **thesis title**. He has not seen the current title and did not
  comment on it, so it is left for that conversation.

---

## §12 — What was applied, and the verification that followed

Applied 2026-08-15 in this order: the Doc section by section; then `thesis/chapters/04_results.md`
(rewritten with a `v3` version note) and the relabel in `01_abstract.md`, `02_introduction.md`,
`03_methods.md`, `05_discussion.md`; then `thesis/figure_manifest.md` (three caption rewrites, the F1 /
F4 / Table 2 relabels, and the F1 heading retargeted from *Methods/Results overview* to *Results*).

Verification, all passing:

- Doc re-read in full and compared against the chapter — §4.1–4.6 match.
- Doc contains no `#REF`, no `~X`, no `0.015-0.03`, no "scalars", no "no post-hoc tests".
- Reference list still **29 entries in the same order** — nothing renumbered.
- Table 1's header cell (row 0, col 3) confirmed via `getTableStructure` to read **aMCI**.
- Every plain "MCI" left in the Doc was checked against §8's leave-list; all are the general-condition
  or other-people's-studies cases, except the two flagged in §11.
- Every number re-checked against `demographics_V3/`, `demographics_V4/` and `three_groups_V11/`.
- `[TO SUPPLY]` markers in the chapters: **3, all pre-existing from the Methods pass** —
  `03_methods.md:15` (young-cohort recruitment; Sydney healthy-older recruitment) and
  `03_methods.md:17` (Sydney aMCI recruitment, diagnostic criteria, diagnosing clinician).

Still outstanding, for you to do by hand in the browser: **§10** — move the Figure 1 image and caption
from Methods 3.2 into Results 4.1, and paste Table 2 in after the sleep-continuity paragraph.

---

## §13 — Second round: §13.1–§13.3 APPLIED, §13.4 still open

You approved three more changes on 2026-08-15 after the pass above. The `google-docs` MCP dropped before
they could be applied; it reconnected and **§13.1, §13.2 and §13.3 are now in the Doc and mirrored.**
**§13.4 is still waiting on your decision.**

Verification of this round:

- Renumbering applied **descending** (29→30 first, 23→24 last), so no two markers ever collided; each
  replacement was anchored on surrounding words, and each reported exactly one match.
- The Maris entry sits between the Gramfort (MNE-Python) and Vallat (YASA) entries — confirmed by index.
  The reference list is an auto-numbered Docs list, so it renders as **23** and the rest follow on.
- The Maris entry's DOI is plain text, not a live hyperlink, unlike the neighbouring entries. Cosmetic;
  fix by hand if it matters.

**Already mirrored in the repo:** `03_methods.md:59` (Maris citation) and `05_discussion.md:21`
(the closing sentence). Nothing else in the repo is affected — the title exists only in the Doc, since
`00_front_matter.md` still carries a `_TBD with Prof. Nir_` placeholder.

### 13.1 — Cite `maris2007nonparametric`, and renumber

Maris E, Oostenveld R. *Nonparametric statistical testing of EEG- and MEG-data.* Journal of Neuroscience
Methods. 2007;164(1):177–190. https://doi.org/10.1016/j.jneumeth.2007.03.024

⚠ **This is not a free append.** The reference list is numbered by order of first appearance, and Maris
belongs in Methods 3.6 — ahead of YASA, Ohayon and the whole Discussion block. It becomes **23**, and
**every reference from 23 to 29 shifts up by one.** (An earlier note in this file said it would append as
30 and renumber nothing; that was wrong.)

**Apply in this order, so no two markers ever collide:**

1. Renumber the existing markers **descending**: `²⁹`→`³⁰` (Grollero), `²⁸`→`²⁹` (Niethard, inside
   `¹²,²⁸`), `²⁷`→`²⁸` (André), `²⁶`→`²⁷` (Schmitz), `²⁵`→`²⁶` (Chen), `²⁴`→`²⁵` (Ohayon, Results 4.1),
   `²³`→`²⁴` (YASA, Methods 3.7). Anchor each on surrounding words, not the bare glyph.
2. Then the Methods 3.6 sentence:
   **BEFORE** `Spatial group differences were tested with cluster-based permutation tests (MNE-Python²²).`
   **AFTER** `Spatial group differences were tested with cluster-based permutation tests (MNE-Python²², following Maris and Oostenveld²³).`
3. Then insert the Maris entry into the reference list **immediately after the Gramfort (MNE-Python)
   entry**, so the list order matches the new numbering.

*Reason:* the canonical methods citation for cluster-based permutation testing was missing; Methods 3.6
credited only the software. Not a Yuval item — your call, taken 2026-08-15.

### 13.2 — aMCI in the title

**BEFORE**
> Aging, but not mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle power

**AFTER**
> Aging, but not amnestic mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle power

*Reason:* the title states our finding, so it takes the cohort label. Note he has never seen this title
(it postdates the copy he reviewed) and did not comment on it, so it is still an open conversation.

### 13.3 — aMCI in the Discussion's closing sentence

**BEFORE**
> It did not, in contrast, distinguish patients with mild cognitive impairment from healthy older adults
> of the same age.

**AFTER**
> It did not, in contrast, distinguish patients with amnestic mild cognitive impairment from healthy
> older adults of the same age.

*Reason:* our result, so the same rule applies. Both were flagged in §11 as deliberately left plain; you
overturned that.

### 13.4 — APPLIED: "aMCI" was being used before it was defined

The relabel has left the abbreviation undefined at first use. `aMCI` first appears in the **Abstract**
("30 patients with aMCI") and again in the **Introduction** ("three groups … and patients with aMCI"),
but it is only spelled out in **Methods 3.1**. The Abstract meanwhile still defines the *other*
abbreviation, `mild cognitive impairment (MCI)`, in the preceding sentence — correctly, because that
sentence states the general research question.

**Applied, one definition per context, no nested parentheses:**

- **Abstract** (standalone, so it carries its own definition): `30 patients with aMCI` →
  `30 patients with amnestic MCI (aMCI)`. The later `Patients with aMCI were indistinguishable…` in the
  same paragraph then uses a defined abbreviation.
- **Introduction**: both mentions spelled out and left **unabbreviated** — `and patients with amnestic
  MCI)` and `whether the ISFS of patients with amnestic MCI is comparable to…`. Defining it inside the
  first one would have nested a parenthesis inside a parenthesis, and the comma workaround
  (`…amnestic MCI, aMCI)`) reads as a fourth item in the list.
- **Methods 3.1** therefore carries the single body definition it already had: `patients with amnestic
  mild cognitive impairment (aMCI; n = 30; …)`. Results and Discussion use `aMCI` freely from there.

The general "whether mild cognitive impairment (MCI) adds a signal beyond that of age" sentence in the
Abstract, and the equivalent in the Introduction, are untouched — so the reader meets MCI first, then
aMCI, each defined once per context, in order.

*On the "why twice" question:* abstracts are read and indexed standalone, so an abbreviation defined
there is conventionally re-defined at first use in the body. This thesis already does exactly that for
`non-rapid eye movement (NREM)`, `infra-slow fluctuation of sigma power (ISFS)` and `mild cognitive
impairment (MCI)`.
