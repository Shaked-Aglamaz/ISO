# Paper outline — ISFS in NREM2 sleep across aging and MCI

> Paper-style (~6000 words, IMRAD, Intro ~1.5 pages). Conclusion folded into Discussion; lean Supplementary instead of Appendix. **Locked figure set (2026-06-05):** Table 1 + Figures 1–5 + Supp S1–S2 (see `figure_manifest.md`). Statistics are the **V7 / N=35** values.

---

## Scientific story (four-sentence lock)

1. **Gap** — Infra-slow fluctuations of sigma power (ISFS) during NREM2 sleep were characterized in young adults (Dimitriades 2024) but had not been studied in aging or mild cognitive impairment (MCI) at the time of this work.
2. **Approach** — High-density (256-ch) EEG ISFS analysis in three groups — young controls (n=35), healthy elderly (n=38), MCI (n=31) — extracting per-channel peak frequency, bandwidth, and AUC of the sigma-envelope spectrum, compared whole-scalp (violins), topographically (cluster-based permutation), and within a pre-defined central-parietal ROI.
3. **Finding** — With age the rhythm becomes **faster** (peak frequency Young < Elderly = MCI, p = 0.003; bandwidth trends broader but is not significant, p = 0.052) and its strength is **preserved globally yet reduced focally** over central-parietal cortex (one significant cluster, p = 0.016, 9 electrodes; the young-adult hotspot flattens). **MCI is statistically indistinguishable from healthy elderly on every measure, and no ISFS metric correlates with MoCA.**
4. **Why it matters** — ISFS is a sensitive marker of how healthy aging reshapes the thalamocortical/spindle infrastructure of NREM2 sleep, but in this cohort it does not add an MCI-specific signal — tempering, rather than supporting, its use as a standalone early-MCI biomarker.

_Biomarker framing note: keep honest. The headline is an **aging** effect; the MCI=Elderly and null-MoCA results are themselves findings and must not be buried._

---

## Locked figures & tables (map of every display item)

| # | Title | Section | Source |
|---|-------|---------|--------|
| **T1** | Participant demographics + data quality | Methods (Participants) | `demographics_V1/demographics_table.png` (N=35) |
| **F1** | Sleep overview: (a) hypnospectrograms, (b) sleep-stage pies, (c) sleep/N2 comparison | Methods/Results | `figures/hypno_sleep_stages_N35.png` |
| **F2** | ISFS concept + feature extraction + ROI definition, panels (a)–(d) | Methods | `figures/methods_flow_roi_v3.png` |
| **F3** | Whole-scalp ISFS parameter comparison (violins) | Results | `three_groups_V7/group_comparison_violin.png` |
| **F4** | AUC topographies + cluster-permutation result (ROI = green dots) | Results | `three_groups_V7/three_group_topo_auc.png` (+ raw, per-group topos) |
| **F5** | ROI AUC comparison (normalized violins) | Results | `three_groups_V7/group_comparison_violin_extended_ROI_normalized_auc.png` |
| **S1** | Peak-frequency & bandwidth topographies | Supplementary | `three_groups_V7/three_group_topo_{peak_frequency,bandwidth}.png` |
| **S2** | ISFS metrics × MoCA correlation grid | Supplementary | `moca_correlation_V2/moca_correlations_grid.png` |
| T2 | Whole-scalp ISFS stats (omnibus + post-hoc) | Results/Supp | `three_groups_V7/three_group_statistics.txt` → table |
| T3 | ROI normalized-AUC stats | Results/Supp | `three_groups_V7/three_group_statistics_extended_ROI_normalized_auc.txt` → table |

> **ROI naming rule:** to the reader it is only "the ROI" / "the central-parietal ROI" — never "extended" (internal label only).

---

## 1. Abstract
~250 words, flowing prose (no labeled sections). Write last. Lead with the aging effect (faster + focal central-parietal AUC loss), state MCI=Elderly and null MoCA, close on the tempered-biomarker implication.

## 2. Introduction (funnel, ~1.5 pages)
- 2.1 NREM2 sleep, sigma activity, and spindles — why NREM2 matters. _Claim: spindles/sigma are central to NREM2 function._
- 2.2 Infra-slow organization of sleep (human + rodent); LC/noradrenergic pacing of fragile/stable substates (Lecci 2017, Lázár 2019, Osorio-Forero 2021). _Claim: an ~0.02 Hz rhythm organizes spindle expression._ → concept introduced in **F2a**.
- 2.3 ISFS — definition and prior characterization in young adults (Dimitriades 2024). _Claim: ISFS is a quantifiable feature with peak-frequency/bandwidth/AUC parameters._ → **F2b–c**.
- 2.4 Aging and MCI sleep-EEG changes; spindle/sigma decline (Niethard 2023, Champetier 2023, Chen 2025); MCI biomarker context (Zhang 2022, André 2025, Schmitz 2018). _Claim: aging degrades spindle infrastructure; MCI sleep-EEG markers are sought._
- 2.5 Knowledge gap + aims + hypotheses. _Claim: ISFS in aging/MCI is unstudied; we test whether ISFS parameters and their topography differ across the three groups._ Cite **Grollero 2026 as concurrent/convergent work, not motivation.**

## 3. Methods
- 3.1 Participants — three cohorts (Tel-Aviv + Sydney), inclusion/exclusion. → **T1**.
- 3.2 Sleep recording — EGI 256-ch, scoring; sleep architecture comparable, N2 plentiful in all groups. → **F1a–c**.
- 3.3 Preprocessing — resample 250 Hz, notch, 0.1–40 Hz bandpass, average reference, automated bad-channel/epoch cleaning.
- 3.4 N2 bout extraction — clean N2 bouts ≥ 300 s (≥ 3–5 ISFS cycles). → **F1c** (bout properties).
- 3.5 ISFS feature extraction — sigma power (13–16 Hz, Gabor-Morlet) → amplitude envelope → FFT → Gaussian fit yielding peak frequency, bandwidth, AUC. → **F2a–c**.
- 3.6 Pre-defined central-parietal ROI — electrodes from the Dimitriades 2024 young-adult AUC hotspot, mapped to the 256-ch montage. → **F2d**.
- 3.7 Statistics — per-subject normalization (over all channels, then restrict to ROI), whole-scalp omnibus + post-hoc, cluster-based spatiotemporal permutation tests; displayed group means are mean-of-subject-means.
- 3.8 Software & reproducibility — MNE-Python, yasa; outputs versioned (`three_groups_V7`).

## 4. Results
- 4.1 Cohort & sleep architecture — groups differ in age, comparable data quality; aging reduces deep/REM but N2 remains plentiful (so group ISFS differences are not an N2-quantity artifact). → **T1**, **F1**.
- 4.2 Temporal parameters shift with age — peak frequency faster (Young < Elderly = MCI, ANOVA p = 0.003; post-hoc Y vs E p = 0.018, Y vs MCI p = 0.005); bandwidth same direction, not significant (KW p = 0.052); whole-scalp AUC unchanged (p = 0.53); Elderly = MCI throughout. → **F3**, **T2**.
- 4.3 Spatial AUC loss is focal — young adults concentrate AUC over central-parietal cortex; a single significant cluster (p = 0.016, 9 electrodes) shows reduced AUC in Elderly and MCI vs Young (Y vs E at 7/9, Y vs MCI at 6/9 electrodes; Elderly = MCI). → **F4**.
- 4.4 ROI averaging dilutes the focal effect — normalized ROI AUC follows the expected order (Young 1.10 > Elderly 1.05 > MCI 1.02) but the three-group omnibus is not significant (ANOVA p = 0.194). → **F5**, **T3**.
- 4.5 Peak-frequency/bandwidth topographies are diffuse — frontal emphasis of peak frequency flattens with age; no significant spatial cluster for either metric. → **S1**.
- 4.6 No cognitive correlation — within pooled Elderly+MCI (n = 43), no ISFS scalar tracks MoCA (all |r| ≤ 0.12, p ≥ 0.45). → **S2**.

## 5. Discussion (Conclusion folded in)
- 5.1 Summary — aging makes the infra-slow sigma rhythm faster and focally weaker (central-parietal), without abolishing it; MCI mirrors healthy aging.
- 5.2 Interpretation — faster/broader rhythm + flattened hotspot as reduced precision of the thalamocortical infra-slow clock and spindle-generator topography.
- 5.3 MCI = Elderly and null MoCA — ISFS indexes age, not (in this sample) cognitive status; implications for biomarker hopes (temper).
- 5.4 Convergence with Grollero 2026 — concurrent, postdates this analysis.
- 5.5 Mechanisms — LC/noradrenergic pacing, cholinergic decline, spindle-generator aging (Osorio-Forero 2021, Schmitz 2018, André 2025).
- 5.6 Limitations — modest/heterogeneous samples (two sites), borderline effects (bandwidth p = 0.052, ROI omnibus ns), cross-sectional, MoCA available for a subset.
- 5.7 Future directions + brief conclusion.

## 6. References
Pandoc-generated from `references/library.bib`.

## 7. Supplementary
- S1, S2 figures (above).
- T2, T3 stats tables (if not in main text).
- Per-subject summary; exclusion list with reasons.

---

## Title candidates (for Yuval)
1. Infra-slow fluctuations of sigma power in NREM2 sleep are altered by aging but not further by mild cognitive impairment
2. Aging speeds and spatially flattens the infra-slow sigma rhythm of NREM2 sleep
3. Topography of infra-slow sigma-power fluctuations across healthy aging and mild cognitive impairment
4. The NREM2 infra-slow sigma rhythm as a marker of healthy aging: a three-group high-density EEG study
5. Central-parietal weakening and temporal acceleration of infra-slow sigma fluctuations in aging sleep
