# Defense Q&A preparation

Working file for the question-preparation sessions. Speaker notes per slide are in
`speaker_notes.md`; this file is about the examiners and the questions to expect.

## The examiners

**Prof. Riva (Rivi) Tauman** — Sieratzki-Sagol Institute for Sleep Medicine, TASMC; Department of
Pediatric Pulmonology, Critical Care and Sleep Medicine. Clinical sleep medicine and polysomnography.
Her published work is largely obstructive sleep apnea and sleep-disordered breathing, much of it
paediatric (adenotonsillectomy outcomes, obesity, OSA and metabolic measures), plus recent adult work:
she led the sleep-breathing assessment for **Sharon et al. 2025**, the slow-wave synchrony and prodromal
Alzheimer's paper on the cohort that overlaps ours, and she is on a 2026 validation of AI algorithms for
OSA diagnosis and sleep staging from oximetry at TASMC. She also guided the PSG monitoring for this
thesis.

Likely to probe: the apnea screening (we kept AHI ≤ 15, so mild OSA is present in the sample and is not
covariate-adjusted); scoring reliability and who scored; whether ISFS could be tracking respiratory
events or arousals rather than the LC; clinical usefulness of the measure; why slow-wave synchrony
separated impairment in her own paper while the ISFS did not.

**Dr. Michal Kahn** — School of Psychological Sciences, TAU (PhD under Avi Sadeh); Kahn Sleep Lab, and
adjunct at Flinders. Developmental and clinical sleep: pediatric insomnia, infant sleep and parental
tolerance for crying, sleep and emotion, evening technology use. Multi-method by design: actigraphy,
auto-videosomnography, PSG, plus observational and report measures. Strong on behavioral intervention
trials and on measurement.

Likely to probe: measurement validity and reliability (is one night enough? night-to-night stability of
the ISFS?); what a null means with these sample sizes (power, effect sizes, equivalence rather than
absence); the developmental angle, since the reference framework is a developmental study
(Dimitriades 2024) and the rhythm is measurable in early childhood; whether the measure relates to
anything behavioural (memory, MoCA, daytime function); the heterogeneity of the aMCI group; the choice
of a cross-sectional design.

**Prof. Yuval Nir** — supervisor. Mechanism, LC-NE, and everything already in the review record.

## Priority items to prepare

### 1. Champetier et al. 2023 already measured the ISFS in aging (highest priority)

`thesis/references/Age_changes_spindle_memory_consolidation_Champetier_2023.pdf`, Fig 2 and its
results text:

- 32 young-middle aged adults (20 to 52 years, 34.5 ± 10.9) versus 147 cognitively unimpaired older
  adults (65 to 83, 69.3 ± 4.1).
- Fast-spindle band power at **C3 and C4 only**, FFT, Gaussian fit, exactly our parameterization of the
  peak.
- Grand-average peaks: **0.021 Hz young-middle aged, 0.022 Hz older**. Individual peaks: **no group
  difference, F(1,175) = 0.164, p = .69** (two-way ANOVA controlled for sex).
- They also report that the proportion of clustered fast spindles falls with age, and that a slower
  infra-slow oscillation goes with larger spindle clusters.

Why this matters: the thesis states that the ISFS "has not been characterized in older adults or in
aMCI". For peak frequency at two central electrodes, that is too strong, and our positive peak-frequency
result (0.0199 vs 0.0226/0.0232 Hz, p = 0.0026) sits against their null.

Answer to give:
1. **Their means move the same way ours do** (0.021 → 0.022 Hz). The difference is significance, not
   direction.
2. **Their age contrast is compressed.** Their younger group averages 34.5 years and extends to 52; ours
   averages 27.1 (range 21 to 39). Purcell's lifespan curves put most of the spindle change after about
   40, so a 34-year-old reference group absorbs part of the effect.
3. **Two electrodes versus 176.** C3/C4 cannot show a topography, and the effect we report for strength
   is focal; for frequency our effect is diffuse, which is why a whole-scalp average detects it.
4. **What was genuinely open**: the whole-scalp parameters, the topography, a young reference group
   recorded on the same system, and aMCI, on which there were no data at all.

Background slide 7 now shows their Figure 2 and says this out loud, which is the safest way to handle
it: raise it before an examiner does.

### 2. Sleep-disordered breathing (expect this from Tauman)

Every older participant has AHI ≤ 15, so mild OSA is in the sample. We did not enter AHI as a covariate,
and AHI is not available for every recording (a binary clinical judgment was made where the recording
did not permit an index). Micro-arousals are part of the phenomenon the ISFS indexes, so respiratory
arousals are a plausible confound in principle. What we can say: the artifact rejection removes
arousal-contaminated epochs from the analysed N2, the share of N2 surviving cleaning is identical across
groups (51.9 / 52.3 / 51.5 %), and elderly and aMCI do not differ on any ISFS measure although they
differ in WASO. An AHI-covariate analysis in the two older groups is a run we could do if asked.

### 3. Why the ISFS did not track impairment when slow-wave synchrony did

Sharon et al. 2025 (Tauman is a co-author) found slow-wave synchrony tracks cognitive impairment in
prodromal Alzheimer's disease in an overlapping cohort. Two measures of the same nights behaving
differently is a result, not an embarrassment: it says the two index different things. Cyclic
alternating pattern is reduced in MCI and predicts incident dementia, so the infra-slow domain does
carry cognitive information that this particular parameterization does not capture.

### 4. Standard set (already in the slide notes)

- Bandwidth tracks analyzed N2 duration (r = 0.386), so it is not read as an age effect.
- The ROI null (p = 0.143) versus the cluster result: 9 electrodes inside a 36-electrode region.
- MoCA available for 44 of 69 older participants.
- Two sites, and Sydney recruitment is not documented at the same level of detail.
- One night per participant, cross-sectional, no test-retest estimate for the ISFS parameters.
- Detection rate is lowest in young adults (74.5 %) yet their hotspot is strongest, so the group effects
  are not an artefact of fit success.
