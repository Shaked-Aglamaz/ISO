# Speaker notes: ISFS_defense_V2.pptx

Generated from the deck by `code/defense/dump_notes.py`. 54 slides.

Edit the notes in `code/defense/build_deck.py` and rebuild, not here.

## 1. Aging, but not amnestic mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle power

Placeholder date on the title slide: replace with the real defense date. Open by naming the object of the talk: a rhythm inside N2 sleep, about one cycle every 50 seconds, and the question of what aging and amnestic MCI do to it.

## 2. What this talk covers

One sentence per box, then move on. Do not read the boxes out. Flag that the aMCI null is a result, not a missing result: it is what makes the aging interpretation clean.

## 3. Background

_(no notes)_

## 4. Sleep spindles

Start here rather than at what N2 is: everyone in the room scores sleep. Spindles come from thalamic reticular and thalamocortical interaction, they define N2, and they are tied to plasticity, overnight consolidation and sensory disconnection. This talk is about the fast subtype, because that is where the infra-slow rhythm is most pronounced. Purcell is the largest spindle dataset there is, 11,630 individuals aged 4 to 97; the topography panel comes from its paediatric subsample with 18 electrodes, and the slow-frontal / fast-central pattern is the standard adult one too (Molle 2011). Right panel: density falls and the fast peak flattens with each decade past about 40.

## 5. Spindles appear in a rhythm

Talk over the top figure instead of reading it: each bar is one detected spindle in one channel, the green window is opened out below, and the three traces are the raw channel, the 13 to 16 Hz sigma, and the sigma amplitude envelope. What fluctuates is that envelope. The Lazar panel is the human characterization: the infra-slow modulation is confined to the fast sigma band and to central-parietal and occipital electrodes, exactly where fast spindles are largest. On the naming, say once that the same phenomenon is published as an infra-slow fluctuation of sigma power (ISFS) and as an infra-slow oscillation (ISO). The rate depends on what is measured: sigma power puts it near 0.01 Hz, spindle events near 0.02 Hz (Lazar 2019; Lecci 2017).

## 6. Two substates, paced by the locus coeruleus

Top: our schematic of the published account, not our data. Bottom: the same cycle as it is described from rodent work, with LC activity, spindle density, micro-arousals and autonomic arousal aligned on it. In rodents the LC sets the phase; thalamic noradrenaline suppresses spindle generation, so LC peaks mark micro-arousals and LC troughs the spindle trains (Lecci 2017; Osorio-Forero 2021, 2025). Acetylcholine, serotonin and dopamine run on the same cycle, and silencing the LC abolishes the noradrenergic oscillation (Kjaerby 2022, 2026). Lecci also showed the phase gates arousability: mice wake to a tone on the declining sigma phase and sleep through it on the rising phase. The bottom panel is from Yuval's own unpublished review, so clear it with him; the published alternative already in the assets folder is the mouse and human spectra from Lecci 2017.

## 7. What aging does to spindles, and to the rhythm

This slide positions the thesis, so be precise. Aging: spindle density falls, slow-wave to spindle coupling loosens, and the temporal clustering of spindles breaks down (Mander 2017; Helfrich 2018; Champetier 2023). Champetier came closest to our question: 32 young-middle aged adults, 20 to 52 years, against 147 cognitively unimpaired older adults, fast-spindle power at C3 and C4 only, FFT and a Gaussian fit, peak 0.021 versus 0.022 Hz, no group difference (F(1,175) = 0.164, p = .69). They also showed the proportion of clustered fast spindles falls with age. So what was open was not whether the rhythm survives into old age, but its whole-scalp parameters and its topography, against a genuinely young reference group, and whether aMCI adds anything. Expect to be asked why our peak-frequency result is positive where theirs is null, and answer with the age contrast (their younger group averages 34.5 years and reaches 52, ours averages 27.1), with two central electrodes versus 176, and with the direction: their means differ the same way ours do.

## 8. Amnestic MCI

Petersen 1999 and Albert 2011 for the criteria: the syndrome is diagnosed first and its cause judged separately, so a group defined on cognitive grounds is aetiologically mixed. Most people with MCI never progress (Mitchell 2009) and about a quarter of amnestic cases revert to normal cognition (Malek-Ahmadi 2016). Candidate NREM markers: Taillard 2019, Liu 2020, and the parietal fast-spindle deficit of Gorgoni 2016. How to read a null here belongs in the discussion, so do not pre-empt it.

## 9. Open questions

Hypotheses as they were stated: an age effect on the parameters and on the topography, and, if the ISFS indexes early neurodegeneration, a further change in aMCI. What was missing: whole-scalp parameters and topography in older adults, a young reference group recorded the same way, and any data at all in aMCI. Grollero et al. (2026) measured the same features in diagnosed Alzheimer's disease concurrently with this work: convergent, not the motivation, and it returns in the discussion.

## 10. Methods

_(no notes)_

## 11. Cohort: 104 participants, two sites

aMCI patients in Tel Aviv came through the Memory and Attention Disorders Center and were diagnosed clinically at the Cognitive Neurology Unit; patients already diagnosed with Alzheimer's disease were not included. Moderate or severe apnea was excluded, so every older participant has an AHI of 15 or below. MoCA was collected in Tel Aviv only, which is why the MoCA analysis has n = 44 rather than 69. Age: Welch t = -0.56, p = 0.57; MoCA: Mann-Whitney U = 375.5, p < 0.001.

## 12. Recording and scoring

Same protocol at both sites, which is what makes pooling defensible. The excluded electrodes are the ones not over the scalp and most prone to muscle and movement artifact.

## 13. Preprocessing and artifact rejection

Technical exclusions applied on top of the diagnostic criteria: more than 20 % of electrodes bad, more than 30 % of N2 artifactual, fewer than three clean bouts, or total sleep time under 210 min. The 210 min floor means each analyzed night holds more than two full NREM-REM cycles, so N2 is sampled across the night and not only from its first hours. Backup slide B3 has the exclusion counts.

## 14. Clean N2 bouts: the unit of analysis

Splitting around artifacts rather than rejecting whole N2 periods is what preserves usable data. The three-bout minimum is an inclusion criterion, and it excluded nine recordings across the three groups.

## 15. From the EEG to the sigma envelope

One example participant and channel. The envelope comes from a Gabor-Morlet wavelet transform, 13 to 16 Hz in 0.2 Hz steps, 4 cycles wide, implemented as convolution in the frequency domain. Then, for each bout: subtract the mean of the envelope so its constant offset does not dominate the 0 Hz bin, Fourier transform, take the squared magnitude, put it on a frequency axis common to all bouts, and divide by its own mean power so channels and bouts of different signal strength are comparable.

## 16. Three parameters per channel

The parameterization is Dimitriades et al. (2024), adapted from their 128-channel montage to ours; they followed the ISFS across development and linked these three measures to arousal and memory reactivation. Point at the curve while naming the parameters: the peak, the width of the orange band, the area of the shaded band. Normalized bout spectra are averaged within each channel before the fit. Why the baseline matters: broadband noise puts a channel-specific offset under the peak, which would inflate any peak measured on top of it, and the 0.06 to 0.10 Hz range contains no ISFS signal, so it estimates that offset cleanly. The acceptance rule is the reference pipeline's plus the positive-area gate, which rejects degenerate fits. Leaving failures missing rather than zero matters: a zero would be a measurement, a missing value is the absence of one.

## 17. The region of interest was defined a priori

Say the ROI, or the central-parietal ROI, and nothing else: the internal variant labels are not for the audience. It is used for the AUC comparison and appears as green dots on the topography slides. Because it was fixed in advance, the ROI test is confirmatory while the cluster test is exploratory over the whole scalp.

## 18. Statistics at three levels of spatial detail

Whole-scalp AUC failed the normality check, so that one parameter is tested non-parametrically. The order of normalization matters: normalizing inside the ROI would force its mean to 1 by construction, which is why it is done over all channels first. Cluster-forming threshold at the p = 0.05 F-critical value, one-tailed.

## 19. Results

_(no notes)_

## 20. Sleep changes with age, but N2 stays plentiful

N3: ANOVA F = 10.85, p = 0.0001. REM: F = 36.70, p < 0.0001. Both are driven by young versus each older group, with elderly and aMCI indistinguishable. N2 is 34.1 %, 44.5 % and 39.0 % of recording time. This is the normative aging pattern, which is the point: it is the baseline against which any aMCI effect has to show itself.

## 21. One whole night per group

Hypnogram over the spectrogram of one electrode, one representative participant per group. Subject codes are cropped on purpose: two of the codes are misleading about group membership. Point out the fragmentation and the loss of deep sleep in the two older nights, then move on: the group statistics are on the next slide.

## 22. Sleep architecture and continuity

Percentages are of total recording time; mean ± SD. Sleep onset latency did not differ across groups (17.4, 15.6, 19.9 min; p = 0.38) and REM latency did (92.8, 125.4, 130.3 min; p = 0.005). Effect sizes: WASO eta squared 0.291, sleep efficiency 0.170. The full 14-row table with tests and post-hoc columns is backup slide B2.

## 23. Is the N2 that entered the analysis comparable?

Raise this before showing any ISFS result: it is the first thing an examiner asks. Total analyzed N2 does not differ significantly (Kruskal-Wallis p = 0.081). Bandwidth is the one parameter that does track this, and it gets its own slide. Full ANCOVA output is backup slide B4.

## 24. The ISFS is present in nearly every channel

This is what the group averages rest on: well-formed single-participant spectra, not an average that only looks clean once pooled. Each panel is one channel of one participant, with its own vertical scale, drawn with the same conventions as the methods figure. If asked what a failed fit means: no peak clearing the threshold inside the ISFS band, so no ISFS measurable at that channel, left missing rather than zero.

## 25. The rhythm runs faster in both older groups

Headline temporal result. The effect survives the N2-duration covariate (ANCOVA p = 0.0034, partial eta squared 0.108) with both pairwise contrasts surviving Holm correction. If asked how big it is: about a 15 % higher rate, or a cycle roughly 7 s shorter.

## 26. Bandwidth: no group effect, and it tracks analyzed N2

The covariate is the stronger predictor: F = 14.73, p = 0.0002, partial eta squared 0.128, against group's 0.031. So this is not interpretable as an effect of age, and the thesis does not call it a trend. Say why being explicit here is a strength: the same covariate analysis is what makes the peak-frequency result credible, because there it changed nothing.

## 27. Overall strength, averaged over the scalp, is unchanged

Deliberate setup for the topography slides. If the talk is running long, this slide and the next can be merged, but do not skip the point: the null at whole-scalp level is what makes the focal result interesting rather than just a global decline.

## 28. Where the strength sits: the young-adult hotspot

Raw per-group maps; the number above each map is the mean of the per-participant means. Worth saying out loud: the young-adult topography reproduces Dimitriades 2024 and Lazar 2019 in an independent cohort, which is a validation of the whole pipeline before any group claim is made.

## 29. One significant central-parietal cluster

Electrodes E130, E143, E144, E153, E154, E155, E184, E185, E197 (mean F = 5.15, max F = 10.62). Post-hoc: young vs elderly significant at 5 of 9, young vs aMCI at 7 of 9, elderly vs aMCI at 1 (E197). These maps are per-participant normalized, so the result is about where a participant's strength is concentrated, not about the overall level, which did not differ. Headline spatial result. Backup slide B5 has the electrode lists.

## 30. Averaging over the whole ROI dilutes the effect

Do not present this as support for the cluster result. The honest reading is that the a-priori ROI, taken from young adults, is larger than the region that actually changes with age. Values are per-participant normalized, so 1.10 means 10 % above that participant's own whole-scalp mean.

## 31. The temporal changes are diffuse, not regional

In young adults peak frequency shows a frontal emphasis that flattens in the older groups, and bandwidth has no consistent spatial pattern at all. The interpretation to offer: peak frequency changes everywhere rather than anywhere in particular, which is what a change in the pacemaker rather than in a cortical generator would look like.

## 32. No association with cognitive score

All four scalars were tested; two are shown, the other two behave the same way. Say the limitation before an examiner does: MoCA was collected in Tel Aviv only, so this analysis has n = 44 out of 69 older participants and is the least powered in the thesis.

## 33. Discussion

_(no notes)_

## 34. What changed, and what did not

Say the summary in three sentences and resist elaborating: the next four slides do the elaborating. The framing of the whole thesis is in the title: aging, not amnestic MCI.

## 35. A loss of temporal and spatial precision

The last bullet is the careful reading, and it is the one to volunteer rather than defend: safer to read the effect as a change in where the rhythm is expressed than as evidence about the spindle generators themselves. The peak-frequency map blurs in the same direction without reaching significance.

## 36. A noradrenergic account, and its limits

Lecci 2017, Osorio-Forero 2021 and 2025, Kjaerby 2022 and 2026; Braak 2011, Zarow 2003, Dahl 2019, Van Egroo 2022, Galgani 2023, Luthi 2025. The prediction our data does test: a mechanism anchored in a structure that degenerates early and in the general population, not only in patients, predicts a change complete in healthy older adults and no further advanced in aMCI. That is the pattern we found.

## 37. No evidence for an effect of amnestic MCI beyond age

Grollero 2026 found reduced peak amplitude in clinically diagnosed Alzheimer's disease, a parameter we did not measure, in a group defined by diagnosis rather than by cognitive complaint. So our null says the rhythm does not track an aMCI diagnosis; it does not say the rhythm is preserved in early Alzheimer's disease. Two measures of the same nights behaving differently is informative about what each one indexes.

## 38. The ISFS in other populations, and why comparison is hard

So the opposite directions, reduced in schizophrenia and in aging but elevated in chronic fatigue, cannot yet be read as a contradiction. If the rhythm indexes how deeply arousal is modulated across the night, deviation in either direction may be what matters, which is what an inverted-U on noradrenergic fluctuation amplitude would predict (Luthi 2025). Two of our three measures have essentially no comparison literature outside the approach they come from.

## 39. Limitations

Volunteer these rather than waiting to be asked. The Sydney recruitment and diagnostic route are not documented in the same detail as the Tel Aviv one, which is a real gap in the methods and is stated as such in the thesis.

## 40. Future directions

Keep this short. If asked what the single most informative next experiment would be: repeated overnight recordings years apart in the same older and aMCI participants, because only a within-subject design separates aging trajectories from cross-sectional differences.

## 41. Conclusion

Land on the two sentences and stop. Then the acknowledgements slide, then invite questions.

## 42. Acknowledgements

Thank you. Questions.

## 43. Backup slides

_(no notes)_

## 44. B1. Full demographics and data quality

Bad channels 6.3 %, 1.4 %, 3.4 %; bad N2 epochs 4.1 %, 3.3 %, 1.8 %. The header of this rendering still says MCI rather than aMCI; the thesis table is a native table that reads aMCI.

## 45. B2. Full sleep table, with tests and effect sizes

Dashes mark comparisons that were not run because the omnibus test was not significant. Bold marks p < 0.05.

## 46. B3. Exclusions: 26 recordings

The criteria are stated in the methods: more than 20 % of electrodes bad, more than 30 % of N2 artifactual, fewer than three clean bouts, or total sleep time under 210 min.

## 47. B4. ANCOVA with analyzed N2 duration as covariate

AUC failed the per-group normality check in the main analysis, so Kruskal-Wallis remains its primary test and the ANCOVA row is a covariate-adjusted supplement. Source: three_group_ancova_statistics.txt.

## 48. B5. The cluster, electrode by electrode

12 candidate clusters were formed, one survived. The direction is lower AUC in both older groups than in young adults. Across re-runs on slightly different cohorts the cluster stays central-parietal and keeps its direction; the electrode count moves by one or two.

## 49. B6. Bandwidth against analyzed N2 duration

A longer recording gives more bouts and a more finely resolved spectrum, which plausibly broadens the fitted peak. Whatever the mechanism, duration is the stronger predictor and it is not balanced across groups.

## 50. B7. Feature extraction, in detail

If asked what differs from the reference implementation: the positive-area gate on the fit, which rejects degenerate solutions, and the montage adaptation from 128 to 256 channels. If asked about the absolute-value on c: the fit can return a negative width parameter, which is the same Gaussian, so its magnitude is what enters the bandwidth.

## 51. B8. Detection rates, and what a failed fit means

Young adults have the lowest detection rate and the strongest hotspot, which rules out a story in which the group differences are driven by how often the fit succeeds. If pressed on why young adults fit less often: they also contributed the least analyzed N2 sleep.

## 52. B9. Was the aMCI group too heterogeneous?

Outputs are kept under results/no_naMCI. The reason for rejecting the sensitivity run was that it traded a defensible full sample for a smaller one without changing the conclusion, and the exclusion criteria were already fixed. Do not quote its p values as results.

## 53. B10. Alongside Grollero et al. (2026)

Concurrent work, not the motivation for this thesis. The two studies agree on frequency, which distinguished neither clinical contrast, and are broadly consistent on a weakening of the rhythm over central regions. They diverge on whether cognitive impairment leaves a mark beyond age, and the two strength measures are not the same quantity, so the divergence is not a direct contradiction.

## 54. B11. Where the hypnograms and annotations come from

This is the answer to any question about how sleep-architecture percentages were computed and why they do not match a naive count of scored epochs. The full note is in the repository.
