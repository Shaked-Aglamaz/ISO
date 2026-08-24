"""Build the thesis-defense deck (16:9 .pptx) from the outline in thesis/defense/deck_outline.md.

Content lives in SLIDES below, one dict per slide; layout, typography and image
fitting are handled by the renderers, so editing text or swapping a figure means
editing one entry and re-running. Every number here is taken from
thesis/chapters/04_results.md and the V10 statistics files; nothing is computed.

Written directly against python-pptx rather than through the powerpoint skill's
JSON spec because the deck needs per-cell table fonts, aspect-ratio-preserving
image fitting and captions, none of which that spec exposes.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/defense/build_deck.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn  # noqa: F401
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

OUT = Path("thesis/defense/ISFS_defense_V2.pptx")

# Shown on the title slide. Shaked typed a date into V1 by hand; put the real one here.
DEFENSE_DATE = "[defense date]"

SLIDE_W, SLIDE_H = 13.333, 7.5
FONT = "Segoe UI"

INK = RGBColor.from_string("1A1A1A")
MUTED = RGBColor.from_string("5A5A5A")
NAVY = RGBColor.from_string("1F3A5F")
WHITE = RGBColor.from_string("FFFFFF")
RULE = RGBColor.from_string("D8DEE6")
BAND = RGBColor.from_string("F2F5F8")

YOUNG = RGBColor.from_string("2F9C86")
ELDERLY = RGBColor.from_string("D0402C")
AMCI = RGBColor.from_string("2F7FAE")

# table header cells that name a group are filled with that group's colour
GROUP_HEAD = {"Young": YOUNG, "Elderly": ELDERLY, "aMCI": AMCI}

FOOTER = "Shaked Aglamaz  ·  ISFS in aging and aMCI"

A = Path("results/defense_slides_V1")            # assets built for this deck
R = A / "refs"                                   # panels cropped from published figures
G11 = Path("results/group_comparison_results/three_groups_V11")
FIG = Path("thesis/figures")

# --------------------------------------------------------------------------- content

SLIDES = [
    # ---------------------------------------------------------------- 1 title
    dict(layout="title",
         title="Aging, but not amnestic mild cognitive impairment, "
               "reshapes the infra-slow rhythm of sleep spindle power",
         subtitle=["Shaked Aglamaz",
                   "Supervised by Prof. Yuval Nir",
                   "Sagol School of Neuroscience, Tel Aviv University",
                   f"M.Sc. thesis defense  ·  {DEFENSE_DATE}"],
         notes="Placeholder date on the title slide: replace with the real defense date. "
               "Open by naming the object of the talk: a rhythm inside N2 sleep, about one "
               "cycle every 50 seconds, and the question of what aging and amnestic MCI do to it."),

    # ---------------------------------------------------------------- 2 roadmap
    dict(layout="cards", title="What this talk covers",
         cards=[("The rhythm", "Spindles come in trains about every 50 s: an infra-slow "
                               "fluctuation of sigma power (ISFS)", NAVY),
                ("What we measured", "256-channel EEG, full nights, 104 participants: "
                                     "peak frequency, bandwidth, strength (AUC)", NAVY),
                ("What changed with age", "Faster rhythm, and the central-parietal "
                                          "hotspot of strength loses its focus", ELDERLY),
                ("What aMCI added", "Nothing detectable beyond age, and nothing that "
                                    "tracks cognitive score", AMCI)],
         notes="One sentence per box, then move on. Do not read the boxes out. "
               "Flag that the aMCI null is a result, not a missing result: it is what makes "
               "the aging interpretation clean."),

    # ---------------------------------------------------------------- 3 divider
    dict(layout="section", title="Background",
         subtitle="N2 sleep, spindles, and the infra-slow rhythm that organizes them"),

    # ---------------------------------------------------------------- 4 spindles
    dict(layout="figure", title="Sleep spindles",
         image=R / "purcell_spindle_topo.png",
         labels=["thalamo-cortical bursts, 11 to 16 Hz",
                 "slow: frontal   ·   fast: central-parietal",
                 "tied to consolidation and to sensory disconnection"],
         source="Purcell et al. 2017, n = 11,630",
         notes="Start here rather than at what N2 is: everyone in the room scores sleep. "
               "Spindles come from thalamic reticular and thalamocortical interaction, they "
               "define N2, and they are tied to plasticity, overnight consolidation and "
               "sensory disconnection. This talk is about the fast subtype, because that is "
               "where the infra-slow rhythm is most pronounced. Purcell is the largest "
               "spindle dataset there is, 11,630 individuals aged 4 to 97; the topography "
               "panel comes from its paediatric subsample with 18 electrodes, and the "
               "slow-frontal / fast-central pattern is the standard adult one too (Molle "
               "2011). Right panel: density falls and the fast peak flattens with each "
               "decade past about 40."),

    # ---------------------------------------------------------------- 5 the rhythm
    dict(layout="figure_plus", title="Spindles appear in a rhythm",
         image=A / "methods_panel_AB.png", image_h=3.6,
         small_image=Path("code/debug/YA_AUC.png"),
         small_caption="ISFS strength in young adults (Dimitriades et al. 2024)",
         labels=["one bar = one detected spindle",
                 "trains recur every ~50 s   (~0.02 Hz)",
                 "fast sigma, central-parietal",
                 "ISFS = ISO"],
         notes="Talk over the top figure instead of reading it: each bar is one detected "
               "spindle in one channel, the green window is opened out below, and the three "
               "traces are the raw channel, the 13 to 16 Hz sigma, and the sigma amplitude "
               "envelope. What fluctuates is that envelope. The Lazar panel is the human "
               "characterization: the infra-slow modulation is confined to the fast sigma "
               "band and to central-parietal and occipital electrodes, exactly where fast "
               "spindles are largest. On the naming, say once that the same phenomenon is "
               "published as an infra-slow fluctuation of sigma power (ISFS) and as an "
               "infra-slow oscillation (ISO). The rate depends on what is measured: sigma "
               "power puts it near 0.01 Hz, spindle events near 0.02 Hz (Lazar 2019; Lecci "
               "2017)."),

    # ---------------------------------------------------------------- 6 substates + LC
    dict(layout="two_figures", title="Two substates, paced by the locus coeruleus",
         images=[(A / "lc_ne_schematic.png", "our schematic of the published account"),
                 (R / "nir_lc_infraslow.png",
                  "the same cycle from rodent work (Nir et al., review)")],
         labels=["offline: spindle trains, sensory gating",
                 "fragile: micro-arousals",
                 "noradrenaline in anti-phase with sigma"],
         notes="Top: our schematic of the published account, not our data. Bottom: the same "
               "cycle as it is described from rodent work, with LC activity, spindle "
               "density, micro-arousals and autonomic arousal aligned on it. In rodents the "
               "LC sets the phase; thalamic noradrenaline suppresses spindle generation, so "
               "LC peaks mark micro-arousals and LC troughs the spindle trains (Lecci 2017; "
               "Osorio-Forero 2021, 2025). Acetylcholine, serotonin and dopamine run on the "
               "same cycle, and silencing the LC abolishes the noradrenergic oscillation "
               "(Kjaerby 2022, 2026). Lecci also showed the phase gates arousability: mice "
               "wake to a tone on the declining sigma phase and sleep through it on the "
               "rising phase. The bottom panel is from Yuval's own unpublished review, so "
               "clear it with him; the published alternative already in the assets folder "
               "is the mouse and human spectra from Lecci 2017."),

    # ---------------------------------------------------------------- 7 aging
    dict(layout="two_figures", title="What aging does to spindles, and to the rhythm",
         images=[(R / "purcell_spindle_age.png",
                  "fewer spindles with age (Purcell et al. 2017)"),
                 (R / "champetier_iso_age.png",
                  "the rhythm is present in older adults; at C3/C4 its peak frequency does "
                  "not differ (Champetier et al. 2023)")],
         labels=["spindles: fewer, less clustered",
                 "parietal fast-spindle deficit in aMCI and AD (Gorgoni 2016)",
                 "the rhythm: two electrodes, one null, no topography"],
         notes="This slide positions the thesis, so be precise. Aging: spindle density "
               "falls, slow-wave to spindle coupling loosens, and the temporal clustering "
               "of spindles breaks down (Mander 2017; Helfrich 2018; Champetier 2023). "
               "Champetier came closest to our question: 32 young-middle aged adults, 20 to "
               "52 years, against 147 cognitively unimpaired older adults, fast-spindle "
               "power at C3 and C4 only, FFT and a Gaussian fit, peak 0.021 versus 0.022 "
               "Hz, no group difference (F(1,175) = 0.164, p = .69). They also showed the "
               "proportion of clustered fast spindles falls with age. So what was open was "
               "not whether the rhythm survives into old age, but its whole-scalp "
               "parameters and its topography, against a genuinely young reference group, "
               "and whether aMCI adds anything. Expect to be asked why our peak-frequency "
               "result is positive where theirs is null, and answer with the age contrast "
               "(their younger group averages 34.5 years and reaches 52, ours averages "
               "27.1), with two central electrodes versus 176, and with the direction: "
               "their means differ the same way ours do."),

    # ---------------------------------------------------------------- 8 aMCI
    dict(layout="text", title="Amnestic MCI",
         bullets=["Cognitive complaint plus objective impairment, daily function preserved",
                  "Amnestic subtype: memory impaired, closest to Alzheimer's disease",
                  "Enriched for Alzheimer's pathology, not equivalent to it",
                  "NREM sleep features proposed as early markers"],
         notes="Petersen 1999 and Albert 2011 for the criteria: the syndrome is diagnosed "
               "first and its cause judged separately, so a group defined on cognitive "
               "grounds is aetiologically mixed. Most people with MCI never progress "
               "(Mitchell 2009) and about a quarter of amnestic cases revert to normal "
               "cognition (Malek-Ahmadi 2016). Candidate NREM markers: Taillard 2019, Liu "
               "2020, and the parietal fast-spindle deficit of Gorgoni 2016. How to read a "
               "null here belongs in the discussion, so do not pre-empt it."),

    # ---------------------------------------------------------------- 9 gap and aims
    dict(layout="cards", title="Open questions",
         cards=[("Aim 1", "Does the ISFS change with healthy aging, in its parameters and "
                          "in its topography?", NAVY),
                ("Aim 2", "Does amnestic MCI add anything beyond age?", NAVY),
                ("Approach", "256-channel EEG, three groups: whole scalp, channel by "
                             "channel, and an a-priori ROI", NAVY)],
         notes="Hypotheses as they were stated: an age effect on the parameters and on the "
               "topography, and, if the ISFS indexes early neurodegeneration, a further "
               "change in aMCI. What was missing: whole-scalp parameters and topography in "
               "older adults, a young reference group recorded the same way, and any data at "
               "all in aMCI. Grollero et al. (2026) measured the same features in diagnosed "
               "Alzheimer's disease concurrently with this work: convergent, not the "
               "motivation, and it returns in the discussion."),

    # ---------------------------------------------------------------- 10 divider
    dict(layout="section", title="Methods",
         subtitle="From a full night of 256-channel EEG to three numbers per channel"),

    # ---------------------------------------------------------------- 11 cohort
    dict(layout="table", title="Cohort: 104 participants, two sites",
         table=dict(
             rows=[["Group", "n  (Tel Aviv / Sydney)", "Age (years)", "MoCA"],
                   ["Young", "35  (35 / 0)", "27.1 ± 4.3", "not assessed"],
                   ["Elderly", "39  (30 / 9)", "66.5 ± 9.7", "27.2 ± 2.6  (n = 30)"],
                   ["aMCI", "30  (14 / 16)", "67.8 ± 9.1", "21.5 ± 4.3  (n = 14)"]],
             col_w=[2.6, 3.3, 3.0, 3.2], font=20, row_h=0.62),
         bullets=["Identical EEG setup and identical scoring at both sites: TASMC Tel Aviv, "
                  "and the CIRUS Centre, Woolcock Institute, Sydney",
                  "The two older groups are matched in age (p = 0.57) and differ in MoCA as "
                  "expected (p < 0.001)"],
         notes="aMCI patients in Tel Aviv came through the Memory and Attention Disorders "
               "Center and were diagnosed clinically at the Cognitive Neurology Unit; "
               "patients already diagnosed with Alzheimer's disease were not included. "
               "Moderate or severe apnea was excluded, so every older participant has an "
               "AHI of 15 or below. MoCA was collected in Tel Aviv only, which is why the "
               "MoCA analysis has n = 44 rather than 69. Age: Welch t = -0.56, p = 0.57; "
               "MoCA: Mann-Whitney U = 375.5, p < 0.001."),

    # ---------------------------------------------------------------- 12 recording
    dict(layout="text", title="Recording and scoring",
         bullets=["256-channel EGI system, Cz reference, digitized at 1000 Hz, impedances "
                  "below 50 kOhm",
                  "Full-night PSG: EEG with EOG and submental EMG",
                  "Scored in 30 s epochs by AASM criteria (Visbrain within SleepEEGpy), on "
                  "frontal, central and occipital derivations against contralateral mastoids",
                  "Verified against the Pz spectrogram with the hypnogram overlaid",
                  "Face, neck and ear electrodes dropped: 176 channels enter the analysis",
                  "The ISFS analysis is restricted to N2 sleep"],
         notes="Same protocol at both sites, which is what makes pooling defensible. The "
               "excluded electrodes are the ones not over the scalp and most prone to muscle "
               "and movement artifact."),

    # ---------------------------------------------------------------- 13 preprocessing
    dict(layout="flow", title="Preprocessing and artifact rejection",
         steps=["Resample\n250 Hz", "Notch\n50, 100 Hz",
                "Band-pass\n0.1 to 40 Hz", "Reject bad\nchannels, epochs",
                "Average reference\n(good channels)", "Interpolate\nbad channels"],
         bullets=["Semi-automatic rejection, every recording inspected visually over its "
                  "full duration",
                  "Bad channels: 6.3 %, 1.4 %, 3.4 % of channels (young, elderly, aMCI)",
                  "N2 time rejected as artifactual: 4.1 %, 3.3 %, 1.8 %",
                  "The reference is computed over good channels only, so a bad channel "
                  "never enters it; interpolation comes afterwards"],
         notes="Technical exclusions applied on top of the diagnostic criteria: more than "
               "20 % of electrodes bad, more than 30 % of N2 artifactual, fewer than three "
               "clean bouts, or total sleep time under 210 min. The 210 min floor means each "
               "analyzed night holds more than two full NREM-REM cycles, so N2 is sampled "
               "across the night and not only from its first hours. Backup slide B3 has the "
               "exclusion counts."),

    # ---------------------------------------------------------------- 14 bouts
    dict(layout="text", title="Clean N2 bouts: the unit of analysis",
         bullets=["Everything not scored as N2 is discarded",
                  "Each N2 period is split around every overlapping artifact annotation",
                  "Segments of at least 300 s are kept; each bout is analyzed independently",
                  "300 s is the floor because the lowest frequency of interest, 0.0075 Hz, "
                  "has a period of about 133 s, so a bout must hold two full cycles",
                  "At least three clean bouts were required per participant",
                  "Across the cohort: 1023 clean bouts, on average 9.8 per participant "
                  "(range 3 to 21)"],
         notes="Splitting around artifacts rather than rejecting whole N2 periods is what "
               "preserves usable data. The three-bout minimum is an inclusion criterion, and "
               "it excluded nine recordings across the three groups."),

    # ---------------------------------------------------------------- 15 signal chain
    dict(layout="figure", title="From the EEG to the sigma envelope",
         image=A / "methods_panel_AB.png", image_h=4.35,
         claim="A) Spindles over one 745 s N2 bout. B) The 45 s window opened out: raw "
               "channel, 13 to 16 Hz sigma, and the sigma amplitude envelope, with detected "
               "spindles shaded.",
         notes="One example participant and channel. The envelope comes from a Gabor-Morlet "
               "wavelet transform, 13 to 16 Hz in 0.2 Hz steps, 4 cycles wide, implemented "
               "as convolution in the frequency domain. Then, for each bout: subtract the "
               "mean of the envelope so its constant offset does not dominate the 0 Hz bin, "
               "Fourier transform, take the squared magnitude, put it on a frequency axis "
               "common to all bouts, and divide by its own mean power so channels and bouts "
               "of different signal strength are comparable."),

    # ---------------------------------------------------------------- 16 fit
    dict(layout="text_figure", title="Three parameters per channel",
         image=A / "methods_panel_C.png",
         bullets=["envelope  >  FFT  >  Gaussian fit",
                  "Peak frequency: how fast",
                  "Bandwidth: how sharply defined",
                  "AUC: how strong",
                  "Baseline from 0.06 to 0.10 Hz subtracted first",
                  "Accepted if: peak > 1.5 SD, peak inside 0.0075 to 0.04 Hz, area > 0",
                  "Failures left missing, not zero"],
         notes="The parameterization is Dimitriades et al. (2024), adapted from their "
               "128-channel montage to ours; they followed the ISFS across development and "
               "linked these three measures to arousal and memory reactivation. Point at "
               "the curve while naming the parameters: the peak, the width of the orange "
               "band, the area of the shaded band. Normalized bout spectra are averaged "
               "within each channel before the fit. "
               "Why the baseline matters: broadband noise puts a channel-specific offset "
               "under the peak, which would inflate any peak measured on top of it, and the "
               "0.06 to 0.10 Hz range contains no ISFS signal, so it estimates that offset "
               "cleanly. The acceptance rule is the reference pipeline's plus the "
               "positive-area gate, which rejects degenerate fits. Leaving failures missing "
               "rather than zero matters: a zero would be a measurement, a missing value is "
               "the absence of one."),

    # ---------------------------------------------------------------- 17 ROI
    dict(layout="figure", title="The region of interest was defined a priori",
         image=A / "methods_panel_ROI.png", image_h=3.9,
         claim="Taken from the young-adult AUC hotspot of Dimitriades et al. (2024) and "
               "mapped onto the 256-channel montage. Fixed before any group comparison, and "
               "identical for every participant.",
         notes="Say the ROI, or the central-parietal ROI, and nothing else: the internal "
               "variant labels are not for the audience. It is used for the AUC comparison "
               "and appears as green dots on the topography slides. Because it was fixed in "
               "advance, the ROI test is confirmatory while the cluster test is exploratory "
               "over the whole scalp."),

    # ---------------------------------------------------------------- 18 statistics
    dict(layout="text", title="Statistics at three levels of spatial detail",
         bullets=["Whole scalp: one value per participant, the mean over channels with an "
                  "accepted fit",
                  "One-way ANOVA with eta squared where normality and variance homogeneity "
                  "held, Kruskal-Wallis otherwise; Tukey HSD post-hoc when the omnibus test "
                  "was significant",
                  "Topography: cluster-based permutation test (F statistic, 5000 "
                  "permutations, EGI-256 adjacency), Tukey-Kramer post-hoc at each electrode "
                  "of a significant cluster",
                  "ROI: per-channel values normalized by the participant's whole-scalp mean "
                  "first, and only then restricted to the ROI",
                  "Where the amount of analyzed N2 sleep could matter, it was entered as a "
                  "covariate (ANCOVA)"],
         notes="Whole-scalp AUC failed the normality check, so that one parameter is tested "
               "non-parametrically. The order of normalization matters: normalizing inside "
               "the ROI would force its mean to 1 by construction, which is why it is done "
               "over all channels first. Cluster-forming threshold at the p = 0.05 "
               "F-critical value, one-tailed."),

    # ---------------------------------------------------------------- 19 divider
    dict(layout="section", title="Results",
         subtitle="Young 35  ·  Elderly 39  ·  aMCI 30"),

    # ---------------------------------------------------------------- 20 sleep pies
    dict(layout="figure", title="Sleep changes with age, but N2 stays plentiful",
         image=Path("results/demographics_V3/sleep_stage_pies.png"), image_h=4.5,
         claim="Less N3 and less REM sleep in both older groups (both p < 0.001), while N2, "
               "the stage the ISFS lives in, remains the largest stage in all three groups.",
         notes="N3: ANOVA F = 10.85, p = 0.0001. REM: F = 36.70, p < 0.0001. Both are driven "
               "by young versus each older group, with elderly and aMCI indistinguishable. "
               "N2 is 34.1 %, 44.5 % and 39.0 % of recording time. This is the normative "
               "aging pattern, which is the point: it is the baseline against which any aMCI "
               "effect has to show itself."),

    # ---------------------------------------------------------------- 21 hypnos
    dict(layout="three_images", title="One whole night per group",
         images=[(A / "hypno_young.png", "Young"),
                 (A / "hypno_elderly.png", "Elderly"),
                 (A / "hypno_amci.png", "aMCI")],
         notes="Hypnogram over the spectrogram of one electrode, one representative "
               "participant per group. Subject codes are cropped on purpose: two of the "
               "codes are misleading about group membership. Point out the fragmentation and "
               "the loss of deep sleep in the two older nights, then move on: the group "
               "statistics are on the next slide."),

    # ---------------------------------------------------------------- 22 sleep table
    dict(layout="table", title="Sleep architecture and continuity",
         table=dict(
             rows=[["", "Young", "Elderly", "aMCI", "omnibus p"],
                   ["Wake (%)", "9.9 ± 6.6", "15.2 ± 7.5", "21.1 ± 11.8", "< 0.0001"],
                   ["N1 (%)", "3.4 ± 2.5", "9.3 ± 6.2", "10.0 ± 5.6", "< 0.0001"],
                   ["N2 (%)", "34.1 ± 13.2", "44.5 ± 11.7", "39.0 ± 12.3", "0.0009"],
                   ["N3 (%)", "27.9 ± 8.7", "18.4 ± 9.7", "19.3 ± 9.9", "0.0001"],
                   ["REM (%)", "23.9 ± 10.4", "11.4 ± 5.0", "10.3 ± 5.2", "< 0.0001"],
                   ["WASO (min)", "23.2 ± 19.6", "51.9 ± 30.0", "71.0 ± 46.3", "< 0.0001"],
                   ["Sleep efficiency (%)", "89.2 ± 6.7", "83.6 ± 8.5", "78.6 ± 11.7", "0.0001"]],
             col_w=[3.3, 2.25, 2.25, 2.25, 2.05], font=17, row_h=0.5),
         claim="The normative aging pattern, and on every measure elderly and aMCI are "
               "statistically indistinguishable (all p >= 0.11).",
         notes="Percentages are of total recording time; mean ± SD. Sleep onset latency did "
               "not differ across groups (17.4, 15.6, 19.9 min; p = 0.38) and REM latency "
               "did (92.8, 125.4, 130.3 min; p = 0.005). Effect sizes: WASO eta squared "
               "0.291, sleep efficiency 0.170. The full 14-row table with tests and post-hoc "
               "columns is backup slide B2."),

    # ---------------------------------------------------------------- 23 analyzed N2
    dict(layout="table", title="Is the N2 that entered the analysis comparable?",
         table=dict(
             rows=[["", "Young", "Elderly", "aMCI", "omnibus p"],
                   ["N2 bouts (n)", "7.7 ± 3.8", "11.0 ± 4.2", "10.8 ± 4.7", "0.0013"],
                   ["Mean bout length (s)", "629 ± 175", "561 ± 109", "501 ± 77", "0.0022"],
                   ["Analyzed N2 (min)", "80.7 ± 44.2", "106.0 ± 50.1", "90.8 ± 39.9", "0.081"],
                   ["Share of N2 kept (%)", "51.9 ± 15.9", "52.3 ± 17.4", "51.5 ± 15.5", "0.98"]],
             col_w=[3.3, 2.25, 2.25, 2.25, 2.05], font=17, row_h=0.5),
         bullets=["The share of each participant's N2 that survived cleaning is identical "
                  "across groups",
                  "The imbalance that does exist runs opposite to the effects: young adults "
                  "contributed the least analyzed N2, yet show the strongest hotspot",
                  "With analyzed N2 duration as a covariate, the peak-frequency effect holds "
                  "(p = 0.0034) and the covariate explains none of its variance (p = 0.94)"],
         notes="Raise this before showing any ISFS result: it is the first thing an examiner "
               "asks. Total analyzed N2 does not differ significantly (Kruskal-Wallis "
               "p = 0.081). Bandwidth is the one parameter that does track this, and it gets "
               "its own slide. Full ANCOVA output is backup slide B4."),

    # ---------------------------------------------------------------- 24 detection
    dict(layout="figure", title="The ISFS is present in nearly every channel",
         image=FIG / "s3_example_spectra_V11.png", image_h=4.3,
         claim="A valid fit in 80.8 % of channels overall: 74.5 % in young adults, 85.6 % in "
               "elderly, 82.0 % in aMCI. Two example channels per group.",
         notes="This is what the group averages rest on: well-formed single-participant "
               "spectra, not an average that only looks clean once pooled. Each panel is one "
               "channel of one participant, with its own vertical scale, drawn with the same "
               "conventions as the methods figure. If asked what a failed fit means: no peak "
               "clearing the threshold inside the ISFS band, so no ISFS measurable at that "
               "channel, left missing rather than zero."),

    # ---------------------------------------------------------------- 25 peak frequency
    dict(layout="text_figure", title="The rhythm runs faster in both older groups",
         image=A / "group_comparison_violin_peak_frequency.png",
         claim="Young 0.0199, Elderly 0.0226, aMCI 0.0232 Hz. One-way ANOVA p = 0.0026, "
               "eta squared 0.111.",
         bullets=["Young < Elderly = aMCI",
                  "Post-hoc: young vs elderly p = 0.013, young vs aMCI p = 0.005",
                  "Elderly vs aMCI: p = 0.86",
                  "Across all 104 participants the peak sits at 0.0219 Hz on average "
                  "(range 0.0095 to 0.0314), where the literature puts it",
                  "Each dot is one participant's mean over the fitted electrodes"],
         notes="Headline temporal result. The effect survives the N2-duration covariate "
               "(ANCOVA p = 0.0034, partial eta squared 0.108) with both pairwise contrasts "
               "surviving Holm correction. If asked how big it is: about a 15 % higher rate, "
               "or a cycle roughly 7 s shorter."),

    # ---------------------------------------------------------------- 26 bandwidth
    dict(layout="two_figures", title="Bandwidth: no group effect, and it tracks analyzed N2",
         images=[(A / "group_comparison_violin_bandwidth.png", None),
                 (A / "bandwidth_vs_n2_duration.png", None)],
         claim="0.0236, 0.0281, 0.0276 Hz, ANOVA p = 0.061, not significant. Bandwidth "
               "correlates with how much N2 sleep a participant contributed (r = 0.386), and "
               "with duration as a covariate the group difference weakens to p = 0.206.",
         notes="The covariate is the stronger predictor: F = 14.73, p = 0.0002, partial eta "
               "squared 0.128, against group's 0.031. So this is not interpretable as an "
               "effect of age, and the thesis does not call it a trend. Say why being "
               "explicit here is a strength: the same covariate analysis is what makes the "
               "peak-frequency result credible, because there it changed nothing."),

    # ---------------------------------------------------------------- 27 AUC whole scalp
    dict(layout="text_figure", title="Overall strength, averaged over the scalp, is unchanged",
         image=A / "group_comparison_violin_auc.png",
         claim="AUC 6.46, 7.31, 7.48. Kruskal-Wallis p = 0.38.",
         bullets=["No group difference in ISFS strength when every channel is averaged "
                  "together",
                  "Tested non-parametrically because AUC failed the normality check",
                  "Averaging over the whole scalp can hide a regional effect, which is "
                  "exactly what happens next"],
         notes="Deliberate setup for the topography slides. If the talk is running long, "
               "this slide and the next can be merged, but do not skip the point: the null "
               "at whole-scalp level is what makes the focal result interesting rather than "
               "just a global decline."),

    # ---------------------------------------------------------------- 28 raw topo
    dict(layout="figure", title="Where the strength sits: the young-adult hotspot",
         image=G11 / "three_group_topo_auc_raw.png", image_h=3.9,
         claim="Young adults concentrate ISFS strength over central-parietal electrodes. In "
               "both older groups the hotspot is less focal and more diffuse.",
         notes="Raw per-group maps; the number above each map is the mean of the "
               "per-participant means. Worth saying out loud: the young-adult topography "
               "reproduces Dimitriades 2024 and Lazar 2019 in an independent cohort, which "
               "is a validation of the whole pipeline before any group claim is made."),

    # ---------------------------------------------------------------- 29 cluster
    dict(layout="figure", title="One significant central-parietal cluster",
         image=G11 / "three_group_topo_auc.png", image_h=3.9,
         claim="Cluster-based permutation test: p = 0.023, 9 electrodes, lower AUC in both "
               "older groups. Yellow rings mark significant post-hoc electrodes, green dots "
               "the pre-defined ROI.",
         notes="Electrodes E130, E143, E144, E153, E154, E155, E184, E185, E197 (mean "
               "F = 5.15, max F = 10.62). Post-hoc: young vs elderly significant at 5 of 9, "
               "young vs aMCI at 7 of 9, elderly vs aMCI at 1 (E197). These maps are "
               "per-participant normalized, so the result is about where a participant's "
               "strength is concentrated, not about the overall level, which did not differ. "
               "Headline spatial result. Backup slide B5 has the electrode lists."),

    # ---------------------------------------------------------------- 30 ROI
    dict(layout="text_figure", title="Averaging over the whole ROI dilutes the effect",
         image=A / "group_comparison_violin_extended_ROI_normalized_auc.png",
         claim="Same order as the topography (1.099, 1.040, 1.012) but the three-group "
               "comparison is not significant: ANOVA p = 0.143.",
         bullets=["The cluster is 9 electrodes; the ROI is 36",
                  "Averaging over the whole region mixes the affected electrodes with "
                  "unaffected neighbours",
                  "This is a null, and it is reported as one; a larger sample may resolve "
                  "whether the ROI effect is genuine"],
         notes="Do not present this as support for the cluster result. The honest reading is "
               "that the a-priori ROI, taken from young adults, is larger than the region "
               "that actually changes with age. Values are per-participant normalized, so "
               "1.10 means 10 % above that participant's own whole-scalp mean."),

    # ---------------------------------------------------------------- 31 PF/BW topos
    dict(layout="figure", title="The temporal changes are diffuse, not regional",
         image=FIG / "s1_topo_composite_V11.png", image_h=4.4,
         claim="Neither peak frequency (A) nor bandwidth (B) produced a significant cluster "
               "anywhere on the scalp.",
         notes="In young adults peak frequency shows a frontal emphasis that flattens in the "
               "older groups, and bandwidth has no consistent spatial pattern at all. The "
               "interpretation to offer: peak frequency changes everywhere rather than "
               "anywhere in particular, which is what a change in the pacemaker rather than "
               "in a cortical generator would look like."),

    # ---------------------------------------------------------------- 32 MoCA
    dict(layout="two_figures", title="No association with cognitive score",
         images=[(A / "moca_panel_peak_frequency.png", None),
                 (A / "moca_panel_roi_auc.png", None)],
         claim="Pooled elderly and aMCI, n = 44: no ISFS measure tracks the MoCA "
               "(all |r| <= 0.15, all p >= 0.34, Pearson and Spearman). Elderly in red, "
               "aMCI in blue.",
         notes="All four scalars were tested; two are shown, the other two behave the same "
               "way. Say the limitation before an examiner does: MoCA was collected in Tel "
               "Aviv only, so this analysis has n = 44 out of 69 older participants and is "
               "the least powered in the thesis."),

    # ---------------------------------------------------------------- 33 divider
    dict(layout="section", title="Discussion",
         subtitle="What changed, what did not, and what would explain it"),

    # ---------------------------------------------------------------- 34 summary
    dict(layout="cards", title="What changed, and what did not",
         cards=[("Faster", "Peak frequency higher in both older groups (p = 0.0026), "
                           "everywhere on the scalp", ELDERLY),
                ("Less focal", "The central-parietal hotspot of strength flattens "
                               "(cluster p = 0.023) while overall strength is preserved",
                 ELDERLY),
                ("aMCI = Elderly", "No difference on any measure, and no correlation with "
                                   "cognitive score", AMCI)],
         claim="Bandwidth is not part of the story: what difference there was tracked how "
               "much N2 sleep we analyzed.",
         notes="Say the summary in three sentences and resist elaborating: the next four "
               "slides do the elaborating. The framing of the whole thesis is in the title: "
               "aging, not amnestic MCI."),

    # ---------------------------------------------------------------- 35 precision
    dict(layout="text", title="A loss of temporal and spatial precision",
         bullets=["In young adults the ISFS is strongest in fast spindles and over "
                  "centro-parieto-occipital cortex (Molle 2011; Lazar 2019), and our "
                  "young-adult hotspot reproduces that",
                  "With age the rhythm that organizes spindle trains speeds up, and its "
                  "energy is no longer concentrated where fast-spindle generators are densest",
                  "AUC at an electrode reflects how deeply spindle amplitude rises and falls "
                  "there, not how many spindles occur",
                  "So what was lost is regional focus, not overall strength: a drive that "
                  "has become less precise rather than weaker",
                  "An infra-slow rhythm contributes to, but does not dominate, spindle "
                  "timing (Chen 2025), so this is not direct evidence that spindle "
                  "generation has degraded"],
         notes="The last bullet is the careful reading, and it is the one to volunteer rather "
               "than defend: safer to read the effect as a change in where the rhythm is "
               "expressed than as evidence about the spindle generators themselves. The "
               "peak-frequency map blurs in the same direction without reaching significance."),

    # ---------------------------------------------------------------- 36 noradrenergic
    dict(layout="text", title="A noradrenergic account, and its limits",
         bullets=["In rodents the LC does not merely accompany the rhythm, it sets its "
                  "phase; noradrenaline in thalamus, basal forebrain and brainstem rises and "
                  "falls on the same 50 s cycle",
                  "That structure ages badly: the earliest abnormal tau appears in the LC "
                  "(Braak 2011), and it loses proportionally more neurons in Alzheimer's "
                  "disease than the nucleus basalis or the substantia nigra (Zarow 2003)",
                  "In living humans, LC integrity tracks memory and sleep regulation in "
                  "older adults, and an LC abnormality in aMCI separates those who progress "
                  "to dementia from those who do not",
                  "A whole-scalp frequency effect fits a change in the pacemaker better "
                  "than a change in the cortex where spindles appear",
                  "Not shown here: neither noradrenergic nor thalamic activity was measured. "
                  "What our data can do is test the account's prediction"],
         notes="Lecci 2017, Osorio-Forero 2021 and 2025, Kjaerby 2022 and 2026; Braak 2011, "
               "Zarow 2003, Dahl 2019, Van Egroo 2022, Galgani 2023, Luthi 2025. The "
               "prediction our data does test: a mechanism anchored in a structure that "
               "degenerates early and in the general population, not only in patients, "
               "predicts a change complete in healthy older adults and no further advanced "
               "in aMCI. That is the pattern we found."),

    # ---------------------------------------------------------------- 37 aMCI null
    dict(layout="text", title="No evidence for an effect of amnestic MCI beyond age",
         bullets=["The two groups were age-matched and differed in cognitive score, yet "
                  "were indistinguishable on every ISFS measure",
                  "Reading one: the change is already well advanced in healthy older adults, "
                  "and the transition to aMCI does not measurably accelerate it",
                  "Reading two: the rhythm changes before cognition does, in which case a "
                  "group defined by cognitive score cannot separate",
                  "Dilution: aMCI is aetiologically mixed, so an Alzheimer's-specific effect "
                  "would be diluted by the patients who do not have that pathology",
                  "A contrast worth noting: in an overlapping cohort, slow-wave synchrony "
                  "does track impairment (Sharon 2025), and the cyclic alternating pattern "
                  "is reduced in MCI and predicts incident dementia"],
         notes="Grollero 2026 found reduced peak amplitude in clinically diagnosed "
               "Alzheimer's disease, a parameter we did not measure, in a group defined by "
               "diagnosis rather than by cognitive complaint. So our null says the rhythm "
               "does not track an aMCI diagnosis; it does not say the rhythm is preserved in "
               "early Alzheimer's disease. Two measures of the same nights behaving "
               "differently is informative about what each one indexes."),

    # ---------------------------------------------------------------- 38 other populations
    dict(layout="text", title="The ISFS in other populations, and why comparison is hard",
         bullets=["Reduced central-parietal strength in childhood- and early-onset "
                  "schizophrenia, uncorrelated with clinical characteristics "
                  "(Dimitriades 2025): structurally the same result as ours, in a different "
                  "condition and a much younger sample",
                  "Elevated infra-slow sigma power in myalgic encephalomyelitis and chronic "
                  "fatigue syndrome (Sun 2026)",
                  "Already measurable in children aged one to five, with no difference "
                  "between autistic and typically developing children (Liu 2026)",
                  "But the rhythm is not measured the same way: a Gaussian fit to the "
                  "envelope spectrum (ours) or relative band power in a fixed window, which "
                  "yields no frequency or bandwidth at all",
                  "Even strength is three different quantities across studies: area of the "
                  "fitted peak, amplitude of the fitted peak, or relative band power"],
         notes="So the opposite directions, reduced in schizophrenia and in aging but "
               "elevated in chronic fatigue, cannot yet be read as a contradiction. If the "
               "rhythm indexes how deeply arousal is modulated across the night, deviation "
               "in either direction may be what matters, which is what an inverted-U on "
               "noradrenergic fluctuation amplitude would predict (Luthi 2025). Two of our "
               "three measures have essentially no comparison literature outside the "
               "approach they come from."),

    # ---------------------------------------------------------------- 39 limitations
    dict(layout="text", title="Limitations",
         bullets=["Modest samples, two recording sites, and patients recruited through "
                  "memory clinics rather than assembled as a uniform research cohort",
                  "Patients reach a clinic at different points in their illness, so an "
                  "effect confined to more advanced patients could have been missed",
                  "The ROI reduction fell short of significance (p = 0.143) and would "
                  "benefit from a larger sample",
                  "Cognitive scores were available for only part of the older sample (n = 44)",
                  "Cross-sectional: individual trajectories, and which patients progress, "
                  "cannot be identified from these data"],
         notes="Volunteer these rather than waiting to be asked. The Sydney recruitment and "
               "diagnostic route are not documented in the same detail as the Tel Aviv one, "
               "which is a real gap in the methods and is stated as such in the thesis."),

    # ---------------------------------------------------------------- 40 future
    dict(layout="text", title="Future directions",
         bullets=["Longitudinal recordings: does ISFS topography predict conversion, or "
                  "track decline within individuals?",
                  "Coupling: sleep is built from nested rhythms, and the timing of spindles "
                  "relative to the slow oscillation predicts overnight retention and loosens "
                  "with age (Helfrich 2018)",
                  "So ask what the ISFS is coupled to: the slow oscillations inside its "
                  "windows, and the infra-slow haemodynamic fluctuations that share its "
                  "timescale",
                  "The LC cannot be recorded in a sleeping human, so the origin of the human "
                  "rhythm will stay inferred; converging indirect evidence is the route"],
         notes="Keep this short. If asked what the single most informative next experiment "
               "would be: repeated overnight recordings years apart in the same older and "
               "aMCI participants, because only a within-subject design separates aging "
               "trajectories from cross-sectional differences."),

    # ---------------------------------------------------------------- 41 conclusion
    dict(layout="statement", title="Conclusion",
         statement="The infra-slow sigma rhythm of N2 sleep is a sensitive marker of how "
                   "healthy aging reshapes the spindle infrastructure of sleep: it becomes "
                   "faster and loses its central-parietal focus.",
         statement2="It did not distinguish patients with amnestic MCI from healthy older "
                    "adults of the same age.",
         notes="Land on the two sentences and stop. Then the acknowledgements slide, then "
               "invite questions."),

    # ---------------------------------------------------------------- 42 acknowledgements
    dict(layout="text", title="Acknowledgements",
         bullets=["Prof. Yuval Nir, for the guidance and the standards",
                  "Noa Bregman, and Rivi Tauman and Jenny Zitser, for the clinical and "
                  "sleep-medicine side",
                  "Maria E. Dimitriades, whose analysis pipeline this work is built on",
                  "Angela D'Rozario and Rick Wassing, for the recordings at the CIRUS "
                  "Centre in Sydney",
                  "Rotem Falach and Flavio Schmidig, who taught me EEG analysis, and the "
                  "whole Nir lab",
                  "The participants and their families"],
         notes="Thank you. Questions."),

    # ---------------------------------------------------------------- 43 divider
    dict(layout="section", title="Backup slides",
         subtitle="Referenced by number from the notes of the slide each one defends"),

    # ---------------------------------------------------------------- B1
    dict(layout="figure", title="B1. Full demographics and data quality",
         image=Path("results/demographics_V3/demographics_table.png"), image_h=2.6,
         claim="Sample sizes, age, MoCA, bad channels and bad N2 epochs per group, with "
               "exclusion counts.",
         notes="Bad channels 6.3 %, 1.4 %, 3.4 %; bad N2 epochs 4.1 %, 3.3 %, 1.8 %. The "
               "header of this rendering still says MCI rather than aMCI; the thesis table "
               "is a native table that reads aMCI."),

    # ---------------------------------------------------------------- B2
    dict(layout="figure", title="B2. Full sleep table, with tests and effect sizes",
         image=Path("results/demographics_V4/table2_sleep_architecture.png"), image_h=4.4,
         claim="Sleep architecture, sleep continuity and N2 bout properties, with the "
               "omnibus test, eta squared and post-hoc p values for every row.",
         notes="Dashes mark comparisons that were not run because the omnibus test was not "
               "significant. Bold marks p < 0.05."),

    # ---------------------------------------------------------------- B3
    dict(layout="table", title="B3. Exclusions: 26 recordings",
         table=dict(
             rows=[["Reason", "Young", "Elderly", "aMCI"],
                   ["Fewer than 3 clean N2 bouts", "4", "2", "3"],
                   ["Too many bad channels", "4", "1", "4"],
                   ["Too many bad N2 epochs", "1", "1", "2"],
                   ["Total sleep time under 210 min", "1", "0", "3"],
                   ["Total excluded", "10", "4", "12"]],
             col_w=[5.4, 2.2, 2.2, 2.2], font=18, row_h=0.55),
         claim="Analyzed: 35, 39 and 30 recordings. Patients who had progressed to a "
               "clinical diagnosis of Alzheimer's disease were not included in the analysis "
               "at all, and are not counted here.",
         notes="The criteria are stated in the methods: more than 20 % of electrodes bad, "
               "more than 30 % of N2 artifactual, fewer than three clean bouts, or total "
               "sleep time under 210 min."),

    # ---------------------------------------------------------------- B4
    dict(layout="table", title="B4. ANCOVA with analyzed N2 duration as covariate",
         table=dict(
             rows=[["Parameter", "unadjusted p", "group p", "covariate p", "verdict"],
                   ["Peak frequency", "0.0026", "0.0034", "0.94", "group effect survives"],
                   ["Bandwidth", "0.061", "0.206", "0.0002", "duration, not group"],
                   ["AUC (whole scalp)", "0.38", "0.45", "0.49", "no effect either way"]],
             col_w=[3.0, 2.2, 1.9, 2.1, 2.9], font=17, row_h=0.55),
         bullets=["Covariate mean-centred at the cohort grand mean of 93.1 min; N = 104, "
                  "residual df = 100",
                  "Peak frequency: partial eta squared 0.108 for group, 0.0001 for the "
                  "covariate; adjusted means identical to raw means",
                  "Bandwidth: partial eta squared 0.128 for the covariate against 0.031 for "
                  "group",
                  "Homogeneity of regression slopes held for all three parameters "
                  "(p >= 0.11)"],
         notes="AUC failed the per-group normality check in the main analysis, so "
               "Kruskal-Wallis remains its primary test and the ANCOVA row is a "
               "covariate-adjusted supplement. Source: three_group_ancova_statistics.txt."),

    # ---------------------------------------------------------------- B5
    dict(layout="table", title="B5. The cluster, electrode by electrode",
         table=dict(
             rows=[["", "n", "electrodes"],
                   ["Cluster (F test), p = 0.023", "9",
                    "E130, E143, E144, E153, E154, E155, E184, E185, E197"],
                   ["Young vs Elderly", "5", "E143, E144, E154, E155, E184"],
                   ["Young vs aMCI", "7", "E143, E144, E153, E155, E184, E185, E197"],
                   ["Elderly vs aMCI", "1", "E197"]],
             col_w=[3.6, 0.9, 7.5], font=15, row_h=0.55),
         claim="Mean F = 5.15, max F = 10.62 across the cluster. Post-hoc tests are "
               "Tukey-Kramer at each cluster electrode.",
         notes="12 candidate clusters were formed, one survived. The direction is lower AUC "
               "in both older groups than in young adults. Across re-runs on slightly "
               "different cohorts the cluster stays central-parietal and keeps its "
               "direction; the electrode count moves by one or two."),

    # ---------------------------------------------------------------- B6
    dict(layout="figure", title="B6. Bandwidth against analyzed N2 duration",
         image=A / "bandwidth_vs_n2_duration.png", image_h=4.3,
         claim="r = +0.386 across all 104 participants. This is why the bandwidth difference "
               "is not read as an effect of age.",
         notes="A longer recording gives more bouts and a more finely resolved spectrum, "
               "which plausibly broadens the fitted peak. Whatever the mechanism, duration "
               "is the stronger predictor and it is not balanced across groups."),

    # ---------------------------------------------------------------- B7
    dict(layout="text", title="B7. Feature extraction, in detail",
         bullets=["Gabor-Morlet wavelets, 13.0 to 16.0 Hz in 0.2 Hz steps, fixed width of "
                  "4 cycles, applied as frequency-domain convolution via FFT",
                  "Envelope = mean magnitude of the complex coefficients across those "
                  "wavelet frequencies",
                  "Per bout: subtract the envelope mean, FFT, take the squared magnitude, "
                  "resample onto a common frequency axis, divide by the bout's own mean power",
                  "If the spectral maximum fell in the lowest bins (0 to 0.006 Hz), those "
                  "bins were discarded as residual drift",
                  "Baseline subtracted from the channel-mean spectrum, estimated over 0.06 "
                  "to 0.10 Hz",
                  "Gaussian a·exp(-((f-b)/c)^2) by nonlinear least squares; bandwidth is the "
                  "width of b ± |c|, AUC the integral over that band"],
         notes="If asked what differs from the reference implementation: the positive-area "
               "gate on the fit, which rejects degenerate solutions, and the montage "
               "adaptation from 128 to 256 channels. If asked about the absolute-value on c: "
               "the fit can return a negative width parameter, which is the same Gaussian, "
               "so its magnitude is what enters the bandwidth."),

    # ---------------------------------------------------------------- B8
    dict(layout="text", title="B8. Detection rates, and what a failed fit means",
         bullets=["Valid Gaussian fit in 80.8 % of channels overall: 74.5 % young, 85.6 % "
                  "elderly, 82.0 % aMCI",
                  "1023 clean N2 bouts entered the analysis, on average 9.8 per participant",
                  "A failure means no peak clearing 1.5 SD inside 0.0075 to 0.04 Hz, or a "
                  "non-positive integrated area",
                  "Failed channels are left missing, so they neither raise nor lower a "
                  "participant's mean",
                  "The reference study reports comparable rates on its own criteria; the "
                  "numbers are not directly comparable because our acceptance rule is "
                  "slightly stricter"],
         notes="Young adults have the lowest detection rate and the strongest hotspot, which "
               "rules out a story in which the group differences are driven by how often the "
               "fit succeeds. If pressed on why young adults fit less often: they also "
               "contributed the least analyzed N2 sleep."),

    # ---------------------------------------------------------------- B9
    dict(layout="text", title="B9. Was the aMCI group too heterogeneous?",
         bullets=["A sensitivity analysis restricting the patient group to purely amnestic "
                  "cases was run, and then rejected: the full aMCI group (n = 30) is what "
                  "the thesis reports",
                  "In that run the focal central-parietal AUC cluster survived, the "
                  "young-versus-patient peak-frequency contrast lost significance at the "
                  "reduced sample size, and the MoCA nulls were unchanged",
                  "It was run on the pre-final cohort, so its p values are not the numbers "
                  "of record",
                  "Reported instead in the discussion as a limitation: a mixed group dilutes "
                  "any disease-specific effect"],
         notes="Outputs are kept under results/no_naMCI. The reason for rejecting the "
               "sensitivity run was that it traded a defensible full sample for a smaller "
               "one without changing the conclusion, and the exclusion criteria were already "
               "fixed. Do not quote its p values as results."),

    # ---------------------------------------------------------------- B10
    dict(layout="table", title="B10. Alongside Grollero et al. (2026)",
         table=dict(
             rows=[["", "This thesis", "Grollero et al. 2026"],
                   ["Sample", "39 elderly, 30 aMCI, 35 young",
                    "10 Alzheimer's disease, 20 age-matched controls"],
                   ["Recording", "In-lab 256-channel EEG", "Home EEG"],
                   ["Strength measure", "Area under the fitted peak (AUC)",
                    "Amplitude of the fitted peak"],
                   ["Frequency", "Faster with age, no clinical difference",
                    "No clinical difference"],
                   ["Strength", "Central-parietal focus lost with age",
                    "Reduced in AD, tracks plasma amyloid"],
                   ["Group definition", "Clinical aMCI, mixed aetiology",
                    "Confirmed Alzheimer's disease"]],
             col_w=[2.8, 4.6, 4.6], font=15, row_h=0.62),
         notes="Concurrent work, not the motivation for this thesis. The two studies agree "
               "on frequency, which distinguished neither clinical contrast, and are "
               "broadly consistent on a weakening of the rhythm over central regions. They "
               "diverge on whether cognitive impairment leaves a mark beyond age, and the "
               "two strength measures are not the same quantity, so the divergence is not "
               "a direct contradiction."),

    # ---------------------------------------------------------------- B11
    dict(layout="text", title="B11. Where the hypnograms and annotations come from",
         bullets=["Sleep stages and artifact marks are read from each recording's cleaned "
                  "annotation file, which is the file the analysis itself used",
                  "Two independent labels can coincide: an unscored epoch from the scorer, "
                  "and an amplifier or acquisition gap",
                  "Both are excluded from the numerator and the denominator of total sleep "
                  "time and wake after sleep onset, so neither inflates the other",
                  "Stage labels appear in two dialects across files and are normalized "
                  "before any counting"],
         notes="This is the answer to any question about how sleep-architecture percentages "
               "were computed and why they do not match a naive count of scored epochs. The "
               "full note is in the repository."),
]

# --------------------------------------------------------------------------- rendering


def textbox(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def _bullet_paragraph(para, indent=0.30):
    """Real bullet glyph plus a hanging indent, so wrapped lines line up.

    python-pptx exposes neither, so both are set on the paragraph properties
    element directly.
    """
    pPr = para._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    bu_font = OxmlElement("a:buFont")
    bu_font.set("typeface", "Arial")
    bu_char = OxmlElement("a:buChar")
    bu_char.set("char", "▪")
    pPr.append(bu_font)
    pPr.append(bu_char)


def write(tf, lines, size=20, color=INK, bold=False, italic=False, space_after=10,
          line_spacing=1.05, align=PP_ALIGN.LEFT, first=True, bullet=False, rtl=False):
    """Write a list of strings as paragraphs into a text frame."""
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if (first and i == 0) else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        para.line_spacing = line_spacing
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.name = FONT
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        if bullet:
            _bullet_paragraph(para)
        if rtl:
            para._p.get_or_add_pPr().set("rtl", "1")
    return tf


def slide_title(slide, text, color=NAVY, size=28):
    tf = textbox(slide, 0.6, 0.32, 12.2, 0.95)
    write(tf, [text], size=size, color=color, bold=True, line_spacing=0.95, space_after=0)
    # thin rule under the title
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.62), Inches(1.24),
                                  Inches(12.1), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False


def claim_text(slide, text, top=1.36, width=12.1, size=16):
    tf = textbox(slide, 0.62, top, width, 0.9)
    write(tf, [text], size=size, color=MUTED, line_spacing=1.0, space_after=0)


def bullet_list(slide, lines, left, top, width, height, size=19):
    tf = textbox(slide, left, top, width, height)
    write(tf, lines, size=size, color=INK, space_after=13, line_spacing=1.02,
          bullet=True)


def fit_image(slide, path, left, top, width, height, caption=None):
    """Place an image scaled to fit the box, centred, with an optional caption."""
    iw, ih = Image.open(path).size
    scale = min(width / iw, height / ih)
    w, h = iw * scale, ih * scale
    l = left + (width - w) / 2
    t = top + (height - h) / 2
    slide.shapes.add_picture(str(path), Inches(l), Inches(t), width=Inches(w), height=Inches(h))
    if caption:
        tf = textbox(slide, left, t + h + 0.06, width, 0.4)
        write(tf, [caption], size=13, color=MUTED, italic=True, align=PP_ALIGN.CENTER,
              space_after=0)
    return l, t, w, h


def add_table(slide, spec, left, top):
    rows = spec["rows"]
    col_w = spec["col_w"]
    row_h = spec.get("row_h", 0.55)
    font = spec.get("font", 18)
    width = sum(col_w)
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(row_h * len(rows)))
    table = shape.table
    table.first_row = True
    for c, w in enumerate(col_w):
        table.columns[c].width = Inches(w)
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                # a group column keeps its group colour in the header
                cell.fill.fore_color.rgb = GROUP_HEAD.get(str(val).strip(), NAVY)
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else BAND
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for run in para.runs:
                run.font.size = Pt(font)
                run.font.name = FONT
                run.font.bold = (r == 0)
                run.font.color.rgb = WHITE if r == 0 else INK
    return top + row_h * len(rows)


def chrome(slide, number):
    """Footer and slide number."""
    tf = textbox(slide, 0.6, 7.02, 7.0, 0.32)
    write(tf, [FOOTER], size=10, color=MUTED, space_after=0)
    tf = textbox(slide, 12.1, 7.02, 0.7, 0.32)
    write(tf, [str(number)], size=10, color=MUTED, align=PP_ALIGN.RIGHT, space_after=0)


# --------------------------------------------------------------------------- layouts

def render_title(slide, s):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                 Inches(SLIDE_W), Inches(3.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False

    tf = textbox(slide, 0.9, 0.85, 11.5, 2.2, anchor=MSO_ANCHOR.MIDDLE)
    write(tf, [s["title"]], size=30, color=WHITE, bold=True, line_spacing=1.08,
          space_after=0)
    tf = textbox(slide, 0.9, 4.1, 11.5, 2.4)
    write(tf, s["subtitle"], size=19, color=INK, space_after=10)


def render_section(slide, s):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                 Inches(SLIDE_W), Inches(SLIDE_H))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = textbox(slide, 1.0, 2.9, 11.3, 1.9, anchor=MSO_ANCHOR.MIDDLE)
    write(tf, [s["title"]], size=40, color=WHITE, bold=True, space_after=8)
    if s.get("subtitle"):
        write(tf, [s["subtitle"]], size=19, color=RGBColor.from_string("C9D5E4"),
              first=False, space_after=0)


def render_text(slide, s):
    slide_title(slide, s["title"])
    top = 1.5
    if s.get("claim"):
        claim_text(slide, s["claim"], top=1.36)
        top = 2.15
    bullet_list(slide, s["bullets"], 0.75, top, 11.9, 6.8 - top, size=20)


def render_figure(slide, s):
    slide_title(slide, s["title"])
    top = 1.4
    if s.get("claim"):
        claim_text(slide, s["claim"], top=top)
        top = 2.28 if len(s["claim"]) > 120 else 2.05
    labels = s.get("labels")
    iw, ih = Image.open(s["image"]).size
    if labels and iw / ih < 1.35:
        # a portrait-ish figure leaves the right half of the slide empty, so the
        # labels go down the side instead of across the bottom
        fit_image(slide, s["image"], 0.7, top, 6.1, 6.7 - top, caption=s.get("caption"))
        height = 0.95
        y = top + 0.4
        for text in labels:
            pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2),
                                          Inches(y), Inches(5.5), Inches(height - 0.14))
            pill.fill.solid()
            pill.fill.fore_color.rgb = BAND
            pill.line.color.rgb = RULE
            pill.shadow.inherit = False
            tf = pill.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = Inches(0.20)
            write(tf, [text], size=19, color=INK, space_after=0, line_spacing=0.95)
            y += height
        if s.get("source"):
            source_line(slide, s["source"], top=y + 0.05, left=7.4, width=5.3)
        return
    limit = 5.60 if labels else 6.75
    img_h = s.get("image_h", limit - top)
    _, t, _, h = fit_image(slide, s["image"], 0.7, top, 11.95, img_h,
                           caption=s.get("caption"))
    if labels:
        label_row(slide, labels, limit + 0.28)
    if s.get("source"):
        source_line(slide, s["source"])
    if s.get("bullets"):
        bullet_list(slide, s["bullets"], 0.85, t + h + 0.30, 11.7, 6.85 - (t + h + 0.30),
                    size=16)


def render_text_figure(slide, s, image_right=True):
    slide_title(slide, s["title"])
    top = 1.5
    if s.get("claim"):
        claim_text(slide, s["claim"], top=1.36)
        top = 2.3 if len(s["claim"]) > 110 else 2.05
    text_left, img_left = (0.75, 6.95) if image_right else (7.0, 0.6)
    if s.get("bullets"):
        bullet_list(slide, s["bullets"], text_left, top, 5.6, 6.8 - top, size=18)
    fit_image(slide, s["image"], img_left, 1.45, 5.9, 5.25, caption=s.get("caption"))


def label_row(slide, labels, top, size=17):
    """Short phrases in a row of pills: the terse alternative to bullet sentences."""
    n = len(labels)
    gap = 0.25
    width = (12.1 - gap * (n - 1)) / n
    for i, text in enumerate(labels):
        left = 0.62 + i * (width + gap)
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                      Inches(top), Inches(width), Inches(0.72))
        pill.fill.solid()
        pill.fill.fore_color.rgb = BAND
        pill.line.color.rgb = RULE
        pill.shadow.inherit = False
        tf = pill.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.10)
        tf.margin_right = Inches(0.10)
        write(tf, [text], size=size, color=INK, align=PP_ALIGN.CENTER, space_after=0,
              line_spacing=0.95)


def source_line(slide, text, top=6.72, left=0.62, width=12.1):
    tf = textbox(slide, left, top, width, 0.35)
    write(tf, [text], size=13, color=MUTED, italic=True, space_after=0)


def render_two_figures(slide, s):
    slide_title(slide, s["title"])
    top = 1.4
    if s.get("claim"):
        claim_text(slide, s["claim"], top=top)
        top = 2.5 if len(s["claim"]) > 150 else 2.15
    labels = s.get("labels")
    bottom = 6.75 if not labels else 5.60
    boxes = [(0.55, 6.1), (6.75, 6.1)]
    # captions are drawn under the image, so reserve room for them inside the box
    cap_room = 0.55 if any(cap for _, cap in s["images"]) else 0.0
    for (left, width), (path, cap) in zip(boxes, s["images"]):
        fit_image(slide, path, left, top, width, bottom - top - cap_room, caption=cap)
    if labels:
        label_row(slide, labels, bottom + 0.28)
    if s.get("source"):
        source_line(slide, s["source"])


def render_figure_plus(slide, s):
    """One main figure, with a small supporting panel and short labels beneath it."""
    slide_title(slide, s["title"])
    top = 1.42
    img_h = s.get("image_h", 3.6)
    fit_image(slide, s["image"], 0.7, top, 11.95, img_h, caption=s.get("caption"))
    row_top = top + img_h + 0.28
    if s.get("small_image"):
        # leave room under the panel for its caption, above the footer
        fit_image(slide, s["small_image"], 0.62, row_top, 4.4, 6.35 - row_top,
                  caption=s.get("small_caption"))
    if s.get("labels"):
        n = len(s["labels"])
        height = min(0.78, (6.75 - row_top) / n)
        for i, text in enumerate(s["labels"]):
            pill = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.4), Inches(row_top + i * height),
                Inches(7.3), Inches(height - 0.12))
            pill.fill.solid()
            pill.fill.fore_color.rgb = BAND
            pill.line.color.rgb = RULE
            pill.shadow.inherit = False
            tf = pill.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = Inches(0.18)
            write(tf, [text], size=18, color=INK, space_after=0, line_spacing=0.95)


def render_two_stacked(slide, s):
    """Two figures stacked, with short labels in a row underneath."""
    slide_title(slide, s["title"])
    top = 1.40
    labels = s.get("labels")
    bottom = 6.75 if not labels else 5.85
    span = bottom - top
    heights = [span * 0.52, span * 0.42]
    y = top
    for (path, cap), h in zip(s["images"], heights):
        fit_image(slide, path, 1.1, y, 11.1, h, caption=cap)
        y += h + (span * 0.06)
    if labels:
        label_row(slide, labels, bottom + 0.22)


def render_three_images(slide, s):
    slide_title(slide, s["title"])
    # two on the top row, one centred below; heights leave room for the captions
    # and keep the bottom one clear of the footer
    top1, h1 = 1.45, 2.25
    fit_image(slide, s["images"][0][0], 0.5, top1, 6.1, h1, caption=s["images"][0][1])
    fit_image(slide, s["images"][1][0], 6.75, top1, 6.1, h1, caption=s["images"][1][1])
    fit_image(slide, s["images"][2][0], 3.6, top1 + h1 + 0.60, 6.1, h1,
              caption=s["images"][2][1])


def render_table(slide, s):
    slide_title(slide, s["title"])
    top = 1.5
    if s.get("claim") and not s.get("bullets"):
        claim_text(slide, s["claim"], top=1.36)
        top = 2.25
    spec = s["table"]
    left = (SLIDE_W - sum(spec["col_w"])) / 2
    bottom = add_table(slide, spec, left, top)
    if s.get("claim") and s.get("bullets"):
        claim_text(slide, s["claim"], top=bottom + 0.15)
        bottom += 0.75
    if s.get("bullets"):
        bullet_list(slide, s["bullets"], 0.85, bottom + 0.25, 11.7,
                    6.9 - (bottom + 0.25), size=16)


def render_cards(slide, s):
    slide_title(slide, s["title"])
    top = 1.6
    if s.get("claim"):
        claim_text(slide, s["claim"], top=1.36)
        top = 2.15
    n = len(s["cards"])
    gap = 0.35
    width = (12.1 - gap * (n - 1)) / n
    height = 2.9
    for i, (head, body, colour) in enumerate(s["cards"]):
        left = 0.62 + i * (width + gap)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                      Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = BAND
        card.line.color.rgb = RULE
        card.shadow.inherit = False
        card.text_frame.text = ""
        # head and body share one frame, so a two-line head pushes the body down
        # instead of overlapping it
        tf = textbox(slide, left + 0.25, top + 0.26, width - 0.5, height - 0.5)
        write(tf, [head], size=20, color=colour, bold=True, space_after=12)
        write(tf, [body], size=16, color=INK, line_spacing=1.05, space_after=0,
              first=False)


def render_flow(slide, s):
    slide_title(slide, s["title"])
    steps = s["steps"]
    top, height = 1.55, 1.15
    n = len(steps)
    gap = 0.34                      # room for the arrow glyph between boxes
    width = (12.1 - gap * (n - 1)) / n
    for i, step in enumerate(steps):
        left = 0.62 + i * (width + gap)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                     Inches(top), Inches(width), Inches(height))
        box.fill.solid()
        box.fill.fore_color.rgb = NAVY if i % 2 == 0 else RGBColor.from_string("2F5D8C")
        box.line.fill.background()
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.06)
        tf.margin_right = Inches(0.06)
        tf.margin_top = Inches(0.03)
        tf.margin_bottom = Inches(0.03)
        # one paragraph per line: a literal newline inside a run is not portable
        write(tf, step.split("\n"), size=13, color=WHITE, bold=True,
              align=PP_ALIGN.CENTER, space_after=0, line_spacing=0.95)
        if i < n - 1:
            arrow = textbox(slide, left + width, top, gap, height,
                            anchor=MSO_ANCHOR.MIDDLE)
            write(arrow, ["›"], size=22, color=MUTED, bold=True,
                  align=PP_ALIGN.CENTER, space_after=0)
    bullet_list(slide, s["bullets"], 0.75, top + height + 0.55, 11.9, 3.7, size=19)


def render_statement(slide, s):
    slide_title(slide, s["title"])
    tf = textbox(slide, 1.1, 2.2, 11.1, 3.0, anchor=MSO_ANCHOR.MIDDLE)
    write(tf, [s["statement"]], size=26, color=NAVY, bold=True, line_spacing=1.15,
          space_after=26)
    write(tf, [s["statement2"]], size=22, color=INK, line_spacing=1.15, first=False,
          space_after=0)


RENDERERS = {
    "title": render_title,
    "section": render_section,
    "text": render_text,
    "figure": render_figure,
    "text_figure": lambda sl, s: render_text_figure(sl, s, image_right=True),
    "figure_text": lambda sl, s: render_text_figure(sl, s, image_right=False),
    "two_figures": render_two_figures,
    "figure_plus": render_figure_plus,
    "two_stacked": render_two_stacked,
    "three_images": render_three_images,
    "table": render_table,
    "cards": render_cards,
    "flow": render_flow,
    "statement": render_statement,
}


def main():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]

    missing = []
    for s in SLIDES:
        for path in ([s["image"]] if s.get("image") else []) + \
                    [p for p, _ in s.get("images", [])]:
            if not Path(path).exists():
                missing.append(str(path))
    if missing:
        raise SystemExit("Missing image assets:\n  " + "\n  ".join(missing))

    for i, s in enumerate(SLIDES, start=1):
        slide = prs.slides.add_slide(blank)
        RENDERERS[s["layout"]](slide, s)
        if s["layout"] not in ("title", "section"):
            chrome(slide, i)
        if s.get("notes"):
            slide.notes_slide.notes_text_frame.text = s["notes"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved: {OUT}  ({len(SLIDES)} slides)")


if __name__ == "__main__":
    main()
