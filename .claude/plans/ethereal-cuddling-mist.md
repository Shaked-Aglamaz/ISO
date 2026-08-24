# Defense deck: general rules + background rebuild

## Context

`thesis/defense/ISFS_defense_V2.pptx` is a 54-slide draft built from `code/defense/build_deck.py`.
This session fixes (a) the rules that govern every future deck session, and (b) the Background block.
Methods, Results and Discussion are separate sessions.

The Background currently runs spindles → the rhythm → LC → aging → aMCI → aims. That is the thesis'
*section order*, not the *argument* that motivated the study. The rebuild puts the motivation on the
slides in the order it was actually made: rodent ISFS-LC → spindles and memory → memory breaks in aMCI →
the ISFS is being measured everywhere except aging and aMCI → so measure it, and if it is altered the
scalp gives a read-out of the locus coeruleus, a nucleus that is rarely the focus in human sleep work.

**The deck is open in PowerPoint right now** (`~$ISFS_defense_V2.pptx` exists). Output goes to a new
`ISFS_defense_V3.pptx`; V2 is not touched.

---

## Standing rules for all defense sessions (to be saved to memory)

1. **No new literature.** Every citation, figure and concept comes from `thesis/references/library.bib`
   and from the story the thesis already tells. New papers are not introduced for the talk.
2. **Approve before applying.** Every change is written into an `.md` first (this file for this round),
   approved, and only then applied to `build_deck.py` and rebuilt. Same workflow as the manuscript.
3. As few full sentences on a slide as possible; sentences live in the speaker notes.
4. Never overwrite a pptx that PowerPoint holds open; bump the version.
5. `DEFENSE_DATE` = **30 August 2026**.

---

## Answers to your questions

**Is Purcell in the thesis background?** Yes, once: §2.2, `@purcell2017characterizing`, for spindles being
the most reliable signature of N2 and for density/topography varying between individuals. Note that the
*age curves* we show on the aging slide are from the same paper but are not a claim the thesis makes with
Purcell (aging spindles are cited to Mander 2017, Helfrich 2018, Champetier 2023). Same paper, same bib,
so it stays; just be ready to attribute it correctly.

**Slide 6, "our schematic of the published account".** Badly worded caption. It means: *we drew this
figure ourselves to illustrate what the rodent literature reports; it is not our data and not a panel
lifted from a paper*. Being explicit matters here because the panel beside it *is* from a paper. New
caption: **"schematic: sigma power and noradrenaline in anti-phase (after Lecci 2017; Osorio-Forero
2021)"**, and the notes say "our drawing, not our data" in those words.

**Champetier: how the thesis uses it, and the options.** Three citations, all in the Introduction, none
of them about their ISFS result:

| Where | What it is cited for |
|---|---|
| §2.2 | how tightly spindles cluster on the infra-slow timescale relates to overnight consolidation |
| §2.3 | the same, as the reason a rhythm that organizes spindles is of interest |
| §2.4 | temporal clustering of spindles breaks down with age, and that tracks cognitive decline |
| Discussion | **not cited at all** (verified by grep) |

So the thesis never engages their Figure 2, and §2.4 states "the ISFS itself has not been characterized
in older adults or in aMCI", which their Fig 2 partly contradicts: 32 young-middle aged (34.5 ± 10.9 y)
vs 147 older adults, fast-spindle power at C3/C4, FFT, Gaussian fit, peaks 0.021 vs 0.022 Hz,
F(1,175) = 0.164, p = .69.

- **Option A (recommended).** Main slide uses Champetier for exactly what the thesis uses it for: their
  Fig 5A (clustered fast-spindle proportion falls with age) on the aging slide, their Fig 8 (cluster size
  vs overnight memory change) on the memory slide. Their Fig 2 and our four-point answer move to a new
  backup slide **B12**, and stay in `qa_prep.md`. Nothing in the main talk invites the comparison; the
  answer is one click away if it comes.
- **Option B.** Keep V2's behaviour: show their Fig 2 in the Background and raise the null yourself.
  Safest against being ambushed, but it spends main-talk time arguing against a paper the thesis reads as
  supportive, and it tells the room the gap statement was too strong.
- **Option C.** Champetier for clustering only; no Fig 2 anywhere; the answer lives in `qa_prep.md` only.
  Least attack surface, no slide to fall back on.

Yuval read the thesis with this citation in it and did not flag it, which argues for A or C over B.

**Deferred (2026-08-23): you will decide later.** A and C agree on everything in the main talk, so this
round builds what they share: their Fig 5A on the aging slide, their Fig 8 on the memory slide, no mention
of their peak-frequency null anywhere on a slide. The only thing left open is whether backup B12 gets
built (A) or not (C); B is a one-entry revert if you go that way. This sits in **Pending decisions** below.

**Fast spindles only: added to `qa_prep.md`** as an open item for you to answer, with the thesis material
gathered: Mölle 2011 (slow ≈11-13 Hz frontal vs fast ≈13-16 Hz centroparietal, §2.2); Lázár 2019 (the
infra-slow modulation is confined to the high/fast sigma band and centro-parieto-occipital electrodes,
§2.2 and §2.4); Dimitriades 2024 used the fast band and we replicated their pipeline (§2.5, Methods §3.4);
Gorgoni 2016 (the aMCI/AD deficit falls on the same subtype and region, §2.4).

**MCI outcome figure.** Mitchell 2009 (Wiley) and Malek-Ahmadi 2016 are both paywalled, no local PDF, and
their figures are forest plots that do not project. So we draw one from the published numbers, with each
number's citation printed on it. No new papers.

**Raw spindles for slide 4.** Settled 2026-08-23: **you will supply a screenshot from our own data.** For
the record, no figure in Purcell has a time-domain trace; within the bib the only raw-trace candidates were
Andrillon 2011 Fig 1F (raw + 9-16 Hz + envelope, but intracranial) and Mölle 2011 Fig 1 (scalp, but
grand-mean waveforms rather than raw data), so a screenshot of our own N2 EEG is the better panel anyway.
Until the file arrives, slide 4 is built with the Purcell panel alone and its labels; dropping the
screenshot in is a two-line change. This sits in **Pending decisions** below.

---

## The new Background

Slides 3-11 replace slides 3-9. Slide 2 (roadmap) is deleted. Main deck 42 → 43 slides; total 54 → 55
(56 with backup B12).

| # | Slide | Visual | Status |
|---|---|---|---|
| 3 | Section: Background | — | unchanged |
| 4 | Sleep spindles | your Purcell file + Andrillon raw-trace panel | rebuilt |
| 5 | Spindles appear in a rhythm | our timeline + zoom, Lázár topography panel | YA topo removed |
| 6 | Two substates, paced by the locus coeruleus | our schematic + rodent cycle | caption fixed |
| 7 | **The ISFS, measured channel by channel** | Dimitriades Fig 2A+2B + YA AUC topo | **new** |
| 8 | **Why the timing matters** | Champetier Fig 8 | **new** |
| 9 | What aging does to spindles | Purcell age curves + Champetier Fig 5A | reframed |
| 10 | Amnestic MCI | new outcomes figure | figure added |
| 11 | The gap, and why the locus coeruleus | 3 cards | reframed |

Time: background goes from ~10 to ~12 min for 9 figure-led slides. Methods (8 slides) can give back the
two minutes when we get to it.

### Slide 4 — Sleep spindles

Layout `figure` with side labels now, `two_figures` once your screenshot arrives.

- Figure: **`thesis/defense/figure_from_outside/Purcell_2017_spindles.png`** (your crop), caption
  "slow spindles are frontal, fast spindles central-parietal (Purcell et al. 2017)". The old stacked
  crop `purcell_spindle_topo.png` is retired.
- Second panel, **pending your screenshot**: a few spindles in raw N2 EEG from our own recordings, saved
  to `thesis/defense/figure_from_outside/` (or anywhere, tell me the path). When it lands the slide
  switches to `two_figures` and the caption reads "raw N2 EEG, one channel, one of our recordings".
- Labels: `thalamo-cortical bursts, 11 to 16 Hz` · `slow (11 to 13 Hz): frontal · fast (13 to 16 Hz):
  central-parietal` · `tied to consolidation and to sensory disconnection`
- Source line: `Purcell et al. 2017` (the n = 11,630 is dropped from the slide and kept in the notes).

### Slide 5 — Spindles appear in a rhythm

Layout stays `figure_plus`. The Dimitriades young-adult AUC map moves to slide 7; the small panel becomes
**`refs/lazar_iso_topography.png`** (already built), caption "the modulation is confined to fast sigma and
to centro-parieto-occipital electrodes (Lázár et al. 2019)". This keeps the thesis order, Lázár before
Dimitriades, and it is what the current speaker notes already describe. Labels unchanged. The notes lose
the sentence that described the YA map as the Lázár panel.

### Slide 6 — Two substates, paced by the locus coeruleus

Text only: caption 1 becomes "schematic: sigma power and noradrenaline in anti-phase (after Lecci 2017;
Osorio-Forero 2021)"; the notes open with "our drawing, not our data, and no panel is taken from a paper".
The rodent panel still needs Yuval's OK, with the Lecci 2017 spectra as the published fallback.

### Slide 7 — The ISFS, measured channel by channel (new)

Layout `figure_plus`.

- Main image **`refs/dimitriades_fig2AB.png`**: their Fig 2A (mean ISFS spectrum per age group) and 2B
  (peak frequency, bandwidth, AUC per group, with the significance marks). One crop, because A and B are
  one embedded image on p.11 of the preprint.
- Small image **`refs/dimitriades_ya_auc.png`**: the young-adult AUC topography, cut from Fig 2C at
  220 dpi. Caption "young-adult AUC hotspot (Dimitriades et al. 2024)". This replaces the low-resolution
  `code/debug/YA_AUC.png` that slide 5 used; the methods ROI panel (slide 17) can be upgraded to the same
  crop in the methods session.
- Labels: `detectable in every participant` · `peak frequency · bandwidth · AUC` · `all three differ
  across development` · `the pipeline we applied`
- Source: `Dimitriades et al. 2024, Figure 2 (CC BY 4.0)`
- Notes: detectable in at least 20% of electrodes in every individual (74.7 / 75.7 / 82.1 / 80.9% of
  electrodes by group); peak frequency higher in young adults than children and early adolescents
  (F(3,150) = 4.60, p = 0.004); bandwidth late adolescents > children (F = 6.32, p < 0.001); AUC late
  adolescents and young adults > children (F = 5.38, p = 0.002); topographically a frontal seven-electrode
  peak-frequency cluster, and a central AUC maximum, which is the region our ROI came from. Say that the
  method is theirs and that it comes in the methods section, so do not walk through it here.

### Slide 8 — Why the timing matters (new)

Layout `text_figure`, image right.

- Image **`refs/champetier_memory.png`**, their Fig 8: overnight memory change against mean spindle
  cluster size in older adults.
- Short lines: `slow oscillation groups spindles, spindles group ripples` · `plasticity windows, not
  spindle counts` · `tighter clustering, better overnight retention` · `the trains recur on the ISFS
  timescale`
- Notes: systems consolidation and the nested hierarchy (Rasch 2013; Klinzing 2019), the ~50 s train
  framework (Boutin 2020), and Champetier 2023 for the clustering-to-retention link. This is the leg of
  the argument that makes an aMCI cohort the right test: what breaks in aMCI is memory.

### Slide 9 — What aging does to spindles

Layout `two_figures`, unchanged geometry, new right-hand panel.

- Left: `refs/purcell_spindle_age.png` (density falls, fast peak flattens per decade past ~40).
- Right: **`refs/champetier_clustering_age.png`**, their Fig 5A, caption "the proportion of clustered
  fast spindles falls with age (Champetier et al. 2023)". This is the claim the thesis actually cites
  them for, and it replaces their Fig 2.
- Labels: `fewer spindles, lower fast peak past ~40` · `less tightly clustered with age` · `parietal
  fast-spindle deficit in aMCI and AD (Gorgoni 2016)`
- Notes: Mander 2017, Helfrich 2018, Champetier 2023; then one line pointing at backup B12 for their
  peak-frequency null, so you can go there if asked. (Option A above; B and C change this line only.)

### Slide 10 — Amnestic MCI

Layout `text_figure`, image right, the four bullets shortened.

- Image **`mci_outcomes.png`**, drawn by us: a stacked bar of what follows an MCI diagnosis (39.2%
  progress in specialist settings, 21.9% in population samples, Mitchell 2009; about a quarter revert to
  normal cognition, 14% in clinic samples, Malek-Ahmadi 2016), and beside it the aMCI subtype rates
  (17 vs 1.5 events per 100 person-years to probable AD vs dementia with Lewy bodies, Ferman 2013).
  Every number carries its citation on the figure.
- Lines: `complaint plus objective impairment, function preserved` · `amnestic subtype: memory, closest
  to AD` · `most never progress; about one in four revert` · `enriched for AD pathology, not equivalent`
- Notes: Petersen 1999 and Albert 2011 for the two-step diagnosis; Jicha 2006 for 10 of 34 autopsied
  progressors carrying a non-AD primary diagnosis; and the consequence, that a group recruited on
  cognitive criteria is aetiologically mixed enough to dilute a disease-specific effect. How to read our
  null belongs in the Discussion, so do not pre-empt it.

### Slide 11 — The gap, and why the locus coeruleus

Layout `cards`, with a claim line above them.

- Claim: `The ISFS has been measured across development, in mice, and in disease. Not in aging, and not
  in aMCI.`
- Card 1 **Aim 1**: does the ISFS change with healthy aging, in its parameters and in its topography?
- Card 2 **Aim 2**: does amnestic MCI add anything beyond age?
- Card 3 **Why it matters**: the earliest tau pathology is in the locus coeruleus, decades before symptoms
  (Braak 2011). An LC-paced rhythm would be a scalp read-out of a nucleus that is hard to reach in humans.
- Notes: state the hypotheses as posed, no directional prediction; name what was missing (whole-scalp
  parameters, topography, a young reference group on the same system, any aMCI data); and Grollero et al.
  2026 as concurrent and convergent, returning in the Discussion.

### Backup B12 — Champetier's ISFS result (held, pending your call)

`refs/champetier_iso_age.png` (already built) plus the four-point answer from `qa_prep.md`: their means
move the same way ours do; their young reference group averages 34.5 y and reaches 52 against our 27.1;
two central electrodes cannot show topography; and whole-scalp parameters, topography and aMCI were open
either way.

---

## Pending decisions (I will remind you)

Both go into the open-items list of `thesis/defense/deck_outline.md` so the next session sees them.

1. **Slide 4 raw-spindle screenshot from our data.** You are supplying it. Slide 4 ships with the Purcell
   panel alone until then.
2. **Champetier backup slide B12.** Build it (Option A) or leave it out (Option C)? Nothing in the main
   talk changes either way, and `qa_prep.md` keeps the answer regardless.

---

## Files to change

| File | Change |
|---|---|
| `code/defense/build_deck.py` | `OUT` → `ISFS_defense_V3.pptx`; `DEFENSE_DATE = "30 August 2026"`; delete the roadmap dict; rewrite slides 4-9 into the nine entries above; B12 held |
| `code/defense/make_ref_figures.py` | add crops: Dimitriades Fig 2A+B and the YA AUC cell (render p.11 of `ISFS_Development_Dimitriades_2024.pdf` at 220 dpi, page rect 612x792, figure images at y 72-417 and y 417-590 pt); Champetier Fig 5A (p.9, image bbox 60,54-541,291) and Fig 8 (p.10, bbox 310,498-540,662). Note in the docstring that `purcell_spindle_topo.png` is superseded by your own crop |
| `code/defense/make_slide_assets.py` | add `mci_outcomes()` writing `mci_outcomes.png`, same style as `lc_ne_schematic()` (matplotlib, fonts sized for projection, group colours unused here) |
| `thesis/defense/qa_prep.md` | add the fast-spindle question with the four thesis sources; move the Champetier item to "prepared, backup B12" |
| `thesis/defense/deck_outline.md` | new slide map, new numbers table rows, the two standing rules |

Nothing is recomputed: no statistics file, no manuscript figure and no result asset is touched.

## Verification

1. `PYTHONIOENCODING=utf-8 python code/defense/make_ref_figures.py` and `make_slide_assets.py`, then eyeball
   the new PNGs (`refs/_contact_sheet.png` style check) before they go on a slide.
2. `PYTHONIOENCODING=utf-8 python code/defense/build_deck.py` → writes V3, prints the slide count (55, or
   56 with B12). The build already fails loudly on a missing image path.
3. `powershell -File code/defense/render_deck.ps1` (PowerPoint COM; if it produces 0 files, kill the stuck
   `POWERPNT.exe` first) and read slides 1-11 as PNGs: check that no pill box overflows, that the Purcell
   and Andrillon panels are legible at slide size, and that the footer slide numbers are 1-55.
4. `PYTHONIOENCODING=utf-8 python code/defense/dump_notes.py` → refresh `thesis/defense/speaker_notes.md`.
5. Report back the rendered background slides for your review before the methods session starts.
