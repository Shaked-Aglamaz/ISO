---
name: reference_yael_gat_thesis
description: "Yael Gat's M.Sc. thesis (same lab, same cohort) — the template Yuval points at for Introduction structure and the 'sleep' definition"
metadata: 
  node_type: memory
  type: reference
  originSessionId: f5953e15-b1e5-4783-a40e-a6008e2ab67a
  modified: 2026-08-16T15:17:37.450Z
---

`thesis/references/YaelG_MSc_thesis.pdf` — **Yael Gat, M.Sc., Sagol School of Neuroscience, TAU,
January 2023, supervised by Yuval Nir.** 52 pages. *"Sleep in healthy elderly and amnestic Mild
Cognitively Impaired (aMCI) patients due to neurodegeneration."* Same lab, same TASMC memory-clinic
recruitment route, overlapping cohort (Omer Sharon collected/scored part of her data).

**This is the template Yuval hands people for Introduction structure.** Her Introduction contents
page is, word for word, the sub-topic list he wrote into Shaked's margin: Sleep · Sleep, Learning and
Memory · Sleep and Aging · Sleep and Neurodegeneration · Alzheimer's Disease (AD) · Sleep in
Alzheimer's Disease · Mild Cognitive Impairment (MCI) · Amnestic Mild Cognitive Impairment · Sleep in
MCI. When he asks for Intro sections, check here first. See [[project_intro_revision_yuval]].

**The "sleep" definition he likes** (p. 8, attributed to Markov & Goldman 2006 = our
`markov2006normalsleep`): *"Sleep is a state of reduced consciousness and immobility, defined by a
reversible disconnection from the environment that is homeostatically regulated."*

**She does NOT list per-stage scoring criteria** — no K-complexes, no 30 s epochs, no AASM rules. She
describes signatures narratively instead (waking EEG slows; high-amplitude >75 µV, 0.5–4 Hz slow
waves; N3 = >20% slow-wave activity; REM = wake-like EEG + rapid eye movements + atonia). Useful
calibration for how much scoring detail an Intro needs at this lab.

**Proportions:** ~9-page Introduction; AD gets ~2 pages (amyloid cascade vs tau, BPSD, cholinergic
basal forebrain, drugs), sleep-in-AD ~1.5. So one paragraph each in the ISFS thesis is *leaner* than
the precedent, not excessive.

**Her framing of aMCI is what C571 now rejects** — she states repeatedly that aMCI is "the preclinical
or very early stage of AD" / "widely considered as a prodromal phase of AD", with conversion rates
30–60% at 3 y and up to 80% at 6 y (Morris 2001). Yuval's C571 is him **updating his own lab's earlier
position**, not correcting Shaked. Expect the conversion numbers not to match ours
(`mitchell2009progression`: 39.2% specialist / 21.9% population, from 41 inception cohorts).

**Her result, for context:** aMCI vs age-matched healthy controls differed *only* in REM sleep
(longer REM latency), and REM integrity correlated with overnight memory — a REM-side null-elsewhere
result structurally similar to the ISFS thesis's aMCI = elderly finding.

**No PDF renderer on this machine** (`pdftoppm` missing, so the Read tool cannot page it). Extract
text with `pypdf`, which is installed both system-wide and in `eeg_clean`.
