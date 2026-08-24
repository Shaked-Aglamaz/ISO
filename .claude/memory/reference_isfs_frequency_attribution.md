---
name: reference_isfs_frequency_attribution
description: "Who reported which ISFS rate: Lázár 2019 = ~0.01 Hz for sigma power, ~0.02 Hz for spindle events; the 0.02 Hz belongs to Lecci 2017"
metadata: 
  node_type: memory
  type: reference
  originSessionId: caaebecc-a92c-4f35-893a-dd34fd3458c1
  modified: 2026-08-16T15:22:54.781Z
---

**A mis-citation that was in the manuscript Introduction until 2026-08-06.** The text read "the ISFS appears during N2 as a modulation of the sigma amplitude envelope at approximately **0.02 Hz** … [Lázár]" — but that is **not** what Lázár, Dijk & Lázár 2019 report.

Their actual findings (`thesis/references/ISFS_humans_Lazar_2019.pdf`, *J Neurosci Methods* 316:22–34, "Infraslow oscillations in human sleep spindle activity"):
- Abstract, verbatim: *"We confirm the existence of ISO in sigma activity albeit with a frequency **below** the previously reported 0.02 Hz."* Their dominant rate for the **sigma-power** fluctuation is **≈ 0.01 Hz**.
- They **do** recover ≈ 0.02 Hz, but from a **different measure**: individual spindles reduced to a binary on/off event signal (their methodological novelty), contrasted against permuted spindle/inter-spindle intervals.
- They attribute the **0.02 Hz** figure to **Lecci et al. 2017** (our reference 6, `lecci2017infraslow`) — mice and humans.
- Sample: **34 healthy young adults**, baseline sleep. Stages: NREM **stage 2 and slow-wave sleep** (not N2 only). Band: fast sigma **13–15 Hz**, effect strongest in high sigma. Topography: **centro-parieto-occipital**, left hemisphere, stronger in the second half of the night.

The fixed Introduction sentence now says the rate "depends on what is measured", placing the sigma-power estimate somewhat below 0.02 Hz and the spindle-event estimate near it. The user chose to **drop the sample size and the author names** (bare superscripts).

**The other "roughly 0.02 Hz" statements in the manuscript are correct and were left alone** — Abstract, Intro ¶3 and the Discussion describe *our own* measurements (young 0.0199 Hz, older ≈ 0.023 Hz) and the rodent literature, both genuinely ≈ 0.02 Hz. Only the Lázár attribution was wrong. Useful because it pre-empts the obvious examiner question: why do our peaks sit near 0.02 Hz when the paper cited for the human phenomenon reports 0.01 Hz for the same quantity.

**⚠ Yuval's returned .docx contains an edit that would reinstate the error — do not apply it.**
Found 2026-08-16. His tracked change to the Introduction's human-ISFS paragraph keeps the V1 sentence
*"the ISFS appears during N2 as a modulation of the sigma amplitude envelope at approximately
0.02 Hz"* with Lázár attached, and merely splits it in two (*"…0.02 Hz. **Such modulations are** most
prominent in fast spindles…"*). He reviewed V1, so he never saw the correction. The Introduction pass
declined the restructure and took only his *"for each EEG channel"* phrasing. If a later pass
re-applies his Intro edits wholesale, this is the one that must be skipped — and it is worth a line in
the reply to him, since it also answers the examiner question above before he asks it.

See [[project_flavio_review]] (#9), [[reference_dimitriades_citation_status]],
[[project_intro_revision_yuval]].
