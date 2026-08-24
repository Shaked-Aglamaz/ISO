---
name: project_defense_deck
description: "Defense-deck pipeline (code/defense/ → thesis/defense/ISFS_defense_V2.pptx); the slide-text rule, why slide figures are regenerated into results/defense_slides_V1, how to render slides for QA on this machine, and the open decisions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1ae97b22-eae8-4b40-a397-c2e972d1ba1a
  modified: 2026-08-23T13:07:51.028Z
---

Thesis defense prep began **2026-08-19** (thesis V2 already sent to Yuval, see [[project_final_pass_sent]]).
Current deck: **`thesis/defense/ISFS_defense_V2.pptx`**, 16:9, 54 slides for a ~45 min talk (42 main +
backup divider + 11 backup slides B1–B11); V1 is the first draft, kept. Statistics are V10 / cohort
35-39-30; nothing is recomputed for the deck. Examiners and expected questions: see
[[project_defense_examiners]].

- **Never overwrite the deck in place while Shaked has it open.** A `~$ISFS_defense_*.pptx` lock file in
  `thesis/defense/` means PowerPoint holds it and his hand edits live only in that session (that is how V1
  ended up with the Hebrew line removed and a date typed in, none of it on disk). Bump to a new V instead,
  and mirror his manual edits back into `build_deck.py`.
- **Open decisions as of 2026-08-20:** the real defense date (set `DEFENSE_DATE` in `build_deck.py`);
  whether to keep the roadmap slide 2 (I recommended cutting it); whether Yuval agrees to the panel from
  his unpublished review on slide 6.

- **Content lives in code, not in the pptx.** `code/defense/build_deck.py` holds a `SLIDES` list (text,
  images, tables, speaker notes) plus renderers; edit and re-run to rebuild. `code/defense/dump_notes.py`
  exports `thesis/defense/speaker_notes.md`, which is the study document for the Q&A-prep sessions.
  `thesis/defense/deck_outline.md` is the slide map plus a number→source table.
- **The V11 manuscript figures cannot go on a slide.** They were rebuilt portrait for a Doc page (the
  three-metric violin is 0.87 aspect). Slide-geometry variants are regenerated into
  **`results/defense_slides_V1/`**: `make_violin_panels.py` (one panel per metric via
  `metrics_filter=[...]`, which needed **no change** to `step6_groups_comparison.py`),
  `make_methods_panels.py` (imports `panel_A/B/C/D` + `prepare_data`/`prepare_panelC` from
  `make_f2_figure.py`, caches the expensive example-subject data under `_cache/`), and
  `make_slide_assets.py` (LC-NE schematic drawn by us, bandwidth × analyzed-N2 scatter, MoCA-grid crops,
  hypnogram crops). The three V11 topo PNGs and `spindle_timeline.png` are already landscape and reused as is.
- **Slide QA renders through PowerPoint, not LibreOffice.** `soffice` and `pdftoppm` are absent on this
  machine, so the powerpoint skill's `pptx_render.py` cannot run. Use
  `powershell -File code/defense/render_deck.ps1` (PowerPoint COM `Presentation.Export`). Gotcha that cost
  time: a COM-started `POWERPNT.exe` survives `Quit()` and the next run silently attaches to that stuck
  instance and produces nothing; the script now records pre-existing PIDs and kills only the one it started.
  If a render produces 0 files, look for a leftover POWERPNT first.
- **Slide-text rule (Shaked, 2026-08-20): as few full sentences on a slide as possible** - figures plus short
  phrases, the sentences go in the speaker notes. Applied to the background slides; methods, results and
  discussion still need the same sweep. Background also starts deeper now (spindle subtypes and topography,
  not "what is N2"), because all three examiners are sleep researchers.
- **Published figure panels** are cropped by `code/defense/make_ref_figures.py` into
  `results/defense_slides_V1/refs/` (Purcell 2017 and Lecci 2017 downloaded from PMC; Champetier 2023 and
  Lazar 2019 rendered from the local PDFs; one panel from Yuval's unpublished LC-NE review, which needs his
  OK). Examiner profiles and the questions to prepare are in `thesis/defense/qa_prep.md`.
- **python-pptx cannot do animations** - no API for the timing tree. Build-ups are done by duplicating a
  slide and adding one element per copy, or added by hand in PowerPoint afterwards.
- **Crop the subject code off the hypnospectrograms** before showing them: `MCI27` is an Elderly participant
  and `YS0` an aMCI patient, so the code names the wrong group (same reason the manuscript figure crops them).
