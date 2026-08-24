# Final revision session — front matter, Abstract, Hebrew abstract, TOC

## Context

Yuval Nir returned a marked-up review of the thesis on 2026-08-11 (`thesis/reviews/Shaked's Thesis_YN.docx`,
17 comments + 364 tracked insertions). Six revision passes have since closed the Methods, Analysis,
Results, Introduction, Figures and Discussion items. One row of the status table in
`thesis/reviews/yuval_review_triage.md:20` is still **OPEN**:

> **Writing, remaining** (Hebrew abstract, TOC, Abstract additions, title page, acknowledgements)

This session closes that row. Everything here is something Yuval explicitly asked for — email items
#1 (missing TOC) and #2 (Hebrew abstract), and his tracked front-matter insertions (title-page
department correction, Acknowledgements stub, Abstract rewrite). Scope rule for this revision stands:
**only what he asked**; ⚑ items from our own internal check stay out.

Manuscript = the Google Doc **"Shaked's Thesis V2"**, `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`.
Edit in place, never export a local copy. Mirror all prose into `thesis/chapters/*.md`.

---

## Decisions taken (this session, by Shaked)

| Question | Answer |
|---|---|
| Abstract | Adopt **his whole rewrite**, adapted to V2 facts |
| Bandwidth in Abstract | Keep, with the N2-duration caveat |
| Title-page date | **August, 2026** |
| Table of contents | **Native auto-TOC** (Shaked clicks Insert → Table of contents) |
| Acknowledgements | Use the drafted prose below; **no funding statement** |
| Out-of-scope extras | **Hebrew title page: yes.** Student ID: no. Ethics/consent statements: no |

---

## Current state of the Doc

Page break at index 298 separates the title page from `Abstract` (HEADING_2 at 300). Body is
HEADING_2 / HEADING_3 / NORMAL_TEXT / TITLE only. 63 reference entries. Second page break at 62241
before `Supplementary figures`.

```
TITLE PAGE  →  [pagebreak 298]  →  Abstract  →  Introduction (2.1–2.5)  →  Methods (3.1–3.7)
            →  Results (4.1–4.6)  →  Discussion  →  [pagebreak]  Supplementary figures  →  References
```

Target structure, following the TAU house format of `thesis/references/YaelG_MSc_thesis.pdf`
(the thesis Yuval modelled his pasted example on — same lab, same supervisor):

```
English title page → Hebrew title page → Acknowledgements → Table of Contents
→ Abstract → תקציר → Introduction → … (unchanged)
```

---

## Work items

### 1. Title page (his tracked correction, verbatim)

`thesis/chapters/00_front_matter.md` is a stale 2026-05-27 placeholder — the live title page exists
only in the Doc. Replace both lines:

| Before | After |
|---|---|
| `Department of Physiology and Pharmacology (Medicine)` | `Department of Neuroscience and Brain Disorders`<br>`Gray Faculty of Medical and Health Sciences` |
| `July, 2026` | `August, 2026` |

Keep the existing 14 pt / CENTER run styling. Everything else on the page is untouched — he edited
nothing else there.

### 2. Hebrew title page (new page 2)

Mirror of the English page, RTL, following Yael Gat's page 2 exactly in structure:
school / department / faculty / title / `מאת` / name / `החיבור בוצע בהנחייתו של` / supervisor / date.

- **Verify the official TAU Hebrew names** for "Department of Neuroscience and Brain Disorders" and
  "Gray Faculty of Medical and Health Sciences" before writing them — do not translate by ear.
  Yael's page carries the old `המחלקה לפיזיולוגיה ופרמקולוגיה (רפואה)`, which no longer applies.
- Draft the Hebrew title, put it in the review file, and get Shaked's sign-off before it enters the Doc.
- RTL is **not settable through the MCP** (`applyParagraphStyle` has no direction field). Set
  `paragraphStyle.direction = RIGHT_TO_LEFT` through a direct Docs API `batchUpdate` — token-refresh
  recipe is in the `reference_manuscript_gdoc` memory and is already proven working this session.

### 3. Acknowledgements (new, page 3)

His stub is skeletal (`I would like to thank … / e.g. / Noa Bregman … for`). Approved prose:

> I would like to thank Prof. Yuval Nir for his guidance, his scientific standards, and for trusting
> me with this project; Noa Bregman for professional support at the Tel Aviv Sourasky Medical Center,
> and for referring patients to the study; Rivi Tauman and Jenny Zitser for their guidance in sleep
> medicine and PSG monitoring; Rotem Falach and Flavio Schmidig for teaching me EEG analysis and for
> their advice throughout; and the members of the Nir lab for their help and good company. Finally, I
> thank the participants and their families for their time and willingness to take part.

No funding line. All five of his names are preserved in the roles he assigned them.

### 4. Table of Contents (email #1)

The Docs API has **no request type that creates a TOC** — only a person clicking
Insert → Table of contents can. So:

- I insert a `Table of Contents` page after the Acknowledgements and confirm every heading is a real
  HEADING_2/HEADING_3 (verified: they all are, and the Discussion is a single unbroken block with no
  subheadings, which is correct and matches the chapter file).
- **Shaked's one manual step:** put the cursor on that page, Insert → Table of contents → *with page
  numbers*.
- `Acknowledgements`, `Table of Contents` and `תקציר` are styled as **bold centred NORMAL_TEXT, not
  HEADING_2**, so the generated TOC starts at `Abstract` — matching the model he pasted, whose first
  row is `Abstract 6`.

### 5. Abstract (his tracked rewrite, adapted)

Replaces the single paragraph at doc index 309 and line 7 of `thesis/chapters/01_abstract.md`.
His rewrite accepted wholesale, with three adaptations:

- **bandwidth** — his `and with a trend towards broader bandwidth (p = 0.061)` becomes the caveated
  clause, per C429 / Results 4.2 (`04_results.md:23`), which says it "should not be read as an effect of age";
- **`(aMCI)` defined at first use** — his text says "amnestic MCI" then later "aMCI". The abstract is
  standalone, so it carries the definition (abbreviation policy, triage §8.2.4);
- **mean ages given to one decimal** — he wrote 27 / 66 / 67; the true values are **27.1 / 66.5 /
  67.8**, so his third figure is wrong by a year on rounding. One decimal matches Methods 3.1 exactly
  and avoids the dispute. *Flag in the review file; revert to bare integers if you prefer his form.*

> During non-rapid eye movement (NREM) sleep stage 2 (N2), the occurrence of sleep spindles in the
> sigma (13–16 Hz) band fluctuates following an infra-slow timescale of roughly 0.02 Hz (oscillation
> with period of ~50 sec). This infra-slow fluctuation of sigma power (ISFS) is associated with
> changes in locus-coeruleus noradrenergic (LC-NE) activity and other arousal systems and with
> related autonomic measures. In humans, the ISFS has been characterized in young adults. Here, we set
> out to investigate whether the ISFS changes with aging and mild cognitive impairment (MCI). To this
> end, we performed full-night polysomnography (PSG) including high-density (256-channel) EEG in 35
> young adults (mean age 27.1), 39 healthy older adults (mean age 66.5), and 30 patients with amnestic
> MCI (aMCI) referred from a cognitive neurology clinic (mean age 67.8). We characterized the ISFS
> during N2 sleep by applying an established analysis pipeline to the EEG data, quantifying the peak
> frequency, bandwidth, and area under the spectral peak (AUC) of the sigma-envelope spectrum at every
> scalp EEG channel. We found that older age was associated with a significantly higher peak frequency
> (p = 0.0026); bandwidth showed a tendency toward broader peaks (p = 0.061), but this tracked how much
> N2 sleep each participant contributed rather than age. Overall AUC across all scalp channels was
> preserved, but the central-parietal ISFS hotspot, prominent in young adults, was significantly less
> focal in older age (cluster p = 0.023). We could not reveal significant differences in ISFS
> parameters between individuals with aMCI and healthy older adults, or correlations between ISFS
> parameters and cognitive status. Thus, the ISFS is reshaped by aging, becoming faster and losing its
> central-parietal focus, rather than by mild cognitive impairment.

Two of his words dropped, both noted in the review file: **"(smeared)"** after "less focal" (a
coinage, against the recorded prose rules) and the sentence he deleted, *"The ISFS is a sensitive
read-out of how aging reorganizes the machinery that generates spindles…"* — deleting it was his
edit and the chosen option, but it is the locked closing of `project_scientific_story`, so it goes in
the reply to him.

### 6. Hebrew abstract — תקציר (email #2)

Placed immediately after the English Abstract, before the Introduction — Yael's page 7. Heading
`תקציר`, RTL, same batchUpdate mechanism as item 2. **Translated from the final English abstract in
item 5, so it is written last**, and Shaked reviews the Hebrew before it enters the Doc.

---

## Files

| File | Change |
|---|---|
| Google Doc `1YpXrDGFlzRk…` | all six items above |
| `thesis/reviews/front_matter_edits_before_after.md` | **new** — the approval record, in the format of the five existing `*_edits_before_after.md` passes |
| `thesis/chapters/00_front_matter.md` | replace the stale placeholder with the real title page, Hebrew title page, acknowledgements, and a TOC note |
| `thesis/chapters/01_abstract.md` | new abstract body + a `v4 (2026-08-17)` provenance note; add the `תקציר` under its own heading |
| `thesis/reviews/yuval_review_triage.md` | status table row → APPLIED; §8 carryover for the reply session |

---

## Order of operations

Docs indices shift on every insert, so **work backwards through the document**:

1. Write `front_matter_edits_before_after.md` with every before/after, the Hebrew drafts, and the
   flagged items. **Wait for approval** — per the standing rule, nothing enters the Doc first.
2. תקציר at index 1747 (start of the `Introduction` heading) → restyle the inserted paragraphs to
   NORMAL_TEXT, then set RTL. Inserting at a heading's start makes paragraphs inherit the heading
   style; this is expected and is fixed with `applyParagraphStyle`.
3. Abstract body: replace the paragraph at 309.
4. Front-matter block at index 300: Hebrew title page + Acknowledgements + Table of Contents page,
   with `insertPageBreak` between each.
5. Title-page department / faculty / date (indices 34–88, 288–300).
6. Mirror everything into the three markdown files; update the triage.

---

## Verification

- **Re-fetch the Doc as JSON via the Docs API** (same script pattern used to survey it this session)
  and assert the paragraph order is: English title page → Hebrew title page → Acknowledgements →
  Table of Contents → Abstract → תקציר → Introduction, with page breaks between each.
- **Nothing else moved:** diff the extracted body text against the pre-edit dump already captured at
  `%TEMP%\gdoc_v2.json`; the only differing ranges must be the ones edited.
- **Reference invariant still holds:** regex every superscript run over `[⁰¹²³⁴⁵⁶⁷⁸⁹⁻,]+`, expand `⁻`
  ranges and `,` lists, and assert the n-th distinct number encountered top-to-bottom is n, with
  N = 63 list entries. This catches any accidental damage to the citation stream.
- **Hebrew renders RTL:** confirm `paragraphStyle.direction == "RIGHT_TO_LEFT"` on every Hebrew
  paragraph in the re-fetched JSON.
- **TOC:** after Shaked inserts it, re-fetch and confirm the generated TOC's first row is `Abstract`
  and that all 5 Introduction, 7 Methods and 6 Results subsections appear.
- Confirm the abstract paragraph is still `JUSTIFIED` / 12 pt and that no run picked up stray bold.

---

## Spotted, not done (per the scope rule)

- **Student ID on the title page** — the TAU template carries it under the author's name; ours does
  not. Declined this session.
- **Ethics / informed-consent / conflict-of-interest statements** — long-standing finding S4
  (`thesis/final_check_report.md:106`). Still absent; a two-site human study normally carries them.
  Declined this session.
- **Yuval's copy has the old title.** He has never seen *"Aging, but not amnestic mild cognitive
  impairment, reshapes the infra-slow rhythm of sleep spindle power."* Already logged for the reply
  session (triage §8.3.6); not touched here.
- **His pasted TOC contains a stray one-character row `T`** and mis-levels `Sleep in MCI`,
  `Supplementary` and `References`. Artefacts of his copy-paste; the native TOC makes them moot.
