---
name: project_front_matter_pass
description: "Front-matter pass DONE 2026-08-17 — title page, Hebrew title page + תקציר, acknowledgements, TOC page, Abstract rewrite, Discussion 5.1–5.7; TOC inserted and thesis SENT 2026-08-19, only the reply to Yuval is left"
metadata: 
  node_type: memory
  type: project
  originSessionId: 595a7c0c-ce82-4ca2-bdb0-be4046b226c1
  modified: 2026-08-17T06:05:39.497Z
---

**The last content pass on the thesis is DONE (2026-08-17).** Record:
`thesis/reviews/front_matter_edits_before_after.md` (§11 is the closing state; §1–§10 describe the
*proposal*, which was amended). Triage status table is fully green — **emails #1–#9 and all 17 doc
comments resolved**. The only remaining pass is **the reply to Yuval** ([[project_yuval_review]] §8.3,
now carrying six new items 8–13).

What went in, in [[reference_manuscript_gdoc]]:

- **Title page** — his verbatim correction: `Department of Neuroscience and Brain Disorders` +
  a new third line `Gray Faculty of Medical and Health Sciences`. Date `July, 2026` → **August, 2026**.
- **Hebrew title page** (page 2, not his ask — Shaked approved it because the TAU house format pairs
  it with the Hebrew abstract). Official Hebrew, verified on TAU's site, not translated by ear:
  `בית הספר סגול למדעי המוח` / `החוג למדעי העצב ומחלות נוירולוגיות` /
  `הפקולטה למדעי הרפואה והבריאות ע"ש גריי`. **Careful: TAU's official Hebrew for that department
  back-translates as "Neuroscience and Neurological *Diseases*", not "Brain Disorders".**
- **Acknowledgements** (page 3) — built from his 5-name stub, **plus three he omitted**: Angela
  D'Rozario and Rick Wassing (Sydney/CIRUS recordings) and **Maria E. Dimitriades for the analysis
  pipeline and personal help**. No funding statement — nothing in the project records a funder.
- **Abstract** — **his whole tracked rewrite adopted**, with four deliberate departures: bandwidth
  carries the C429 duration caveat; `(aMCI)` defined at first use; `(smeared)` dropped as a coinage;
  mean ages **27.1 / 66.5 / 67.8** because **his "67" is wrong** (true mean 67.8). Shaked also added
  `amnestic` to the research question and the closing, and had the "characterized in young adults"
  sentence name the other populations. **His rewrite deletes the "sensitive read-out" closing
  sentence** — the locked ending of [[project_scientific_story]] is therefore gone from the Abstract.
- **תקציר** — Hebrew abstract, own page, right after the English Abstract (Yael Gat's layout, see
  [[reference_yael_gat_thesis]]). `NREM` left untranslated as she did; `תחום סיגמא` not `פס סיגמא`;
  `הלוקוס סרולאוס` not `קוארולאוס`.
- **Discussion subheadings 5.1–5.7** — Shaked's request, so the TOC is not one bare row. Topic-style,
  numbered to match 2.x/3.x/4.x. **No prose rewritten**; ¶1–¶2 stay unheaded.

**TOC: DONE 2026-08-19** — Shaked inserted it by hand and sent the thesis (see
[[project_final_pass_sent]]). Kept below because the constraint recurs on every future version:
the TOC list itself is a manual step. **No API — MCP or raw Docs — can create
a table of contents.** The page is prepared after the Acknowledgements; Shaked clicks
Insert → Table of contents (with page numbers). `Acknowledgements`, `Table of Contents` and `תקציר`
are deliberately **bold centred NORMAL_TEXT, not HEADING_2**, so the generated list starts at
`Abstract`, matching the model Yuval pasted (which is Yael's contents page, complete with a stray
one-character row `T`).

Declined, on the scope rule: student ID on the title page, and the S4 ethics/consent/COI statements.
Pre-existing oddity left alone: two English title-page paragraphs are already `RIGHT_TO_LEFT`
(invisible, both centred).

**Verified 2026-08-19, when the TOC was finally inserted:** the heading styles are all correct and the
generated list populates properly — title = HEADING_1, the six chapters (Abstract, Introduction, Methods,
Results, Supplementary figures, References) = HEADING_2, all 25 numbered subsections 2.1–5.7 = HEADING_3.
`Acknowledgements`, `Table of Contents` and `תקציר` are confirmed still NORMAL_TEXT, so the list does
start at `Abstract` as intended. **One consequence not anticipated above:** because the thesis title is
HEADING_1, it appears as the top row of the generated TOC. Restyling the title paragraph to `Title`
instead of `Heading 1` drops it while looking identical — flagged to Shaked, who left it as-is.

Reference list unchanged at **63**, invariant re-verified after the pass.
