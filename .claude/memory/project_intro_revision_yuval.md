---
name: project_intro_revision_yuval
description: "Introduction expansion for Yuval's review — APPLIED 2026-08-16; his headings came from Yael Gat's thesis; ref list rebuilt 30 → 49 with Intro refs 1–39 now permanent"
metadata: 
  node_type: memory
  type: project
  originSessionId: f5953e15-b1e5-4783-a40e-a6008e2ab67a
  modified: 2026-08-16T15:17:19.041Z
---

**Introduction pass DONE 2026-08-16.** Email #3 (too ISFS-focused), both margin notes, C571-in-Intro,
and his in-place Intro tracked edits. Applied to the Doc and mirrored to `02_introduction.md` +
`03_methods.md`. Full record: `thesis/reviews/intro_edits_before_after.md`. 782 → ~2,200 words.

**He wrote the section headings himself**, as tracked insertions the triage §3 only summarised.
Extract them by walking `w:ins` over `word/document.xml` (paragraphs 62–84). Final structure, his
order, numbered to match Methods 3.x: **2.1 Human sleep and scalp EEG · 2.2 Infra-slow fluctuation of
sigma power (ISFS) in NREM sleep · 2.3 Changes in sleep and EEG across aging and MCI (8 ¶, one per
sub-topic) · 2.4 Changes in sleep spindles and ISFS across aging and MCI · 2.5 The present study**,
with the Flavio-#6 hook paragraph kept unheaded above 2.1.

**The single most useful finding: his sub-topic list is copied verbatim from [[reference_yael_gat_thesis]].**
All nine headings match her contents page word for word. He was handing over the structure of the
last thesis he supervised on this cohort — so *her* proportions are the yardstick, and her AD /
sleep-in-AD sections are longer than ours. Also: her thesis asserts aMCI is "the preclinical or very
early stage of AD", which is exactly what **C571 now corrects** — he is updating his own lab's earlier
position, not flagging a mistake of Shaked's. Write the C571 paragraph as current evidence, not as a
correction of anyone.

**REFERENCE LIST IS NOW 49 ENTRIES (was 30), and Introduction refs 1–39 are PERMANENT.** The list is
auto-numbered by order of first appearance and the Introduction is the first chapter, so nothing a
later chapter adds can move them. Methods holds 40–44; only **45–49** (Chen, Schmitz, André, Niethard,
Grollero) can still shift. This is why the Intro could be numbered without waiting for the Discussion.

**⚠ `thesis/reviews/discussion_edits_before_after.md` §9 is VOID** — computed against the 30-entry
baseline, assumed "refs 1–26 do not move". A banner on that file lists what changed, chiefly that
**six of its 23 "new" refs are already cited in the Introduction** (`braak2011stages` 25,
`ferman2013nonamnestic` 31, `mitchell2009progression` 32, `malekahmadi2016reversion` 33,
`jicha2006neuropathologic` 34, `drozario2020objectivesleep` 35 — the C571 evidence set, which he asked
for in both chapters) and must not be inserted twice. Projected final count **64**. `galgani2023locuscoeruleus`
was deliberately left for the Discussion, where it serves C557 and C571 at once.

**Renumbering method that worked — reuse it.** The "renumber descending" rule in
[[reference_manuscript_gdoc]] assumes numbers only increase; here `ohayon2004metaanalysis` moved
*down* (25 → 19, first appearance shifted from Results 4.1 to Intro 2.3). Instead **anchor every
replacement on surrounding words**, never a bare glyph — unique strings, so order is irrelevant and
collisions are impossible. 22 sites outside the Intro.

**Prose rules this pass nearly broke** (all caught by an audit before applying, all recorded in
`intro_edits_before_after.md` §9):
- **Stage labels** — `feedback_thesis_prose_rules` requires N1/N2/N3 + REM, "NREM" only for the
  general category. First draft used "slow-wave sleep"; converted to N3.
- **Em dashes** — the Abstract and Discussion were humanized to zero and the Intro had none. First
  draft added seven; all removed.
- **The Lázár mis-citation** — see [[reference_isfs_frequency_attribution]]. **His own tracked edit to
  Intro ¶3 would reinstate it**: he keeps the V1 sentence attributing ~0.02 Hz to Lázár and merely
  splits it. Declined; only "for each EEG channel" was taken. Worth a line in the reply to him.
- Grollero — [[project_thesis]] and [[project_scientific_story]] both say Discussion-only, never
  background. A draft had it in 2.4; the user caught it. Removed.

**Abbreviations, settled by the user 2026-08-16:** define `Alzheimer's disease (AD)` and
`amnestic MCI (aMCI)` **once each**, at first use in 2.3, abbreviate thereafter. Consequence:
**Methods 3.1 no longer defines aMCI and Methods 3.2 no longer expands PSG** (both defined earlier
now). Do not reintroduce either.

Related: [[project_yuval_review]], [[project_bibliography_expansion]], [[project_results_revision_yuval]],
[[project_methods_revision_yuval]]
