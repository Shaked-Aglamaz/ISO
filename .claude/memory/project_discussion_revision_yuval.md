---
name: project_discussion_revision_yuval
description: "Discussion pass for Yuval's review — APPLIED 2026-08-17; ref list 49 → 63 and email #9 cleared; three of his own citations removed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 51608906-5e0e-4335-9089-3ec13373a321
  modified: 2026-08-16T21:20:58.714Z
---

**Discussion pass DONE 2026-08-17.** C557, C558, C571, both halves of his `##` margin note, his
surviving tracked edits, and the bandwidth consequence of his own C429. Applied to the Doc and mirrored
to `05_discussion.md` (+ one sentence of `02_introduction.md`, one citation in `03_methods.md`).
Full record: **`thesis/reviews/discussion_edits_before_after.md`**. Discussion grew 10 → 17 paragraphs.

**REFERENCE LIST IS NOW 63.** Intro **1–39**, Methods **40–45**, Discussion **46–63**. This pass added
17 and **removed 3** — `schmitz2018cholinergic`, `andre2025remslowing`, `niethard2023spindleaging`, each
cited only in the ¶4 that was replaced. `tallonbaudry1997gamma` entered at **Methods 3.4 as ref 42**
(the Morlet "4 cycles" convention had no source), which shifted MNE/Maris/YASA. **Email #9 (≥50) is
cleared.** See [[project_intro_revision_yuval]] for the 30 → 49 step.

**Verify renumbering by script, not by eye.** Parse every superscript run in the body (handling `⁻`
ranges and `,` lists), collect order of first appearance, and assert it equals `1..N` and that N equals
the list length. That single check catches orphans, duplicates and dangling numbers at once. It passed
exactly here. Also check for hyperlink bleed: dump `readDocument` format=json to a file and flag any
reference-entry run that carries a `link` but is not a DOI — appending after a URL-ending paragraph is
the risk, and it did **not** happen this time (the 17 new entries came in plain and were then linked
with `applyTextStyle`, house style `#1155CC` + underline).

**Two content traps worth not re-deriving:**

1. **Grollero reports peak AMPLITUDE, not AUC.** The Doc says so itself in the same paragraph. A draft
   that called their measure "AUC" would have contradicted the manuscript two sentences earlier. The
   final text says their result "has no direct counterpart in our data" rather than forcing an
   equivalence — and that change then required softening "Their result **parallels** our cluster" →
   "**is broadly consistent with**", four sentences earlier in the same paragraph.
2. **The LC does not release acetylcholine** (basal forebrain does). That is *why* the C557 paragraph
   needs `kjaerby2026neuromodulators`: ACh oscillates infra-slowly *under LC control*, so cholinergic
   decline is downstream rather than a rival account. `schmitz2018cholinergic` was still dropped —
   basal-forebrain degeneration is AD-specific and would predict an aMCI effect this thesis did not find.

**Prose rules enforced, all inherited from earlier passes:** zero em dashes; "ISO" only when describing
other groups' work, never ours; "paces" only in rodent contexts (human-facing uses softened to
"organizes"); `locus coeruleus (LC)` defined at first Discussion use, because the Intro defines only
"LC-NE system".

**How the user works, confirmed over ~6 review rounds:** they read the before/after file line by line
and reject anything that (a) cross-references section numbers in prose, (b) defends the thesis against
Yuval's comments inside the text, (c) opens a paragraph with a question, or (d) states a conclusion
twice. They asked for a plain-English `▸ Plain version` bullet box after each drafted paragraph and used
those to check the logic — keep that convention. **§4.3 they rewrote themselves in a separate session**
and it was taken verbatim.

**Left for the reply session** (in `discussion_edits_before_after.md` §10): three of his own citations
now gone; the Discussion openly concedes that CAP separated MCI where the ISFS did not
(`maestri2015capmci`); and Yael Gat's thesis reports an aMCI REM-latency difference our C390 analysis
does not reproduce — same lab, overlapping cohort, and he supervised both. See
[[reference_yael_gat_thesis]].

Related: [[project_yuval_review]], [[project_intro_revision_yuval]], [[project_bibliography_expansion]],
[[feedback_doc_edits_review_first]], [[reference_manuscript_gdoc]]
