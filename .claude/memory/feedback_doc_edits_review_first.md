---
name: feedback_doc_edits_review_first
description: "Never edit the manuscript Doc directly — write a before/after .md with context first, wait for approval; and use [TO SUPPLY] for missing facts instead of stalling"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 98f7363a-0941-45c2-b541-544d8166489a
  modified: 2026-08-16T21:24:54.927Z
---

**Before applying any change to the manuscript Google Doc, write the full before/after into a
markdown file and stop for the user to read it.** Each entry needs the section, a sentence or two of
surrounding context so the location is recognisable, the exact current text, the exact proposed
text, and a one-line reason naming the comment it answers. Only after the user approves does
anything touch the Doc. Instruction given 2026-08-14; the pattern file is
`thesis/reviews/methods_edits_before_after.md`, and `thesis/reviews/flavio_edits_before_after.md`
is the older example of the same format.

**Why:** the user reviews the wording before it lands, and wants to do that *without* having to
Ctrl-F around the Doc to work out where each change sits. Editing first and reporting afterwards
forces them to reconstruct the context from a diff they cannot see.

**How to apply:** treat the review file as the deliverable of the drafting pass. Fold any open
decisions into it as numbered options with a recommendation, so a single read resolves both the
wording and the decisions — rather than asking questions in chat and then producing the file
separately. Mark it APPLIED afterwards and record which option was chosen.

**Sharpened 2026-08-16 — it is step 1 of the plan, not one of the outputs.** A plan for the
Introduction pass listed `intro_edits_before_after.md` in the "files to change" table alongside the
Doc and the chapters, and the user pushed back: *"dont forget to start with the before and after md
(like methods, results, discussion) before touching the doc."* Write the review file **first**, as
its own gated phase, with the Doc and chapter mirrors explicitly downstream of approval. If a plan
is being written, the phase table has to show that ordering — burying it in a deliverables list
reads as "produced along the way".

**Add a `▸ Plain version` box after every drafted paragraph (asked for 2026-08-17, and it worked).**
His words: *"write in the chat a bullet version of this paragraph in simple words so it would be easier
for me to get the flow, and the chronology of the processes"* — then *"add the simple words lists
explanations after the thesis phrasing (only for me)"*. So: numbered bullets, ordinary language, in the
order the argument moves, plus a one-line "changed this round" note. Label them clearly as not-thesis-text.
They are what he actually reads to check the logic, and they surfaced several real errors — the ambiguous
autism sentence and the aMCI-vs-AD framing trap were both caught from the bullets, not the prose.

**Expect ~6 rounds on a chapter, and keep the decision ledger inside the file.** The Discussion pass ran
six review rounds. Carry a `§0` with three tables — *what changed this round*, *what is still open*, and
*what is settled* — and move items between them each round. When everything closes, replace the open
table with an explicit "ALL DECISIONS CLOSED" line so the next session does not reopen them. Answer any
"what is this reference?" question with a real one-paragraph explanation plus a recommendation; he asks
because he is deciding, not because he is testing.

**Companion rule — `[TO SUPPLY]` over stalling.** When a fact is missing (recruitment details,
criteria, a name), write the literal marker `[TO SUPPLY: what is needed]` into the prose and finish
everything else. The user collects the gaps and fills them in a later dedicated session. Do not
block a whole section on one unknown, and do not guess. Grep for the marker at the end and hand back
the list.

Related: [[reference_manuscript_gdoc]] for the Docs API mechanics,
[[project_methods_revision_yuval]] for the pass this came from, [[feedback_diagnose_dont_fix]] for
the same instinct applied to code.
