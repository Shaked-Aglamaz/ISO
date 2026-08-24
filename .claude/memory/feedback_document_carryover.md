---
name: feedback_document_carryover
description: "At the end of a revision pass, write every cross-session finding into the file that session will open — not just into chat"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d88c91d8-c7a4-41f8-92aa-e74d366dffef
  modified: 2026-08-16T15:22:41.730Z
---

**When a pass ends, document everything it hands on to a later session, in the place that session will
actually open.** Asked for explicitly 2026-08-15, after the Results pass: *"make sure that all the comments
you had throughout this session (in the before/after md or in our chat) about other sessions will be well
documented for the relevant sessions."*

**Why:** the thesis revision runs as separate passes (methods → results → figures → discussion → reply),
often days apart and with no shared context. Anything that lives only in chat, or only in a section of a
review file nobody will reopen, is lost. Observations that cost real work to find — a contradiction the
pass created, a stale caption, a numbering rule — are exactly what gets re-derived or missed.

**How to apply:**
- **One master record, per-session sections.** `thesis/reviews/yuval_review_triage.md` is that record: a
  STATUS block at the top, and a `§8 Carry-over` split into §8.1 figures / §8.2 discussion+writing /
  §8.3 reply-to-Yuval / §8.4 anything touching the Doc.
- **Plant a local pointer where the work happens** — a "START HERE" box at the top of
  `thesis/figure_manifest.md`, a `v3` version note on each `thesis/chapters/*.md` the pass touched. A
  future session opens the chapter, not the triage.
- **Correct rows that would mislead.** Marking an item DONE matters as much as adding new ones; a stale
  "OPEN" causes duplicated work.
- **Flag contradictions the pass itself created**, not just leftovers. The sharpest example: answering
  C429 made the Results call bandwidth a duration artefact, which contradicts the Discussion still calling
  it a tendency "pointing the same way" as the age effect.
- Say plainly what was deliberately *not* done and why, so nobody "fixes" it — e.g. the general-condition
  "MCI" mentions left unrelabelled on purpose.

**The mirror rule, added 2026-08-16 — READ the other passes before starting yours.** Carry-over only
works if the next session actually looks. Before planning any chapter pass, **list
`thesis/reviews/*_edits_before_after.md` and check the status line at the top of each**: a file can be
fully drafted and *unapplied*, which the triage STATUS table will not necessarily show. Hit this on
the Introduction pass — `discussion_edits_before_after.md` had been drafted the same day, was
unapplied, and renumbered the same reference list on a now-false premise. It was found only by
noticing the file's mtime while opening a different file. Two passes silently competing for one
auto-numbered list is the failure mode to watch for.

**When your pass invalidates another's plan, write the correction into *their* file, at the top.**
Not a line in the triage, not chat — a banner on the file that session will open, stating what changed
and what to recompute. The Introduction pass put a `🛑 §9 IS VOID` block at the head of
`discussion_edits_before_after.md` listing the six references it must no longer insert.

Related: [[feedback_doc_edits_review_first]] (the review-before-edit rule this pairs with),
[[project_results_revision_yuval]], [[project_methods_revision_yuval]], [[project_yuval_review]],
[[project_intro_revision_yuval]].
