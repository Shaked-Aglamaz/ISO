---
name: feedback_paper_format_pivot
description: "Writing-format decision (2026-05-28) — paper-style not full thesis. IMRAD, ~6000 words, Intro ~1.5 pages, no separate Conclusion, no full front matter, no TAU template"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d7e0bf61-82d0-4afc-bb1a-ad690a607b82
  modified: 2026-08-12T10:13:19.588Z
---

> **REVERTED 2026-08-12 — the deliverable is the FULL THESIS again.** Yuval's review demands TOC, Hebrew
> abstract, acknowledgements, a thesis-scale Introduction (sleep/functions, PSG + EEG signatures, memory,
> aging, AD, MCI/aMCI, sleep in MCI) and ≥50 references — and his own margin note reads *"for thesis
> (unnecessary for paper)"*. The paper-format decision below was Shaked's own call and was never agreed with
> the PI, which is why the draft he received had no front matter. **Everything below is historical**; apply
> thesis proportions, not paper proportions. See [[project_yuval_review]].

**The write-up is paper-format, not a full Master's thesis.** _(superseded — see the note above)_

Decided 2026-05-28 during session kickoff. Replaces the earlier "~50–80 page TAU thesis" assumption in the planning document.

Concrete implications:
- **Length target:** ~6000 words total (journal-paper ballpark). Intro target: ~1.5 pages.
- **Structure:** IMRAD only — Introduction, Methods, Results, Discussion. No separate Conclusion chapter (fold a short closing paragraph into Discussion).
- **Front matter:** paper-style title block (title + author + affiliation + abstract). No declaration page, no acknowledgments page, no table of contents.
- **Appendix:** lean supplementary materials only. Anything that belongs in a thesis appendix needs to justify its place in a paper.
- **No TAU template** is needed. Output is one paper-style PDF + .docx via Pandoc.
- **Chapter stubs** in `thesis/chapters/00–08*.md` will be restructured at the start of Phase 2 — keep the existing skeleton on disk but rewrite for paper format when drafting begins.

**Why:** Shaked decided a paper-like structure is the right scope — shorter, tighter narrative, also positions the work for journal submission later. The full-thesis scope was over-engineered for what the analysis supports.

**How to apply:** Whenever drafting prose or proposing structure, target paper-paper proportions (Intro short and funneled, Methods compact and methods-first, Results figure-driven, Discussion focused on a few key implications + limitations). If the user reverts to full-thesis later, update this memory.

Related: [[project_thesis]], [[feedback_thesis_prose_rules]]
