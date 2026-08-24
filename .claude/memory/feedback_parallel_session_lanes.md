---
name: feedback_parallel_session_lanes
description: "How to split a big revision across several Claude Code sessions: partition by file ownership, and only ONE session may ever write to the manuscript Doc"
metadata:
  node_type: memory
  type: feedback
---

**The user runs several Claude Code sessions in parallel on this project.** When a job is big enough to
split, hand each session a kickoff prompt that names the files it **owns** and the files it must **not
touch** — otherwise two sessions write the same file and one silently loses.

**The hard rule: only one session may write to the manuscript Google Doc.** Docs edits are index-based, so
a second concurrent writer computes ranges against text the first has already shifted and corrupts both.
Prose passes therefore run **sequentially**, never in parallel. See [[reference_manuscript_gdoc]].

**The split that worked for the Yuval revision (2026-08-13 → 19, all passes closed):**

| Lane | Owns | Ran |
|---|---|---|
| **A · analyses** | stats scripts + new `results/*_V{N}/` dirs | parallel |
| **B · figures** | plotting scripts, `thesis/figures/`, `figure_manifest.md` | parallel |
| **C · literature** | `library.bib`, `library_status.md`, a notes file | parallel |
| **D · prose** | `thesis/chapters/*.md` **and sole access to the Doc** | sequential, 6 sub-passes |

D was split by document section — Methods, Results, Discussion, Introduction, figures/captions, then
front matter — each its own session with a shared preamble.

**Dependency order that matters:** analyses before figures (figures consume the CSVs); literature before the
Introduction (it writes from the ref→section map, not from a bare .bib); Abstract after Results and
Discussion; **front matter and TOC last**, because the TOC depends on final headings.

**Put in every kickoff prompt:** repo conventions (run from repo root, venv `eeg_clean`,
`PYTHONIOENCODING=utf-8` — see [[feedback_pythonioencoding]], versioned output dirs — see
[[feedback_versioned_output_dirs]]), "do not commit", the current scope rule, and a pointer to the record
file the pass must read and write. Tell analysis lanes to **report numbers, not rewrite prose** — anything
they find has to flow through the single Doc-writing session.

**Verify each lane's output on disk before writing the next lane's prompt.** Two prompts of mine named
paths that did not exist (an invented `results/yuval_revision_V11/`, and a C432 composite figure that was
never actually assembled); a 30-second `ls` would have caught both. Also carry forward findings that change
a later lane's job — the ANCOVA turning bandwidth into a duration artefact changed what the Discussion and
Abstract could claim. Related: [[feedback_document_carryover]].
