# Triage Yuval's review against V2, and draft the reply

## Context

The manuscript was sent to Yuval on 2026-08-12 as the Google Doc **"Shaked's Thesis V2"**
(`1YpXrDGFlzRk…`). He returned `thesis/reviews/Shaked's Thesis_YN.docx` — **an export of V1**, the same
pre-Flavio draft Flavio reviewed — plus a 9-item list in the email body. The worry is that a large part of
the review is already obsolete.

**It mostly isn't.** The version mixup cost real work, but it did not invalidate the review:

| | Yuval's items | Resolved by V2 | Partly | Still open |
|---|---|---|---|---|
| Email list | 9 | 0 | 1 (intro, marginally) | 8 |
| Doc comments | 17 | 2* | 1 | 14 |
| Tracked line-edits | 364 ins / 111 del, 30 paragraphs | see below | | |

\* the two are gaps rather than comments: Methods 3.3 and Results 4.7 are **empty** in his copy and are
full sections in V2.

The redundancy is concentrated in his **in-place line-edits**, where Flavio's V2 rewrite had already made
several of the same changes independently (Results openers, Discussion "First… Second…", FFT spelled out).
Those are worth listing to him precisely, because they explain why his edits will not appear verbatim in
the next version he sees.

**Format decision (this session):** Yuval is reviewing this as a **thesis**, not a paper — TOC, Hebrew
abstract, acknowledgements, expanded intro, ≥50 references, and his own margin note *"for thesis
(unnecessary for paper)"*. The 2026-05-28 paper-format decision is **reverted**: full thesis format is the
deliverable. This alone accounts for 4 of his 9 email items, and `feedback_paper_format_pivot` memory must
be updated.

### Evidence that his copy is V1 (for the reply, and so we never re-litigate it)

- Abstract still has the pre-Flavio *"has been characterized almost exclusively in young adults, leaving
  open whether…"* framing and the *"tempers its use as a standalone marker"* closer (Flavio #4 replaced both).
- Results 4.1 and 4.2 unmerged; old method-first headings (V2 merged them, Flavio #46, and renumbered to 4.1–4.6).
- Methods 3.3 empty (V2: full N2-bout section, Flavio #17). Results 4.7 reduced to a bare `.` — Flavio #73
  anchored to a real sentence there, so the text existed in V1 and was removed in his copy, probably edited
  with Track Changes off.
- Methods 3.6 still carries the display/interpolation/MoCA paragraph that Flavio #39 relocated to Results.
- None of the four pre-send fixes are present: Niethard still ref 15 (S3), no "total sleep time under
  210 min" clause (S5), Fig 1 caption still says both older groups have shorter bouts (S8), Results 4.3
  lacks the "mean of the per-subject means" sentence (B2).

## Deliverables

1. **`thesis/reviews/yuval_review_triage.md`** — every item (9 email + 17 comments + his inline `##` notes +
   the tracked-edit paragraph map), each with: verbatim text, anchor, **V2 status** (resolved / partly /
   open), what it costs to fix, and where it overlaps Flavio's review or our own `final_check_report.md`.
   This is the working checklist for the revision *and* the evidence base for the reply.
2. **`thesis/reviews/yuval_reply_draft.md`** — the email, for Shaked to edit and send.

Both new files; nothing else is edited in this task.

## Steps

### 1. Pin down V2's exact text (input for the diff)

The triage below was built from `thesis/chapters/*.md` (v2, Flavio applied) + `thesis/final_check_report.md`,
which match the doc on all four applied fixes and on every sentence the check quotes. To make each
"already fixed" claim airtight against the file Yuval will actually receive:

> **Ask Shaked to download "Shaked's Thesis V2" as .docx into `thesis/reviews/Shaked's Thesis_V2.docx`**
> (File → Download → Microsoft Word). No session restart needed, and it is more reliable than the MCP
> (`readDocument` hides table cells). The `google-docs` MCP server is configured in `~/.claude.json` but did
> not start this session; a restart is only worth it if we later want to apply edits in the doc directly.

Fall back to the chapters if he'd rather not — the residual risk is mislabelling doc-only wording drift.

### 2. Extract and align both documents

Reuse the extraction already working in this session (zip + `ElementTree` over `word/document.xml`,
walking `w:ins` / `w:del` / `w:delText` / `w:commentRangeStart`, and `word/comments.xml` for author + date +
anchored span). Same recipe as the Flavio pass — per `project_flavio_review`, **a returned .docx always
needs the tracked changes read, not just the comments**.

Then align V1 ¶ ↔ V2 ¶ by fuzzy match (`difflib.SequenceMatcher` on normalized text) and classify each
paragraph Yuval touched as **unchanged / lightly edited / rewritten in V2**. A tracked edit landing on a
rewritten paragraph is where redundancy lives; one landing on an unchanged paragraph still applies verbatim.

### 3. Write the triage doc

Findings already established, to be carried in rather than re-derived:

- **C395 — "interpolate before average referencing, or re-run".** No re-run needed, and the manuscript's
  description is accurate. `code/step2_auto_bad_channels.py:402-411` sets `info['bads']`, then
  `set_eeg_reference('average', projection=False)`, then `interpolate_bads`. Verified empirically on MNE
  1.6.1 in `eeg_clean`: with 4 channels at 1/2/3/100 µV and `D` marked bad, the output is
  `[-1, 0, 1, 100]` — i.e. the average is over **good channels only** (mean 2.0), and the bad channel is
  left untouched until interpolation. The noise he is worried about never enters the reference. Offer a
  sensitivity re-run on 2–3 subjects (interpolate-then-reref) if he wants it in writing, and add one
  Methods clause saying bads are excluded from the average.
- **C429 — "how do we know the differences aren't the amount of N2?"** He is right and it is the same
  problem our own check raised as **S7** (still pending). The paragraph's logic is a non-sequitur, and N2
  share *does* differ across groups (KW p = 0.0009; young 34.1 % vs elderly 44.5 %, p = 0.0006 — never
  mentioned). The two facts that do support the claim are already computed: proportion of each subject's N2
  retained as clean bouts is equal across groups (ANOVA p = 0.98) and total analyzed bout duration does not
  differ (KW p = 0.081). His second ask — include N2 amount **as a factor** — is a new ANCOVA on the
  whole-scalp parameters, cheap to run.
- **C390 — sleep efficiency / WASO / REM latency.** Sleep efficiency is already in the subjects sheet
  (`sleep_efficiency_pct`, used by `code/sleep_stage_pies.py:61`). WASO, SOL and REM latency are not
  extracted anywhere but come straight from the existing hypnograms via `yasa.sleep_statistics`. Feasible.
- **C389 + email #5 — Fig 1 panel C.** `code/make_f1_figure.py` pastes the table as a bitmap panel in a
  13 × 16.3 in composite (`fig.text(..., "C)", ...)` at y = 0.318). Promote it to a standalone **Table 2**
  built by `code/n2_bouts_table.py`, and fold in the C390 metrics. This satisfies both his asks at once.
- **C432 — per-subject example spectra.** Nearly free: `code/find_example_gaussians.py` /
  `find_clean_gaussian.py` already produce this figure (see `reference_rd43_alias_and_example_gaussians`).
- **C345 / his `> X% / > Y% / > Z%` table inserts.** Our own **S11** already flags that the semi-automatic
  bad-channel/epoch criteria are never stated. The thresholds live in `code/step1_auto_cleaning.py`
  (`GFP_SPIKE_MADS`, `PTP_THRESHOLD`, `OUTLIER_TIME_FRACTION`, …) and CLAUDE.md documents them.
- **C344 / his apnea note** (*"You can't do sleep research in elderly without addressing this"*) — the
  hardest item. Nothing in the repo touches AHI, apnea, or respiratory channels (grep is empty), and the
  subjects sheet exposes no such column. Needs Shaked to say whether respiratory data or clinical AHI exists
  at either site; if not, this becomes an explicit limitation. Same paragraph is our **S6** (MCI never
  defined diagnostically).
- **C571 + email #8 — aMCI is not just early AD.** Open. V2 still says *"perhaps because MCI is an earlier
  disease stage"* (Discussion ¶15). Note that V2 *did* add Flavio's alternative reading (#112), which is
  adjacent but not the same point. He also wants it fixed in the Intro, and he silently inserted "**a**MCI"
  throughout — a relabelling pass to *amnestic* MCI is implied.
- **C557 / C558 — Discussion mechanism.** Both open, verbatim in V2 ¶11. He names the replacements: the LC
  degeneration review (from Noa R.) and Omer's slow-wave paper — the latter is already ref 20
  (`sharon2025slowwaves`), cited only for the cohort.
- **Redundant, and worth naming to him:** Results 4.3–4.6 openers (his "First, we examined…/Next, we
  examined…" is V2's "We first asked…/We then asked…"), his rewrite of "On every parameter … indistinguishable"
  (V2: "Neither peak frequency, bandwidth, nor strength differed significantly between the elderly and MCI
  groups"), Discussion ¶1 "First… Second…" (V2 already reads that way), FFT spelled out at first mention
  (Flavio #25), "clean **N2** bout" in Methods 3.4, "Electrical Geodesics, Inc. (EGI)" at first mention,
  and "~50 sec" in the abstract (V2: "roughly every 50 seconds").
- **Cross-check every item against `thesis/final_check_report.md`.** Yuval independently found S7 (C429) and
  S11 (C345, Sydney scoring) — saying so in the reply shows the draft was already under this scrutiny, and
  the pending list (B1/S2 citations, S1 site confound, S4 ethics statements) should be folded into the same
  revision plan rather than tracked separately.

### 4. Draft the reply

Tone: **one factual line, no blame**, then straight into substance. Structure:

1. Thanks; note in one sentence that the copy he opened was the pre-review draft (it predates Flavio's
   comments), so a handful of his line-edits are already in the current text — list is attached — and
   everything else is being taken up.
2. **Own the format question.** The draft was written to paper proportions; his list makes clear it should
   be the full thesis. Confirm the conversion: TOC, Hebrew abstract, acknowledgements, thesis-scale Intro
   (sleep and its functions, PSG/EEG signatures, memory, aging, AD, MCI/aMCI, sleep in MCI), reference list
   from 29 to 50+.
3. **The one place he is technically mistaken**, stated plainly and without triumph: bad channels are
   excluded from the average reference by construction, so no re-run is needed — with the offer of a
   sensitivity check.
4. Point-by-point: adopted / adopted-with-a-question / already in the current draft. Group the five
   figure-legibility comments into one commitment (fonts, shared green/red/blue group palette across all
   figures, panel C promoted to its own table).
5. Two genuine questions back to him: does respiratory/AHI data exist for either cohort, and does he want
   the N2-amount covariate as a supplementary analysis or in the main text.
6. Timeline for the revised thesis.

## Follow-on work (not this task, but scoped so the reply can promise it)

Ordered by cost. Items marked ⚑ are also on our own pending list.

1. Text-only, ~1 day: aMCI relabelling; the three `#REF` placeholders he marked; abstract additions (mean
   ages, "full-night PSG", LC-NE softened to "associated with"); peak-frequency range sentence in Results;
   N2 percentage in 4.1; ⚑ S7 rewrite (C429); ⚑ S10 "normalized"; ⚑ S12/S13; ⚑ B1+S2 citation updates;
   ⚑ S9 Lázár values; ⚑ S4 ethics/consent/funding/COI statements; ⚑ S11 reproducibility gaps; Methods
   clause on bads-excluded-from-average (C395); Fig 1 moved to Results (C412 — Flavio raised the same as
   #18 and it was declined; the PI overrides).
2. Analysis, ~1 day: N2-amount ANCOVA (C429); ⚑ S1 site check (elderly 30 TASMC vs 9 Sydney); WASO / SOL /
   REM-latency extraction (C390); optional interpolation-order sensitivity on 2–3 subjects.
3. Figures, ~2 days: shared group palette + font sizes across `make_f1_figure.py`, `replot_f3_no_title.py`,
   `make_topo_composites.py`, `replot_f5_no_title.py`, `replot_roi_violins.py`; Table 2 from
   `n2_bouts_table.py`; supplementary example-spectra figure from `find_example_gaussians.py`.
4. Writing, ~1–2 weeks: thesis Intro expansion + 20+ new references (`literature-review` skill,
   `thesis/references/library.bib` at 31 entries); two new Discussion paragraphs (broader aging/MCI sleep
   changes incl. CAP; ISFS in other conditions); C557/C558 mechanism rewrite; aMCI-vs-AD framing;
   Hebrew abstract; TOC + front matter.
5. Memory: update `feedback_paper_format_pivot` (reverted to thesis), `project_pre_send_check` (Yuval's
   review received and triaged), and add a `project_yuval_review` record.

## Verification

- **Triage completeness:** assert the doc accounts for all 17 comment IDs from `word/comments.xml`
  (C344, C345, C371, C389, C390, C395, C412, C429, C432, C487, C502, C557, C558, C571, C591, C592 — note
  C-ids are non-contiguous), all 9 email items, and every paragraph carrying a `w:ins`/`w:del`.
- **Every "already fixed in V2" claim** must quote the V2 sentence that fixes it, from
  `Shaked's Thesis_V2.docx` (or the chapter file if we stay on the proxy). No claim without a quote — that
  is the one part of the reply that would be embarrassing to get wrong.
- **The C395 claim** is already reproduced above; re-run the 4-channel MNE check and paste the output into
  the triage doc so the argument travels with it.
- **Numbers reused from `final_check_report.md`** (p = 0.98, p = 0.081, p = 0.0009, 34.1 / 44.5 %) were
  verified against `three_groups_V10` / `sleep_stage_stats.txt` on 2026-08-08; re-check the two N2 ones
  against the source files before they go in an email to the PI.
- **Read the reply once as Yuval:** it should read as "took the review seriously, has a plan" and nowhere as
  "argues about which file I opened."
