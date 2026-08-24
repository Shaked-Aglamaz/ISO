---
name: project_yuval_review
description: "Yuval's (PI) review returned 2026-08-11 on the WRONG version (V1) — triaged 2026-08-12; ~all of it still applied; all passes closed + V2 sent 2026-08-19, only the reply is left"
metadata: 
  node_type: memory
  type: project
  originSessionId: cc98b112-bfbc-40d5-834d-0c254a6f2975
  modified: 2026-08-16T15:17:54.457Z
---

**Yuval reviewed V1, not the V2 that was sent to him** (`thesis/reviews/Shaked's Thesis_YN.docx`, returned
2026-08-11: 17 comments, 364 insertions / 111 deletions, plus a 9-item email list). It is the *same*
pre-Flavio draft Flavio reviewed — proven by `reviews/flavio_comments_mapping.md:192` and `:96` quoting his
paragraphs verbatim as Flavio's anchors, and by all four pre-send fixes (B2/S3/S5/S8) being absent.

**The mixup buys almost nothing — do not build the reply on it.** Of his 9 email items **0** are resolved by
V2; of 17 comments, 2 (Methods 3.3 and Results 4.7, both *empty* in his copy) and 1 partly. Only his
in-place line-edits are partly superseded: of 29 paragraphs he touched, 5 identical / 11 lightly edited /
12 rewritten / 1 dissolved in V2 → **55% still land verbatim.**

**Full record: `thesis/reviews/yuval_review_triage.md`** — read it rather than re-deriving. **No reply has
been sent.** A drafted reply was deleted 2026-08-12: the user decided to answer only once, at the end, with
all fixes in hand, since the version mismatch is not worth a standalone email. Write the reply fresh then. The V2 side of the triage was checked against a temporary .docx export
that the user had me delete — **always read V2 from the Google Doc itself**, see [[reference_manuscript_gdoc]].

**Two things worth remembering beyond the docs:**

1. **C395 — his one factual error, and it saves a full re-run.** He demanded interpolation *before* average
   referencing "or re-run". `step2_auto_bad_channels.py:402-411` does reref-then-interpolate, but MNE's
   `set_eeg_reference('average')` **excludes `info['bads']` from the average** — verified empirically on
   MNE 1.6.1 (4 channels at 1/2/3/100 µV, `D` bad → `[-1, 0, 1, 100]`, i.e. mean of the good 3). Bad
   channels never enter the reference. Only a Methods clause is missing.
2. **C429 = our own pending S7, and the real answer is stronger than the draft's.** Young adults contributed
   the *least* analyzed N2 (80.7 vs 106.0 / 90.8 min) yet the *strongest* central-parietal hotspot, so the
   confound runs **opposite** to the effect. Supporting: N2 share does differ (KW p = 0.0009), but
   proportion of N2 analyzed is equal (ANOVA p = 0.9808) and total analyzed duration ns (KW p = 0.0811) —
   all in `results/demographics_V3/{sleep_stage_stats,n2_bouts_table}.txt`.

**He also silently deleted Results 4.7 with Track Changes off** — so a returned .docx can contain losses no
`w:ins`/`w:del` walk will find. Only the paragraph-level diff against the current version catches those.

**Gotcha for future docx diffs:** `difflib.SequenceMatcher` at *character* level with default
`autojunk=True` destroys ratios on long paragraphs (scored a 95%-identical paragraph at 0.16). Compare
**word lists with `autojunk=False`**; short paragraphs still under-score, so eyeball those.

**SCOPE RULE (user, 2026-08-12): do ONLY what Yuval explicitly asked. No voluntary work.** Stated when
dropping the ⚑S1 site check, which he never requested. So the ⚑ items from [[project_pre_send_check]] are
**out of scope unless he asked for the same thing** — S7 stays only because it is his C429. Don't quietly
re-add them.

**METHODS ITEMS ALL DONE 2026-08-15** — C344, C345, C371, C395, C412 and his apnea / Sydney-scoring /
identical-setups / mixing-Methods-and-Results notes. See [[project_methods_revision_yuval]] and
`thesis/reviews/methods_edits_before_after.md`.

**INTRODUCTION DONE 2026-08-16** (email #3, both margin notes, C571-in-Intro) — see
[[project_intro_revision_yuval]]. The reference list was rebuilt **30 → 49** in the same pass, which
**voided `discussion_edits_before_after.md` §9** (since recomputed — see below).

**ALL CONTENT PASSES CLOSED, THESIS V2 SENT 2026-08-19** — Discussion, figures, Abstract, title page,
Hebrew abstract and TOC all landed; `discussion_edits_before_after.md` §9 was recomputed and applied, so it
is no longer void. See [[project_final_pass_sent]], [[project_front_matter_pass]],
[[project_discussion_revision_yuval]], [[project_v11_figure_overhaul]]. **The reply to Yuval is the only
thing left**, and it must be written fresh (the 2026-08-12 draft was deleted). The N2-covariate placement
question resolved itself when C429 was applied in Results; whether he accepts the post-Flavio title is
still unasked and belongs in the reply.

**Extracting his tracked edits works — later sessions will need it again.** python-docx is not
installed; walk `word/document.xml` inside the .docx with `zipfile` + `xml.etree`, treating `w:ins`
as insertions and `w:del`/`w:delText` as deletions, and reconstruct before/after per `w:p`. §3 of the
triage only *summarises* the 364 edits, so the purely stylistic ones exist nowhere but the file
itself. This yielded 16 changed Methods paragraphs; do the same for Intro/Results/Discussion when
those passes come. Also dump the *unchanged* paragraphs and eyeball them — that is the only way to
catch his Track-Changes-off deletions (Methods had none).

**Apnea question RESOLVED 2026-08-14 (user):** all participants had **AHI ≤ 15**, following the
Sharon 2025 protocol — moderate/severe OSA, central apnea and hypoventilation were exclusions. It is
now a stated screening criterion in Methods 3.1, not a limitation.

Related: [[project_pre_send_check]], [[project_flavio_review]], [[feedback_paper_format_pivot]],
[[project_scientific_story]]
