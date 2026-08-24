---
name: project_flavio_review
description: "Flavio's 66-comment thesis review: applied 2026-08-06 to the doc COPY; what was declined, deferred, and still open"
metadata: 
  node_type: memory
  type: project
  originSessionId: caaebecc-a92c-4f35-893a-dd34fd3458c1
  modified: 2026-08-12T09:45:40.606Z
---

Flavio Schmidig reviewed the ISFS manuscript (`Shaked's Thesis_FS.docx`, in `C:\Users\Shaked\Downloads`) and left **66 comments** plus 2 email-level asks. **Applied 2026-08-06** to the doc **copy** only (`1YpXrDGFlzRk…`, see [[reference_manuscript_gdoc]]), and mirrored into `thesis/chapters/0[1-5]*.md` + the three affected caption blocks in `thesis/figure_manifest.md`.

**Artifacts (moved to `thesis/reviews/` on 2026-08-12, previously loose in `thesis/`):** `flavio_comments_mapping.md`/`.pdf` (ID → anchor → verbatim comment — **never edited, IDs stay stable across versions**), `flavio_comments_review.md`/`.pdf` (V1 categories), `flavio_comments_review_V2.md`/`.pdf` (post-apply status for all 66), `flavio_edits_before_after.md` (full before→after, Part A = items the user reviewed, Part B = pre-approved). `thesis/answers.txt` (the user's decisions) stayed put in `thesis/`. `thesis/reviews/` also holds Yuval's returned copy `Shaked's Thesis_YN.docx`.

**He also edited in place as tracked changes, not comments — 38 insertions / 27 deletions across 8 paragraphs, which had been missed entirely by the V1 categorisation.** Extract them with a zip+ElementTree walk of `word/document.xml` looking for `w:ins`/`w:del`/`w:delText`. They reduced to 6 edits (C5-1…C5-6), all taken; C5-1 partly (FFT spelled out, his "transform**ation**" declined as non-standard). **If a reviewer returns a .docx, always check tracked changes, not just comments.**

**STILL OPEN:**
1. ~~**Humanizer pass on the Discussion**~~ — **DONE 2026-08-08**, and widened to the whole manuscript at the
   user's choice. Suggestions were delivered but **not applied** (the manuscript went to Yuval without them);
   the before→after list is in `thesis/final_check_report.md` §3. See [[project_pre_send_check]].
2. **Bayesian equivalence analysis** (#98 + email ask A) — deferred as paper-level. Flavio's point is valid: frequentist non-significance can't support "elderly and MCI are indistinguishable". Mitigated for now by wording only (#109: "negative result" → "no evidence"); **nothing about equivalence/power was added to the limitations, per user instruction**.
3. **Results restructuring** (email ask B) — deferred. #46 was the carved-out piece and is done (old 4.1+4.2 merged; sections renumbered 4.1→4.6).
4. Two cosmetic paragraph breaks in Results 4.1 (after "Table 1).") and 4.2 (after "no post-hoc tests).") where the approved text was one paragraph — left split because `findAndReplace` can't fuse paragraphs. User hasn't said whether to join.

**Declined by the user:** #18 (Fig 1 sentence reads as a result — leave it in Methods) and #36 ("whole scalp" → "main/overall ISFS" — user disagrees, "whole-scalp" retained).

**Judgement calls the user endorsed:** (a) Flavio's suggested title "Healthy, but not pathological, aging alters…" **misstates the result** — it implies pathological aging doesn't alter ISFS, when MCI shows the *same* changes; new title is "Aging, but not mild cognitive impairment, reshapes the infra-slow rhythm of sleep spindle power" ("spindle", not "sigma", chosen for a broad audience). (b) #40 — Flavio wanted the demographics *result* stated in Methods; refused because it's already in Results 4.1, so the sentence was just shortened. (c) #9 — author names dropped from the Intro in favour of bare superscripts, leaving **Dimitriades as the only name in the Introduction** (the study we adopt), which is the right convention for a numbered citation style.

Nothing committed.
