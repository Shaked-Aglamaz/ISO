# Applying Flavio's review to the thesis

## Context

Flavio left **66 comments** (categorized in `thesis/flavio_comments_review.md`, anchored in `thesis/flavio_comments_mapping.md`) plus 2 email-level asks. Your decisions are in `thesis/answers.txt`, refined over this session. We are submitting to the school now, so the two paper-level asks (Bayesian equivalence testing, full Results restructuring) stay parked.

Also folded in: Flavio's **tracked in-place edits**, extracted from `Shaked's Thesis_FS.docx` — 38 insertions and 27 deletions across 8 paragraphs, none attached to a comment, so none appeared in the original review. Six distinct edits, now **Category 5**.

## Target

**Only the new Doc copy: `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`.** Verified reachable and byte-identical to the original (39,583 chars). The original `1miqNSnMpbX0…` is **not touched** — I will not open it for writing at any point.

## → Full before → after for every edit

**`I:\Shaked\ISO\thesis\flavio_edits_before_after.md`** — reviewed and signed off. Part A holds the new material and the items you flagged; Part B holds everything approved earlier. Your revisions from this session are marked ✎ in the file:

- **A4** (#9) — names dropped (bare superscripts), Lázár sample size dropped, Variant 2 adopted. Answered your convention question: naming authors in running text is normally reserved for studies you adopt or argue with, so **Dimitriades ends up the only name in the Introduction** — internally consistent. This edit also corrects a real mis-citation: our text attributes "approximately 0.02 Hz" to reference 10, whose headline result is a rate *below* 0.02 Hz (≈ 0.01 Hz) for sigma power, with 0.02 Hz belonging to reference 6.
- **A5** (#25) — checked the FFT first mention as you asked. It *is* first in Methods 3.4, but the expansion is **already duplicated** in the Figure 2 caption. So: expand in 3.4, reduce the caption to bare "FFT" (added to B19). Also using "fast Fourier trans**form**", not Flavio's "transformation" — the standard term, and what your own caption already says.
- **A7** (#46) — "A detectable ISFS was the rule rather than the exception" cut; replaced with "The ISFS was present in the vast majority of the data", and the next sentence collapsed into it to avoid "vast majority" / "large majority" two clauses apart.
- **A9** (#116) — "Several limitations", not "Three".
- **#109** — applied as wording only ("negative result" → "no evidence"), nothing added to the limitations.

## Files to modify

| File | What changes |
|---|---|
| Google Doc `1YpXrDGFlzRk…` | Every edit — the new copy only |
| `thesis/chapters/01_abstract.md` … `05_discussion.md` | Mirror of the prose edits |
| `thesis/figure_manifest.md` | F2 title + FFT fix, F3 caption, F4 caption — keeps the manifest the caption source of truth |
| `thesis/flavio_comments_review_V2.md` + `.pdf` | New review artifact (below) |

`thesis/flavio_comments_mapping.md` / `.pdf` is **not** touched — comment IDs stay stable.

## Execution order

1. **Title + Abstract** first — neither contains citation superscripts, so it is the safest place to confirm the write path end to end on the new copy.
2. **Introduction** → **Methods** → **Results** → **Discussion**.
3. **Captions** in the Doc, then mirror into `figure_manifest.md`.
4. Mirror all prose edits into `thesis/chapters/*.md`.
5. Build the V2 review artifact: `thesis/flavio_comments_review_V2.md` → PDF via pandoc (already on PATH, same route as the existing PDFs). Contents: per-comment status (applied / declined-by-you / deferred), the Category 5 table, and the resolved answers on #2, #9, #40.

### Known risk and mitigation

The Doc renders citations as **superscript runs** (`¹⁻²`, `⁶⁻⁷`, `¹⁴`). A find-and-replace whose range crosses one can silently drop the superscript formatting. So **every replacement is chunked at superscript boundaries** — the Intro #9 paragraph, which now carries `¹⁰ … ⁶ … ¹⁰ … ¹¹ … ¹²`, is applied as several separate replacements rather than one. Where a paragraph must be replaced whole (Abstract, Discussion ¶1), the target has no citations, or the superscript is re-applied and verified afterwards.

## Verification

1. `readDocument` after each section and diff against the intended text — confirming every reference marker `¹`–`²⁹` still renders as superscript and the reference list is untouched.
2. Confirm the original Doc `1miqNSnMpbX0…` is unmodified at the end (its `lastModifyingUser` / revision unchanged).
3. Re-run the tracked-changes extraction against the `_FS.docx` and confirm all 6 Category 5 items are accounted for.
4. Grep the final Doc text and `thesis/chapters/*.md` for terms that must be **absent**: "ISO" as a standalone term, "frequency shifts", "extended" near ROI, "kernel density", "earlier subject set", "negative result", the stricter-detection-criterion sentence, and a second expansion of "fast Fourier transform".
5. Confirm the numbers still match `results/group_comparison_results/three_groups_V10` — p = 0.0026, p = 0.061, cluster p = 0.023 / 9 electrodes, ROI p = 0.143, N = 35/39/30 — since several sentences were re-worded around them.
6. Open the V2 PDF and check all 66 comment IDs appear with a status.
