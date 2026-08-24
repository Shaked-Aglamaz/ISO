---
name: project_results_revision_yuval
description: "Results pass on Yuval's review DONE 2026-08-15 (C412/C429/C390 + aMCI relabel); three gotchas worth not re-deriving"
metadata: 
  node_type: memory
  type: project
  originSessionId: d88c91d8-c7a4-41f8-92aa-e74d366dffef
  modified: 2026-08-15T15:48:03.921Z
---

**Yuval's Results items are APPLIED to the Doc and mirrored** (2026-08-15). Record with full
before/after: `thesis/reviews/results_edits_before_after.md`. Covered C412 (Figure 1 moves to Results,
his own rewrite of the overview sentence reinstated), C429 (the N2-amount confound answered properly +
the ANCOVA), C390 (WASO / SOL / REM-latency / sleep-efficiency paragraph), his `~X %` and
peak-frequency-range placeholders, both `#REF` fills, and the aMCI relabel across the whole thesis.

**Three things that cost real work and should not be re-derived:**

1. **The `#REF` in §4.5 is supportable — Dimitriades 2024 says it outright.** From its Results:
   *"Peak frequency and area under the curve showed local minima and maxima in central regions,
   respectively, while bandwidth displayed no clear topographical pattern"*, plus a frontal
   peak-frequency cluster highest in young adults. An earlier draft had flagged this claim as possibly
   unsupportable; it is not. Extract the PDF with **pypdf** (installed; no poppler on this machine).
2. **Grollero 2026 can never support a topographic claim.** Dreem-2 headband, *"a frontal–central signal
   of interest was obtained by averaging the two available bipolar derivations"* — one channel, and
   "topograph" appears zero times in the paper. It reports bandwidth, but never spatially. So it is not
   an alternative to `dimitriades2024isfs` in Results, and citing it there would renumber five refs for
   nothing.
3. **A double-rounding bug made Table 2 disagree with the prose.** `sleep_statistics_extended.py` writes
   `sleep_statistics_table.csv` rounded to 2 dp (WASO MCI SD → 46.25); `make_table2_sleep.py` then
   formatted that to 1 dp and banker's rounding gave 46.2, while `sleep_statistics_stats.txt` prints
   46.3 from the raw value. Fixed 2026-08-15 by making `make_table2_sleep.py` format mean ± SD once,
   from `sleep_statistics_per_subject.csv`. Watch for the same pattern anywhere a summary CSV is
   re-formatted downstream.

**Reference numbering held at 29, unchanged** — 10 = Lázár, 11 = Dimitriades, confirmed from the Doc's
own list, so both `#REF` fills renumbered nothing.

**Second round (`§13`) also APPLIED 2026-08-15:** `maris2007nonparametric` now cited in Methods 3.6, and
"amnestic" added to the title and the Discussion's closing sentence. **The Maris citation renumbered refs
23–29 → 24–30** — the Doc's reference list is an *auto-numbered Docs list ordered by first appearance*, so
a Methods citation is never a free append; renumber existing superscripts **descending** (29→30 first) so
no two markers collide. **Reference count is now 30.**

**§13.4 also fixed:** the relabel had left `aMCI` used before it was defined. Now the **Abstract** defines
it (`30 patients with amnestic MCI (aMCI)`) because abstracts are standalone, the **Introduction** spells
it out unabbreviated both times (defining it inside its existing parenthetical would nest brackets, and
the `…amnestic MCI, aMCI)` workaround reads as a fourth list item), and **Methods 3.1** keeps the single
body definition. The general "mild cognitive impairment (MCI)" research-question sentences are untouched,
so the reader meets MCI then aMCI, each defined once per context. This matches how the thesis already
handles NREM / ISFS / MCI.

**Left deliberately undone:** the Figure 1 image + caption move and the Table 2 paste are **manual jobs
for the user in the browser** (`insertImage` needs a Drive URI and risks re-encoding page-tuned assets).
The Doc still holds V10 figure assets and Figure 1's caption still describes the removed panel C — that
is the figures pass (C487/C502/C591/C592). `maris2007nonparametric` is still cited nowhere.

**All cross-session carry-over now lives in `thesis/reviews/yuval_review_triage.md` §8**, split by session
(§8.1 figures, §8.2 discussion/writing, §8.3 reply-to-Yuval, §8.4 anything touching the Doc), with a
STATUS block at the top of the triage. Local pointers were also planted where each session will actually
look: a "START HERE" box at the top of `thesis/figure_manifest.md`, and `v3` version notes on
`01_abstract.md`, `02_introduction.md` and `05_discussion.md`. **The sharpest carry-over is that the
Discussion still calls bandwidth a tendency "pointing the same way" as the age effect, which the revised
Results now contradict.**

Related: [[project_yuval_review]], [[project_methods_revision_yuval]], [[feedback_doc_edits_review_first]],
[[reference_manuscript_gdoc]], [[project_c429_c390_results]].
