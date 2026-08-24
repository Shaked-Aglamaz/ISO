# Results pass — Yuval's review (C412, C429, C390, ~X %, peak-freq range, 2 × #REF, aMCI)

## Context

Yuval returned a reviewed copy of the thesis on 2026-08-11 (`thesis/reviews/Shaked's Thesis_YN.docx`),
triaged in `thesis/reviews/yuval_review_triage.md`. The Methods items were applied 2026-08-15. This pass
closes the **Results** items: doc sections 4.1–4.6 of **"Shaked's Thesis V2"**
(`1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`), mirrored into `thesis/chapters/04_results.md`, plus the
aMCI relabel across the whole thesis and three caption fixes his edits force.

A previous session drafted all of this into `C:\Users\Shaked\.claude\plans\graceful-crunching-wreath.md`
but could not touch the Doc (MCP never registered). That draft is the starting point; this session has
now **verified it end to end** and resolved its open items.

**Scope rule in force:** only what he explicitly asked for. ⚑ items from our own pending list stay out
(S7 counts — it is his C429). Anything else spotted gets reported, not fixed.

### Verification done this session (all against live sources, nothing assumed)

| Checked | Result |
|---|---|
| Doc Results 4.1–4.6 vs `04_results.md` | **identical** — every BEFORE block in the draft matches the Doc verbatim |
| Doc reference list | **10 = Lázár 2019, 11 = Dimitriades 2024** confirmed → the #REF superscripts renumber nothing |
| C390 numbers | `demographics_V4/sleep_statistics_stats.txt` — all four measures match exactly |
| C429 numbers | `demographics_V3/{sleep_stage_stats,n2_bouts_table}.txt` + `three_groups_V11/three_group_ancova_statistics.txt` — all match |
| Peak-freq range | recomputed from `three_group_ancova_per_subject.csv`: min 0.00953, max 0.03138, mean 0.02188 (N=104) |
| N2 % of recording | 34.1 ± 13.2 / 44.5 ± 11.7 / 39.0 ± 12.3 (`table2_sleep_architecture.csv`) |
| His tracked changes ¶172–200 | re-extracted from the .docx (zipfile + ElementTree over `w:ins`/`w:del`) — draft's readings confirmed |
| **`[VERIFY]` on #REF (b)** | **RESOLVED — the claim IS supportable.** Dimitriades 2024 (p. Results): *"Peak frequency and area under the curve showed local minima and maxima in central regions, respectively, while bandwidth displayed no clear topographical pattern"*, plus a frontal peak-frequency cluster highest in young adults. Both halves of our §4.5 sentence match. Cite ref ¹¹. |

### Decisions locked with the user

1. Peak-frequency range → report the **full range 0.0095–0.0314 Hz**.
2. Keep **"(Table 2)"** pointers in the prose; **user inserts the Table 2 PNG by hand** together with the
   Figure 1 move.
3. **Fix all three captions** (F3, F5, S2) in the Doc and in `thesis/figure_manifest.md`.
4. aMCI relabel = **whole thesis**, per the handoff's per-sentence classification (relabel where the
   sentence means *our cohort*; keep plain "MCI" for the condition in general or others' studies).

Taken by me, flagged not asked: the AUC ANCOVA is reported without the non-normal-residuals qualifier
(it is null before and after, and the Kruskal–Wallis remains the stated primary test elsewhere); §4.2's
elderly-vs-aMCI sentence and §4.4's ROI sentence keep V2's phrasing with only his additions folded in,
rather than swapping in his sentences wholesale.

---

## Phase A — write the before/after review file, then stop

**Deliverable: `thesis/reviews/results_edits_before_after.md`**, in the format of
`thesis/reviews/methods_edits_before_after.md`: section, surrounding context, exact current text, exact
proposed text, one-line reason naming the comment answered. Nothing touches the Doc until this is read
and approved (standing rule).

It carries the entries below, all four locked decisions applied, and re-verifies each BEFORE block
against the Doc immediately before writing.

### §4.1 — heading, Figure 1 lands, N2 %, C390, C429

- **Heading** → `4.1 Sleep differs with age, but the N2 sleep entering the analysis does not`
  ("N2 sleep is plentiful" restates the exact non-sequitur C429 attacks).
- **¶1** insert his `significantly` into the MoCA clause (his ¶173).
- **¶2 (new, first)** — the Figure 1 overview sentence reinstated **in his own rewrite**, "N2 bout
  properties" repointed to Table 2 (that panel moved there 2026-08-13):
  > An overview of the recorded sleep across the three groups, including whole-night hypnograms
  > (time-course of sleep stage dynamics) superimposed with EEG spectrograms (time-frequency dynamics),
  > and the distribution of sleep stages, is shown in Figure 1; group statistics for every measure below
  > are reported in Table 2.
- **¶3 sleep architecture** — his wording ("Time spent in N3 and in REM sleep", "evident when comparing",
  "highest among all sleep stages"), real F/p added, his `(~X %)` filled with **34.1 ± 13.2 / 44.5 ± 11.7 /
  39.0 ± 12.3 % of total recording time**, and the trailing non-sequitur **deleted**.
- **¶4 NEW — sleep continuity (C390)**: WASO (23.2 / 51.9 / 71.0 min; KW H=31.44, p<0.0001, η²=0.291),
  REM latency (92.8 / 125.4 / 130.3 min; H=10.53, p=0.005), sleep efficiency (89.2 / 83.6 / 78.6 %;
  H=19.19, p=0.0001) — each young-vs-both-older, elderly=aMCI (all p ≥ 0.11); sleep onset latency
  reported as null (17.4 / 15.6 / 19.9 min; H=1.93, p=0.38) because he asked for it.
  Uses **46.3** for the WASO MCI SD (the value `sleep_statistics_stats.txt` prints); Table 2's PNG shows
  46.2 — a rounding-only 0.1 difference, flagged not silently reconciled.
- **¶5 NEW — the C429 answer** replacing the non-sequitur. Four moves, in order: N2 share *does* differ
  (KW H=13.93, p=0.0009; young-vs-elderly p=0.0006); the analysed **proportion** of each subject's N2 is
  equal (51.9 / 52.3 / 51.5 %; ANOVA F=0.019, p=0.98) and total analysed duration does not differ
  (80.7 / 106.0 / 90.8 min; H=5.02, p=0.081); the imbalance **runs opposite to the effect** (young
  contribute the least analysed N2 yet show the strongest hotspot); and the ANCOVA — peak frequency
  survives (F=6.03, p=0.0034, partial η²=0.108; both contrasts survive Holm; covariate p=0.94), AUC
  unaffected, **bandwidth does not** (covariate r=0.386, F=14.73, p=0.0002 — a larger effect than group;
  group p 0.061 → 0.206), stated plainly as *not interpretable as an effect of age*.
- **¶6** detection rates — relabel only.

### §4.2 — whole-scalp parameters

- Peak frequency: prepend his range sentence with the **full computed range** —
  *"Across all participants the ISFS peak frequency lay around 0.02 Hz, as expected (mean 0.0219 Hz,
  range 0.0095–0.0314 Hz across the 104 participants)."*
- Bandwidth: his gloss ("the extent to which the ISFS was tightly or loosely locked around its peak
  frequency"), the "best read as a trend / no post-hoc tests" hedge deleted per his ¶181, and a
  back-reference to §4.1 stating the duration adjustment removes what remained (p = 0.206).
- AUC: his ¶182 split — AUC named at first use, "significantly" added.
- Elderly-vs-aMCI summary: relabel only.

### §4.3 — topography + #REF (a)

His ¶188 edits ("when averaged across all electrodes", "By contrast", "less focal", "performed on these
electrodes revealed"), and the citation: **`lazar2019infraslow` + `dimitriades2024isfs` → ¹⁰,¹¹**
(both already in the numbered list — nothing renumbers; the Discussion already makes the same pairing).

### §4.4 — ROI

His ¶192 motivation ("the proportion of a participant's overall ISFS strength that falls within the
region") and the explicit ordering, his typo `central-parietel` corrected. His word "trend" **not** used
at p = 0.143.

### §4.5 — peak-freq / bandwidth topographies + #REF (b)

His ¶198 sentence added, cited **`dimitriades2024isfs` → ¹¹** — now verified against the source PDF
rather than assumed. Backup `grollero2026iso` deliberately avoided: citing it here moves its first
appearance ahead of the Discussion and renumbers five references.

### §4.6 — MoCA

Relabel only.

### Captions (decision 3)

- **Figure 3** — "Bandwidth trended in the same direction but did not reach significance (…; no post-hoc
  tests)" → "Bandwidth did not differ significantly across groups (one-way ANOVA, p = 0.061)"; "Elderly
  and MCI did not differ on any parameter" → aMCI.
- **Figure 5** — "the group means followed the expected order" reworded per his ¶192; aMCI.
- **Figure S2** — "ISFS scalars" / "No scalar correlated" → **metrics** (his ¶200 edit; the draft missed
  that its anchor survives in this caption, not in the §4.6 body).

### aMCI relabel — whole thesis (decision 4)

Per-sentence list already classified in the handoff §7. Targets: `01_abstract.md` (our sample, and the
MCI-vs-elderly result), `02_introduction.md` (the three-group sentence and our research question),
`03_methods.md` (define **aMCI** at first use in 3.1, plus the Sydney/TASMC sentences, the MoCA n=14, the
3.4% data-quality figure), `05_discussion.md` (~7 sentences about our cohort/result), all Results
captions, and the **Table 1 header cell** (row 0, col 3, `table:body:0`). Left as plain "MCI":
the general research question in the abstract, the Intro literature sentences, André's study, and the
general disease-stage claims in the Discussion.

---

## Phase B — apply, after approval

1. **Doc first**, section by section, re-reading each target range before editing.
   Mechanics: `findAndReplace` for in-paragraph swaps (safe across the superscript citation glyphs, which
   carry no formatting); `findElement` → `deleteRange(start, textEnd + 1)` to remove a whole paragraph;
   `insertText` + `applyParagraphStyle NORMAL_TEXT` for new paragraphs following a heading;
   `getTableStructure` (not `readDocument`) for the Table 1 cell. Nothing in Results is italic.
2. **Mirror** into `thesis/chapters/04_results.md` with a `v3` note at the top in the style of the
   existing version note, and into `01_abstract.md` / `02_introduction.md` / `03_methods.md` /
   `05_discussion.md` for the relabel.
3. **`thesis/figure_manifest.md`**: the three caption rewrites, plus line 105 — retarget F1 from
   *Methods/Results overview* to ***Results*** (Methods no longer references it).
4. Mark `thesis/reviews/results_edits_before_after.md` **APPLIED**, recording the four chosen options.

### Left to the user, by hand in the browser (30 seconds, zero risk)

- **Move the Figure 1 image + its caption** from Methods 3.2 into Results 4.1, after the new overview
  paragraph. The API route (`deleteRange` + `insertImage`) needs a Drive-hosted URI and risks a re-encode
  on a page-tuned asset.
- **Paste in Table 2** (`results/demographics_V4/table2_sleep_architecture.png` + its caption from
  `figure_manifest.md:101`) at the same time — the new prose points at it.

---

## Out of scope — reported, not acted on

- **C432** (his `#` in §4.1) — supplementary figure of example spectra. Figures pass.
- The Doc still holds **V10** figure assets while the manifest is at V11, and the Doc's Figure 1 caption
  still describes the removed panel C. Figures pass (C487/C502/C591/C592, email #6).
- **Dimitriades is now published** (*Sci Rep*, 18 Jun 2026) but the Doc's reference 11 still reads
  "bioRxiv [preprint]". Pending item B1/S2, not a Yuval ask — say the word and it folds in.
- `05_discussion.md:17` calls the MCI cohort "clinically and etiologically heterogeneous"; 9 of our 30
  are non-amnestic, which now sits next to a blanket "aMCI" label. Discussion pass.
- `maris2007nonparametric` appears cited nowhere despite the cluster-permutation Methods section.

## Verification

- Re-read Doc §4.1–4.6 after applying and diff against `thesis/chapters/04_results.md` — they must be
  character-identical apart from the citation superscripts (the .md uses `[@key]`).
- Grep the Doc text for `#REF`, `~X`, `0.015-0.03`, `scalars`, `no post-hoc tests` → all must be gone.
- Grep the repo chapters for `[TO SUPPLY:` and hand back the list.
- Confirm the reference list is still 29 entries in the same order (nothing renumbered).
- Spot-check every number written into the Doc against the source .txt files one final time.
