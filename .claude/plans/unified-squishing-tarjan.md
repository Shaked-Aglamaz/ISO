# Final pre-submission check — "Shaked's Thesis V2"

## Context

The manuscript is believed ready to send to the PI (Yuval Nir) today. It lives in the **new** Google Doc `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY` ("Shaked's Thesis V2", owner Shaked Aglamaz, last modified 2026-08-06 15:08) — this is the copy that Flavio Schmidig's 66 comments + 6 tracked-change edits were applied to. The older doc (`1miqNSnMpbX0…`, recorded in memory) is now superseded and must not be touched.

The user wants a single read-only sweep: every relevant academic-writing check we have, plus the humanizer pass that was deliberately deferred to the end, plus a list of any other verification worth doing before hitting send. **Nothing is to be edited — not the doc, not the chapter files.** Output is findings only.

Already established (read-only, this session):
- Doc reads clean end-to-end: 41,991 chars (~6,300 words incl. references), IMRAD, Abstract → Intro → Methods 3.1–3.7 → Results 4.1–4.6 → Discussion → 2 supplementary figures → 29 numbered references.
- `listComments` returns **zero** open or unresolved comments.
- One table object (10 rows × 4 cols) = Table 1; its cell contents do not come back in the plain-text read and still need to be inspected.
- Headline stats in the doc match the V10 source of truth from memory (peak freq p = 0.0026, BW p = 0.061 ns, AUC cluster p = 0.023 / 9 electrodes, ROI p = 0.143, cohort 35/39/30) — but every number still needs auditing against the files, not against memory.
- Ground-truth outputs exist at `results/group_comparison_results/three_groups_V10/` (`three_group_statistics.txt`, `three_group_topo_statistics.txt`, `three_group_statistics_extended_ROI_normalized_auc.txt`), `results/demographics_V3/`, `results/moca_correlation_V3/`.

Issues already spotted on the first read, to be confirmed and expanded during the passes:
1. **Intro, Lázár paragraph is self-contradictory** — "placed the dominant rate … somewhat below the 0.02 Hz described in rodents, and closer to 0.02 Hz when it tracked … individual spindles." Per `reference_isfs_frequency_attribution`, the sigma-power figure is ~0.01 Hz; the sentence currently says "below 0.02" and "closer to 0.02" of the same comparison.
2. NREM is expanded twice (Intro ¶1 and ¶2); "(LC)" is defined and never used again; the locus-coeruleus pacing claim is made twice in three paragraphs.
3. Title page reads **"July, 2026"**.
4. Methods 3.1 contains **no ethics approval / informed consent statement**, and the manuscript has no data- or code-availability statement.
5. Section numbers 3.x / 4.x hang off unnumbered headings ("Methods", "Results") — Intro and Discussion are unnumbered, so the numbering has no visible parent.
6. Reference 15 (Niethard, SLEEP 2023;46(5):zsad011) is listed with a **single author** and sits in the same issue as Champetier zsac282 — likely a commentary on it, yet it is cited as documentation of spindle-timing breakdown with age.
7. Grollero 2026 carries DOI prefix `10.64898`, not the usual bioRxiv `10.1101`.

## Passes to run

**A. Academic writing review** — `scientific-writing`, `academic-researcher`, and `scientific-critical-thinking` skills over the full text: IMRAD compliance, Abstract↔Results↔Discussion claim consistency, over/under-claiming, hedging calibration, methods reproducibility, limitation coverage, caption conventions (per `feedback_caption_conventions`: `*Label. Title.* body`, stats-in/claims-out, define-ROI-once), figure/section cross-reference integrity, and terminology hygiene (never "ISO" in prose; ISFS = Infra-Slow Fluctuations of **Sigma Power**).

**B. Humanizer pass — whole manuscript** (user's choice), Abstract through Discussion plus all seven captions. Report AI tells with the offending phrase quoted and a suggested rewrite; do not apply.

**C. Numbers audit vs source files.** Every numeric claim in the doc traced to a file: group means ± SD and test statistics → `three_groups_V10/three_group_statistics.txt`; cluster p and the nine electrode labels + post-hoc counts → `three_group_topo_statistics.txt`; ROI value → the extended-ROI stats file (extended 36-ch ROI is the locked choice); demographics, MoCA, site splits, exclusion counts → `demographics_V3/`; MoCA correlations → `moca_correlation_V3/`; bout counts (1023 bouts, mean 9.8, range 3–21) and ISFS-detection percentages → the per-subject outputs. Any V9-era leftover is a blocker.

**D. Citation and DOI check (web).** All 29 references: DOI resolves, author list complete, year/volume/pages right, style internally consistent — and, for the ~10 references carrying an interpretive claim, whether the cited paper actually supports the sentence. Priority: refs 5, 10, 11, 12, 15, 16, 17, 26, 28, 29. Also confirms whether the three preprints (Dimitriades, André, Grollero) are still preprints, per `reference_dimitriades_citation_status`.

**E. Figure and table render check.** Confirm all seven figures are genuinely embedded (not orphan captions), read Table 1's cells and check its site splits and exclusion counts against `demographics_V3`, and check legibility at page width per `feedback_doc_figure_legibility`. Also verify the doc's figures are the V10 assets (`hypno_sleep_stages_V10.png`, `f4_auc_composite_V10.png`, `s1_topo_composite_V10.png`, `methods_flow_roi_v3.png`), not their V9 predecessors.

**F. Pre-send checklist** — the "anything else?" the user asked for: ethics/consent statement, funding + conflict-of-interest, data/code availability, author affiliations, title-page date, keywords, the two still-open Flavio items (Bayesian/TOST equivalence, Results restructuring — flagged only, per the user's deselection), and whether the PI should receive the doc as a link with comment rights or as an exported PDF.

## Deliverable

- Prioritized findings in chat, grouped **blocker / should-fix-before-sending / optional**.
- Full line-by-line report written to `thesis/final_check_report.md` (the only file created; no edits to the doc or the chapter markdown).

## Verification

- Re-read the report against the doc to confirm every quoted phrase appears verbatim and every file citation is a real path.
- `getDocumentInfo` before and after to confirm `modifiedTime` is unchanged — proof nothing in the doc was touched.
- Cross-check the numbers table in the report against `three_group_statistics.txt` a second time before reporting a blocker.
