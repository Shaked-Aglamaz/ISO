# Fix B2 — remove the incorrect missing-channel claim (minimal)

## Context

`thesis/final_check_report.md` item **B2** flags the last sentence of Results 4.3:

> "The group maps show the mean of the per-subject means, and channels without a valid fit were
> interpolated from their neighbours for visualization purposes only, never for the statistics."

Verified against the code. The neighbour-interpolation clause is true — that routine
(`utils/topo_aggregation.py::neighbor_impute`) only builds the displayed raw maps (Figure 4A).
What is false is **"never for the statistics"**: a second, different fill enters every topographic
statistic. `step4_distribution_analysis.py::normalize_subject_channels` (lines 83–87) replaces a
missing channel with that subject's mean over accepted channels and then divides by the scalp mean,
so those cells land at ≈1.0 — and that path feeds the cluster permutation test, the electrode-wise
ANOVA, the post-hoc Tukey (`step5_topo_comparison.py:74`), the normalized maps in F4B/S1, and the
ROI violin F5 (`step6_groups_comparison.py:990-991`). Raw whole-scalp violins, displayed means and
the MoCA correlations use plain `nanmean` and impute nothing.

The display/stats split is a deliberate V5 decision (memory `project_raw_topo_aggregation_consistency`)
and **no code changes here**. The imputed fraction differs by group whole-scalp (young 25.5%,
elderly 14.4%, MCI 18.0%) but is comparable at the 9 cluster electrodes (11.7 / 10.8 / 14.1%), and
excluding imputed ROI channels moves the F5 ANOVA from p = 0.143 to p = 0.126 — the fill is
conservative. Those numbers are kept here as backup for a PI question; the user decided **not** to
put them in the manuscript.

**Decision: remove the incorrect claim rather than explain the mechanism.** No new Methods paragraph,
no sensitivity run. Applied to `thesis/chapters/*.md` **and** to the Google Doc "Shaked's Thesis V2"
(id `1YpXrDGFlzRk…`, memory `reference_manuscript_gdoc`).

## Edits

### 1. `thesis/chapters/04_results.md` §4.3 — replace the final sentence

Remove:

> The group maps show the mean of the per-subject means, and channels without a valid fit were
> interpolated from their neighbours for visualization purposes only, never for the statistics.

Replace with:

> The value above each map in Figure 4A is the mean of the per-subject means.

Rationale for keeping the short clause rather than deleting outright: Figure 4A prints
`mean=6.457 / 7.314 / 7.478` in the panel titles, and this sentence is the only text explaining them;
it is also where Flavio's comment #39 asked the topography convention to live. The replacement makes
no claim about missing channels, and dropping `neighbours` also closes report item **H7** (it was the
manuscript's only British spelling).

### 2. `thesis/figure_manifest.md` — F5 caption, one word out

After normalization no channel is missing, so all 36 ROI electrodes are averaged, not only the
fitted ones. Change:

> Each dot is one subject's mean across the fitted electrodes of the pre-defined central-parietal ROI.

to:

> Each dot is one subject's mean across the electrodes of the pre-defined central-parietal ROI.

### 3. Google Doc sync

Apply both edits in "Shaked's Thesis V2" via the google-docs MCP — Results §4.3 body text and the
F5 caption (caption title stays bold + blue; only the body sentence changes). If the MCP returns
`invalid_grant`, the user re-runs `npx @a-bonus/google-docs-mcp auth` (ticking Drive) and restarts
Claude Code.

## Verified as leaving no other false claim

- Methods 3.5 "Channels whose fit failed any of these checks … left missing" — accurate at the
  fit-acceptance stage, says nothing about the later fill.
- Methods 3.6 whole-scalp "the mean across all channels with an accepted fit" — matches the code
  (pandas `.mean()` skips NaN).
- Methods 3.6 ROI/normalization paragraph — describes the normalization order only, no claim about
  missing channels.
- Methods 3.2 "channels rejected as bad were interpolated from neighboring channels" — unrelated
  (EEG preprocessing), true.
- Results 4.2 detection rates (74.5 / 85.6 / 82.0%) — correct, and they are the complement of the
  imputed fractions.

## Flagged, not changed

`step5_topo_comparison.py` still passes `min_detection_rate=0.2`, the criterion dropped thesis-wide.
It excludes nobody in the final cohort (lowest rates EL3033 20.5%, RY42 21.0%, MCI07 22.7%), so no
reported result is affected and no code edit is in scope.

## Verification

1. `grep -n "neighbour" thesis/chapters/*.md` returns nothing.
2. `grep -n "fitted electrodes" thesis/figure_manifest.md` returns nothing.
3. Read §4.3 end to end — the paragraph must still flow into the new closing sentence.
4. Read the edited passages back from the Google Doc to confirm the MCP replaced in place and did
   not leave a duplicate of the old sentence.
5. Mark B2 and H7 addressed in `thesis/final_check_report.md`.
