---
name: project_no_namci_sensitivity
description: "no_naMCI aMCI-only sensitivity run (REJECTED, code stripped, full MCI=31 kept) — cohort mapping + full with/without results comparison"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3a84b674-a2e1-4639-a4b3-ecc80afb2625
---

aMCI-only sensitivity analysis (added 2026-06-15): re-run of the 3-group comparison
with the 10 non-amnestic MCI (naMCI) subjects dropped from the MCI group.

**DECISION (2026-06-15): KEEP the full MCI cohort (N=31); naMCI exclusion REJECTED.**
Removing naMCI did not help — see findings below. The `ISFS_NO_NAMCI` gated code was
then STRIPPED from all scripts (step4/5/6 + moca_correlation.py + config.NAMCI_EXCLUDE);
output figures/CSVs retained on disk in `results/no_naMCI/` as the record. To redo,
re-add the gate (drop NAMCI_EXCLUDE list below from mci_subjects, redirect output dir).

**Cohort:** naMCI/aMCI labels live in Google Sheet `1QqDGN7dSCCI-qtLFD_LYhyI4mYnk8wP7yeu-MnQwUlk`
(single tab "Sheet1", `group` col). Sheet "New ID" → result-dir name drops the middle
zero (MCI001→MCI01). Sheet lists 12 naMCI; only 10 are active (MCI008="no PSG" never
recorded; MCI012/MCI12 already in MCI_clean/excluded/). naMCI to drop =
MCI01,02,07,13,14,16,17,22,28,39. Result: MCI 31→21 (14 TASMC + 7 Sydney aMCI:
MCI03/04/24/26/35/37/41). "Keep all others" = TASMC and Young(36)/Elderly(38)
unaffected. Note Sydney "MCI" group label was only ever aMCI+naMCI present (no
control/SMC-labeled subjects made it into our active cohort). See [[project_two_site_cohort]].

**RESULTS COMPARISON — full V9 (MCI=31) vs no_naMCI (MCI=21); Y=36, E=38 both:**
- *Peak freq* (whole-scalp ANOVA): omnibus 0.0020→0.0119 (still sig). Pairwise:
  Y-vs-E 0.0112→0.0087 (* both, AGING ROBUST); **Y-vs-MCI 0.0039→0.26 (sig→NS, main casualty)**;
  E-vs-MCI ns both.
- *Bandwidth* (whole-scalp ANOVA): omnibus 0.0376→0.0520 — flips just-sig→just-ns,
  borderline & fragile either way.
- *AUC whole-scalp* (Kruskal): 0.348→0.378, NS both (effect is focal not global).
- *AUC focal cluster* (spatiotemporal perm F-test) = the headline ISFS finding:
  **SURVIVES** — 1 sig cluster, p=0.014→0.043, 10→7 electrodes, same E155 hotspot
  (min-p ~2e-5 both). Cluster post-hoc unchanged pattern: Y≠E (5 ch), Y≠MCI (7→4 ch),
  **E=MCI (0 ch)**.
- *AUC normalized ext-ROI* (ANOVA): 0.182→0.264, NS both.

**Conclusion:** doesn't add value. Focal AUC effect + Elderly=MCI + aging-driven story
all hold for aMCI-only; restricting just costs power on the secondary Y-vs-MCI peak-freq
and BW contrasts. Consistent with AGING-not-MCI framing in [[project_scientific_story]].

**MoCA:** `moca_correlation.py` was also gated, but **0 naMCI dropped** — the 14
MCI-with-MoCA are ALL TASMC (ED9/KS5/SM*/VZ9/YC8/YS0); no Sydney MCI0x subject has a
MoCA in the subjects sheet. So no_naMCI MoCA == full-cohort moca_correlation_V2, all
null: with-HE pooled N=43 (HE=29,MCI=14) all |r|<0.1 p>0.55; MCI-only N=14 weak positive
r~0.19-0.27 all p>0.36. Don't re-investigate. See [[project_demographics_tables]].

**Reproduction (code was STRIPPED — re-add to redo):** env gate `ISFS_NO_NAMCI=1` in
step4/5/6 + moca_correlation.py: drop the naMCI list from mci_subjects and redirect
output dir `three_groups_V9`→`results/no_naMCI`. step4 needed a `save_dir` param on
`plot_topographies()` (read data from source dir, write PNG to no_naMCI) to avoid
clobbering the canonical source-dir topo PNG. Run with PYTHONIOENCODING=utf-8.
Outputs preserved in `results/no_naMCI/` (topos, violins, F-stat/cluster topos, 3
stats txts, 2 MoCA grids + 4 CSVs).
