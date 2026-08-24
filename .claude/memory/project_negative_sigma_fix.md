---
name: project_negative_sigma_fix
description: "|sigma| fix in isfs_presence.py (2026-06-06) + full-cohort re-run to results/sigma_fix_{YA,HE,MCI} (DONE 2026-06-08, 0 crossers in active cohort)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4d4e7585-dae6-453e-9521-9e360f440c6a
---

**Fix (2026-06-06, permanent):** `code/new_iso/isfs_presence.py` `fit_gaussian` now does `sigma_fit = abs(sigma_fit)` right after `curve_fit`. The Gaussian `a*exp(-((x-b)/c)^2)` depends on sigma only via sigma^2, so curve_fit can return a negative sigma identical to its positive twin; the downstream AUC band `[b-sigma, b+sigma]` then collapsed empty → `trapz`=0 → spurious "Invalid AUC" rejection of good wide peaks. (MATLAB reference `f_ISFS_PresenceParamaters.m` never hit this — its `gauss1` bounds c>0. MATLAB code lives at `C:/Users/Shaked/Downloads/Infraslow-Fluctuation-main/`, DO NOT EDIT.)

**Why investigated:** the 4 <20%-detection excluded young controls (EL3017/EL3018/EL3021/EL3033, in `ISO_data/control_clean/a_excluded/`). V2=before fix, V3=after, in `results/new_iso_results/a_excluded_V2` and `_V3`:
- EL3017 15.3%→15.3%, EL3018 14.2%→14.2%, EL3021 9.1%→9.7% — rejects were true needle-spikes (bandwidth ~2 mHz vs real ISFS ~8-19 mHz); **exclusions VALID**.
- **EL3033 15.3%→20.5% (+9 ch): crosses 20% → should move out of a_excluded into the cohort.** Was a pipeline artifact, not biology. (Not yet relocated; already has V3 results so no need to rerun it in the cohort batch.)
- Needle-vs-real diagnostics saved in `a_excluded_V2/EL30*_invalid_auc_diagnostics/`.

**Full-cohort re-run (task bti2gfdd8, COMPLETED 2026-06-08, exit 0, 0 errors):** `main_loop.py` was edited into a 3-group batch (`GROUPS` dict + `run_group()` + `SKIP_DIRS`), iterating ALL subjects per group → `results/sigma_fix_{group}` for {YA,HE,MCI} (YA=control_clean 35, HE=elderly 38, MCI=MCI_clean 31; 104 total). New dirs, no overrides. Slow (~hours) due to many dpi=300 PNGs per channel, not compute.

**Run result:** mean detection YA 75.3% / HE 86.6% / MCI 79.2%. **[STALE 2026-06-08 snapshot — superseded; do not cite.]** Paper-locked per-group means are **YA 73.7 / HE 85.5 / MCI 79.2** (re-verified from sigma_fix CSVs 2026-06-14): YA fell after EL3033 (20.5%) was added (n35→36), HE fell after RY42's projector fix (→21.0%); MCI unchanged. Full reconciliation in [[project_dropped_20pct_criterion]]. **18 subjects gained channels (all gains, 0 regressions — fix is monotonic); 86 identical.** Biggest: SM0019 +20 (MCI), EL3015 +12 (YA). **ZERO subjects cross the 20% bar → no membership change in the active cohort.** Only inclusion change anywhere is EL3033 (in a_excluded, 15.3→20.5%, still to relocate).

**Downstream re-run DONE 2026-06-09 → `three_groups_V9` + `demographics_V2`:** step4/5/6 input dirs repointed `new_*_results`→`sigma_fix_{YA,HE,MCI}`, outputs bumped to `three_groups_V9`; EL3033 relocated into the Young cohort (N 35→36; fixed run COPIED from `a_excluded_V3/EL3033`→`sigma_fix_YA/EL3033`, not re-run). New standalone catalog `thesis/figures_V9_sigma_fix.md`. **Stat changes vs N=35/V7:** peak-freq still strong (ANOVA p=0.0020), AUC cluster holds (p=0.016, now **10** electrodes vs 9), ROI ns (p=0.182), Elderly=MCI — BUT **bandwidth whole-scalp flipped ns→SIG (ANOVA p=0.038, was KW p=0.052)**: the test switched KW→ANOVA because all 3 groups now pass normality, and only Young-vs-Elderly post-hoc survives — report as borderline, revisit the "BW not significant" wording in [[project_scientific_story]]/[[project_paper_figure_set]]. See [[project_topo_projection_fix]] (option-B clip baked into the V9 topos).

**Still TODO:** (1) `main_loop.py` config is single-subject (HE/RY42 + `subjects_filter`); restore 3-group batch if a full re-run is needed; (2) F1 sleep-overview composite needs a final manual stitch (`hypno_sleep_stages_N36.png`) — component panels refreshed in `demographics_V2`; (3) `thesis/figure_manifest.md` still points at V7/V8B — update to V9/demographics_V2 once reviewed.

Related: [[project_isfs_definition]], [[feedback_edit_main_loop_directly]].
