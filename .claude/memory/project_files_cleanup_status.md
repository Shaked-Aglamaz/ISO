---
name: Files cleanup & subject status (2026-04-17)
description: Subject counts per group, exclusion reasons, processing status, 196-ch backup to G:, elderly annotation deduplication
type: project
originSessionId: aa491c21-6c1d-4a0f-9787-8dbd24523a46
modified: 2026-08-15T08:25:14.590Z
---
> **⚠ COUNTS BELOW ARE SUPERSEDED.** The per-group actives here (36 / 38 / 31) predate the
> 2026-06-16 exclusions overhaul. Verified from the folders 2026-08-15: **YA 35, HE 39, MCI 30**
> (HG78, MCI13, SM07 out on TST < 210 min; DS6 into HE; MR5 into MCI). Current cohort truth =
> [[project_v10_regeneration]] and [[project_cohort_change_mr5_tst210]]. Everything else in this
> file — the 196-ch backup, the FIF-stage anomalies (RS5, RY42), the annotation dedup, the
> `a_the_rest` batch notes — is still accurate and is why the file is kept.
>
> Also superseded: the "ISFS exclusion criteria" section at the bottom. The **20 % detection-rate
> criterion was dropped thesis-wide 2026-06-14** — see [[project_dropped_20pct_criterion]]. The
> exclusion criteria now in Methods §3.1 are numeric and different; see
> [[project_methods_revision_yuval]].

## Data cleanup performed 2026-04-08, updated 2026-04-15/17

### 196-channel FIF backup
All `*196*` FIF files (111 files, ~175 GB) moved from `I:/Shaked/ISO_data/` to `G:/Shaked_ISO_data_backup/` preserving folder hierarchy. These are old intermediate files (196 channels before final 176-channel pipeline).

### Young Controls (control_clean → results/new_iso_results)
- **Active**: **36 subjects** as of 2026-06-09 (was 35; **+EL3033** relocated `a_excluded/EL3033`→`control_clean/EL3033`, now passes 20% after the |sigma| fix at 36/176=20.5%; its fixed run was COPIED from `new_iso_results/a_excluded_V3/EL3033` → `results/sigma_fix_YA/EL3033` rather than re-run). Demographics/sleep regenerated at N=36 in `results/demographics_V2` (EL3033's 18 derived sheet columns computed + written to the "subjects" tab row 37; method validated to reproduce DG1/EL3002 exactly). Group-comparison regenerated in `three_groups_V9`. See [[project_negative_sigma_fix]].
- Prior 35-subject roster (28 original + 6 new: EL3031, EL3034–EL3037, EL3044; + EL3029)
  - **+EL3029** added 2026-06-01: ran step2 + main_loop, moved `a_excluded/EL3029`→`control_clean/EL3029`. ISFS 133/176 (75.6%), 7 clean N2 bouts (~58.7 min). Healthy addition. Now picked up automatically by step6 (Young N=35 confirmed in three_groups_V7, 2026-06-03). Note its **mean BW=0.0437 Hz is high** (group mean ~0.024) — it pulled Young BW up enough to flip the BW omnibus to ns (see [[project_paper_figure_set]]).
- **Excluded (ISFS <20%)**: 3 — EL3017, EL3018, EL3021 (EL3033 was the 4th but now included, see above)
  - In `ISO_data/control_clean/a_excluded/`
- **Excluded (other)**: EL3045 — moved to `a_excluded/` on 2026-04-15
- **The "a_the_rest" batch** (EL3026, EL3030, EL3032, EL3040, EL3042) — these ARE young controls. Their hypnos are in `scoring/young_control/{ID}.txt` and step2 outputs (avgref_interpolate FIF) exist in `a_excluded/{ID}/`, but most lack an intermediate pre-interp FIF (only raw .mff + step2 output survive).
  - **EL3026**: poor ISFS candidate — only 31.5 min N2 (8.2% of recording, no N1) and **0 clean bouts ≥300 s**. Its sigma_boxplot was misfiled under `new_elderly_results/` (it's young) → moved to `new_iso_results/sigma_boxplot/` on 2026-06-01.
  - Clean N2 bouts ≥300s (from `annotations.txt`, mult_chan logic): EL3030=2, EL3032=1, EL3040=2 — all marginal.

### Elderly Controls (elderly_control_clean → results/new_elderly_results)
- **Active**: 38 subjects (29 original + 9 new: MCI20, MCI25, MCI27, MCI31, MCI32, MCI34, MCI40, MCI42, MCI43)
- **Excluded**: 5 — DS6, EG5, HR72, NE32, SG27 (in `a_excluded/`)
- **Note**: Elderly subjects are named MCI* due to data source naming — confirmed not an issue
- **Annotation dedup**: EB34, LS56, NE36, SB00 had duplicate dirs. Resolved (see prior notes).
- **FIF-stage anomalies (found 2026-06-06)**: most subjects' `find_subject_fif_file` picks `..._avgref_interpolate_raw.fif`, but **RS5** has only `RS5_cleaned_raw.fif` — pre-referenced upstream (average-referenced; VREF-verified 2026-06-09: VREF std ~15µV non-flat + per-sample channel mean ≈0, independent of the custom_ref flag). OK to run as-is. Spot-checked E70/E90/E132/E192 with step1 detectors → all clean and **RY42** had only `RY42_176-head-ch_...bad-epochs.fif`. **RY42 RESOLVED 2026-06-09**: that bad-epochs file was actually data-unreferenced (VREF flat, custom_ref=0) but carried an inactive "Average EEG reference" SSP projector — see [[project_ry42_projector_fix]]. Stripped projector, ran manual step1 (4 bad ch) + step2 (avg-ref+interp) + main_loop → now has `RY42_176-channels_..._avgref_interpolate_raw.fif`; `results/sigma_fix_HE/RY42` overridden (ISFS 37/176=21.0%, 3 bouts). HE now lockable.

### MCI (MCI_clean → results/new_MCI_results)
- **Active**: 31 subjects (25 original + 6 new: KS5, SM0016, SM0018, SM0019, SM0020, SM09)
- **Excluded in results**: 3 — AB7, MR5, SK6
- **Excluded in data**: 11 — AH3, MCI11, MCI12, MG4, MR5, NT3, SC5, SM03, SM09(old), SM12, YS2
  - SC5 excluded 2026-04-17 — only 23.3% ISFS detection
- **Excluded AD**: 4 — AB7, AY1, SK6, YG9 (in `excluded_AD/`)
- **SM0017**: excluded — only 1 clean bout >= 300s
- **SM006**: concatenated but no hypnogram available — cannot run pipeline

### ISFS exclusion criteria
`filter_subjects_by_detection_rate()` in `step4_distribution_analysis.py` — excludes subjects with <20% of channels having valid ISFS parameters (peak_frequency, bandwidth, auc all non-NaN).
