---
name: project_clean_bout_criterion
description: "2026-06-16 inclusion criterion: >=3 clean N2 bouts; 'not enough N2' renamed 'not enough clean bouts'; DS6 re-included (pending pipeline), EG5 reason->bad channels"
metadata: 
  node_type: memory
  type: project
  originSessionId: 230bceb3-3eff-4a50-945b-1d56191ff97b
---

2026-06-16, after manually verifying the 6 borderline subjects that had >2 bouts in the old table (EG5/NE32/MCI12/AH3/MCI11/DS6):

- **New inclusion criterion: a subject must have >= 3 clean N2 bouts (>=300 s, BAD-split + merged).** Replaces the vague "not enough N2". Added to Methods 3.1 (exclusion grounds) + 3.3 (`thesis/chapters/03_methods.md`), and the T1 category label in `code/demographics_table.py` ("Not enough N2" → "Not enough clean bouts"; internal key still `not_enough_n2`). Retained-floor = 3 (lowest retained: EL3003/EL3020/EL3006/el3007/RY42 all =3).
- **Re-counted clean bouts from `cleaned_annotations.txt`** (BAD-split, the ISFS-pipeline definition in `new_iso/mult_chan.py`) for the 6 subjects → corrected `thesis/low_bout_and_excluded_n2_table.md`: NE32 3→1, AH3 4→2, MCI11 5→2, MCI12 3→2; EG5 (3) and DS6 (7) unchanged. The old MCI11/MCI12 counts were hypno-only upper bounds (no BAD-split); they now have proper cleaned-annotation files (MCI12 scoring added this session in step1 cell 8 from `scoring/MCI/MCI12_hypno.txt`). N2% in that table = N2/TST and reproduced exactly, confirming parsing.
- **DS6 RE-INCLUDED** (7 clean bouts, 40.5% N2; the lone subject above the floor with only a bogus "not enough N2" reason). Files moved `elderly_control_clean/a_excluded/DS6` → `elderly_control_clean/DS6` (active). **PROCESSED 2026-06-16: step2 (0 bad ch) + main_loop → `results/sigma_fix_HE/DS6` (7 bouts, ISFS 163/176 = 92.6%).** HE N 38→39. STILL TODO: add DS6 row to subjects sheet (compute ~18 derived demographics cols like EL3033 did, see [[project_v9_regeneration]]) + move excluded→subjects + remove from `EXCL_SUBJECTS_TO_DROP` in demographics_table.py.
- **MR5 PROCESSED 2026-06-16** (was already in subjects sheet row 104, MCI N=32→30 after TST exclusions): re-ran step2 (overrode prior output; **0 bad channels — user decision, sheet W/X/Y = TRUE/0/0**) + main_loop → `results/sigma_fix_MCI/MR5` (11 bouts, ISFS 176/176 = 100%). step2 main() + main_loop main() edited in place to run the DS6/MR5 pair (group-aware).
- **EG5 stays excluded but reason recategorized → bad channels** (it clears the 3-bout floor; real disqualifier is bad channels). Added `"EG5": "bad_channels"` to `EXCL_CATEGORY_OVERRIDES`; sheet free-text reason kept (per [[project_dropped_20pct_criterion]] convention).
- NE32/MCI12/AH3/MCI11 stay excluded — now genuinely under "not enough clean bouts" (all <3), which is finally accurate.

Builds on [[project_cohort_change_mr5_tst210]] and [[project_demographics_tables]]. T1 regeneration still pending (also still needs the TST-too-short category fix noted in the cohort-change memory).
