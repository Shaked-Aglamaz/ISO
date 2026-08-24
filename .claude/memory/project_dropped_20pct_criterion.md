---
name: project_dropped_20pct_criterion
description: "20%-ISFS-detection exclusion criterion dropped thesis-wide (2026-06-14); 3 young subjects re-presented under 'too many bad channels'"
metadata: 
  node_type: memory
  type: project
  originSessionId: 597b9b42-1ebe-40b1-a3a6-0ebec416d39b
---

**Decision (2026-06-14):** The "valid fit in <20% of channels → ISFS-absent → excluded" rule was removed as an *exclusion criterion* throughout the thesis. Detection is now reported descriptively (Dimitriades-style): mean rate per group + "all subjects exceeded 20% of channels (~35 of ~176)". The 20% number still appears, but only as an observation, never as a pass/fail gate.

**The 3 affected young subjects = EL3017, EL3018, EL3021** (low-ISFS excluded young controls; see [[project_negative_sigma_fix]]). They remain excluded but are now **presented under "Too many bad channels"** in the demographics table — a deliberate framing decision by Shaked, NOT a re-derivation from their actual bad-channel counts. Their true recorded reason in the Google Sheet `excluded` tab ("ISFS rate - X%") is **unchanged** (ground truth preserved).

**How it was implemented:**
- `code/demographics_table.py`: added EL3017/EL3018/EL3021 → `"bad_channels"` in `EXCL_CATEGORY_OVERRIDES` (bypasses `categorize_reason`, Sheet untouched) AND removed the `("low_isfs_rate", ...)` row from `EXCL_CATEGORIES`. Regenerated → overwrote `results/demographics_V2/demographics_table.{png,csv,txt}`. Young "Too many bad channels" now = 4 (EL3045 + the 3); no low-ISFS row. Totals unchanged (YA 9 / HE 5 / MCI 10, AD not counted).
- `thesis/chapters/03_methods.md` §3.4 and `thesis/chapters/04_results.md` §4.2: exclusion sentences replaced with the descriptive "all subjects >20%" wording (no Dimitriades citation, no "no subject was excluded" clause — per explicit user instruction).
- `thesis/cohort_table.md` YA exclusions row: dropped "Low ISFS detection rate", bumped bad channels 1→4.
- Figure captions (`figure_manifest.md` T1, `figures_V9_sigma_fix.md` Table 1) needed no edit — they never named "low ISFS", just "exclusion counts and reasons summarized below", still accurate.
- `thesis/chapters/05_discussion.md` line 15 (stricter *channel-level* narrow-peak criterion caveat) intentionally KEPT — it is about per-channel validation, not subject exclusion.

**Locked numbers in §4.2 (the replacement framing):** denominator = **exactly 176 analyzed channels** per subject (256 − 80 face/neck/ear; 20% ≈ 35 channels). Valid-fit detection: mean **79.6%** overall; per group **YA 73.7%** (range 20.5–100%), **HE 85.5%** (21.0–100%), **MCI 79.2%** (22.7–100%). Re-verified 2026-06-14 directly from the `sigma_fix_{YA,HE,MCI}/*/*_all_channels_summary.csv` headers (mean of per-subject %): YA 73.73 (n=36), HE 85.46 (n=38), MCI 79.16 (n=31), all denominators = 176. **These are correct and paper-locked.**

**Discrepancy with [[project_negative_sigma_fix]] RECONCILED (2026-06-14):** that memory's console means (YA 75.3 / HE 86.6 / MCI 79.2) are a *stale 2026-06-08 batch snapshot*, taken before two later changes. YA: batch was n=35; **EL3033 (20.5%, group min) added afterward** → (73.73×36 − 20.5)/35 = 75.25 ✓ explains 75.3→73.7. HE: **RY42 reprocessed (projector strip, 2026-06-09) to 21.0%** (was ~64% in the batch) → 86.6→85.5 ✓. MCI: no changes → 79.2 unchanged ✓. Mean-of-per-subject-% throughout (not pooled).

Related: [[project_isfs_method_vs_dimitriades]], [[project_demographics_tables]], [[project_negative_sigma_fix]].
