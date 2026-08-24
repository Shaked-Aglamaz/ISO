---
name: project_pending_verifications
description: "RESOLVED 2026-06-17 — the bad_channels.txt verification (SM13/SM14/VZ9) is confirmed fine; no open verifications currently"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e0bf61-82d0-4afc-bb1a-ad690a607b82
---

**RESOLVED 2026-06-17.** No open pre-finalization verifications remain. (If new ones arise, log them here.)

**Bad-channels file missing for MCI subjects** — closed:
- Originally flagged 4 (`SM07`, `SM13`, `SM14`, `VZ9`) with `has_bad_channels_file = FALSE` (discovered 2026-05-28).
- `SM07` is now **excluded** from the cohort (TST < 210 rule, see [[project_cohort_change_mr5_tst210]]), so it no longer matters.
- `SM13`, `SM14`, `VZ9` remain in the final MCI=30 cohort (have `results/sigma_fix_MCI/` dirs); **user confirmed 2026-06-17 that running them without an explicit bad_channels.txt is fine** (nothing to mark). Methods §3.2 semi-automatic bad-channel statement stands. Do not re-raise.

Related: [[project_thesis]], [[reference_subjects_sheet]]
