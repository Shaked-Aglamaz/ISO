---
name: feedback_edit_main_loop_directly
description: "For ad-hoc ISFS re-runs, edit main_loop.py config in place — don't create a separate driver script"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4d4e7585-dae6-453e-9521-9e360f440c6a
---

When re-running the ISFS pipeline on a different subject set / output dir, edit the config globals at the top of `code/new_iso/main_loop.py` directly (`SUBJECTS`, `source_folder`, `RESULTS_DIR`) instead of writing a new wrapper/driver script that imports `main_loop` and overrides them.

**Why:** User prefers keeping one canonical entry point; extra driver scripts clutter `code/new_iso/` and diverge from the main script.

**How to apply:** Modify `main_loop.py` in place for the run. (Per CLAUDE.md, never commit — local edits only.) Related: [[project_new_mci_subjects]] (main_loop is the standard ISFS entry point).
