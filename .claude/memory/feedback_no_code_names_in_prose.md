---
name: feedback_no_code_names_in_prose
description: "Thesis/paper prose rule (2026-06-07): never name the user's own scripts, files, functions, or internal code identifiers in prose; external packages are fine"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 10d13e89-eb06-49fb-9325-3482985949df
---

In all ISFS paper prose (Methods/Results/Intro/Discussion/captions), **do NOT use any of the user's own code names** — no script filenames (`step0_mff_cleaning.py`, `main_loop.py`), no module/file paths (`code/new_iso/...`), no internal function names (`normalize_subject_channels`, `spatio_temporal_cluster_test` as a *function ref*), and no reference-implementation filenames (`f_GaborWavelet.m`).

**Allowed:** names of external/third-party packages and platforms the analysis used — MNE-Python, YASA, Visbrain, SleepEEGPy, NumPy, SciPy, pandas, statsmodels. These are standard to cite in a Methods section.

**Why:** the manuscript describes *what was done*, not the repository layout; internal code identifiers are implementation detail that doesn't belong in a paper.

**How to apply:** describe the method/algorithm in plain scientific language. If a verification note must point at code, keep it in a `> [VERIFY]` blockquote marker, not in the prose body. Strip code refs when revising older drafts on sight.

Related: [[feedback_thesis_prose_rules]], [[feedback_caption_conventions]]
