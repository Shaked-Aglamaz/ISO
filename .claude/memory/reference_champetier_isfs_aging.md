---
name: reference_champetier_isfs_aging
description: "Champetier 2023 already measured the ISFS peak frequency in aging (C3/C4, Gaussian fit) and found NO group difference - the closest prior work to our headline result; the \"not characterized in older adults\" framing is too strong"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 1ae97b22-eae8-4b40-a397-c2e972d1ba1a
  modified: 2026-08-20T12:51:54.446Z
---

`Age_changes_spindle_memory_consolidation_Champetier_2023.pdf` (already in `library.bib`, cited in the
thesis for spindle clustering and memory) is much closer to our own headline result than the citation
suggests. From its Fig 2 and results text:

- **32 young-middle aged adults (20-52 y, 34.5 ± 10.9) vs 147 cognitively unimpaired older adults
  (65-83 y, 69.3 ± 4.1)**, three pooled cohorts.
- Fast-spindle band power at **C3 and C4 only** (the only electrodes shared across cohorts), FFT of the
  power time course, **Gaussian fit** - the same parameterization we use for peak frequency.
- Grand-average peaks **0.021 Hz (young-middle) vs 0.022 Hz (older)**; individual peaks
  **not different: F(1,175) = 0.164, p = .69**.
- Also: proportion of clustered fast spindles **falls with age** (stable 20-50, then drops), and a
  slower infra-slow oscillation goes with larger spindle clusters (beta = -0.37, p < .001).

Why it matters: our peak-frequency result (0.0199 vs 0.0226 / 0.0232 Hz, ANOVA p = 0.0026) stands against
their null, and Introduction 2.4's "the ISFS has not been characterized in older adults" is too strong
for peak frequency at central electrodes. The thesis is already sent, so this is a **defense-answer**
issue, not a text fix.

The answer (also written into `thesis/defense/qa_prep.md` and the notes of background slide 7, which now
shows their Fig 2): their means move in the **same direction** as ours; their young reference group
averages 34.5 y and reaches 52 while ours averages 27.1 (21-39), so the age contrast is compressed; two
central electrodes cannot show topography, and our strength effect is focal; and nothing was known about
whole-scalp parameters, topography, or aMCI. Raise it before an examiner does.
