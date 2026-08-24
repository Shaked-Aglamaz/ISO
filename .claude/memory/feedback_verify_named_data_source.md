---
name: feedback_verify_named_data_source
description: "When a note/spec names a data source, don't silently substitute an equivalent-looking one — verify equivalence over the whole cohort and say so up front"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6349a5e9-7b93-4a57-af24-3a6b05200f69
  modified: 2026-08-13T12:49:00.666Z
---

2026-08-13: `notes/tst_waso_unknown_handling.md` said to derive TST/WASO from the hypnogram
(`ISO_data/scoring/{group}/{sub}.txt`). I used `*_cleaned_annotations.txt` instead — defensible, and
documented in the approved plan — but validated it on **one** subject (AT36) and reported the numbers
without flagging the substitution. The user caught it: *"i thought the note told you to read the hypno
file!"*

**Why:** the substitution turned out to be exactly equivalent (89/93 subjects 100.000% identical
second-by-second; SOL and REM latency identical for all 93), so the *result* was fine — but the user
had to spend a turn discovering that, and if it had NOT been equivalent the numbers would already have
been reported as final. A one-line "I read X instead of Y because Z, verified equivalent on N
subjects" would have cost nothing.

**How to apply:** when a spec, note, or CLAUDE.md names a specific input file, either read that file,
or (a) verify equivalence across the **whole cohort**, not one spot-check, and (b) state the
substitution and its verification in the same message as the results — not only in a plan file the
user skims. When challenged on a choice like this, test it rather than re-arguing the reading of the
note; the empirical check settled it in one pass. Related: [[project_unknown_vs_acq_skip]],
[[feedback_diagnose_dont_fix]].
