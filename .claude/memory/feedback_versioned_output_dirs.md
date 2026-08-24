---
name: Always save to versioned output dirs
description: Group comparison outputs must go to _V{X} dirs, never to unversioned base dirs like three_groups/
type: feedback
originSessionId: 2e5f2b7c-6b26-4b80-8d23-504b6be2622c
---
When saving group-comparison outputs under `results/group_comparison_results/`, always write to the currently-active versioned directory (e.g. `three_groups_V2/`), never to an unversioned base like `three_groups/`. Same rule applies to any sibling workflow that uses `_V{X}` suffixes (e.g. `a_group_plots_V2/`).

**Why:** The user keeps parallel V1/V2/... dirs to track evolution of analyses across parameter/method changes. Writing to the base dir pollutes the workspace, creates ambiguity about which run a file belongs to, and forces manual cleanup. Happened on 2026-04-18 when outputs landed in `three_groups/` instead of `three_groups_V2/`.

**How to apply:** Before running any script that writes to `results/group_comparison_results/` (or similar versioned output areas), check the output path it targets. If it points at an unversioned base dir, update it to the current `_V{X}` dir (ask the user which V is current if unclear — V4 active 2026-04-27, V5 active 2026-04-29). Do not create a new unversioned dir "temporarily."

**Each V is a complete snapshot.** When iterating to a new V, re-run *all* scripts that produce plots in that dir, even ones whose logic is unchanged this iteration — the user wants every V to be self-contained for side-by-side comparison. (Example: 2026-04-29 step6 wasn't being modified, but the user still asked for it to run into V5 alongside the changed step4/step5.)
