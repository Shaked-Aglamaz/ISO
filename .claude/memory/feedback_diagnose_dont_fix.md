---
name: Diagnose, then wait — don't fix without confirmation
description: After investigating a bug/discrepancy, present findings and stop; do not apply fixes until the user confirms direction
type: feedback
originSessionId: 31c46e93-6be4-4f9f-b4dd-31cb8bb42140
---
When investigating a bug or numerical discrepancy, report the root-cause analysis and stop. Do NOT make code changes to "fix" the cause without explicit user approval, even if the fix seems obvious.

**Why:** The user wants to see the diagnosis and decide on the right fix together. Sometimes the "right" fix is a different code path entirely, or it changes downstream interpretation of existing results.

**How to apply:** After finishing the investigation, write a concise summary of: what each pipeline does differently, which difference accounts for which numerical gap, and (if helpful) what a fix would look like. Then wait — don't proceed to edit. This was stated in the context of the AUC-mean discrepancy across step4 / step5 / step6 plots in 2026-04-27.
