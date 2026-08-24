---
name: feedback_match_effort_to_problem
description: "Don't over-investigate setup/config problems — ask the user what they did or what the error said first"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: de7dfea8-edf4-4a4c-8c33-640bc4398acb
  modified: 2026-08-04T08:54:48.995Z
---

For environment/config/auth problems, **ask the user the one question that would resolve it before mounting an investigation.** They usually already know what happened — they performed the setup step.

Called out verbatim 2026-08-04: "you overkilled, it was me who gave access only to 2 of the 6." I had read the MCP package source, minted an access token to enumerate granted scopes, and written a full Google Cloud Console walkthrough plus a three-option tradeoff table — to explain a checkbox the user had knowingly left unticked. One question ("did you tick Drive on the consent screen?") would have replaced all of it.

**Why:** the user is a competent operator of their own machine, not a bug report. Long branching guides for a problem they can name in five words waste their reading time and bury the one thing they need to do.

**How to apply:**
- Config/setup/auth misbehaving → ask what they clicked or paste-me-the-error, *then* dig if the answer doesn't explain it.
- Don't pre-write both branches of a fork the user can collapse instantly. One question beats a decision table.
- Verifying a claim with a real probe is still right (the scope check *did* prove the cause); the excess was the unsolicited remediation guide around it.
- Same instinct as [[feedback_answer_before_acting]] and [[feedback_diagnose_dont_fix]]: default to answering, not to producing.
- Applies to prose length in chat too — when the user asks "check now", lead with the result, not the reasoning that produced it.
