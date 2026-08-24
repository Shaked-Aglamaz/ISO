---
name: Answer diagnostic questions in chat before changing code
description: When the user asks "why", "do we", "what's happening" — explain first, don't preemptively modify code or files.
type: feedback
originSessionId: b5e4b0d5-37ca-457e-af8c-10aeb0e211d6
---
When the user asks a diagnostic or clarifying question ("do we exclude X?", "why is Y this way?", "how does Z work?"), respond with an explanation in chat. Do not reach for Edit/Write as the first action.

**Why:** During the ROI plotting work the user asked "do we exclude some of the electrodes in the plot?" and I immediately edited code to "fix" it. The user corrected me: *"i didnt say to change anything just answer in the chat"*. They were observing behavior and wanted understanding, not a change — my preemptive edit wasted a round trip and forced a revert.

**How to apply:** Treat question-shaped messages as requests for explanation unless they also contain an explicit instruction ("change X", "fix Y", "make it do Z"). Only propose or make code edits after the user either (a) explicitly asks for a change, or (b) confirms that a change is what they want after I've described the situation.
