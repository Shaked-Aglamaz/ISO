---
name: feedback_stop_if_gdoc_unreadable
description: "Asked to read a Google Doc and the docs MCP isn't available: STOP and say so immediately — never read local chapters/exports instead"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 78a0da0a-d560-4f35-ab38-2eba463dd0b8
  modified: 2026-08-14T11:05:43.780Z
---

2026-08-14: asked to read "thesis V2" (the Google Doc) and check aMCI vs MCI usage. The
`google-docs` MCP tools were not present at session start, so I read the local
`thesis/chapters/*.md` sources instead and answered from those, noting the substitution. The user
replied: *"try again the google doc"* — the server WAS available on the second turn — then:
*"when i tell you to read a google doc and you cant do it - just stop and tell me that you cant so
ill fix that!!! dont read anything else instead"*.

**Why:** the doc is the live artifact and can be ahead of the local files (V2 was modified
2026-08-13, the chapters on 2026-08-08), and only the doc has the title page, table/figure captions,
and reference list. A "close enough" local proxy costs the user a round trip and risks an answer
based on stale text. The user can fix the MCP in seconds — they just need to be told. Note this is
STRICTER than [[feedback_verify_named_data_source]]: for Google Docs the rule is not
"substitute and verify", it is **do not substitute at all**.

**How to apply:** when a Google Doc is the named source and the `google-docs` MCP tools are absent
or erroring — stop, in one short sentence say the docs MCP isn't available and ask them to fix it.
Do not read local chapters, .docx exports, review notes, or anything else as a stand-in, and do not
answer the question from them. Recovery steps live in [[reference_manuscript_gdoc]] (server missing
at session start ≠ auth; `invalid_grant` → user re-runs `npx @a-bonus/google-docs-mcp auth` with
the Drive scope ticked, then restarts Claude Code) — surface those, but let the user act.
