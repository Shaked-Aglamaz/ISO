---
name: reference_manuscript_gdoc
description: "Manuscript Google Doc id + full read/write/Drive access, and how its OAuth breaks"
metadata: 
  node_type: memory
  type: reference
  originSessionId: e4373938-8e3c-4435-af4c-1c92698d7513
  modified: 2026-08-17T06:06:02.360Z
---

The ISFS manuscript lives in a Google Doc the user maintains. **CURRENT VERSION = "Shaked's Thesis V2",
id `1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`** (~42k chars, post-Flavio-review; the user confirmed
2026-08-08 that this is the copy going to the PI). Table 1 is a real 10×4 table object; all 7 figures are
embedded as images (V10 assets, verified by image-matching against the source PNGs).

**The old doc `1miqNSnMpbX0MTCSGPKCWCquxpomu2szl5MLkkJwRteU` ("Shaked's Thesis") is SUPERSEDED** — it is the
**pre-review** version (old title, "kernel density", "negative result") and was deliberately left untouched.
Don't edit it. See [[project_flavio_review]].

**ALWAYS work inside the Google Doc — never make a local copy of the manuscript.** Instruction from the user
2026-08-12, after I exported V2 to `thesis/reviews/Shaked's Thesis_V2.docx` to diff it against Yuval's
returned file; the export was deleted on request. The doc is the single source of truth, and a local .docx
goes stale the moment either side is edited, which is exactly the confusion that caused the
[[project_yuval_review]] mess. Read with `readDocument` / `getTableStructure`; **edit in place** with the
Docs tools. Reviewer copies people send back (`Shaked's Thesis_YN.docx`, `_FS.docx`) are fine to keep — they
are received artifacts, not exports.

**Reading the doc:** `readDocument` (format `text`) **silently omits table cell contents** — use
`listDocumentTables` + `getTableStructure` for Table 1. Inspecting embedded images is the one job that needs
a `downloadFile` .docx export (unzip `word/media/*`; Google re-encodes to max 2048 px, so match by
downscaled-pixel correlation, not by hash; figure display sizes and page setup come from `<wp:extent>` /
`<w:pgSz>` in `word/document.xml`) — **write that export to the scratchpad, never into the repo, and delete
it when done.**

**Drive scope now GRANTED (2026-08-04) — full toolset works:** `getDocumentInfo`, `listComments`/`addComment`/`replyToComment`, `insertImage`, `searchDriveFiles`, `downloadFile`. Doc = "Shaked's Thesis", created 2026-06-22, owner **shaked.aglamaz@gmail.com** (personal Gmail, so the Workspace "Internal" OAuth route is NOT available), shared=true. No comments on it as of 2026-08-04 (`listComments` → `[]`) — any Yuval feedback would be on a different copy. Edits I make appear as the user (`lastModifyingUser` = Shaked).

**If any Drive-backed tool 403s as "Permission denied": it's a scope problem, NOT sharing — never ask the user to re-share.** The package relabels every HTTP 403 as that. `getDocumentInfo` → `drive.files.get`; the Drive-requiring tools all call `getDriveClient()`. Diagnose properly instead of guessing: refresh the on-disk token against `https://oauth2.googleapis.com/token` and read the `scope` field in the response (a real 403 body says `ACCESS_TOKEN_SCOPE_INSUFFICIENT`). The diagnostic was a throwaway scratchpad script, not kept — re-derive it if needed: client id/secret from `~/.claude.json`, refresh_token from the token path below.

**The auth flow silently accepts partial grants.** It requests 6 scopes (documents, drive, spreadsheets, gmail.modify, calendar.events, script.external_request) and only logs what came back. The 2026-08-04 Docs-only state happened because the **user left the Drive checkbox unticked** on the granular consent screen (NOT a console misconfiguration — I wrongly guessed console config first). So: when re-authing, tell them to tick Drive. Docs+Sheets+Drive is the working set; Gmail/Calendar intentionally ungranted.

**Also: the MCP loads the token ONCE at server startup.** After a re-auth, tools keep failing until Claude Code restarts or `/mcp` reconnects `google-docs`. Check `token.json` mtime vs. when the session started before re-diagnosing anything.

**If the server never connects at session start, that is NOT an auth problem — don't send the user to re-auth.** Hit this 2026-08-08: the startup reminder listed `google-docs` as "still connecting" and it never registered any tools, so no `mcp__google-docs__*` existed all session (ToolSearch found nothing; MCP tools register only at startup, so nothing recovers mid-session). Diagnose in one step by launching the server yourself with the client id/secret from `~/.claude.json`: `timeout 45 npx -y @a-bonus/google-docs-mcp </dev/null 2>&1`. It printed `Using saved credentials` / `authorized successfully` — token fine, it had just missed the connect window on a cold `npx` resolve. Fix = user restarts Claude Code (the manual launch also warms the npx cache, so the reconnect is fast). Concurrent edits from another session are NOT a cause.

**Correction to the line above, learned 2026-08-15: "nothing recovers mid-session" is wrong.** The server can also **drop mid-session and come back by itself.** It did: all 119 `mcp__google-docs__*` tools vanished partway through a working session (system-reminder: "MCP server disconnected"), then a later reminder announced them as available again and `getDocumentInfo` worked immediately — no restart, no re-auth. So on a mid-session disconnect: say so plainly, do the repo-side work, stage the exact Doc edits, and try again a turn or two later before sending the user to restart. The startup-window failure in the paragraph above is a different, genuinely unrecoverable case.

**If every google-docs call returns `invalid_grant` (400), the refresh token expired and only the user can fix it** (hit this 2026-08-04). That expiry came from the consent screen being in *Testing* (~7-day refresh tokens); it has since been published to production, so this should not recur — but if it does, this is the fix. The MCP is `@a-bonus/google-docs-mcp` (user-level in `~/.claude.json`, NOT the project `.mcp.json` which only has google-sheets); refresh token at `C:\Users\Shaked\.config\google-docs-mcp\token.json` (holds only `refresh_token`, so its mtime = last re-auth). Fix — the user runs this in a **PowerShell** window (browser sign-in, can't be automated), then reconnects via `/mcp` or restarts:
```powershell
$env:GOOGLE_CLIENT_ID='<from ~/.claude.json>'; $env:GOOGLE_CLIENT_SECRET='<from ~/.claude.json>'
npx -y @a-bonus/google-docs-mcp auth
```
Permanent cure — **DONE 2026-08-04.** The OAuth consent screen (project number `769851892876`, a *different* project from the google-sheets service account's `refined-spirit-494512-e8`) was published from *Testing* to **In production**, and the token re-minted afterwards (token.json 11:51), so it should no longer carry the 7-day Testing expiry. Unverified production + restricted `auth/drive` did NOT hard-block — the first re-auth attempt failed but a retry succeeded. **Not yet proven:** non-expiry is only confirmed by the token still working after ~2026-08-12. If `invalid_grant` returns anyway, the publish didn't affect token lifetime → just re-auth (command above) and treat it as a weekly chore. Console: https://console.cloud.google.com/auth/audience?project=769851892876 ("Back to testing" reverts). Personal-Gmail owner → Workspace "Internal" user type never available.

Created 2026-06-22 via File → Save as Google Docs from the original **.docx** (id `1ANk_5qAwX-Dzcqhm8_oN_ykEEaPQBFKF`), which could not be read or edited at all — Docs tools reject Office files ("must not be an Office file"). The conversion minted a NEW id (old link unchanged, sharing not inherited). Don't go back to the .docx.

**Editing mechanics — learned the hard way 2026-08-06, saves re-deriving:**
- **Citation markers are literal Unicode superscript glyphs** (`¹⁻²`, `⁶⁻⁷`, `¹⁴`) carrying **no formatting at all** — every run is plain `{"fontSize": 12 PT}`, no `baselineOffset`. Verified via the Docs API. So find-and-replace can span them freely and newly typed markers render identically. Don't waste effort protecting superscript formatting. (Do still check for **italic** runs — the Gaussian formula variables `a`/`f`/`b`/`c` are italic; chunk replacements around them.)
- **`findAndReplace` cannot match across a paragraph mark**, so it can neither delete a paragraph nor fuse two into one. Consequences: to delete a whole paragraph use `findElement` → `deleteRange(start, textEnd+1)` (the +1 eats the `\n`, else you leave an empty paragraph); to *insert* a paragraph, `\n` inside `replaceText` works fine.
- **Inserting a paragraph right after a heading**: `insertText` at the heading-follower's `startIndex` makes the new paragraph inherit the *heading* style — follow with `applyParagraphStyle{namedStyleType: NORMAL_TEXT}`.
- **Replacing a whole chapter: INSERT FIRST, THEN DELETE** (learned 2026-08-16, worked cleanly on the
  Introduction). Insert the new text at the `startIndex` of the *first existing body paragraph* — a
  `NORMAL_TEXT` one — so every inserted paragraph inherits `NORMAL_TEXT` and the
  "inserting-after-a-heading-inherits-the-heading-style" trap never arises. Then locate the old block
  by `findElement` on a string it shares with the new text and taking **instance 2**, and delete
  `[oldStart, nextHeadingStart)`. No index arithmetic, and the doc is never left empty if a step
  fails. Same trick rebuilt the 49-entry reference list: inserting at the start of the first list
  paragraph makes all new paragraphs inherit the **auto-numbered list formatting** (verified: 49/49
  kept one `listId`), then delete `[myLastEntryNewline, oldListEnd)` so the document's final newline
  survives.
- **Bulk `updateTextStyle` is far cheaper through the Docs API than through the MCP.** Re-linking 49
  DOIs would have been 49 `applyTextStyle` calls; one direct `batchUpdate` with 49 `updateTextStyle`
  requests did it in a single call. Ranges stay valid within a batch because text length is unchanged.
  Refresh the token as described at the bottom of this note.
- **Verify numbering on the invariant, not by eye:** fetch the doc via the API, strip the reference
  section, tokenize every Unicode-superscript run (expanding `⁻` ranges and `,` lists) and assert that
  **the n-th distinct number encountered top-to-bottom is n**. That single check catches gaps,
  duplicates, orphans and misordering at once. Script pattern lives in
  `thesis/reviews/intro_edits_before_after.md`.
- **The reference list is an auto-numbered Docs list, ordered by first appearance** (added 2026-08-15). The numbers are list formatting, not text — they never appear in `readDocument` output, and inserting a paragraph into the list renumbers it automatically. Consequence: **a citation added in the Intro or Methods renumbers every superscript after it** — adding `maris2007nonparametric` to Methods 3.6 shifted refs 23–29 to 24–30. Renumber existing markers **descending** (highest first) so no two ever collide, anchor each replacement on surrounding words rather than the bare glyph, then insert the new marker and the list entry last.
- **The doc is now too big for `readDocument` to return inline** (2026-08-17, ~75k chars text / ~418k
  JSON). Both formats error out and dump to a file under the session's `tool-results/`. That is fine and
  is in fact the better workflow: **parse the dump with Python** rather than reading it back. Two scripts
  worth re-deriving — (a) split at the last `\nReferences\n`, count entries and print them numbered, to
  verify the list; (b) regex every superscript run in the body over `[⁰¹²³⁴⁵⁶⁷⁸⁹⁻,]+`, expand `⁻` ranges
  and `,` lists, collect **order of first appearance**, and assert it equals `1..N` with `N` = list
  length. That one assertion catches orphans, duplicates, dangling numbers and mis-ordering at once.
- **Caption anatomy (learned 2026-08-17, needed for every figure pass).** A caption is ONE paragraph,
  `alignment: JUSTIFIED`, every run 12 pt and `#0070C0`, split into exactly two runs: a **bold** title
  sentence ending in a space, then the non-bold body. **Never findAndReplace across both runs** — the
  Docs API gives the whole replacement the first run's formatting, so the entire caption turns bold.
  Replace the title and the body separately and each keeps its own styling. To build a caption from
  scratch: `insertText`, then `applyTextStyle` over the whole range with `foregroundColor #0070C0` +
  `fontSize 12` + `bold false`, then `applyTextStyle {textToFind: <title>}` with `bold true`, then
  `applyParagraphStyle` JUSTIFIED / NORMAL_TEXT.
- **Image geometry cannot be set through the MCP at all.** No tool resizes an image, changes
  inline-vs-floating, or moves one. Every such change is a manual browser step — plan figure passes
  around that, and hand the user exact inch dimensions. **Text area is 468 × 648 pt (6.5 × 9.0 in)**
  on this doc's 612 × 792 pt page with 72 pt margins.
- **A floating image does not appear in the paragraph stream.** It lives in the document's
  `positionedObjects` (layout `WRAP_TEXT`) and is referenced by `positionedObjectIds` on its anchor
  paragraph. An audit that only walks `paragraph.elements` will silently miss it and report the figure
  as absent. Check `positionedObjects` explicitly.
- **Pasting an image with the cursor on a heading anchors it INSIDE the heading paragraph.** Hit
  2026-08-17: a supplementary figure ended up as an `inlineObjectElement` inside the
  "Supplementary figures" HEADING_2. Fix without touching the image: `insertText("\n")` at the image
  element's `startIndex` to split the paragraph, then `applyParagraphStyle` NORMAL_TEXT on the new one.
- **Native tables are the right answer for wide tables.** A 20.5 in table PNG shrinks ~3x at page width
  and renders ~6 pt type. `insertTableWithData` (+ `hasHeaderRow`) then `updateTableColumnWidth` and
  `updateTableCellStyle` reproduces it legibly and selectably. House style: Table 1 is 4 columns at
  ~121 pt, 10 pt type, header row `#D9D9D9` bold and centred; Table 2 is 10 columns at 9 pt with cell
  padding cut to 2 pt so `629.0 ± 174.5` stays on one line, and band rows filled `#DCE6F1`.
  **Per-cell text styling is the one thing to do through the Docs API**, not the MCP — walk
  `table.tableRows[].tableCells[].content[].paragraph.elements[]` for each run's index range and send
  one `batchUpdate` (194 requests in one call, vs ~40 separate `applyTextStyle` calls).
- **Renumbering figures is usually a cycle**, and no ordering of find-and-replace avoids a collision
  (S3→S1, S1→S2, S2→S3). Route every label through a unique temporary token first (`Figure SX1` …),
  then resolve the temporaries. Anchor each first-pass replacement on surrounding words, and expect the
  resolve pass to report exactly 2 hits per label: the caption title and the in-text mention.
- **No API can insert a table of contents.** Not the MCP, not raw Docs `batchUpdate` — there is no
  request type. Reading an existing one works (`tableOfContents` structural element), creating one
  does not. It is always a manual browser step: Insert → Table of contents. Prepare the page and hand
  it over. Corollary: whether a section appears in the generated list is controlled purely by whether
  its heading is a real `HEADING_*` — use bold centred `NORMAL_TEXT` for front-matter headings you
  want excluded.
- **RTL / Hebrew: only through the raw API.** `applyParagraphStyle` in the MCP has **no direction
  field**. Use `updateParagraphStyle` with `paragraphStyle.direction = "RIGHT_TO_LEFT"` and
  `fields` including `direction`. Latin acronyms and `p = 0.023` inside an RTL paragraph are handled
  correctly by the bidi algorithm — do not try to fix them by hand.
- **`pageBreakBefore` beats inserting page-break elements.** A `ParagraphStyle.pageBreakBefore = true`
  is one field on the paragraph you already have to restyle, and it does not add an element that
  later index arithmetic has to step over. This doc's two original breaks are elements; everything
  added 2026-08-17 uses `pageBreakBefore`.
- **`replaceAllText` DOES create a new paragraph from `\n`** (verified 2026-08-17 splitting the
  title-page department line into department + faculty). Both halves keep the original run styling.
  This is the cheap way to split one paragraph into two, and it contradicts the `findAndReplace`
  limitation noted above — that one is the MCP wrapper, this is the raw API.
- **Text inserted into a body paragraph inherits its *explicit* run styling**, which defeats
  `namedStyleType`. This doc's body runs carry an explicit `fontSize: 12`, and its real headings carry
  `textStyle: {}` — so a heading inserted at a body paragraph's start renders at 12 pt until you
  clear it: `updateTextStyle` with `fields: "fontSize,bold,italic"` and an empty `textStyle`. Same for
  the paragraph's `JUSTIFIED` and `spaceBelow`: list them in `fields` so they are cleared, not merged.
- **Batch many index-shifting inserts in ONE `batchUpdate` by ordering them highest-index-first.**
  Requests run in order, so descending targets never invalidate each other and no re-fetch is needed
  between them (21 requests, 7 headings, one call, 2026-08-17).
- **Reference-entry hyperlink style:** `underline: true`, `foregroundColor #1155CC`, 12 pt, plus the
  `link.url`. `applyTextStyle` with `textToFind` = the DOI string sets all four in one call.
- **Correction to the "inserting after a URL run inherits the hyperlink" note below: it did NOT happen**
  when appending nine new entries at the end index of the final reference paragraph (2026-08-17). The
  inserted text came in as plain single runs with no link at all, verified against the JSON dump. So
  append freely at the end of the list, then link each new DOI. Insert *mid*-list at the **start of the
  following paragraph** as before. Deleting several consecutive entries is easiest as one
  `deleteRange(firstEntryStart, nextKeptEntryStart)`.
- **The doc uses straight apostrophes (`'`), not curly** — verified with `findElement` on "each subject's per-channel values". Match and type them straight or `findAndReplace` silently finds nothing.
- **Inserting a list/reference entry after one that ends in a URL**: insert at the *start of the following* paragraph, not at the end of the URL paragraph, or the new text inherits the hyperlink run. Downside: the new entry's own DOI stays plain text — fix by hand if it matters.
- **PDFs cannot be read with the Read tool on this machine** — `pdftoppm` (poppler) is not installed,
  so page rendering fails outright. **`pypdf` is installed** both system-wide and in `eeg_clean`; use
  `PdfReader(...).pages[i].extract_text()`. Matters for every PDF under `thesis/references/`
  (Lázár, Dimitriades, Grollero, the slide deck, [[reference_yael_gat_thesis]]).
- **No LaTeX engine on this machine** (no xelatex/pdflatex; pandoc IS on PATH). The `thesis/reviews/flavio_comments_*.pdf` files were built with **xhtml2pdf** (`markdown` → HTML+inline CSS → `pisa.CreatePDF`); reportlab and xhtml2pdf are installed, weasyprint/wkhtmltopdf are not.
- For direct Docs API work (e.g. inspecting run styles), refresh the token yourself: client id/secret from `~/.claude.json`, refresh_token from `C:\Users\Shaked\.config\google-docs-mcp\token.json`, POST to `https://oauth2.googleapis.com/token`.

Source of truth for figure/table **captions** = `thesis/figure_manifest.md` (the `> *Figure N. Title.* …` blocks). Caption title = bold + blue in the doc; paste as plain text to preserve it. Body prose = `thesis/chapters/*.md`. See [[project_paper_figure_set]], [[feedback_caption_conventions]].
