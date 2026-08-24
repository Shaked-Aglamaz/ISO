---
name: Google Sheets MCP gotchas
description: Setup quirks and access gotchas for the google-sheets MCP on this machine — what trips up reads/writes
type: reference
originSessionId: b644f67d-f7e4-4907-bfdd-a55eb0188a8c
---
The google-sheets MCP is configured at project root in `.mcp.json` (gitignored), running `xing5/mcp-google-sheets` via `uvx` with service account auth.

Service account: `google-sheet-mcp@refined-spirit-494512-e8.iam.gserviceaccount.com`
Key file: `C:\Users\Shaked\.gcp\refined-spirit-494512-e8-059d94c0eb34.json`
uv binary: `C:\Users\Shaked\.local\bin\uvx.exe` (full path needed in `.mcp.json`)

**Sharing options (either works):**
1. Share the sheet directly with the SA email (Viewer for read, Editor for write).
2. Set "Anyone with the link → Viewer/Editor" — the SA gets in via the public link without explicit share.

A SA cannot access via "anyone with link" implicitly without that link-sharing being turned on; it's a 403 otherwise.

**xlsx-in-Drive does NOT work**: if a sheet's URL has `rtpof=true&sd=true` or comes from "Open with Google Sheets" of an uploaded .xlsx, the Sheets API returns "This operation is not supported for this document". Fix: open it → File → Save as Google Sheets → use the new native sheet's URL.

**Workflow that's worked well for this user:**
- Read both source and target tabs with `get_multiple_sheet_data`.
- Match by exact ID. List unmatched IDs back to user; don't guess naming variants without confirmation.
- Show the proposed write plan as a table (rows + values) before calling write tools — user has approved every batch this way.
- Use `batch_update_cells` (one call per sheet, dict of A1-range → 2D values) to bundle related writes; same response covers all ranges.
- Verify ages whenever filling sex from another source — flag mismatches, then update target only on user confirmation that source is authoritative.

**Known recurring translations** (likely to come up again on this project):
- Hebrew sex column: `זכר` = M (male), `נקבה` = F (female).
