---
name: Subjects info Google Sheet
description: The native Google Sheet holding subject-level metadata (age/sex/group/processing status), accessible via the google-sheets MCP
type: reference
originSessionId: b644f67d-f7e4-4907-bfdd-a55eb0188a8c
---
Spreadsheet ID: `1jVZb8vTaOnfH5fcz_KAjtU8qPxKgYs5Oi_-Ji01Yomg` (native Google Sheet, not .xlsx)

Three tabs:
- `all_data ` (note trailing space) — wide layout, 3 groups side-by-side: Young A–J, Elderly L–Q, MCI S–X. Master source of age/gender/exp date/group.
- `data_overview_with_analysis` — single long table per subject with cleaning/processing metadata (cleaning id, hypno, raw mff, cleaning 1/2/3, bad_channels, bad_epochs, total_time, has_unknown, etc).
- `included in analysis` — three groups side-by-side (A–C young, D–F elderly, H–J MCI) with `ID, age, sex`, plus an `excluded` list (col M) with reason (col N) and a `control from maya` list (col R+). User does NOT want columns from M onward touched.

Group cross-listing: some subjects appear in `included in analysis` under a different group than `all_data ` (e.g. EB34, NE36, SB00, LS56 are Young in `all_data ` but listed under Elderly in `included`). Match by ID across all groups when reconciling, not by column position.

Related sheets seen (also accessed via the MCP):
- Hebrew young-controls intake form: `1YP526pF2J8_p8QvvYQtphlKQLgepH6tNjLSEQl7b1o8` — single tab "תגובות לטופס 1". Subject ID col B (e.g. `3010`) maps to `EL3010` in our sheet. Sex col C in Hebrew: `זכר` = M, `נקבה` = F. Age col D.
- MCI001-series source (sex/age, no MoCA): `1QqDGN7dSCCI-qtLFD_LYhyI4mYnk8wP7yeu-MnQwUlk` — uses `MCI0XX` 3-digit format (single I, not "MCII"); has `Old ID`/`New ID`/`age at BL`/`sex`/`group` cols (group values: naMCI, aMCI, SMC, control). Sharing was set to "anyone with link viewer".
- Cognitive scores source (MoCA/MMSE): `1TRuUQwNQLyAK85jbIS8B3KX3DWyOkyOou2flyvp_pTg` — 4 tabs: `MCI\AD`, `Control`, `Returning`, `Analysis`. MCI\AD tab has Subject Code col A, Group col E (values: MCI, AD, ?, X, blank), Gender col F, MMSE col O, MoCA col P. Control tab is structured slightly differently (no Group/Leqembi cols): Subject Code col A, MMSE col M, MOCA col N. Some subjects coded as "X/Y" (combined IDs, e.g. `RV66/RB88`, `SM002/SC5`, `SM005/KS5`). Note: MCI001–MCI043 subjects are NOT in this MoCA source.

ID naming conventions in `included in analysis` (after 2026-04-26 cleanup):
- Young: `EL30XX` (4-digit suffix); Hebrew form has the bare `30XX`.
- MCI new naming: `MCI0XX` (3-digit, single I). User had typo'd shorter versions (`MCI03`, `MCI20`...) — fixed to add leading zero.
- Original SM-series: `SM0XX` (3-digit). User had typo'd both shorter (`SM07`) and over-zeroed (`SM0016`) versions — fixed to canonical 3-digit.
- After 2026-04-26 cleanup all groups complete except 1 still-blank entry (user filled GZ8 and one of VZ9/KS5 manually).

Sheet now has MoCA + comment columns for elderly (cols G, H) and MCI (cols L, M). Young still has only ID/age/sex.

Diagnostic group sanity-check (2026-04-26): user's MCI list contained no AD patients per MCI\AD source. Subjects flagged but kept by user: VZ9 (source: "Excluded High MOCA"), GZ8 (source: "Excluded Low MOCA"), IS74 (source: "Excluded Low MoCA"), KS5/SM005 ("not from Noa's clinic, need to check"), SM014 (Group=?). RB88 has age conflict — `all_data ` says 75, Control source says 77; sheet currently shows 75 per user's call to trust `all_data `.

MCP access: project-level `.mcp.json` at repo root, service account `google-sheet-mcp@refined-spirit-494512-e8.iam.gserviceaccount.com`. Key file at `C:\Users\Shaked\.gcp\refined-spirit-494512-e8-059d94c0eb34.json`. `.mcp.json` is gitignored.
