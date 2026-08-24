---
name: Use PYTHONIOENCODING=utf-8 for background Python runs
description: Windows cp1252 console crashes on Unicode chars (checkmarks, emojis) in print statements — always set PYTHONIOENCODING=utf-8
type: feedback
originSessionId: aa491c21-6c1d-4a0f-9787-8dbd24523a46
---
Always run Python scripts with `PYTHONIOENCODING=utf-8` on this Windows machine when running in background or redirecting stdout. The default cp1252 encoding crashes on Unicode characters like checkmarks and emojis in print statements.

**Why:** Multiple pipeline scripts (step2, main_loop) use `print(f"✓ ...")` which throws `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` on cp1252. The TeeOutput class in step2 was also patched to catch encoding errors, but PYTHONIOENCODING is the cleaner universal fix.

**How to apply:** Prefix Python commands with `PYTHONIOENCODING=utf-8`, e.g.:
```bash
PYTHONIOENCODING=utf-8 python code/new_iso/main_loop.py
```
