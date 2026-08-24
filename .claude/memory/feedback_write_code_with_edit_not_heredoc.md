---
name: feedback_write_code_with_edit_not_heredoc
description: "On this machine a bash heredoc mangles backslash escapes and backticks, so patch Python with Write/Edit instead of python - <<EOF"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1ae97b22-eae8-4b40-a397-c2e972d1ba1a
  modified: 2026-08-23T13:08:16.603Z
---

Patching source files by piping a heredoc into python (`python - <<'PYEOF' ... PYEOF`) is unreliable here:
a `"\n"` inside the heredoc arrives as a real newline, which silently writes a broken string literal into
the file, and backticks inside a double-quoted `python -c` string get run as command substitution (that is
how a MEMORY.md line lost all its paths in one edit).

**Why:** it cost two syntax errors and one corrupted memory line in the defense-deck session (2026-08-20).

**How to apply:** use the Write and Edit tools for any file content containing `\n`, `\t`, backticks or
quotes. Heredocs are fine for pure-text edits with none of those; when a script really must be generated,
write it to a scratch file with Write and splice by line number. Also keep `PYTHONIOENCODING=utf-8` on
every python invocation (see [[feedback_pythonioencoding]]) — writing `→` or `±` without it raises
UnicodeEncodeError on cp1252.
