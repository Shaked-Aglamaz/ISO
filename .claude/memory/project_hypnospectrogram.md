---
name: project_hypnospectrogram
description: hypnospectrogram.py — sleepeegpy SpectralPipe hypnogram+spectrogram figures; per-subject hypno_freq and letter/int coding gotchas
metadata: 
  node_type: memory
  type: project
  originSessionId: dfa281c5-0b02-488b-8cc9-ca5d25b535ec
---

`code/hypnospectrogram.py` generates combined hypnogram+spectrogram figures using `sleepeegpy.pipeline.SpectralPipe.plot_hypnospectrogram`. Plots channel **E101**, win_sec=10, freq 0–25 Hz, cmap Spectral_r; saves PNGs to `results/hypnospectrograms/{sub}_hypnospectrogram.png`. The hardcoded `CASES` list is just a swap-in target — edit it per run. Originally 3 example subjects (EL3011 young, MCI27 elderly, YS0 MCI). **As of 2026-06-01 `CASES` points at 4 excluded young-control subjects** (EL3026, EL3030, EL3032, EL3040) from `control_clean\a_excluded\`; their old EL3011/MCI27/YS0 PNGs (May 6) remain since they weren't re-run.

Run with `PYTHONIOENCODING=utf-8 ./eeg_clean/Scripts/python.exe code/hypnospectrogram.py` (see [[feedback_pythonioencoding]]). The non-`raw.fif` filenames emit a harmless MNE naming-convention warning; sub-second hypno/data length mismatches (<1s) are auto cropped/padded by sleepeegpy.

`code/inspect_hypnos.py` is the diagnostic precursor: reads each fif + scoring file and prints duration, hypno line count, and `sec_per_line` — used to figure out the per-subject `hypno_freq` and label coding before plotting.

**Two non-obvious gotchas the scripts handle (same family as [[project_stage_label_dialects]]):**
1. **hypno_freq differs per subject** — EL3011 and YS0 scored at 1 Hz (`hypno_freq=1.0`, one line/sec); MCI27 scored per 30s-epoch (`hypno_freq=1/30`). Wrong freq misaligns the hypnogram against the spectrogram.
2. **Letter vs integer coding** — some scoring files are letter-coded (`W/1/2/3/R/?` → `LETTER_TO_INT` mapping, `?`=-1), others already integers. MCI27 is letter-coded (`letter=True`); the other two are integer.

Scoring files live at `I:\Shaked\ISO_data\scoring\{group}\` (young-control dir is `young_control\`) with inconsistent naming — `{sub}.txt` (EL3011, all 4 excluded young) vs `{sub}_hypno.txt` (MCI27, YS0).

**FIF preprocessing stage can differ.** Fully-processed subjects have `..._176-channels_..._avgref_interpolate_raw.fif`; excluded/earlier-stage subjects may only have `..._176-head-ch_..._bad-epochs.fif` (NOT avg-referenced, NOT interpolated). For a single-channel hypnospectrogram this is acceptable (user OK'd "use what's available" 2026-06-01). EL3026 had the full file; EL3030/EL3032/EL3040 only the bad-epochs stage. Split FIFs (`-1`/`-2`) auto-load from the base filename.

**Verify E101 isn't a bad/interpolated channel before plotting.** Best source = `pipeline.log` (same method as step1 notebook): parse the LAST `Interpolated channels: [...]` line (a Python list literal, `ast.literal_eval`) from `F:/gennadiy/ella_processed/{sub}/pipeline.log`. Per-subject `{sub}_bad_channels.txt` and FIF `info['bads']` are unreliable for excluded subjects (often absent/empty). On 2026-06-01 E101 was confirmed clean for all 4 (interp counts: EL3026=37, EL3030=35, EL3032=53, EL3040=29 — none included E101). See [[feedback_diagnose_dont_fix]] for the check-and-report-before-running habit.
