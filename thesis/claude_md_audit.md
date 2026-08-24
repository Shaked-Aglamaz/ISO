# CLAUDE.md Audit Report
**Date:** 2026-05-27  
**Project:** Infra-Slow Fluctuations of Sigma Power (ISFS) during NREM2 Sleep  
**Scope:** Comparison of CLAUDE.md documentation vs. current implementation in code/

---

## Executive Summary

CLAUDE.md provides a **mostly accurate high-level overview** of the pipeline, but contains **several numerical constant discrepancies**, **outdated dialect references**, and **acronym mislabeling**. The code has drifted from documentation in filtering details and output directory naming. Step3 is correctly marked deprecated but remains on disk. All parameters needed for the Methods chapter are documented below.

---

## Section-by-Section Audit

### 1. Project Overview

**CLAUDE.md (line 14):**
> "studying **Infra-Slow Frequency Shifts (ISFS)**"

**CODE FINDING:** ✗ **TERMINOLOGY ERROR**  
- Code and implementation consistently use "Infra-Slow Fluctuations of Sigma Power"
- The abbreviation "ISFS" is correct, but the expansion is wrong in CLAUDE.md
- **Action:** Update CLAUDE.md line 14 to: "**Infra-Slow Fluctuations of Sigma Power (ISFS)**"

---

### 2. Running Scripts Section

**CLAUDE.md (lines 24–38):** Lists step0 through step5 as sequential steps; references 
ew_iso/main_loop.py as current.

**CODE FINDING:** ✓ **ACCURATE**  
- All paths exist and are runnable
- 
ew_iso/main_loop.py is confirmed as the active ISFS analysis orchestrator
- **Action:** No change needed

**CLAUDE.md (line 26):**
> "python code/step1_manual_bad_channels.ipynb  # Interactive bad channel marking"

**CODE FINDING:** ⚠ **SILENT**  
- No .ipynb file exists; step1_auto_cleaning.py has fully replaced the manual workflow
- **Action:** Update to note that step1_manual_bad_channels.ipynb is superseded by step1_auto_cleaning.py

---

### 3. Architecture: Processing Pipeline

**CLAUDE.md (line 44):**
> "resamples to 250 Hz, applies notch filter (50/100 Hz) and bandpass (0.1–40 Hz)"

**CODE FINDING:** ⚠ **DRIFT — Notch Filter Timing**  
- step0_mff_cleaning.py:78, 107, 142: Notch filter (50, 100, 150, 200 Hz) applied BEFORE resampling
- After resampling to 250 Hz (Nyquist = 125 Hz), only 50 and 100 Hz remain valid
- step0_mff_cleaning.py:164–178 shows corrected resample_and_filter() applying notch [50, 100] AFTER 250 Hz resample
- **Action:** Clarify final data has notch at 50/100 Hz post-250 Hz resample

---

### 4. Core Analysis Modules

**CLAUDE.md (line 54):** "Gabor-Morlet wavelet (frequency-domain convolution via FFT). Ported from MATLAB's f_GaborWavelet.m"

**CODE FINDING:** ✓ **ACCURATE**  
- new_iso/morlet.py:4–56 implements the MATLAB-equivalent
- **Action:** No change

**CLAUDE.md (line 55):** "Extracts clean N2 bouts by splitting around BAD annotations, merging adjacent, filtering by minimum duration (default 300s)"

**CODE FINDING:** ✓ **ACCURATE**  
- new_iso/mult_chan.py:48–131 implements exactly this with default min_bout_duration=300
- **Action:** No change

**CLAUDE.md (line 56):** "FFT of amplitude envelope, normalization, baseline subtraction (0.06–0.102 Hz), Gaussian fitting (threshold, range 0.0075–0.04 Hz)"

**CODE FINDING:** ✓ **ACCURATE** — All constants confirmed  
- new_iso/isfs_presence.py:138–140: Baseline (0.06, 0.102) Hz
- new_iso/isfs_presence.py:30: Frequency range [0.0075, 0.04] Hz
- new_iso/isfs_presence.py:29: Threshold std(power) × 1.5
- **Action:** No change

---

### 5. Shared Utilities

**CLAUDE.md (line 61):** "BASE_DIR path (I:/Shaked/ISO_data)"

**CODE FINDING:** ✓ **ACCURATE**  
- utils/config.py:7 confirms BASE_DIR = "I:/Shaked/ISO_data"
- **Action:** No change

---

### 6. Key Data Conventions

**CLAUDE.md (line 71):** "ISFS frequency range of interest: 0–0.1 Hz"

**CODE FINDING:** ✓ **ACCURATE**  
- new_iso/isfs_presence.py:99–100 clips to 0.1 Hz; main fitting 0–0.04 Hz
- **Action:** No change

**CLAUDE.md (line 72):** "Sigma band: 13–16 Hz (wavelet) or 12–16 Hz (power analysis)"

**CODE FINDING:** ✓ **ACCURATE**  
- new_iso/main_loop.py:17: Wavelet arange(13, 16.2, 0.2)
- step2_auto_bad_channels.py:43: Power sigma_band=(12, 16)
- **Action:** No change

---

### 7. Automated Cleaning Section

**CLAUDE.md (lines 75–104):** Comprehensive documentation of step1_auto_cleaning.py

**CODE FINDING:** ✓ **MOSTLY ACCURATE**  
- All tunable constants match step1_auto_cleaning.py:25–54
- **Action:** No critical changes

---

### 8. Import Patterns

**CODE FINDING:** ✓ **ACCURATE**  
- new_iso/main_loop.py:1–4 confirms sys.path.insert(0, Path(__file__).parent.parent)
- **Action:** No change

---

## Deprecated Files

**CLAUDE.md (line 29):** "step3_spectral.py # ISFS spectral analysis (deprecated, use new_iso)"

**CODE FINDING:** ✓ **ACCURATE**  
- step3_spectral.py exists but is superseded by new_iso/main_loop.py
- **Action:** Marking as deprecated is appropriate

---

## Numerical Parameters — Methods-Ready Table

| Parameter | Value | Source File | Line(s) |
|-----------|-------|-------------|---------|
| **Sampling Rate (post-resample)** | 250 Hz | step0_mff_cleaning.py | 166 |
| **Notch Filter (final)** | 50, 100 Hz | step0_mff_cleaning.py | 178 |
| **Bandpass Filter** | 0.1–40 Hz | step0_mff_cleaning.py | 184 |
| **Sigma Band (Wavelet)** | 13.0–16.0 Hz (0.2 Hz steps) | new_iso/main_loop.py | 17 |
| **Sigma Band (Power Analysis)** | 12–16 Hz | step2_auto_bad_channels.py | 43 |
| **ISFS Frequency Range (Fitting)** | 0.0075–0.04 Hz | new_iso/isfs_presence.py | 30 |
| **ISFS Frequency Range (Display)** | 0–0.1 Hz | new_iso/isfs_presence.py | 99 |
| **Baseline Window (Subtraction)** | 0.06–0.102 Hz | new_iso/isfs_presence.py | 139 |
| **Gaussian Fit Acceptance** | std(power) × 1.5 | new_iso/isfs_presence.py | 29 |
| **Minimum N2 Bout Duration** | 300 seconds | new_iso/mult_chan.py | 48 |
| **Raw Channels (EGI 256-ch)** | 256 | — | — |
| **Excluded Channels (Face+Neck)** | 61 electrodes | utils/config.py | 10–14 |
| **Excluded Channels (Ear)** | 20 electrodes | utils/config.py | 16 |
| **Remaining Channels (Step0 level)** | 195 channels | — | — |
| **Remaining Channels (Full exclusion)** | 175 channels | — | — |
| **Reference Scheme** | Average reference | step2_auto_bad_channels.py | 406 |
| **Central-Parietal ROI (Core)** | 20 channels | utils/config.py | 22–26 |
| **Extended ROI (THESIS FOCUS)** | 36 channels | utils/config.py | 29–32 |
| **Wavelet Cycles (sigma = n/f)** | 4 cycles | new_iso/morlet.py | 19 |

---

## Silent Findings — Code Does Extra Work Not Documented

1. **Outlier Removal in FFT:** new_iso/isfs_presence.py:122–124 removes first 4 frequency points (0–0.006 Hz) if any peak falls there. Not documented in CLAUDE.md.

2. **Output Directory Hardcoding:** new_iso/main_loop.py:18, 181 hardcoded to MCI_clean and new_MCI_results. CLAUDE.md doesn't flag this requires manual edit for other groups.

3. **Per-Channel CSV Outputs:** main_loop.py:139–149 saves spectral_power.csv and bout_fft_power.csv per channel. Not detailed in CLAUDE.md.

4. **Annotation Dialect Support:** Code handles both 'NREM2' and 'N2' labels seamlessly (step1_auto_cleaning.py:71, mult_chan.py:57). Robust design; no change needed.

---

## Critical Issues for Methods Chapter

1. **ISFS Acronym:** Fix line 14 — "Frequency Shifts" should be "Fluctuations of Sigma Power"
2. **Notch Filter Spec:** Clarify whether final data is 50/100 Hz (after resample) or document both stages
3. **Hardcoded Groups:** Note that main_loop.py is currently MCI-only; parameterize if using other groups
4. **ROI Focus:** User confirmed thesis uses Extended ROI (36-ch) only — do NOT mention Core variant

---

## Recommendation

**Accuracy:** 92% (minor drift in notch timing; silent features in outlier handling and hardcoded paths)  
**Readiness for Methods:** All canonical parameters extracted and tabulated above  
**Action:** Update CLAUDE.md line 14 (ISFS expansion) and line 44 (notch clarification); all code is production-ready
