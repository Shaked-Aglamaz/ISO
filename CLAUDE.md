# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Safety Rules

- **Do NOT delete files** without explicit user permission.
- **Do NOT commit anything** (no git add, git commit, git push).
- **Do NOT change versions** of existing packages.
- You **can** install new packages and change code locally.

## Project Overview

EEG sleep analysis pipeline for studying **Infra-Slow Fluctuations of Sigma Power (ISFS)** during NREM2 sleep. The project processes high-density EEG data (256-channel EGI/MFF format), performs spectral analysis on clean N2 sleep bouts, and compares topographic ISFS features across groups (young controls, elderly controls, MCI patients).

## Running Scripts

All Python scripts are run from the repository root (`I:/Shaked/ISO`). The virtual environment is in `eeg_clean/`.

```bash
# Activate environment
source eeg_clean/Scripts/activate

# Run pipeline steps (each is standalone, run sequentially)
python code/step0_mff_cleaning.py        # MFF→FIF conversion, filtering, annotation
python code/step1_manual_bad_channels.ipynb  # Interactive bad channel marking (Jupyter)
python code/step1_auto_cleaning.py           # Automated bad channel & epoch detection (see below)
python code/step2_auto_bad_channels.py    # Sigma power analysis, interpolation, re-referencing
python code/step3_spectral.py             # ISFS spectral analysis (deprecated, use new_iso)
python code/step4_distribution_analysis.py # Group-level topographic analysis
python code/step5_topo_comparison.py       # Between-group cluster permutation tests

# New ISFS pipeline (current)
python code/new_iso/main_loop.py

# Subject inspection (run from repo root, imports with code. prefix)
python step3_subject_inspection.py
```

## Architecture

### Processing Pipeline (Sequential Steps)

1. **step0_mff_cleaning.py** — Reads raw MFF/FIF files, excludes face/neck/ear electrodes, resamples to 250 Hz, applies notch filter (50/100 Hz) and bandpass (0.1–40 Hz), adds annotations and sleep scoring, saves as FIF.
2. **step1_manual_bad_channels.ipynb** — Interactive notebook for manually identifying bad channels per subject.
   - **step1_auto_cleaning.py** — Automated replacement for step1. Detects bad channels (flat, noisy/inconsistent with neighbors) and bad epochs (GFP spikes/drops, peak-to-peak) during N2 sleep. Evaluates pre-existing BAD annotations and removes insignificant ones. Uses smart marking to preserve clean N2 bouts >= 300s for downstream ISFS analysis. See `## Automated Cleaning (step1_auto_cleaning.py)` section below for usage.
3. **step2_auto_bad_channels.py** — Computes sigma power (12–16 Hz) per channel during N2 vs Wake/REM, identifies outlier channels via IQR, applies average reference, interpolates bad channels.
4. **step3_spectral.py / new_iso/main_loop.py** — Core ISFS analysis: extracts clean N2 bouts, computes Gabor-Morlet wavelet transform, calculates amplitude envelope FFT, fits Gaussian to detect ISFS peak. Outputs per-channel: peak frequency, bandwidth, AUC, peak power.
5. **step4_distribution_analysis.py** — Aggregates per-channel ISFS metrics across subjects, creates topographic maps, performs within-group statistical analysis.
6. **step5_topo_comparison.py** — Between-group comparison using MNE's cluster-based spatiotemporal permutation tests. Imports functions from step4.

### Core Analysis Modules (`code/new_iso/`)

- **morlet.py** — Gabor-Morlet wavelet transform (frequency-domain convolution via FFT). Ported from MATLAB's `f_GaborWavelet.m`.
- **mult_chan.py** — Extracts clean N2 sleep bouts by splitting around BAD annotations, merging adjacent segments, filtering by minimum duration (default 300s).
- **isfs_presence.py** — ISFS parameter extraction: FFT of amplitude envelope per bout, normalization, baseline subtraction (0.06–0.102 Hz), Gaussian fitting with validation (threshold, frequency range 0.0075–0.04 Hz).
- **visualization.py** — Plotting functions for bout FFTs, mean spectrum with Gaussian fit, and bout overlay plots.

### Shared Utilities (`code/utils/`)

- **config.py** — `BASE_DIR` path (`I:/Shaked/ISO_data`), electrode lists for face, neck, ear exclusion.
- **utils.py** — Subject file finding, Google Sheet loading, annotation merging/comparison helpers.

## Key Data Conventions

- EEG data stored as MNE FIF files at 250 Hz after preprocessing
- Raw data lives on `I:/Shaked/ISO_data/` (separate from code repo), organized by group: `control_clean/`, `elderly_control_clean/`, `MCI_clean/`
- Results directories: `new_MCI_results/`, `new_iso_results/`, `new_elderly_results/`
- Electrode naming: EGI 256-channel system (`E1`–`E256`, `VREF`)
- Sleep stages: Wake=0, NREM1=1, NREM2=2, NREM3=3, REM=4
- ISFS frequency range of interest: 0–0.1 Hz (infra-slow oscillations of the sigma amplitude envelope)
- Sigma band: 13–16 Hz (wavelet) or 12–16 Hz (power analysis)
- Minimum N2 bout duration: 300 seconds

## Automated Cleaning (step1_auto_cleaning.py)

Replaces the manual visual inspection in step1. Run from repo root with the virtual environment activated.

### Usage

```bash
# Process a single subject (saves bad_channels.txt and cleaned_annotations.txt)
python code/step1_auto_cleaning.py --group MCI_clean --subject SM09

# Dry-run: compare auto vs existing manual cleaning without saving
python code/step1_auto_cleaning.py --group MCI_clean --subject YC8 --dry-run

# Other groups
python code/step1_auto_cleaning.py --group control_clean --subject EL3002
python code/step1_auto_cleaning.py --group elderly_control_clean --subject BA11
```

### What It Does

1. Loads `{subject}_cleaned_no_avg_ref_raw.fif`, verifies sleep annotations, merges consecutive annotations
2. **Bad channel detection**: flags flat channels (near-zero variance, with special handling for VREF neighbors E9/E45/E81/E132/E186) and "crazy" channels (high amplitude + low neighbor correlation + time-varying outlier fraction)
3. **Bad epoch detection**: uses Global Field Power (GFP) and peak-to-peak amplitude in 2-second windows during N2 to find global artifacts
4. **Existing BAD evaluation**: scores each pre-existing BAD annotation with the same metrics — removes those below severity threshold (insignificant)
5. **Smart marking**: skips borderline BAD annotations that would destroy clean N2 bouts >= 300s needed for ISFS analysis
6. Outputs: `{subject}_bad_channels.txt`, `{subject}_cleaned_annotations.txt` (same format as manual process)

### Tunable Constants (top of file)

Key thresholds that can be adjusted: `GFP_SPIKE_MADS` (default 10), `PTP_THRESHOLD` (300 µV), `OUTLIER_TIME_FRACTION` (15%), `EXISTING_BAD_MIN_SEVERITY` (2.0), `SMART_SEVERITY_THRESHOLD` (15).

## Import Patterns

Scripts in `code/` use relative imports from `utils.config` and `utils.utils`. The root-level `step3_subject_inspection.py` uses `code.` prefix imports. `code/new_iso/main_loop.py` prepends parent directory to `sys.path` for cross-module imports.

## Key Dependencies

MNE-Python, NumPy, SciPy, pandas, matplotlib, yasa, psutil. Memory-aware processing in step0 handles large files via dynamic channel-wise batching.
