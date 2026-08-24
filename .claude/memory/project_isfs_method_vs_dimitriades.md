---
name: project_isfs_method_vs_dimitriades
description: How our ISFS detection differs from the reference Dimitriades MATLAB (stricter; AUC gate kept intentionally) + paper detection-rate numbers
metadata: 
  node_type: memory
  type: project
  originSessionId: 4d4e7585-dae6-453e-9521-9e360f440c6a
---

Comparison done 2026-06-06 against the reference MATLAB (`C:/Users/Shaked/Downloads/Infraslow-Fluctuation-main/`, esp. `f_ISFS_PresenceParamaters.m` 'relative' branch; **DO NOT EDIT**). Our `code/new_iso/isfs_presence.py` matches it on the core rule — fit `a*exp(-((x-b)/c)^2)`, threshold = `1.5*std(baseline-corrected mean spectrum)`, accept iff fitted amplitude ≥ threshold AND peak-freq ∈ [0.0075, 0.04] Hz — but differs in three ways:

1. **Extra `AUC>0` rejection gate (Python only — MATLAB has none).** This makes us STRICTER than the paper. It correctly kills degenerate needle-spike fits (bandwidth ~2 mHz vs real ISFS ~8-19 mHz), so it is **intentionally KEPT** — do NOT remove it. The only bug in it was the negative-sigma sign issue (fixed via `abs(sigma)`, see [[project_negative_sigma_fix]]); the gate itself stays.
2. **`ddof`:** numpy `np.nanstd` uses N (population); MATLAB `std` uses N-1 → MATLAB threshold ~1.2% higher. We did NOT change this (left as numpy default). Minor; can flip razor-thin cases.
3. Smaller: MATLAB `std` returns NaN if spectrum has NaN (latent fall-through); MATLAB clips AUC lower bound to ≥0.0075; MATLAB's `'Exclude'` is a no-op. None acted on.

**Implication for the paper:** our 20%-of-channels exclusion bar is stricter than Dimitriades' (we reject needles they would keep), so detection rates aren't directly comparable to theirs — worth a Methods sentence.

**Dimitriades 2024 detection rates** (preprint, `thesis/references/ISFS_Development_Dimitriades_2024.pdf`), % of electrodes with detectable ISFS (mean±SD): children 74.7±21.1, early-adol 75.7±21.8, late-adol 82.1±18.1, young adults 80.9±24.9. The paper reports **every subject cleared ≥20% of electrodes** — no one was excluded for low detection (the <3-microarousal exclusions are a separate analysis). Our active cohort means (sigma_fix run): YA 75.3 / HE 86.6 / MCI 79.2%. Related: [[project_thesis]].
