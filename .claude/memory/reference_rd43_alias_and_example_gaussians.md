---
name: rd43-26-alias-example-gaussians-figures-mne-plot-scaling
description: "subject numeric aliases in debug/try.ipynb (26=RD43, 31=EL3006), example ISFS-fit figures location, mne_qt_browser +/- scale factors"
metadata: 
  node_type: memory
  type: reference
  originSessionId: e2849f90-e049-43b2-a3ed-b08ba612181e
---

Three reusable facts from the 2026-06 figure-making sessions:

**1. Numeric subject aliases in `code/debug/try.ipynb` & `code/chat_envelope.py`.** `sub_to_hypno_path = {'31':'EL3006', '26':'RD43', '9':'HK5'}`. So files named `26_*` (e.g. the recycled `26_cropped.fif`) belong to **RD43, NOT EL3026** — easy to confuse. RD43 is a young control, processed under `results/new_iso_results/RD43/` (full main_loop outputs incl. `RD43_VREF_output/`). RD43's avgref FIF: `ISO_data/control_clean/RD43/RD43_176-channels_..._avgref_interpolate_raw.fif`.

**2. Representative ISFS FFT+Gaussian figures**: `code/find_example_gaussians.py` → `results/new_iso_results/example_gaussians/{sub}_{ch}_mean_spectrum.png`. Fixed style: `figsize=(6.5, 7)`, `ylim=(-0.5, 2.5)`, `xlim=(0, 0.10)`, baseline-corrected over 0.06–0.102 Hz, Gaussian overlay + peak + bandwidth bar + ±1σ AUC fill, 300 dpi. Auto-selection filters: peak_amplitude in [1.4, 1.7], μ in [0.0075, 0.04], blue_max ≤ 2.4, ranked by off-peak flatness, top 20. To make one for an arbitrary subject/channel (bypassing filters): `import find_example_gaussians as feg`, build a `rec` dict from the subject's `_all_channels_summary.csv` row (fit_peak=peak_amplitude, mu=peak_frequency, sigma=bandwidth_sigma), then `feg.replot(rec, spectral_df)`.

**3. mne_qt_browser interactive amplitude scaling** (`+`/`-` keys): `-` (Decrease) multiplies global `scale_factor` by **4/5**, `+` (Increase) by **5/4**; displayed height ∝ `scale_factor / scalings`. These keys are GLOBAL (all channels at once). To bake per-channel zoom into a `raw.plot(scalings=...)` dict instead: one `-` on a channel ⇒ multiply its scaling by 5/4; one `+` ⇒ multiply by 4/5. Distinct ch_types are required for independent per-channel scaling (one scaling per type).
