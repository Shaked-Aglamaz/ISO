---
name: project_f2_regeneration
description: "F2 methods-flow figure fully rewritten 2026-06-15 — code/make_f2_figure.py now generates all 4 panels (no screenshots); RD43 VREF example, layout/data details"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9d67ca66-7197-429c-bb50-5349a61568f5
---

**2026-06-15: `code/make_f2_figure.py` rewritten from scratch.** It no longer pastes bitmaps (`methods_flow.png` screenshot is gone) — every panel is a real matplotlib plot. Output = `thesis/figures/methods_flow_roi_v3.png` (dpi 300). Older composites preserved, not overwritten. Part of [[project_paper_figure_set]] (F2).

**Example data:** subject **RD43** (YA), channel **VREF**.
- Panels A & B use the single example N2 bout: full raw cropped at **(7418, 8163)** s (745 s). Spindles via `yasa.spindles_detect(freq_sp=(13,16))`. Sigma = 13–16 Hz bandpass. Envelope = Gabor-Morlet (`new_iso/morlet.calculate_gabor_wavelet`, 13–16 Hz @0.2 Hz, mean |transform|) — same as `main_loop.analyze_channel`.
- **Panel C uses ALL of RD43-VREF's clean N2 bouts** (NOT the single crop): pipeline-faithful via `extract_clean_sleep_bouts` → Gabor envelope → `extract_isfs_parameters` (5 bouts / 86.6 min). The single-bout path was tried first and `curve_fit` failed (`maxfev`); the real multi-bout path converges (μ=0.018 Hz, BW 0.030, AUC 10.89).

**Panel B window** = 68 s into the crop, 45 s long (68–113 s); 3 stacked traces: "Raw channel" / "Sigma 13-16 Hz" / "Sigma envelope". Upper two drawn thin (lw 0.4), envelope lw 0.9; raw trace y-squeezed (ylim ×1.8, MNE-minus equivalent). Spindle shading color `#0b2e6b` (dark navy), alpha 0.35.

**Layout (user-tuned):** width A+B = 55% (two cols of 27.5%), C = 25%, D = 20% → `width_ratios=[11,11,10,8]`; `height_ratios=[0.38, 1.18]` (A thin strip on top, B taller below); figsize (16, 4.8). No divider line. Panel A: bottom spine only, legend = "Spindle" only (no zoom-in entry). Panel C: left+bottom spines only. Green `ConnectionPatch` arrows fan from A's zoom box corners into B's top corners ("opening" the window).

**How to apply:** to tweak F2, edit `code/make_f2_figure.py` and re-run `PYTHONIOENCODING=utf-8 python code/make_f2_figure.py`. Per-trace look is controlled by the `traces` list in `panel_B` (label, color, lw, y-squeeze factor). RD43="26" alias context in [[reference_rd43_alias_and_example_gaussians]].
