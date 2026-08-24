---
name: project_ry42_projector_fix
description: Inactive avg-ref SSP projector gotcha in some _bad-epochs.fif files (plot vs get_data disagree); RY42 stripped + reprocessed 2026-06-09; CH53 still has it
metadata: 
  node_type: memory
  type: project
  originSessionId: e57a4801-bbbc-41c6-9d10-507370788cb2
---

Some `*_176-head-ch_..._bad-epochs.fif` files carry an **inactive `'Average EEG reference'` SSP projector** (`active=False`, 257-ch) while the samples are still in the original reference (VREF channel exactly flat, `custom_ref_applied=0`). Consequence: **`raw.plot()` applies the projector on display (looks avg-referenced, VREF non-flat) but `get_data()` does NOT apply it (returns unreferenced data, VREF=0)** — they disagree, which looks contradictory. `pick(["VREF"]).plot()` then errors `"Application of 1 projectors for 1 channels will yield no components"` because the 257-ch projector can't apply to a 1-ch subset.

**The real convention is the data: post-step0 files are VREF-referenced (VREF flat).** The projector is a stray header object never used in the pipeline (analysis reads `get_data()` w/o proj). To verify avg-ref status, don't trust `custom_ref_applied` — check (a) VREF flatness and (b) per-sample mean across channels (~0 ⇒ avg-ref). To strip: `raw.del_proj()` (data untouched); to bake avg-ref in: `del_proj()` then `set_eeg_reference('average', projection=False)`.

**RY42 (HE) fix 2026-06-09:** loaded the file, `del_proj()` only (no sample changed, verified `np.array_equal`), overwrote in place (split FIF, `split_size='2GB'`, same 3 parts). Then full reprocess: manual step1 (auto flagged 36 ch @20.5% — over-flagged a frontal block; user pared to **4**) → step2 avg-ref+interp → main_loop → `results/sigma_fix_HE/RY42` (ISFS 37/176=21.0%). See [[project_negative_sigma_fix]], [[project_files_cleanup_status]].

**Still carries the same leftover projector: CH53** (elderly) — 1 × "Average EEG reference", `custom_ref=0`, VREF flat. AT36/BA11 do NOT (0 projs, custom_ref=1). Header-only scan (preload=False, read `info['projs']`) maps the rest if consistency across the group matters.
