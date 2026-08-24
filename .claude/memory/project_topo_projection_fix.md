---
name: project_topo_projection_fix
description: "EGI-256 topo projection fix (sphere='auto' + extrapolate choice); V8/V8B generated, option-B clip pending supervisor decision"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6403f293-dbf7-4d61-aafa-b49ac83baecb
---

**RESOLVED 2026-06-09 — Option B chosen + implemented.** `step4_distribution_analysis.py` now sets `TOPO_EXTRAPOLATE='head'` (sphere stays `'auto'`) and defines `clip_topo_to_head(ax, info, sphere=None)` — resolves the numeric sphere via `mne.viz.topomap._check_sphere(sphere, info)` and clips `ax.images` + non-`PathCollection` collections (field + contours) to a `Circle((cx,cy), r)`; sensor/mask/ROI dots (PathCollection) stay unclipped. Imported into step5 and called immediately after every `plot_topomap` (1 site in step4, 6 in step5) BEFORE overlays. Verified visually: clean complete circle, no spill, occiput low, ROI green dots + post-hoc circles intact. All topos regenerated into `three_groups_V9` (N=36). V8/V8B superseded (left in place, not deleted).

Topo projection rework started 2026-06-05; option-B decision implemented 2026-06-09.

**Problem (user-spotted in `three_groups_V7/three_group_topo_auc_raw.png`):** occipital electrodes rendered too high (looked parietal); bottom of head looked empty.

**Diagnosis (script `code/topo_montage_test.py`):** production `plot_topomap` calls passed no `sphere`, so MNE used `sphere=None` = fixed `(0,0,0,0.095)`, which crams the WHOLE EGI net — including the face/neck/ear positions that project *inside* the circle — into the upper-central area. Occiput electrode E125 landed at only 60% of head radius. `sphere='auto'` fits `(0, 0.009, 0.042, 0.095)` → E125 at 77% (down at rim), head fills. The fit is identical with/without neck electrodes, so the user's neck-exclusion hunch was right about the *symptom* but the cause is the default sphere. Note the inherent tension: excluding the neck row means the lowest remaining electrodes (occipital, z≈+0.01) sit just above the equator, so there are NO sensors at the very bottom of the head — a fully circular fill there is always extrapolated.

**Second issue (user-spotted):** under `sphere='auto'`, ~19% of kept electrodes project outside the circle (max 127% radius, bottom-heavy). With the MNE EEG default `extrapolate='head'` the colored field expands to the outermost sensor and spills past the outline asymmetrically (lots of color, no dots, at the bottom). `extrapolate='local'` hugs the electrodes (tight) but the fill is non-circular (user dislikes). **Option B** = `auto`+`head`+clip-the-field-to-the-head-circle = clean complete circle, no spill, occiput still low. **Recommended.** Preview only: `results/topo_montage_test/test_clip_options.png` (A=head/V8, B=clip, C=local/V8B).

**Current code state** (`step4_distribution_analysis.py`, `step5_topo_comparison.py`): constants `TOPO_SPHERE='auto'` + `TOPO_EXTRAPOLATE='local'` (defined in step4, imported into step5). `sphere=TOPO_SPHERE` on all 6 step5 `plot_topomap` calls + step4's, AND on the 3 step5 `_find_topomap_coords` calls (ROI dots / cluster circles / ROI ellipse must share the sphere or they drift). `extrapolate=TOPO_EXTRAPOLATE` on all `plot_topomap` calls only. Both output dirs currently point to **V8B**.

**Generated, for side-by-side compare:**
- `three_groups_V8/` = `auto` + `extrapolate='head'`
- `three_groups_V8B/` = `auto` + `extrapolate='local'`
- Option B (clip) NOT generated as a full set yet — only the preview png above.

Each dir has the manifest topo set: `three_group_topo_{auc,auc_raw,peak_frequency,bandwidth}.png` + 6 per-group `*_topographies_avg_{raw,normalized}.png` (+ 3 fstat). Per-group via step4 (writes to `new_*_results/` then copies into the V dir). Stats UNCHANGED by sphere/extrapolate: AUC cluster = same 9 electrodes (E130,E184,E154,E155,E142,E197,E185,E144,E143), p≈0.014–0.016 (pure permutation Monte-Carlo noise, no seed); peak-freq/BW no cluster. See [[project_paper_figure_set]].

**ROI plot:** added optional `sphere=` param to `plot_roi_topomap` in `code/debug/roi_128_to_256_mapping.py` (default None preserves original); new generator `code/debug/make_extended_roi_verification_V2.py` → `code/debug/roi_256_verification_V2.png` (extended 36-ch ROI, `sphere='auto'`).

**TODO when supervisor decides (A=head / B=clip / C=local):**
1. If **B**: implement a small clip helper (clip the field AxesImage + contour lines to a `Circle((sphere[0],sphere[1]), sphere[3])` right after each `plot_topomap`, BEFORE overlays so ROI/sig dots aren't clipped); set `TOPO_EXTRAPOLATE='head'`. If A: set `'head'`. If C: keep `'local'`.
2. Lock both output dirs to ONE final version name, regenerate (one step4 + one step5 run), drop the losing dirs.
3. Update `thesis/figure_manifest.md` — it STILL points to V7 (source-of-truth note + F4/S1 image paths + cluster p). Also [[project_paper_figure_set]] mentions V7.
4. Consider whether step6 / other topo producers need the same sphere/extrapolate.
