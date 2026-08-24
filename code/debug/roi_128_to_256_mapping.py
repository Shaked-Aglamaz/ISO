"""
Reverse-engineer a central-parietal ROI from 128-ch EGI system to 256-ch EGI system.

Uses spatial coordinates (not channel names) to define an ROI around Cz/Pz,
then maps it to the 256-ch system used in this project.

Reference image: YA_AUC.png (128-ch topoplot with central-parietal hotspot)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import FACE_ELECTRODES, NECK_ELECTRODES, EAR_ELECTRODES

# -- Tunable parameters -------------------------------------------------------
ROI_CENTER_WEIGHT = 0.68  # 0 = pure Cz, 1 = pure Pz. 0.68 = shifted toward parietal
ROI_RADIUS_X = 0.062      # left-right (wider)
ROI_RADIUS_Y = 0.044      # anterior-posterior (narrower)
ROI_RADIUS_Z = 0.044      # superior-inferior (narrower)
# -----------------------------------------------------------------------------

EXCLUDED_256 = set(FACE_ELECTRODES + NECK_ELECTRODES + EAR_ELECTRODES)


def load_montage_positions(montage_name):
    """Load montage and return dict of {ch_name: np.array([x, y, z])}."""
    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()['ch_pos']
    return positions, montage


def compute_roi_center(positions_128):
    """Compute ROI center as weighted Cz/Pz midpoint in 128-ch space."""
    # In GSN-HydroCel-129: Cz is 'Cz', Pz-equivalent is E62
    cz_pos = positions_128['Cz']
    pz_pos = positions_128['E62']  # Pz-equivalent in 128-ch system
    center = (1 - ROI_CENTER_WEIGHT) * cz_pos + ROI_CENTER_WEIGHT * pz_pos
    print(f"Cz position:  {cz_pos}")
    print(f"Pz (E62) pos: {pz_pos}")
    print(f"ROI center:   {center}  (weight={ROI_CENTER_WEIGHT})")
    return center


def select_electrodes_in_roi(positions, center, radii, exclude=None):
    """Select all electrodes within ellipsoidal ROI. Returns sorted list.

    radii: (rx, ry, rz) - semi-axes of the ellipsoid in x, y, z directions.
    An electrode is inside if (dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1.
    """
    exclude = exclude or set()
    rx, ry, rz = radii
    selected = []
    for ch, pos in positions.items():
        if ch in exclude:
            continue
        diff = np.array(pos) - np.array(center)
        # Normalized ellipsoidal distance (<=1 means inside)
        ellip_dist = np.sqrt((diff[0]/rx)**2 + (diff[1]/ry)**2 + (diff[2]/rz)**2)
        if ellip_dist <= 1.0:
            selected.append((ch, ellip_dist))
    selected.sort(key=lambda x: x[1])
    return selected


def find_nearest_neighbor(pos_128, pos_256, ch_128):
    """For a 128-ch electrode, find its nearest 256-ch electrode."""
    target = np.array(pos_128[ch_128])
    best_ch = None
    best_dist = np.inf
    for ch, pos in pos_256.items():
        d = np.linalg.norm(np.array(pos) - target)
        if d < best_dist:
            best_dist = d
            best_ch = ch
    return best_ch, best_dist


def plot_roi_topomap(montage, roi_channels, all_channels, title, save_path,
                     ring_channels=None, sphere=None):
    """Plot head outline with all electrodes as black dots and ROI electrodes highlighted.

    ring_channels: optional iterable of channel names to highlight as an outer
    ring in orange (one layer of electrodes just outside the ROI).
    sphere: passed straight to mne.viz.plot_topomap. Default None reproduces the
    original projection; pass 'auto' to match the V8 production topos (step4/step5
    TOPO_SPHERE), which fits the sphere to the EGI-256 cloud so the occipital
    electrodes project down to the rim instead of into the parietal area.
    """
    from matplotlib.colors import ListedColormap

    # Create info with all channels
    info = mne.create_info(ch_names=all_channels, sfreq=250, ch_types='eeg')
    info.set_montage(montage)

    roi_set = set(roi_channels)
    ring_set = set(ring_channels) if ring_channels else set()
    # Zeros so the scalp field is uniformly white
    values = np.zeros(len(all_channels))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # White head (no topography) + default sensor markers (black dots)
    mne.viz.plot_topomap(values, info, axes=ax, show=False,
                         cmap=ListedColormap(['white']), vlim=(0, 1),
                         contours=0, sensors='k.', sphere=sphere)

    # Extract the plotted 2D positions so the ROI dots line up exactly.
    # With sensors='k.', plot_topomap draws the sensors as a Line2D.
    pos = None
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.collections.PathCollection):
            offsets = child.get_offsets()
            if offsets is not None and len(offsets) == len(all_channels):
                pos = np.asarray(offsets).copy()
                break
        elif isinstance(child, plt.matplotlib.lines.Line2D):
            xd = child.get_xdata()
            if len(xd) == len(all_channels):
                pos = np.column_stack([xd, child.get_ydata()])
                break

    if pos is not None:
        # Outer ring (drawn first, underneath the ROI dots)
        ring_idx = [i for i, ch in enumerate(all_channels)
                    if ch in ring_set and ch not in roi_set]
        if ring_idx:
            ring_xy = pos[ring_idx]
            ax.scatter(ring_xy[:, 0], ring_xy[:, 1], s=60, c='orange',
                       edgecolors='black', linewidths=1.0, zorder=9)

        # ROI electrodes as green dots (matches the ROI color in Figure 4)
        roi_idx = [i for i, ch in enumerate(all_channels) if ch in roi_set]
        if roi_idx:
            roi_xy = pos[roi_idx]
            ax.scatter(roi_xy[:, 0], roi_xy[:, 1], s=80, c='#00aa00',
                       edgecolors='black', linewidths=1.0, zorder=10)

    ax.set_title(title, fontsize=12)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("ROI Mapping: 128-ch -> 256-ch EGI System")
    print("=" * 60)

    # 1. Load montages
    pos_128, montage_128 = load_montage_positions('GSN-HydroCel-129')
    pos_256, montage_256 = load_montage_positions('EGI_256')
    print(f"\n128-ch montage: {len(pos_128)} channels")
    print(f"256-ch montage: {len(pos_256)} channels")

    # 2. Define ROI center from 128-ch coordinates
    print(f"\n-- ROI Center --")
    center = compute_roi_center(pos_128)

    # 3. Select 128-ch ROI electrodes
    radii = (ROI_RADIUS_X, ROI_RADIUS_Y, ROI_RADIUS_Z)
    print(f"\n-- 128-ch ROI (radii x={ROI_RADIUS_X}, y={ROI_RADIUS_Y}, z={ROI_RADIUS_Z}) --")
    roi_128 = select_electrodes_in_roi(pos_128, center, radii)
    roi_128_names = [ch for ch, _ in roi_128]
    print(f"Selected {len(roi_128)} electrodes:")
    for ch, dist in roi_128:
        print(f"  {ch:>5s}  edist={dist:.4f}")

    # 4. Select 256-ch ROI electrodes (same spatial criterion)
    print(f"\n-- 256-ch ROI (radii x={ROI_RADIUS_X}, y={ROI_RADIUS_Y}, z={ROI_RADIUS_Z}) --")
    roi_256 = select_electrodes_in_roi(pos_256, center, radii, exclude=EXCLUDED_256)
    roi_256_names = [ch for ch, _ in roi_256]
    print(f"Selected {len(roi_256)} electrodes:")
    for ch, dist in roi_256:
        print(f"  {ch:>5s}  edist={dist:.4f}")

    # 5. Check for excluded electrodes that would have been in ROI
    roi_256_with_excluded = select_electrodes_in_roi(pos_256, center, radii)
    excluded_in_roi = [ch for ch, _ in roi_256_with_excluded if ch in EXCLUDED_256]
    if excluded_in_roi:
        print(f"\nWARNING: Excluded electrodes that fall within ROI: {excluded_in_roi}")
    else:
        print(f"\nOK: No excluded electrodes overlap with ROI")

    # 6. Nearest-neighbor mapping (128->256)
    print(f"\n-- Nearest-Neighbor Mapping (128->256) --")
    print(f"{'128-ch':>8s} -> {'256-ch':>8s}  (distance)")
    for ch128 in roi_128_names:
        ch256, dist = find_nearest_neighbor(pos_128, pos_256, ch128)
        in_roi = "[in ROI]" if ch256 in roi_256_names else "[not in ROI]"
        print(f"  {ch128:>6s} -> {ch256:>6s}  dist={dist:.4f}  {in_roi}")

    # 7. Plot verification images
    print(f"\n-- Generating verification plots --")
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Outer ring (one layer of electrodes just outside the 128-ch ROI)
    ring_scale = 1.35
    ring_radii = (ROI_RADIUS_X * ring_scale,
                  ROI_RADIUS_Y * ring_scale,
                  ROI_RADIUS_Z * ring_scale)
    ring_128 = select_electrodes_in_roi(pos_128, center, ring_radii)
    ring_128_names = [ch for ch, _ in ring_128 if ch not in set(roi_128_names)]
    print(f"\n-- 128-ch outer ring (visual only, {len(ring_128_names)} electrodes): {ring_128_names}")

    # 128-ch plot: use channels that exist in montage
    all_128 = [ch for ch in montage_128.ch_names if ch in pos_128]
    plot_roi_topomap(montage_128, roi_128_names, all_128,
                     f"128-ch ROI ({len(roi_128_names)} electrodes)\nrx={ROI_RADIUS_X}, ry={ROI_RADIUS_Y}, weight={ROI_CENTER_WEIGHT}",
                     os.path.join(out_dir, 'roi_128_verification.png'),
                     ring_channels=ring_128_names)

    # 256-ch outer ring (same expanded radii, with face/neck/ear exclusions)
    ring_256 = select_electrodes_in_roi(pos_256, center, ring_radii, exclude=EXCLUDED_256)
    ring_256_names = [ch for ch, _ in ring_256 if ch not in set(roi_256_names)]
    print(f"-- 256-ch outer ring (visual only, {len(ring_256_names)} electrodes): {ring_256_names}")

    # 256-ch plot: exclude face/neck/ear channels
    all_256 = [ch for ch in montage_256.ch_names if ch in pos_256 and ch not in EXCLUDED_256]
    plot_roi_topomap(montage_256, roi_256_names, all_256,
                     f"256-ch ROI ({len(roi_256_names)} electrodes)\nrx={ROI_RADIUS_X}, ry={ROI_RADIUS_Y}, weight={ROI_CENTER_WEIGHT}",
                     os.path.join(out_dir, 'roi_256_verification.png'),
                     ring_channels=ring_256_names)

    # 8. Print final copy-pasteable list
    print(f"\n{'=' * 60}")
    print(f"FINAL 256-ch ROI ({len(roi_256_names)} electrodes):")
    print(f"{'=' * 60}")
    print(f"ROI_256 = {roi_256_names}")
    print()


if __name__ == '__main__':
    main()
