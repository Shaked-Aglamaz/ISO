"""
Automated EEG cleaning: bad channel and bad epoch detection for N2 sleep.
Replaces the manual visual inspection in step1_manual_bad_channels.ipynb.

Usage:
    python code/step1_auto_cleaning.py --group MCI_clean --subject SM09
    python code/step1_auto_cleaning.py --group MCI_clean --subject YC8 --dry-run
"""

import argparse
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

import mne

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import BASE_DIR, EAR_ELECTRODES
from utils.utils import merge_consecutive_annotations, find_subject_fif_file

# ─── Tunable constants ───────────────────────────────────────────────────────
FLAT_THRESHOLD_RATIO = 0.01        # variance < median * this → flat
VREF_FLAT_THRESHOLD_RATIO = 0.05   # relaxed threshold for VREF neighbors
VREF_NEIGHBORS = ['E9', 'E45', 'E81', 'E132', 'E186']
FLAT_WINDOW_SEC = 30.0             # window size for windowed flat detection
FLAT_WINDOW_FRACTION = 0.50        # flat in >50% of windows → flag
NEIGHBOR_FLAT_RATIO = 0.05         # variance < 5% of neighbor median → flat (all channels)

AMP_ZSCORE_THRESHOLD = 3.5         # robust z-score for crazy detection
AMP_ZSCORE_EXTREME = 5.0           # flag regardless of neighbor correlation
NEIGHBOR_CORR_THRESHOLD = 0.3      # min avg correlation with spatial neighbors
N_NEIGHBORS = 5                    # number of spatial neighbors for correlation
OUTLIER_TIME_FRACTION = 0.15       # if channel is amplitude outlier in >15% of N2 windows → flag
WINDOW_OUTLIER_Z = 2.5             # per-window amplitude z-score to count as outlier
SEGMENTED_OUTLIER_FRACTION = 0.25  # if any temporal segment has >25% outlier windows → flag

GFP_SPIKE_MADS = 10                # MADs above median → GFP spike
GFP_DROP_RATIO = 0.05              # fraction of median → GFP drop
PTP_THRESHOLD = 300e-6             # 300 µV peak-to-peak
PTP_CHANNEL_FRACTION = 0.30        # fraction of channels exceeding PTP
GFP_MIN_CHANNEL_SPREAD = 0.05     # min fraction of channels with elevated PTP for GFP detection
GFP_SPREAD_PTP = 150e-6           # PTP threshold for counting "disturbed" channels in spread check

MIN_BAD_DURATION = 1.0             # minimum BAD annotation length (seconds)
EXISTING_BAD_MIN_SEVERITY = 2.0    # existing BADs below this severity are removed (insignificant)
MIN_BOUT_DURATION = 300            # minimum clean bout duration (seconds)
SMART_SEVERITY_THRESHOLD = 15      # MADs — above this, always mark BAD
WINDOW_SEC = 2.0                   # window size for epoch detection (seconds)
MERGE_GAP_SEC = 1.0                # merge BAD candidates closer than this
BOUNDARY_TRIM_FACTOR = 0.5         # trim boundary windows below this fraction of GFP_SPIKE_MADS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sort_electrode_names(electrode_list):
    """Sort electrode names by numeric index."""
    return sorted(electrode_list, key=lambda x: int(x[1:]) if x.startswith('E') and x[1:].isdigit() else float('inf'))


def get_n2_mask(raw):
    """Build a boolean mask over all samples, True where annotation is NREM2/N2."""
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    mask = np.zeros(n_samples, dtype=bool)
    for ann in raw.annotations:
        desc = ann['description']
        if desc in ('NREM2', 'N2'):
            start = int(ann['onset'] * sfreq)
            end = int((ann['onset'] + ann['duration']) * sfreq)
            start = max(0, start)
            end = min(n_samples, end)
            mask[start:end] = True
    return mask


def get_n2_segments_sec(raw):
    """Return list of (start_sec, end_sec) for all N2 annotations."""
    segments = []
    for ann in raw.annotations:
        if ann['description'] in ('NREM2', 'N2'):
            segments.append((ann['onset'], ann['onset'] + ann['duration']))
    return segments


def get_bad_segments_sec(raw):
    """Return list of (start_sec, end_sec) for all BAD annotations."""
    segments = []
    for ann in raw.annotations:
        if 'BAD' in ann['description'].upper():
            segments.append((ann['onset'], ann['onset'] + ann['duration']))
    return segments


def get_channel_neighbors(raw, n_neighbors=N_NEIGHBORS):
    """Compute k nearest spatial neighbors for each channel using montage positions."""
    montage = raw.get_montage()
    if montage is None:
        raise ValueError("Raw object has no montage — cannot compute spatial neighbors.")

    positions = montage.get_positions()['ch_pos']
    ch_names = raw.ch_names

    # Build position matrix for channels that have positions
    ch_with_pos = [ch for ch in ch_names if ch in positions]
    pos_matrix = np.array([positions[ch] for ch in ch_with_pos])

    # Compute pairwise distances
    dists = cdist(pos_matrix, pos_matrix)

    neighbors = {}
    for i, ch in enumerate(ch_with_pos):
        # argsort and skip self (index 0)
        nearest_idx = np.argsort(dists[i])[1:n_neighbors + 1]
        neighbors[ch] = [ch_with_pos[j] for j in nearest_idx]

    return neighbors


def simulate_clean_bouts(n2_segments, bad_segments, min_bout=MIN_BOUT_DURATION):
    """
    Simulate extract_clean_sleep_bouts logic on seconds (not samples).
    Returns list of (start, end) clean bouts >= min_bout seconds.
    """
    # Split each N2 segment around BAD segments
    clean_parts = []
    for n2_start, n2_end in n2_segments:
        # Find overlapping BADs
        overlapping = []
        for b_start, b_end in bad_segments:
            if not (b_end <= n2_start or b_start >= n2_end):
                overlapping.append((b_start, b_end))
        overlapping.sort()

        if not overlapping:
            clean_parts.append((n2_start, n2_end))
            continue

        pos = n2_start
        for b_start, b_end in overlapping:
            if pos < b_start:
                clean_parts.append((pos, b_start))
            pos = max(pos, b_end)
        if pos < n2_end:
            clean_parts.append((pos, n2_end))

    # Sort and merge adjacent
    if not clean_parts:
        return []
    clean_parts.sort()
    merged = [clean_parts[0]]
    for s, e in clean_parts[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Filter by min duration
    return [(s, e) for s, e in merged if (e - s) >= min_bout]


def merge_bad_candidates(candidates, gap=MERGE_GAP_SEC):
    """Merge overlapping/adjacent BAD candidates. Each is (onset, duration, severity)."""
    if not candidates:
        return []
    # Sort by onset
    candidates = sorted(candidates, key=lambda x: x[0])
    merged = []
    curr_onset, curr_dur, curr_sev = candidates[0]
    curr_end = curr_onset + curr_dur

    for onset, dur, sev in candidates[1:]:
        end = onset + dur
        if onset <= curr_end + gap:
            curr_end = max(curr_end, end)
            curr_dur = curr_end - curr_onset
            curr_sev = max(curr_sev, sev)
        else:
            merged.append((curr_onset, curr_dur, curr_sev))
            curr_onset, curr_dur, curr_sev = onset, dur, sev
            curr_end = onset + dur
    merged.append((curr_onset, curr_dur, curr_sev))
    return merged


# ─── Phase A: Bad Channel Detection ─────────────────────────────────────────

def detect_flat_channels(raw, n2_mask):
    """Detect channels that are flat (near-zero variance) during N2."""
    data = raw.get_data()[:, n2_mask]  # shape: (n_channels, n_n2_samples)
    sfreq = raw.info['sfreq']
    variances = np.var(data, axis=1)
    median_var = np.median(variances)

    flat_channels = {}
    for i, ch in enumerate(raw.ch_names):
        if ch == 'VREF':
            continue

        if ch in VREF_NEIGHBORS:
            threshold = median_var * VREF_FLAT_THRESHOLD_RATIO
            label = 'flat (VREF neighbor)'
        else:
            threshold = median_var * FLAT_THRESHOLD_RATIO
            label = 'flat'

        if variances[i] < threshold:
            flat_channels[ch] = f"{label}, var={variances[i]:.4e} vs median={median_var:.4e}"

    # Windowed flat detection: catches channels that are flat in most windows
    # but have intermittent spikes that inflate overall variance (e.g., KS5 E35).
    # Uses a stricter threshold (0.001) than global flat detection because
    # per-window variance can naturally dip in quiet sleep periods.
    window_samples = int(FLAT_WINDOW_SEC * sfreq)
    n_windows = data.shape[1] // window_samples
    if n_windows > 0:
        flat_threshold = median_var * FLAT_THRESHOLD_RATIO * 0.1  # 0.001 of median
        for i, ch in enumerate(raw.ch_names):
            if ch == 'VREF' or ch in flat_channels:
                continue
            flat_count = 0
            for w in range(n_windows):
                ws, we = w * window_samples, (w + 1) * window_samples
                win_var = np.var(data[i, ws:we])
                if win_var < flat_threshold:
                    flat_count += 1
            flat_frac = flat_count / n_windows
            if flat_frac >= FLAT_WINDOW_FRACTION:
                flat_channels[ch] = (f"flat (windowed: {flat_frac*100:.0f}% of windows), "
                                     f"global_var={variances[i]:.4e} vs median={median_var:.4e}")

    # Neighbor-based flat detection for ALL channels (not just VREF neighbors)
    neighbors = get_channel_neighbors(raw)
    for i, ch in enumerate(raw.ch_names):
        if ch == 'VREF' or ch in flat_channels:
            continue
        neigh_vars = []
        for n_ch in neighbors.get(ch, []):
            if n_ch != 'VREF' and n_ch in raw.ch_names:
                neigh_vars.append(variances[raw.ch_names.index(n_ch)])
        if neigh_vars:
            neigh_median = np.median(neigh_vars)
            if neigh_median > 0 and variances[i] < neigh_median * NEIGHBOR_FLAT_RATIO:
                flat_channels[ch] = (f"flat (neighbor outlier), "
                                     f"var={variances[i]:.4e} vs neighbor_median={neigh_median:.4e}")

    # Additional MAD-based check for VREF neighbors
    for ch in VREF_NEIGHBORS:
        if ch not in raw.ch_names or ch in flat_channels:
            continue
        ch_idx = raw.ch_names.index(ch)
        ch_var = variances[ch_idx]
        neigh_vars = []
        for n_ch in neighbors.get(ch, []):
            if n_ch != 'VREF' and n_ch in raw.ch_names:
                neigh_vars.append(variances[raw.ch_names.index(n_ch)])
        if neigh_vars:
            neigh_median = np.median(neigh_vars)
            neigh_mad = np.median(np.abs(np.array(neigh_vars) - neigh_median))
            if neigh_mad > 0 and ch_var < neigh_median - 3.5 * neigh_mad:
                flat_channels[ch] = f"flat (VREF neighbor outlier), var={ch_var:.4e} vs neighbor_median={neigh_median:.4e}"

    return flat_channels


def detect_crazy_channels(raw, n2_mask, neighbors):
    """Detect channels with high amplitude inconsistent with spatial neighbors during N2."""
    data = raw.get_data()[:, n2_mask]
    ch_names = list(raw.ch_names)
    n_channels = len(ch_names)
    sfreq = raw.info['sfreq']

    # Metric 1: Robust amplitude z-score (global)
    median_abs_amp = np.median(np.abs(data), axis=1)  # per channel
    med = np.median(median_abs_amp)
    mad = np.median(np.abs(median_abs_amp - med))
    mad = max(mad, 1e-15)
    amp_zscores = (median_abs_amp - med) / (mad * 1.4826)

    amp_flagged = set()
    amp_extreme = set()
    for i, ch in enumerate(ch_names):
        if amp_zscores[i] > AMP_ZSCORE_EXTREME:
            amp_extreme.add(ch)
        elif amp_zscores[i] > AMP_ZSCORE_THRESHOLD:
            amp_flagged.add(ch)

    # Metric 2: Neighbor correlation (10s chunks)
    chunk_size = int(10 * sfreq)
    n_n2_samples = data.shape[1]
    n_chunks = max(1, n_n2_samples // chunk_size)

    corr_means = np.zeros(n_channels)
    for i, ch in enumerate(ch_names):
        if ch not in neighbors or ch == 'VREF':
            corr_means[i] = 1.0
            continue
        neigh_indices = [ch_names.index(n) for n in neighbors[ch] if n in ch_names and n != 'VREF']
        if not neigh_indices:
            corr_means[i] = 1.0
            continue

        chunk_corrs = []
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n_n2_samples)
            if end - start < chunk_size // 2:
                continue
            ch_chunk = data[i, start:end]
            ch_std = np.std(ch_chunk)
            if ch_std < 1e-15:
                chunk_corrs.append(0.0)
                continue
            neigh_corrs = []
            for ni in neigh_indices:
                n_chunk = data[ni, start:end]
                n_std = np.std(n_chunk)
                if n_std < 1e-15:
                    continue
                r = np.corrcoef(ch_chunk, n_chunk)[0, 1]
                if not np.isnan(r):
                    neigh_corrs.append(r)
            if neigh_corrs:
                chunk_corrs.append(np.mean(neigh_corrs))
        corr_means[i] = np.mean(chunk_corrs) if chunk_corrs else 1.0

    corr_flagged = set()
    for i, ch in enumerate(ch_names):
        if corr_means[i] < NEIGHBOR_CORR_THRESHOLD:
            corr_flagged.add(ch)

    # Metric 3: Time-varying outlier fraction
    # For each 10s window, compute per-channel std, then z-score across channels.
    # Count what fraction of windows each channel is an outlier.
    outlier_fractions = np.zeros(n_channels)
    valid_chunks = 0
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, n_n2_samples)
        if end - start < chunk_size // 2:
            continue
        valid_chunks += 1
        window_stds = np.std(data[:, start:end], axis=1)
        w_med = np.median(window_stds)
        w_mad = np.median(np.abs(window_stds - w_med))
        w_mad = max(w_mad, 1e-15)
        w_z = (window_stds - w_med) / (w_mad * 1.4826)
        outlier_fractions += (w_z > WINDOW_OUTLIER_Z).astype(float)

    if valid_chunks > 0:
        outlier_fractions /= valid_chunks

    time_flagged = set()
    for i, ch in enumerate(ch_names):
        if ch == 'VREF':
            continue
        if outlier_fractions[i] > OUTLIER_TIME_FRACTION:
            time_flagged.add(ch)

    # Metric 4: Segmented outlier check — catches channels that are crazy in a
    # specific temporal portion (e.g., first quarter, last third) but whose overall
    # outlier fraction is below OUTLIER_TIME_FRACTION.
    # Divide into segments and check if any segment has high outlier fraction.
    n_segments = 4
    seg_size = max(1, valid_chunks // n_segments)
    segment_flagged = set()
    if valid_chunks >= n_segments:
        # Re-compute per-segment outlier fractions
        # We need per-chunk outlier flags per channel
        chunk_outlier_flags = np.zeros((n_channels, valid_chunks), dtype=bool)
        vc = 0
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n_n2_samples)
            if end - start < chunk_size // 2:
                continue
            window_stds = np.std(data[:, start:end], axis=1)
            w_med = np.median(window_stds)
            w_mad = max(np.median(np.abs(window_stds - w_med)), 1e-15)
            w_z = (window_stds - w_med) / (w_mad * 1.4826)
            chunk_outlier_flags[:, vc] = w_z > WINDOW_OUTLIER_Z
            vc += 1

        for i, ch in enumerate(ch_names):
            if ch == 'VREF' or ch in time_flagged:
                continue
            for seg in range(n_segments):
                seg_start = seg * seg_size
                seg_end = min(seg_start + seg_size, valid_chunks)
                if seg_end <= seg_start:
                    continue
                seg_frac = np.mean(chunk_outlier_flags[i, seg_start:seg_end])
                if seg_frac >= SEGMENTED_OUTLIER_FRACTION:
                    segment_flagged.add(ch)
                    break

    # Combine detection rules:
    # - extreme amplitude → always flag
    # - amp_flagged AND corr_flagged → flag
    # - time_flagged → flag (high outlier fraction alone is sufficient)
    # - segment_flagged → flag (crazy in a specific temporal portion)
    # - corr_flagged AND amp > 2.0 → flag
    crazy_channels = {}
    for ch in ch_names:
        i = ch_names.index(ch)
        if ch == 'VREF':
            continue
        if ch in amp_extreme:
            crazy_channels[ch] = f"crazy (extreme amp z={amp_zscores[i]:.1f}, corr={corr_means[i]:.2f})"
        elif ch in amp_flagged and ch in corr_flagged:
            crazy_channels[ch] = f"crazy (amp z={amp_zscores[i]:.1f}, corr={corr_means[i]:.2f})"
        elif ch in time_flagged:
            crazy_channels[ch] = f"crazy (outlier {outlier_fractions[i]*100:.0f}% of time, amp z={amp_zscores[i]:.1f}, corr={corr_means[i]:.2f})"
        elif ch in segment_flagged:
            crazy_channels[ch] = f"crazy (segment outlier, global {outlier_fractions[i]*100:.0f}%, amp z={amp_zscores[i]:.1f}, corr={corr_means[i]:.2f})"
        elif ch in corr_flagged and amp_zscores[i] > 2.0:
            crazy_channels[ch] = f"crazy (low corr={corr_means[i]:.2f}, amp z={amp_zscores[i]:.1f})"

    return crazy_channels


def detect_bad_channels(raw, n2_mask):
    """Run all bad channel detectors and combine results."""
    print("  Detecting flat channels...")
    flat = detect_flat_channels(raw, n2_mask)

    print("  Computing spatial neighbors...")
    neighbors = get_channel_neighbors(raw)

    print("  Detecting crazy channels...")
    crazy = detect_crazy_channels(raw, n2_mask, neighbors)

    # Check for ear electrodes still in data
    ear = {}
    for ch in raw.ch_names:
        if ch in EAR_ELECTRODES:
            ear[ch] = "ear electrode"

    # Combine all
    all_bad = {}
    all_bad.update(flat)
    all_bad.update(crazy)
    all_bad.update(ear)

    if len(all_bad) > 30:
        print(f"  WARNING: {len(all_bad)} bad channels detected — possible data quality issue!")

    return all_bad


# ─── Phase B: Bad Epoch Detection ────────────────────────────────────────────

def detect_bad_epochs(raw, n2_mask, bad_channels):
    """
    Detect bad epochs during N2 using GFP and peak-to-peak metrics.
    Returns list of (onset_sec, duration_sec, severity) candidates.
    """
    sfreq = raw.info['sfreq']
    window_samples = int(WINDOW_SEC * sfreq)

    # Pick good channels only
    good_ch_idx = [i for i, ch in enumerate(raw.ch_names)
                   if ch not in bad_channels and ch != 'VREF']
    data = raw.get_data()[good_ch_idx]
    n_good = len(good_ch_idx)

    # Get N2 sample ranges for iteration
    n2_ranges = []
    in_n2 = False
    start = 0
    for i in range(len(n2_mask)):
        if n2_mask[i] and not in_n2:
            start = i
            in_n2 = True
        elif not n2_mask[i] and in_n2:
            n2_ranges.append((start, i))
            in_n2 = False
    if in_n2:
        n2_ranges.append((start, len(n2_mask)))

    # Compute GFP for all N2 samples
    n2_data_list = []
    n2_sample_offsets = []  # map local index back to global sample
    for rng_start, rng_end in n2_ranges:
        n2_data_list.append(data[:, rng_start:rng_end])
        n2_sample_offsets.append(rng_start)

    if not n2_data_list:
        return []

    # Process each N2 range separately
    all_candidates = []

    for seg_idx, (rng_start, rng_end) in enumerate(n2_ranges):
        seg_data = data[:, rng_start:rng_end]
        n_samples = seg_data.shape[1]
        if n_samples < window_samples:
            continue

        # GFP = std across channels at each time point
        gfp = np.std(seg_data, axis=0)

        # Sliding window stats
        n_windows = n_samples // window_samples
        if n_windows == 0:
            continue

        window_gfp_means = np.zeros(n_windows)
        window_ptp_fractions = np.zeros(n_windows)

        for w in range(n_windows):
            ws = w * window_samples
            we = ws + window_samples
            # GFP mean for this window
            window_gfp_means[w] = np.mean(gfp[ws:we])
            # Peak-to-peak per channel
            ptp = np.ptp(seg_data[:, ws:we], axis=1)
            window_ptp_fractions[w] = np.sum(ptp > PTP_THRESHOLD) / n_good

        # GFP statistics (robust)
        all_gfp_means = window_gfp_means
        gfp_median = np.median(all_gfp_means)
        gfp_mad = np.median(np.abs(all_gfp_means - gfp_median))
        gfp_mad = max(gfp_mad, 1e-15)

        for w in range(n_windows):
            onset_sample = rng_start + w * window_samples
            onset_sec = onset_sample / sfreq
            severity = 0.0
            is_bad = False

            # B1: GFP spike
            gfp_z = (window_gfp_means[w] - gfp_median) / (gfp_mad * 1.4826)
            if gfp_z > GFP_SPIKE_MADS:
                # Channel-spread check: verify disturbance affects enough channels.
                # A GFP spike driven by 1-2 channels (undetected bad) should not
                # mark the epoch as bad for all channels.
                ws = w * window_samples
                we = ws + window_samples
                ptp = np.ptp(seg_data[:, ws:we], axis=1)
                spread_frac = np.sum(ptp > GFP_SPREAD_PTP) / n_good
                if spread_frac >= GFP_MIN_CHANNEL_SPREAD:
                    severity = max(severity, gfp_z)
                    is_bad = True

            # B1: GFP drop
            if window_gfp_means[w] < gfp_median * GFP_DROP_RATIO:
                severity = max(severity, GFP_SPIKE_MADS)  # treat flat as high severity
                is_bad = True

            # B2: Peak-to-peak
            if window_ptp_fractions[w] > PTP_CHANNEL_FRACTION:
                ptp_severity = window_ptp_fractions[w] * 10  # scale to comparable range
                severity = max(severity, ptp_severity)
                is_bad = True

            if is_bad:
                all_candidates.append((onset_sec, WINDOW_SEC, severity))

    # Merge adjacent candidates
    merged = merge_bad_candidates(all_candidates, gap=MERGE_GAP_SEC)

    # Boundary trimming: re-check edge windows of merged epochs and trim
    # if their severity is below threshold. Prevents over-extension.
    if merged:
        trimmed = []
        for onset, duration, sev in merged:
            if duration <= WINDOW_SEC:
                trimmed.append((onset, duration, sev))
                continue
            n_sub = int(round(duration / WINDOW_SEC))
            trim_threshold = GFP_SPIKE_MADS * BOUNDARY_TRIM_FACTOR
            # Trim from the end
            new_end = onset + duration
            for k in range(n_sub - 1, 0, -1):
                win_start_sec = onset + k * WINDOW_SEC
                win_start_sample = int(win_start_sec * sfreq)
                win_end_sample = min(win_start_sample + window_samples, data.shape[1])
                if win_end_sample - win_start_sample < window_samples // 2:
                    break
                win_data = data[:, win_start_sample:win_end_sample]
                win_gfp = np.mean(np.std(win_data, axis=0))
                win_gfp_z = (win_gfp - gfp_median) / (gfp_mad * 1.4826) if gfp_mad > 1e-15 else 0
                if win_gfp_z < trim_threshold:
                    new_end = win_start_sec
                else:
                    break
            # Trim from the start
            new_start = onset
            for k in range(n_sub - 1):
                win_start_sec = onset + k * WINDOW_SEC
                win_start_sample = int(win_start_sec * sfreq)
                win_end_sample = min(win_start_sample + window_samples, data.shape[1])
                if win_end_sample - win_start_sample < window_samples // 2:
                    break
                win_data = data[:, win_start_sample:win_end_sample]
                win_gfp = np.mean(np.std(win_data, axis=0))
                win_gfp_z = (win_gfp - gfp_median) / (gfp_mad * 1.4826) if gfp_mad > 1e-15 else 0
                if win_gfp_z < trim_threshold:
                    new_start = win_start_sec + WINDOW_SEC
                else:
                    break
            new_duration = new_end - new_start
            if new_duration >= MIN_BAD_DURATION:
                trimmed.append((new_start, new_duration, sev))
        merged = trimmed

    # Filter by minimum duration
    merged = [(o, d, s) for o, d, s in merged if d >= MIN_BAD_DURATION]

    return merged


# ─── Evaluate Existing BADs ──────────────────────────────────────────────────

def evaluate_existing_bads(raw, n2_mask, bad_channels):
    """
    Score each pre-existing BAD annotation using the same GFP/PTP metrics
    used for new epoch detection. Returns list of (onset, duration, severity, source)
    where source='existing'.
    """
    sfreq = raw.info['sfreq']
    good_ch_idx = [i for i, ch in enumerate(raw.ch_names)
                   if ch not in bad_channels and ch != 'VREF']
    data = raw.get_data()[good_ch_idx]
    n_good = len(good_ch_idx)

    # Compute baseline GFP stats from clean N2 (excluding all BADs)
    bad_mask = np.zeros(raw.n_times, dtype=bool)
    for ann in raw.annotations:
        if 'BAD' in ann['description'].upper():
            s = max(0, int(ann['onset'] * sfreq))
            e = min(raw.n_times, int((ann['onset'] + ann['duration']) * sfreq))
            bad_mask[s:e] = True
    clean_n2_mask = n2_mask & ~bad_mask
    if np.sum(clean_n2_mask) < int(sfreq * 10):
        # Not enough clean N2 to compute baseline — keep all existing BADs
        results = []
        for ann in raw.annotations:
            if 'BAD' in ann['description'].upper():
                results.append((ann['onset'], ann['duration'], GFP_SPIKE_MADS, 'existing'))
        return results

    clean_gfp = np.std(data[:, clean_n2_mask], axis=0)
    gfp_median = np.median(clean_gfp)
    gfp_mad = np.median(np.abs(clean_gfp - gfp_median))
    gfp_mad = max(gfp_mad, 1e-15)

    results = []
    dropped = []
    for ann in raw.annotations:
        if 'BAD' not in ann['description'].upper():
            continue
        onset = ann['onset']
        duration = ann['duration']
        s = max(0, int(onset * sfreq))
        e = min(raw.n_times, int((onset + duration) * sfreq))
        if e - s < 1:
            continue

        # Compute severity for this existing BAD
        epoch_data = data[:, s:e]
        epoch_gfp = np.std(epoch_data, axis=0)
        mean_gfp = np.mean(epoch_gfp)
        gfp_z = (mean_gfp - gfp_median) / (gfp_mad * 1.4826)

        # PTP check
        ptp = np.ptp(epoch_data, axis=1)
        ptp_frac = np.sum(ptp > PTP_THRESHOLD) / n_good

        # Severity = max of GFP z-score and scaled PTP fraction
        severity = max(gfp_z, ptp_frac * 10)

        # GFP drop (flat)
        if mean_gfp < gfp_median * GFP_DROP_RATIO:
            severity = max(severity, GFP_SPIKE_MADS)

        if severity >= EXISTING_BAD_MIN_SEVERITY:
            results.append((onset, duration, severity, 'existing'))
        else:
            dropped.append((onset, duration, severity))

    if dropped:
        print(f"  Dropped {len(dropped)} existing BADs below severity threshold ({EXISTING_BAD_MIN_SEVERITY})")

    return results


# ─── Smart Marking ───────────────────────────────────────────────────────────

def smart_mark_epochs(candidates, n2_segments):
    """
    Apply smart marking on a unified pool of (onset, duration, severity, source)
    candidates. Source is 'existing' or 'new'.

    Accepts severe artifacts first, then evaluates less severe ones —
    skipping those that would destroy clean N2 bouts >= 300s.
    """
    if not candidates:
        return [], [], []

    # Sort by severity descending — accept the most severe first
    candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)

    accepted = []
    skipped = []
    current_bads = []

    for onset, duration, severity, source in candidates_sorted:
        new_bad = (onset, onset + duration)
        test_bads = current_bads + [new_bad]

        new_bouts = simulate_clean_bouts(n2_segments, test_bads)

        # Check if adding this BAD would completely destroy a clean bout
        bout_destroyed = False
        if severity < SMART_SEVERITY_THRESHOLD:
            current_bouts = simulate_clean_bouts(n2_segments, current_bads)
            for b_start, b_end in current_bouts:
                if (b_end - b_start) < MIN_BOUT_DURATION:
                    continue
                surviving = [
                    (s, e) for s, e in new_bouts
                    if s >= b_start - 1 and e <= b_end + 1
                ]
                if not surviving:
                    bout_destroyed = True
                    break

        if bout_destroyed:
            skipped.append((onset, duration, severity, source))
        else:
            accepted.append((onset, duration, severity, source))
            current_bads.append(new_bad)

    # Separate reporting
    kept_existing = [(o, d, s) for o, d, s, src in accepted if src == 'existing']
    removed_existing = [(o, d, s) for o, d, s, src in skipped if src == 'existing']
    new_accepted = [(o, d, s) for o, d, s, src in accepted if src == 'new']
    new_skipped = [(o, d, s) for o, d, s, src in skipped if src == 'new']

    if removed_existing:
        print(f"  Existing BADs: kept {len(kept_existing)}, removed {len(removed_existing)} (too weak or would kill clean bouts)")
    else:
        print(f"  Existing BADs: kept all {len(kept_existing)}")
    if new_skipped:
        print(f"  New BADs: accepted {len(new_accepted)}, skipped {len(new_skipped)} (borderline, would kill clean bouts)")
    else:
        print(f"  New BADs: accepted {len(new_accepted)}")

    return accepted, kept_existing, removed_existing


# ─── Main Processing ─────────────────────────────────────────────────────────

def process_subject(subject, group, dry_run=False):
    """Process a single subject through automated cleaning."""
    sub_dir = Path(BASE_DIR) / group / subject
    if not sub_dir.exists():
        print(f"ERROR: Subject directory not found: {sub_dir}")
        return False

    # Find the raw file — specifically the pre-avg-ref version
    expected = sub_dir / f"{subject}_cleaned_no_avg_ref_raw.fif"
    if expected.exists():
        raw_path = str(expected)
        print(f"Using file: {expected.name}")
    else:
        # Fallback to find_subject_fif_file
        raw_path = find_subject_fif_file(str(sub_dir), max_length=False)
        if raw_path is None:
            print(f"ERROR: No FIF file found in {sub_dir}")
            return False

    print(f"\n{'='*60}")
    print(f"Subject: {subject} ({group})")
    print(f"{'='*60}")

    # Load raw
    print("Loading raw data...")
    raw = mne.io.read_raw(raw_path, preload=True, verbose='error')
    duration_hours = raw.times[-1] / 3600
    print(f"  {len(raw.ch_names)} channels, {duration_hours:.1f} hours")

    # Verify sleep annotations
    descriptions = set(raw.annotations.description)
    has_n2 = 'NREM2' in descriptions or 'N2' in descriptions
    if not has_n2:
        print(f"ERROR: No NREM2/N2 annotations found. Available: {descriptions}")
        return False
    print(f"  Annotations: {sorted(descriptions)}")

    # Merge consecutive annotations
    merge_consecutive_annotations(raw)

    # Build N2 mask
    n2_mask = get_n2_mask(raw)
    n2_total_sec = np.sum(n2_mask) / raw.info['sfreq']
    print(f"  Total N2 time: {n2_total_sec:.0f}s ({n2_total_sec/60:.1f} min)")

    if n2_total_sec < 60:
        print("ERROR: Less than 60s of N2 data — skipping.")
        return False

    # ─── Phase A: Bad Channels ───
    print("\nPhase A: Bad Channel Detection")
    bad_channels_info = detect_bad_channels(raw, n2_mask)
    bad_channel_names = sort_electrode_names(list(bad_channels_info.keys()))

    # ─── Phase B: Bad Epochs ───
    print("\nPhase B: Bad Epoch Detection")
    n2_segments = get_n2_segments_sec(raw)
    n_original_bads = len([a for a in raw.annotations if 'BAD' in a['description'].upper()])

    # Step 1: Evaluate existing BADs — score them with the same metrics
    print(f"  Evaluating {n_original_bads} existing BAD annotations...")
    evaluated_existing = evaluate_existing_bads(raw, n2_mask, set(bad_channel_names))
    n_severity_dropped = n_original_bads - len(evaluated_existing)

    # Step 2: Detect new BADs
    new_candidates = detect_bad_epochs(raw, n2_mask, set(bad_channel_names))
    new_candidates_tagged = [(o, d, s, 'new') for o, d, s in new_candidates]
    print(f"  Found {len(new_candidates)} new candidate bad epochs")

    # Step 3: Unify and apply smart marking on the combined pool
    all_candidates = evaluated_existing + new_candidates_tagged
    print(f"  Applying smart marking on {len(all_candidates)} total candidates ({len(evaluated_existing)} existing + {len(new_candidates)} new)...")
    accepted, kept_existing, removed_existing = smart_mark_epochs(all_candidates, n2_segments)
    n_total_removed = n_severity_dropped + len(removed_existing)

    # Collect final BAD set
    final_bads = [(o, d, s) for o, d, s, _ in accepted]
    final_bads_sec = [(o, o + d) for o, d, s in final_bads]

    # ─── Summary ───
    bouts_before = simulate_clean_bouts(n2_segments, [])  # no BADs at all
    bouts_after = simulate_clean_bouts(n2_segments, final_bads_sec)
    total_before = sum(e - s for s, e in bouts_before)
    total_after = sum(e - s for s, e in bouts_after)

    new_accepted = [(o, d, s) for o, d, s, src in accepted if src == 'new']

    print(f"\n{'─'*60}")
    print(f"Subject: {subject} ({group})")
    print(f"Recording duration: {duration_hours:.1f} hours")
    print(f"{'─'*60}")

    print(f"\nBad Channels: {len(bad_channel_names)} ({len(bad_channel_names)/len(raw.ch_names)*100:.1f}% of {len(raw.ch_names)})")
    for ch in bad_channel_names:
        print(f"  {ch:6s} — {bad_channels_info[ch]}")

    print(f"\nBad Epochs: {len(kept_existing)} kept + {len(new_accepted)} new = {len(kept_existing) + len(new_accepted)} total")
    print(f"  Original: {n_original_bads} -> kept {len(kept_existing)}, removed {n_total_removed}"
          f" ({n_severity_dropped} insignificant, {len(removed_existing)} would kill clean bouts)")
    if removed_existing:
        for o, d, s in removed_existing:
            print(f"    skipped: t={o:.1f}s dur={d:.1f}s severity={s:.1f} (bout preservation)")
    if final_bads:
        durations = [d for _, d, _ in final_bads]
        print(f"  Duration range: {min(durations):.1f}s — {max(durations):.1f}s (median: {np.median(durations):.1f}s)")
        total_bad_time = sum(durations)
        print(f"  Total BAD time: {total_bad_time:.1f}s")

    print(f"\nClean N2 Impact:")
    print(f"  Before cleaning: {len(bouts_before)} bouts, {total_before:.0f}s total ({total_before/60:.1f} min)")
    print(f"  After cleaning:  {len(bouts_after)} bouts, {total_after:.0f}s total ({total_after/60:.1f} min)")
    if total_before > 0:
        change_pct = (total_after - total_before) / total_before * 100
        print(f"  Change: {change_pct:+.1f}%")
    print(f"{'─'*60}")

    # ─── Dry Run Comparison ───
    if dry_run:
        print("\n[DRY RUN] Comparing with existing manual cleaning...")
        manual_bc_path = sub_dir / f"{subject}_bad_channels.txt"
        manual_ann_path = sub_dir / f"{subject}_cleaned_annotations.txt"

        if manual_bc_path.exists():
            with open(manual_bc_path) as f:
                manual_bads = [l.strip() for l in f if l.strip()]
            auto_set = set(bad_channel_names)
            manual_set = set(manual_bads)
            overlap = auto_set & manual_set
            auto_only = auto_set - manual_set
            manual_only = manual_set - auto_set
            print(f"\n  Bad Channels Comparison:")
            print(f"    Manual: {len(manual_bads)} | Auto: {len(bad_channel_names)} | Overlap: {len(overlap)}")
            if overlap:
                print(f"    Both:       {sort_electrode_names(list(overlap))}")
            if auto_only:
                print(f"    Auto only:  {sort_electrode_names(list(auto_only))}")
            if manual_only:
                print(f"    Manual only: {sort_electrode_names(list(manual_only))}")
        else:
            print(f"  No manual bad channels file found at {manual_bc_path}")

        if manual_ann_path.exists():
            manual_ann = mne.read_annotations(str(manual_ann_path))
            manual_bad_count = sum(1 for d in manual_ann.description if 'BAD' in d.upper())
            manual_bad_dur = sum(dur for dur, d in zip(manual_ann.duration, manual_ann.description) if 'BAD' in d.upper())
            auto_bad_count = len(kept_existing) + len(new_accepted)
            auto_bad_dur = sum(d for _, d, _ in final_bads)
            print(f"\n  Bad Epochs Comparison:")
            print(f"    Manual: {manual_bad_count} BADs ({manual_bad_dur:.1f}s total)")
            print(f"    Auto:   {auto_bad_count} BADs ({auto_bad_dur:.1f}s total)")
            print(f"      - {len(kept_existing)} kept from original, {n_total_removed} removed ({n_severity_dropped} insignificant, {len(removed_existing)} bout preservation)")
            print(f"      - {len(new_accepted)} newly detected")

            # Compare clean bout outcomes
            manual_bad_segs = []
            for ann in manual_ann:
                if 'BAD' in ann['description'].upper():
                    manual_bad_segs.append((ann['onset'], ann['onset'] + ann['duration']))
            manual_bouts = simulate_clean_bouts(n2_segments, manual_bad_segs)
            manual_bout_total = sum(e - s for s, e in manual_bouts)
            print(f"\n  Clean N2 Bout Comparison:")
            print(f"    Manual: {len(manual_bouts)} bouts, {manual_bout_total:.0f}s ({manual_bout_total/60:.1f} min)")
            print(f"    Auto:   {len(bouts_after)} bouts, {total_after:.0f}s ({total_after/60:.1f} min)")
        else:
            print(f"  No manual annotations file found at {manual_ann_path}")

        print("\n[DRY RUN] No files saved.")
        return True

    # ─── Save outputs ───
    # Save bad channels
    bc_path = sub_dir / f"{subject}_bad_channels.txt"
    with open(bc_path, 'w') as f:
        for ch in bad_channel_names:
            f.write(f"{ch}\n")
    print(f"\nSaved bad channels to: {bc_path}")

    # Rebuild annotations: keep all non-BAD annotations, replace BADs with our final set
    non_bad_anns = [(a['onset'], a['duration'], a['description'])
                    for a in raw.annotations if 'BAD' not in a['description'].upper()]
    all_onsets = [o for o, _, _ in non_bad_anns] + [o for o, d, s in final_bads]
    all_durations = [d for _, d, _ in non_bad_anns] + [d for o, d, s in final_bads]
    all_descriptions = [desc for _, _, desc in non_bad_anns] + ['BAD'] * len(final_bads)

    new_annotations = mne.Annotations(
        onset=all_onsets,
        duration=all_durations,
        description=all_descriptions,
        orig_time=raw.annotations.orig_time
    )
    raw.set_annotations(new_annotations)
    merge_consecutive_annotations(raw)

    # Save cleaned annotations
    ann_path = sub_dir / f"{subject}_cleaned_annotations.txt"
    raw.annotations.save(str(ann_path), overwrite=True)
    print(f"Saved annotations to: {ann_path}")

    return True


def main():
    mne.set_log_level("error")

    parser = argparse.ArgumentParser(description="Automated EEG cleaning for N2 sleep")
    parser.add_argument('--group', required=True,
                        help='Subject group directory (e.g., MCI_clean, MCI_clean/a_the_rest)')
    parser.add_argument('--subject', required=True, help='Subject ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compare with existing manual cleaning without saving')
    args = parser.parse_args()

    process_subject(args.subject, args.group, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
