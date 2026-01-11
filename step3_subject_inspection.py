import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from more_itertools import collapse
import numpy as np
from scipy import ndimage

from config import BASE_DIR
from step2_auto_bad_channels import compute_sigma_power_stats
from utils.utils import find_subject_fif_file, get_all_subjects


mne.set_log_level("error")


def plot_psd_by_sleep_stage(
    subject_id,
    raw,
    hypno_path,
    channel_name,
    output_dir,
    fmin=0,
    fmax=25,
    hypno_freq=1/30,
    n_fft=2048,
    n_per_seg=768,
    n_overlap=512,
    window="hamming",
):
    """
    Plot PSD for each sleep stage with different colored lines.
    
    Implementation matches sleepeegpy's approach:
    - Upsamples hypnogram to EEG sampling rate
    - Finds continuous regions for each stage
    - Computes weighted average of PSDs across regions
    
    Args:
        fmin: Minimum frequency to plot (Hz)
        fmax: Maximum frequency to plot (Hz)
        n_fft: FFT length for frequency resolution
        n_per_seg: Segment length in samples
        n_overlap: Overlapping samples between segments
        window: Window function for tapering

    Returns:
        dict: {stage_name: (frequencies, psd_values)} or None if error
    """
    
    print(f"Plotting PSD by sleep stage for {subject_id}, channel: {channel_name}")
    print(f"{'='*60}")

    stage_mapping = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', -1: 'Unknown'}
    stage_colors = {
        'Wake': '#1f77b4',  # blue
        'N1': '#ff7f0e',    # orange
        'N2': '#2ca02c',    # green
        'N3': '#d62728',    # red
        'REM': '#9467bd',   # purple
        'Unknown': '#7f7f7f'  # gray
    }
    
    if channel_name not in raw.ch_names:
        print(f"  ✗ Channel '{channel_name}' not found in data")
        return None
    
    if channel_name in raw.info['bads']:
        print(f"  ✗ Warning: Channel '{channel_name}' is marked as bad! ({raw.info['bads']})")
        return None
    
    print(f"  Loading hypnogram...")
    hypno = np.loadtxt(hypno_path, dtype=int)
    sfreq = raw.info['sfreq']
    repeats = int(sfreq / hypno_freq)
    if not (sfreq / hypno_freq).is_integer():
        raise ValueError(f"EEG sfreq ({sfreq}) / hypno_freq ({hypno_freq}) must be a whole number")
    
    hypno_up = np.repeat(hypno, repeats)
    npts_data = raw.n_times
    diff_sec = np.abs(len(hypno_up) - npts_data) / sfreq
    diff_str = f"{diff_sec:.2f} seconds ({diff_sec/60:.2f} minutes)"
    if len(hypno_up) < npts_data:
        print(f"  Warning: Hypnogram shorter than data by {diff_str}, padding with last value")
        hypno_up = np.pad(hypno_up, (0, npts_data - len(hypno_up)), mode='edge')
    elif len(hypno_up) > npts_data:
        print(f"  Warning: Hypnogram longer than data by {diff_str}, cropping to match")
        hypno_up = hypno_up[:npts_data]
    
    unique_stages = np.unique(hypno_up)
    data = raw.get_data(picks=[channel_name], reject_by_annotation="NaN")  # Shape: (1, n_times)
    n_samples_valid = np.count_nonzero(~np.isnan(data), axis=1)[0]
    print(f"  Valid samples (non-bad): {n_samples_valid}/{npts_data} ({n_samples_valid/npts_data*100:.1f}%)")
    
    stage_psds = {}
    for stage_code in unique_stages:
        if stage_code == -1:  # Skip unknown stages
            continue
            
        # Find continuous regions with this sleep stage
        stage_name = stage_mapping.get(stage_code, f'Stage{stage_code}')
        stage_mask = hypno_up == stage_code
        labeled, _ = ndimage.label(stage_mask)
        regions = list(collapse(ndimage.find_objects(labeled)))
        if len(regions) == 0:
            continue
        
        print(f"  Processing {stage_name}: {len(regions)} continuous regions, {stage_mask.sum() / sfreq:.1f} seconds total")
        
        # Compute PSD for each region and weighted average (like sleepeegpy)
        psds_list, weights = [], []
        n_samples_total = 0
        for region in regions:
            region_data = data[:, region]
            n_valid_samples = np.count_nonzero(~np.isnan(region_data))
            
            # Skip regions with too few valid samples for reliable FFT
            if n_valid_samples < n_fft:
                continue
            
            # Compute PSD for this region using Welch's method. psd_array_welch handles NaN values automatically
            # Using sleepeegpy dashboard parameters for smoother plots
            psds, freqs = mne.time_frequency.psd_array_welch(
                region_data,
                sfreq,
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                n_per_seg=n_per_seg,
                n_overlap=n_overlap,
                window=window,
                verbose=False
            )
            
            if np.all(np.isnan(psds)):
                continue
            
            psds_list.append(psds[0])  # Remove channel dimension
            weights.append(n_valid_samples)  # Weight by valid samples
            n_samples_total += n_valid_samples
        
        if len(psds_list) == 0:
            print(f"  ✗ Warning: {stage_name} has no valid regions (all regions have too many bad epochs)")
            continue
        
        # Weighted average across regions (matching sleepeegpy)
        # If there are NaNs in PSD, mask them for proper averaging
        psds_array = np.array(psds_list)
        if np.any(np.isnan(psds_array)):
            masked_psds = np.ma.masked_array(psds_array, np.isnan(psds_array))
            avg_psd = np.ma.average(masked_psds, weights=weights, axis=0)
            avg_psd = avg_psd.filled(np.nan)  # Convert back to regular array
        else:
            avg_psd = np.average(psds_array, weights=weights, axis=0)
        
        # Calculate percentage of VALID (non-bad) data for this stage
        percent = round(n_samples_total / n_samples_valid * 100, 2)
        stage_psds[stage_name] = {
            'freqs': freqs,
            'psd': avg_psd,
            'percent': percent,
            'n_samples': n_samples_total
        }
    
    # Create plot (matching sleepeegpy's style)
    fig, ax = plt.subplots(figsize=(12, 6))
    for stage_name in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        if stage_name in stage_psds:
            freqs = stage_psds[stage_name]['freqs']
            psd = stage_psds[stage_name]['psd']
            percent = stage_psds[stage_name]['percent']
            
            # Convert from V²/Hz to µV²/Hz (dB) - matching sleepeegpy
            psd_db = 10 * np.log10(10**12 * psd)
            
            ax.plot(
                freqs, 
                psd_db, 
                linewidth=2, 
                color=stage_colors[stage_name],
                label=f'{stage_name} ({percent}%)',
                alpha=0.8
            )
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'$\mu V^{2}/Hz$ (dB)', fontsize=12)
    ax.set_title(f'PSD - Subject: {subject_id}, Channel: {channel_name}', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(fmin, fmax)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'{subject_id}_{channel_name}_psd_by_stage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ PSD plot saved to: {output_path}")
    plt.close(fig)
    
    return stage_psds
        

def plot_topo_and_choose_roi_channels(
    subject_id,
    raw,
    hypno_path,
    output_dir,
    hypno_freq=1,
    n_channels=3,
    sigma_band=(12, 16),
    plot_selected_markers=True,
):
    """
    Find best ROI channels for spindle analysis based on N2 sigma power topography.
    
    Strategy:
    1. Restrict to clean N2 epochs (and optionally N3)
    2. Make sigma-band topography (12-16 Hz average power)
    3. Pick top 2-3 sensors near canonical central/centroparietal and in compact cluster
    4. Prefer Cz-ish, C3-ish, C4-ish or central + 2 centroparietal neighbors
    5. Sanity check: sigma power high in N2 but NOT in wake/REM
    6. Exclude channels marked as outliers in previous boxplot analysis
    
    Returns:
        list: Selected channel names (e.g., ['E11', 'E12', 'E20'])
    """
    print(f"Finding ROI channels for: {subject_id} ({len(raw.ch_names)} channels)")
    print(f"{'='*80}")
    
    print(f"Loading hypnogram...")
    hypno = np.loadtxt(hypno_path, dtype=int)
    
    sfreq = raw.info['sfreq']
    repeats = int(sfreq / hypno_freq)
    hypno_up = np.repeat(hypno, repeats)
    npts_data = raw.n_times
    if len(hypno_up) < npts_data:
        hypno_up = np.pad(hypno_up, (0, npts_data - len(hypno_up)), mode='edge')
    elif len(hypno_up) > npts_data:
        hypno_up = hypno_up[:npts_data]
    
    print(f"Computing sigma power ({sigma_band[0]}-{sigma_band[1]} Hz)...")
    channel_stats = compute_sigma_power_stats(raw, hypno_up, sigma_band)
    
    # Sort by score (outliers already excluded via raw.info['bads'])
    sorted_channels = sorted(channel_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Select top N channels
    selected = [ch for ch, _ in sorted_channels[:n_channels]]
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Top {n_channels} ROI Channels:")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Channel':<12}{'N2 Power':<12}{'Wake/REM':<12}{'Ratio':<10}{'Score':<10}")
    print(f"{'-'*80}")
    
    for rank, (ch, stats) in enumerate(sorted_channels[:n_channels], 1):
        print(f"{rank:<6}{ch:<12}{stats['n2_power']:<12.3f}{stats['wake_rem_power']:<12.3f}"
              f"{stats['ratio']:<10.2f}{stats['score']:<10.2f}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nCreating topography...")
    
    # Create power array (outliers already excluded via raw.info['bads'])
    n2_power_array = np.array([channel_stats[ch]['n2_power'] for ch in raw.ch_names])
    evoked = mne.EvokedArray(n2_power_array[:, np.newaxis], raw.info.copy(), tmin=0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im, _ = mne.viz.plot_topomap(
        evoked.data[:, 0], evoked.info, axes=ax, show=False,
        cmap='RdYlBu_r', vlim=(0, np.percentile(n2_power_array, 95)), contours=6
    )
    
    if plot_selected_markers:
        selected_indices = [raw.ch_names.index(ch) for ch in selected]
        pos = mne.channels.layout._find_topomap_coords(evoked.info, picks=selected_indices)
        ax.plot(pos[:, 0], pos[:, 1], 'g*', markersize=15, markeredgecolor='black', 
                markeredgewidth=1.5, label='Selected')
        
        # Create legend with channel names
        legend_label = f"Selected: {', '.join(selected)}"
        ax.legend([legend_label], loc='upper left', fontsize=10)
    
    ax.set_title(f'N2 Sigma Power ({sigma_band[0]}-{sigma_band[1]} Hz)\n{subject_id}', 
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='RMS Power (µV)', shrink=0.8)
    plt.tight_layout()
    
    plot_path = output_dir / f'{subject_id}_roi_topography.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")
    print(f"{'='*80}\n")
    return selected


def bout_durations(raw, stages, min_duration):
    bouts_count = 0
    bouts_durations = []
    for desc, duration in zip(raw.annotations.description, raw.annotations.duration):
        if desc in stages and duration >= min_duration:
            bouts_count += 1
            bouts_durations.append(duration)

    print(f"{', '.join(stages)} annotations longer than {min_duration} seconds: {bouts_count}")


def main():
    DATA_DIR = Path(BASE_DIR) / "control_clean"
    HYPNO_DIR = Path(BASE_DIR) / "HC_hypno"

    subjects = get_all_subjects(DATA_DIR)
    channels_per_sub = {}
    for sub in subjects:
        sub_dir = DATA_DIR / sub
        hypno_path = f"{HYPNO_DIR}/{sub}.txt"
        try:
            print("-" * 80)
            eeg_path = find_subject_fif_file(sub_dir)
            raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)

            selected = plot_topo_and_choose_roi_channels(sub, raw, hypno_path, output_dir="young_control/topo/")
            channels_per_sub[sub] = selected

            for channel in selected:
                plot_psd_by_sleep_stage(sub, raw, hypno_path, channel, "young_control/psds/", hypno_freq=1)
            
            print("-" * 80)
            bout_durations(raw, ['NREM2'], 300)

        except Exception as e:
            print(f"Error processing subject {sub}: {str(e)}")
            traceback.print_exc()
    
    # Save channels per subject to CSV
    output_csv = Path("young_control/topo/roi_channels_per_subject.csv")
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    with open(output_csv, 'w') as f:
        f.write("subject_id,channel1,channel2,channel3\n")
        for subject_id, channels in channels_per_sub.items():
            channels_padded = channels + [''] * (3 - len(channels))
            f.write(f"{subject_id},{channels_padded[0]},{channels_padded[1]},{channels_padded[2]}\n")
    
    print(f"{'='*80}\n")
    print(f"✓ Saved ROI channels for {len(channels_per_sub)} subjects to: {output_csv}")


if __name__ == "__main__":
    main()
