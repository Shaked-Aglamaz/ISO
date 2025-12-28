from pathlib import Path
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import numpy as np
from config import BASE_DIR, EAR_ELECTRODES
from utils import get_all_subjects, find_subject_fif_file

mne.set_log_level("error")


def compute_sigma_power_stats(raw, hypno_up, sigma_band=(12, 16)):
    """
    Compute sigma power statistics for all EEG channels.
    Returns:
        dict: Channel statistics with keys 'n2_power', 'wake_rem_power', 'ratio', 'score'
    """
    sfreq = raw.info['sfreq']
    n2_mask = (hypno_up == 2)
    wake_rem_mask = (hypno_up == 0) | (hypno_up == 4)
    n2_duration = n2_mask.sum() / sfreq
    wake_rem_duration = wake_rem_mask.sum() / sfreq
    print(f"N2: {n2_duration:.1f}s ({n2_duration/60:.1f} min)")
    print(f"Wake/REM: {wake_rem_duration:.1f}s ({wake_rem_duration/60:.1f} min)")

    # Filter to sigma band
    raw = raw.copy().load_data()
    raw_sigma = raw.filter(l_freq=sigma_band[0], h_freq=sigma_band[1], verbose=False)
    
    # Get data with bad epoch rejection
    data = raw_sigma.get_data(reject_by_annotation="NaN")
    
    # Compute RMS power for each channel
    channel_stats = {}

    for i, ch_name in enumerate(raw.ch_names):
        ch_data = data[i, :]
        
        # N2 sigma power
        n2_data = ch_data[n2_mask]
        n2_valid = n2_data[~np.isnan(n2_data)]
        n2_power = np.sqrt(np.mean(n2_valid**2)) if len(n2_valid) > 0 else 0
        
        # Wake/REM sigma power
        wake_rem_data = ch_data[wake_rem_mask]
        wake_rem_valid = wake_rem_data[~np.isnan(wake_rem_data)]
        wake_rem_power = np.sqrt(np.mean(wake_rem_valid**2)) if len(wake_rem_valid) > 0 else 0
        
        # Ratio (higher = more N2-specific)
        ratio = n2_power / wake_rem_power if wake_rem_power > 0 else 0
        
        channel_stats[ch_name] = {
            'n2_power': n2_power,
            'wake_rem_power': wake_rem_power,
            'ratio': ratio,
            'score': n2_power * ratio  # Combined score
        }
    
    return channel_stats

def get_outliers(data, channel_names):
    """Get outlier channels using IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_channels = []
    for i, val in enumerate(data):
        if val < lower_bound or val > upper_bound:
            outlier_channels.append(channel_names[i])
    
    return outlier_channels


def analyze_sigma_power(
    subject_id,
    raw,
    hypno_path,
    output_dir,
    hypno_freq=1,
    sigma_band=(12, 16),
):
    """
    Create boxplot of sigma power across all channels for a subject.
    
    Shows distribution of N2 sigma power, Wake/REM sigma power, and their ratio
    across all EEG channels.
    
    Args:
        subject_id: Subject identifier
        raw: Raw EEG data object
        hypno_path: Path to hypnogram file
        hypno_freq: Hypnogram sampling frequency (Hz)
        sigma_band: Frequency band for sigma (default: 12-16 Hz)
        output_dir: Optional directory to save plot
    
    Returns:
        dict: Channel statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Analyzing sigma power for: {subject_id}")
    print(f"{'='*80}")
    
    print(f"Loading hypnogram...")
    hypno = np.loadtxt(hypno_path, dtype=int)
    # Upsample hypnogram
    sfreq = raw.info['sfreq']
    repeats = int(sfreq / hypno_freq)
    hypno_up = np.repeat(hypno, repeats)
    
    # Fit to data length
    npts_data = raw.n_times
    if len(hypno_up) < npts_data:
        hypno_up = np.pad(hypno_up, (0, npts_data - len(hypno_up)), mode='edge')
    elif len(hypno_up) > npts_data:
        hypno_up = hypno_up[:npts_data]
    
    print(f"Computing sigma power ({sigma_band[0]}-{sigma_band[1]} Hz) for {len(raw.ch_names)} channels...")
    channel_stats = compute_sigma_power_stats(raw, hypno_up, sigma_band)
    
    # Extract values for plotting (convert to µV if needed - data should already be in µV)
    n2_powers = [stats['n2_power'] for stats in channel_stats.values()]
    wake_rem_powers = [stats['wake_rem_power'] for stats in channel_stats.values()]
    ratios = [stats['ratio'] for stats in channel_stats.values() if stats['ratio'] > 0]
    
    # Get outliers for each metric
    n2_outliers = get_outliers(n2_powers, list(channel_stats.keys()))
    wake_rem_outliers = get_outliers(wake_rem_powers, list(channel_stats.keys()))

    # For ratios, we need to get the channel names that have valid ratios
    ratio_channel_names = [ch for ch, stats in channel_stats.items() if stats['ratio'] > 0]
    ratio_outliers = get_outliers(ratios, ratio_channel_names)

    plot_sigma_power_boxplot(n2_powers, n2_outliers, wake_rem_powers, wake_rem_outliers, ratios, ratio_outliers, sigma_band, subject_id, output_dir)
    # Save channel data and outliers to CSV
    csv_path = output_dir / f'{subject_id}_sigma_power_analysis.csv'
    
    # Create main data: all channels with their N2 power, sorted by N2 power (descending)
    channel_names = list(channel_stats.keys())
    n2_power_values = [channel_stats[ch]['n2_power'] for ch in channel_names]
    
    # Sort channels by N2 power in descending order
    sorted_indices = sorted(range(len(n2_power_values)), key=lambda i: n2_power_values[i], reverse=True)
    channel_names_sorted = [channel_names[i] for i in sorted_indices]
    n2_power_sorted = [n2_power_values[i] for i in sorted_indices]
    
    # Find the maximum length between main data and outlier lists
    max_len = max(len(channel_names_sorted), len(n2_outliers), len(wake_rem_outliers), len(ratio_outliers))
    
    # Pad all lists with empty strings to make them equal length
    channel_names_padded = channel_names_sorted + [''] * (max_len - len(channel_names_sorted))
    n2_power_padded = n2_power_sorted + [''] * (max_len - len(n2_power_sorted))
    n2_outliers_padded = n2_outliers + [''] * (max_len - len(n2_outliers))
    wake_rem_outliers_padded = wake_rem_outliers + [''] * (max_len - len(wake_rem_outliers))
    ratio_outliers_padded = ratio_outliers + [''] * (max_len - len(ratio_outliers))
    
    # Create DataFrame with all columns
    analysis_df = pd.DataFrame({
        'Channel_Name': channel_names_padded,
        'N2_Sigma_Power': n2_power_padded,
        'N2_Sigma_Power_Outliers': n2_outliers_padded,
        'WakeREM_Sigma_Power_Outliers': wake_rem_outliers_padded,
        'N2_WakeREM_Ratio_Outliers': ratio_outliers_padded
    })
    
    analysis_df.to_csv(csv_path, index=False)
    print(f"✓ Saved analysis data: {csv_path}")
    return channel_stats, n2_outliers
    

def plot_sigma_power_boxplot(n2_powers, n2_outliers, wake_rem_powers, wake_rem_outliers, ratios, ratio_outliers, sigma_band, subject_id, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: N2 sigma power
    bp1 = axes[0].boxplot(n2_powers, vert=True, patch_artist=True, showfliers=True, zorder=1)
    bp1['boxes'][0].set_facecolor('lightgreen')
    
    x_n2 = np.random.normal(1, 0.04, size=len(n2_powers))  # Add jitter for visibility
    axes[0].scatter(x_n2, n2_powers, alpha=0.4, s=20, color='darkgreen', edgecolors='black', linewidths=0.5, zorder=2)
    axes[0].set_ylabel('RMS Power (µV)', fontsize=12)
    axes[0].set_title(f'N2 Sigma Power\n({sigma_band[0]}-{sigma_band[1]} Hz)', fontsize=12, fontweight='bold')
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(['All Channels'])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text with proper formatting (scientific notation if needed)
    mean_n2 = np.mean(n2_powers)
    median_n2 = np.median(n2_powers)
    std_n2 = np.std(n2_powers)
    
    # Use scientific notation for very small values
    if mean_n2 < 0.01:
        stats_text = f'Mean: {mean_n2:.2e} µV\nMedian: {median_n2:.2e} µV\nStd: {std_n2:.2e} µV'
    else:
        stats_text = f'Mean: {mean_n2:.3f} µV\nMedian: {median_n2:.3f} µV\nStd: {std_n2:.3f} µV'
    
    axes[0].text(0.98, 0.98, stats_text,
                transform=axes[0].transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add outliers list at bottom
    if len(n2_outliers) > 0:
        outliers_text = f'Outliers ({len(n2_outliers)}): {", ".join(n2_outliers)}'
        axes[0].text(0.5, -0.15, outliers_text, transform=axes[0].transAxes, 
                    fontsize=7, ha='center', style='italic', color='red', wrap=True)
    
    # Plot 2: Wake/REM sigma power
    bp2 = axes[1].boxplot(wake_rem_powers, vert=True, patch_artist=True, showfliers=True, zorder=1)
    bp2['boxes'][0].set_facecolor('lightcoral')
    
    # Add individual data points (strip plot) on top
    x_wr = np.random.normal(1, 0.04, size=len(wake_rem_powers))
    axes[1].scatter(x_wr, wake_rem_powers, alpha=0.4, s=20, color='darkred', edgecolors='black', linewidths=0.5, zorder=2)
    
    axes[1].set_ylabel('RMS Power (µV)', fontsize=12)
    axes[1].set_title(f'Wake/REM Sigma Power\n({sigma_band[0]}-{sigma_band[1]} Hz)', fontsize=12, fontweight='bold')
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(['All Channels'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text with proper formatting
    mean_wr = np.mean(wake_rem_powers)
    median_wr = np.median(wake_rem_powers)
    std_wr = np.std(wake_rem_powers)
    
    if mean_wr < 0.01:
        stats_text = f'Mean: {mean_wr:.2e} µV\nMedian: {median_wr:.2e} µV\nStd: {std_wr:.2e} µV'
    else:
        stats_text = f'Mean: {mean_wr:.3f} µV\nMedian: {median_wr:.3f} µV\nStd: {std_wr:.3f} µV'
    
    axes[1].text(0.98, 0.98, stats_text,
                transform=axes[1].transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add outliers list at bottom
    if len(wake_rem_outliers) > 0:
        outliers_text = f'Outliers ({len(wake_rem_outliers)}): {", ".join(wake_rem_outliers)}'
        axes[1].text(0.5, -0.15, outliers_text, transform=axes[1].transAxes, 
                    fontsize=7, ha='center', style='italic', color='red', wrap=True)
    
    # Plot 3: N2/Wake-REM ratio
    bp3 = axes[2].boxplot(ratios, vert=True, patch_artist=True, showfliers=True, zorder=1)
    bp3['boxes'][0].set_facecolor('lightblue')
    
    # Add individual data points (strip plot) on top
    x_ratio = np.random.normal(1, 0.04, size=len(ratios))
    axes[2].scatter(x_ratio, ratios, alpha=0.4, s=20, color='darkblue', edgecolors='black', linewidths=0.5, zorder=2)
    
    axes[2].set_ylabel('Ratio (N2 / Wake-REM)', fontsize=12)
    axes[2].set_title(f'N2 Specificity Ratio\n(Higher = More N2-specific)', fontsize=12, fontweight='bold')
    axes[2].set_xticks([1])
    axes[2].set_xticklabels(['All Channels'])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    axes[2].text(0.98, 0.98, f'Mean: {np.mean(ratios):.2f}\nMedian: {np.median(ratios):.2f}\nStd: {np.std(ratios):.2f}',
                transform=axes[2].transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add outliers list at bottom
    if len(ratio_outliers) > 0:
        outliers_text = f'Outliers ({len(ratio_outliers)}): {", ".join(ratio_outliers)}'
        axes[2].text(0.5, -0.15, outliers_text, transform=axes[2].transAxes, 
                    fontsize=7, ha='center', style='italic', color='red', wrap=True)
    
    # Add overall title
    fig.suptitle(f'Sigma Power Distribution - Subject: {subject_id}', fontsize=14, fontweight='bold', y=1.02)
    
    # Adjust layout to make room for outlier lists
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave more space at bottom for outlier text
    
    plot_path = output_dir / f'{subject_id}_sigma_power_boxplot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")
    

def reference_and_interpolate(raw, subject_dir, subject_id, output_suffix=None):
    """
    Load EEG data, mark bad channels, interpolate them, and apply average reference.
    1. Load manually marked bad channels 
    2. Load N2 outliers channels
    3. Apply average reference (excluding any remaining bad channels)
    4. Interpolate bad channels using spherical splines
    5. Save the cleaned data to a new file
    """
    print(f"\nProcessing: {subject_id}")
    print(f"{'='*80}")

    # === 1. Load manually marked bad channels ===
    bad_channels_file = subject_dir / f"{subject_id}_bad_channels.txt"
    manual_bad_channels = set()
    if bad_channels_file.exists():
        with open(bad_channels_file, 'r') as f:
            manual_bad_channels = set(line.strip() for line in f if line.strip())
    else:
        print(f"⚠ No manual bad channels file found: {bad_channels_file.name}")
    
    # === 2. Load N2 outlier channels ===
    n2_outliers_file = subject_dir / "n2_outliers.txt"
    n2_outliers = set()
    if n2_outliers_file.exists():
        with open(n2_outliers_file, 'r') as f:
            n2_outliers = set(line.strip() for line in f if line.strip())
    else:
        print(f"⚠ No N2 outliers file found: {n2_outliers_file.name}")

    all_bad_channels = manual_bad_channels | n2_outliers
    existing_bad_channels = [ch for ch in all_bad_channels if ch in raw.ch_names]
    missing_channels = all_bad_channels - set(existing_bad_channels)
    
    if len(missing_channels) > 0:
        print(f"⚠ Warning: {len(missing_channels)} bad channels not found in raw data: {', '.join(sorted(missing_channels))}")
    
    if len(existing_bad_channels) == 0:
        print(f"\n✓ No bad channels to interpolate!")
        print(f"Applying average reference only...")
    
    raw.info['bads'] = existing_bad_channels
    
    # === 3. Apply average reference ===
    print(f"Applying average reference...")
    raw.set_eeg_reference('average', projection=False)

    # === 4. Interpolate bad channels ===
    if len(existing_bad_channels) > 0:
        print(f"Interpolating {len(existing_bad_channels)} bad channels...")
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    # === 5. Save the processed data ===
    if output_suffix is None:
        output_suffix = f"_{len(raw.ch_names)}-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate"
    output_filename = f"{subject_id}{output_suffix}_raw.fif"
    output_path = subject_dir / output_filename
    raw.save(output_path, overwrite=True, verbose=False)
    print(f"✓ Saved: {output_path}")    


def bout_durations(raw):
    long_nrem2_count = 0
    long_nrem2_durations = []

    for desc, duration in zip(raw.annotations.description, raw.annotations.duration):
        if desc in ['NREM2', 'NREM3'] and duration >= 280:
            long_nrem2_count += 1
            long_nrem2_durations.append(duration)

    print(f"NREM2/NREM3 annotations longer than 280 seconds: {long_nrem2_count}")


def bad_durations(raw):
    bad_time = sum(duration for desc, duration in zip(raw.annotations.description, raw.annotations.duration) 
                    if desc.startswith('BAD'))
    total_recording_time = raw.times[-1]  # Total recording duration in seconds
    bad_time_percentage = (bad_time / total_recording_time) * 100

    print(f"BAD time: {bad_time:.1f}s / {total_recording_time:.1f}s ({bad_time_percentage:.1f}%)")


def main():
    """Main function to run PSD analysis for all subjects."""
    DATA_DIR = Path(BASE_DIR) / "control_clean"
    HYPNO_DIR = Path(BASE_DIR) / "HC_hypno"
    
    subjects = get_all_subjects(DATA_DIR)
    for sub in subjects[-2:-1]:
        sub_dir = DATA_DIR / sub
        hypno_path = f"{HYPNO_DIR}/{sub}.txt"
        try:
            print("-" * 80)
            eeg_path = find_subject_fif_file(sub_dir)
            raw = mne.io.read_raw_fif(eeg_path, preload=False, verbose=False)
            channels = [ch for ch in raw.ch_names if ch not in EAR_ELECTRODES]
            raw.pick(channels)
            raw.load_data()

            _, n2_outliers = analyze_sigma_power(sub, raw, hypno_path, output_dir="young_control/sigma_boxplot/")
            with open(f"{sub_dir}/n2_outliers.txt", 'w') as f:
                for channel in n2_outliers:
                    f.write(f"{channel}\n")

            reference_and_interpolate(raw, sub_dir, sub)
            
        except Exception as e:
            print(f"Error processing subject {sub}: {str(e)}")
    
if __name__ == "__main__":
    main()
