import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from utils.config import BASE_DIR
from utils.utils import get_all_subjects


def filter_subjects_by_detection_rate(subjects, min_detection_rate=0.2, dir_path=None):
    """ Filter subjects based on their ISFS detection rate. """
    print(f"\nFiltering subjects by detection rate (>= {min_detection_rate*100:.0f}%)...")
    filtered_subjects = []
    excluded_subjects = []
    
    for subject in subjects:
        subject_data = load_subject_data(subject, dir_path)
        if subject_data is None:
            excluded_subjects.append((subject, 0, 0, 0))
            continue
        
        total_channels = len(subject_data)
        successful_channels = subject_data[['peak_frequency', 'bandwidth', 'auc']].notna().all(axis=1).sum()
        detection_rate = successful_channels / total_channels if total_channels > 0 else 0
        
        if detection_rate >= min_detection_rate:
            filtered_subjects.append(subject)
        else:
            excluded_subjects.append((subject, detection_rate, successful_channels, total_channels))
    
    if excluded_subjects:
        print(f"📊 Detection Rate Summary:")
        print(f"     Excluded {len(excluded_subjects)} subjects below {min_detection_rate*100:.0f}% threshold:")
        for subj, rate, succ, total in excluded_subjects:
            print(f"       - {subj}: {rate*100:.1f}% ({succ}/{total})")
    
    print(f"\nProceeding with {len(filtered_subjects)} subjects that meet detection rate criteria")
    return filtered_subjects, excluded_subjects


def load_subject_data(subject_id, dir_path=None):
    """Load channel summary data for a specific subject."""
    dir_path = f"{dir_path}/{subject_id}" if dir_path is not None else subject_id
    summary_file = f"{dir_path}/{subject_id}_all_channels_summary.csv"

    if not os.path.exists(summary_file):
        print(f"Summary file not found for {subject_id}: {summary_file}")
        return None
    
    try:
        df = pd.read_csv(summary_file, comment='#')
        df['subject'] = subject_id  # Add subject ID column
        return df
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}")
        return None


def collect_all_data(subjects, target_channel, group=None):
    """Collect and combine spectral data from all subjects for target channel."""
    all_data = []
    for subject in subjects:
        subject_data = load_subject_data(subject, group)
        if subject_data is not None:
            channel_data = subject_data[subject_data['channel'] == target_channel]
            if not channel_data.empty:
                all_data.append(channel_data)
            else:
                print(f"No {target_channel} data found for subject {subject}")
        else:
            print(f"Could not load data for subject {subject}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records: {len(combined_df)}")
        return combined_df
    else:
        print(f"No data collected for channel {target_channel}")
        return None


def _create_violin_box_dots_subplot(ax, data, metric_info, legend=False):
    """
    Helper function to create a single violin+box+dots subplot.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : pandas Series or numpy array
        Data to plot
    metric_info : dict
        Dictionary with 'name' and 'unit' keys
    legend : bool
        Whether to add legend (default: False, typically only first subplot)
    """
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12)
        ax.set_title(f'{metric_info["name"]}')
        return
    
    # Calculate statistics
    mean_val = data.mean()
    std_val = data.std()
    median_val = data.median()
    
    # Create half-violin plot using seaborn (single-sided)
    violin = sns.violinplot(y=data, ax=ax, color='#8dd3c7', inner=None, cut=0, linewidth=1.5)
    
    # Modify the violin to show only right half
    for collection in ax.collections:
        if hasattr(collection, 'get_paths'):
            paths = collection.get_paths()
            if len(paths) > 0:
                # Get vertices
                vertices = paths[0].vertices
                
                # Find center x position
                center_x = np.mean(vertices[:, 0])
                
                # Keep only right half (x >= center)
                mask = vertices[:, 0] >= center_x
                vertices[~mask, 0] = center_x  # Set left side to center
                
                collection.set_alpha(0.7)
    
    # Add mean and median lines
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
    ax.axhline(median_val, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Median')

    # Add boxplot overlay centered
    box_parts = ax.boxplot([data], positions=[0], widths=0.15, 
                            vert=True, patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            medianprops=dict(color='orange', linewidth=2))
    
    # Add individual dots with jitter centered at x=0 (aligned with violin/boxplot)
    np.random.seed(42)  # For reproducible jitter
    x_jitter = np.random.normal(0, 0.02, size=len(data))  # Centered at 0 with small jitter
    ax.scatter(x_jitter, data, alpha=0.4, color='darkblue', s=30, zorder=3)
    
    # Formatting
    ax.set_title(f'{metric_info["name"]}', fontsize=14)
    ax.set_ylabel(f'{metric_info["name"]} ({metric_info["unit"]})', fontsize=12)
    ax.set_xlim(-0.4, 0.4)  # Adjust x-limits for half-violin layout
    ax.set_xticks([0])
    ax.set_xticklabels([''])  # Remove x-tick labels for cleaner look
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # Add legend if requested
    if legend:
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean'),
            Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Median'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                  markersize=6, alpha=0.5, label='Data Points')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)


def plot_single_channel_violin(df, output_dir, channel_name):
    """
    Create violin+box+dots plots for a single channel across subjects.
    Each data point represents one subject for the specified channel.
    """
    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Spectral Parameters Distribution - {channel_name} Channel\n(N={len(df)})', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = df[metric].dropna()
        _create_violin_box_dots_subplot(ax, data, info, legend=(i == 0))
    
    plt.tight_layout()
    violin_path = output_dir / f"{channel_name}_violin_box_dots.png"
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin+box+dots plot saved to: {violin_path}")


def load_and_combine_subject_data(subjects, group=None):
    """Load and combine data from all subjects."""
    all_subject_data = []
    for subject in subjects:
        subject_data = load_subject_data(subject, dir_path=group)
        if subject_data is not None:
            all_subject_data.append(subject_data)
    
    if not all_subject_data:
        print("No subject data available for topography")
        return None
    
    return pd.concat(all_subject_data, ignore_index=True)


def filter_eeg_channels(combined_data):
    """Filter non-EEG channels and calculate channel averages, handling failed ISFS channels."""
    eeg_data = combined_data.copy()
    
    # Handle channels that failed Gaussian fitting (missing ISFS)
    if 'gaussian_fit_failed' in eeg_data.columns:
        # For failed channels, the metrics columns should already be empty/NaN in the CSV
        # Let's ensure they are properly set to NaN for visualization
        metric_columns = ['peak_frequency', 'bandwidth', 'auc'] 
        for col in metric_columns:
            if col in eeg_data.columns:
                # Convert empty strings to NaN if they exist
                eeg_data[col] = pd.to_numeric(eeg_data[col], errors='coerce')
    
    if len(eeg_data) == 0:
        print("No EEG channels found for topography")
        return None
    
    # For single subject, just return the data as is (don't average across subjects)
    if len(eeg_data['subject'].unique()) == 1:
        channel_averages = eeg_data[['channel', 'peak_frequency', 'bandwidth', 'auc']].copy()
        # Rename columns to match expected format
        channel_averages = channel_averages.rename(columns={
            'peak_frequency': 'avg_peak_frequency',
            'bandwidth': 'avg_bandwidth', 
            'auc': 'avg_auc'
        })
        # Add gaussian_fit_failed column for tracking
        if 'gaussian_fit_failed' in eeg_data.columns:
            channel_averages['gaussian_fit_failed'] = eeg_data['gaussian_fit_failed']
    else:
        # Multiple subjects: average across subjects (ignoring NaN values)
        channel_averages = eeg_data.groupby('channel')[['peak_frequency', 'bandwidth', 'auc']].mean().reset_index()
        # Rename columns to match expected format
        channel_averages = channel_averages.rename(columns={
            'peak_frequency': 'avg_peak_frequency',
            'bandwidth': 'avg_bandwidth',
            'auc': 'avg_auc'
        })
    
    return channel_averages


def create_electrode_montage(channel_names):
    """Create MNE montage and filter channels with known positions."""
    try:
        montage = mne.channels.make_standard_montage('EGI_256')
        available_channels = [ch for ch in channel_names if ch in montage.ch_names]
        
        if len(available_channels) < 10:
            print(f"Not enough standard electrode positions found ({len(available_channels)})")
            print("Available channels:", available_channels[:10], "...")
            return None, None
            
        return montage, available_channels
        
    except Exception as e:
        print(f"Could not create montage: {e}")
        return None, None


def setup_electrode_info(channel_averages):
    """Setup electrode montage and create MNE info object."""
    channel_names = channel_averages['channel'].tolist()
    
    montage, available_channels = create_electrode_montage(channel_names)
    if montage is None:
        return None, None, None
    
    channel_averages_filtered = channel_averages[channel_averages['channel'].isin(available_channels)]
    channel_names_filtered = available_channels
    info = mne.create_info(ch_names=channel_names_filtered, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    
    return info, channel_averages_filtered, channel_names_filtered


def extract_channel_values(channel_averages, channel_names, metric):
    """Extract values for specific metric from channel averages, handling missing ISFS data."""
    values = []
    missing_channels = []
    failed_channels = []
    for ch in channel_names:
        ch_data = channel_averages[channel_averages['channel'] == ch]
        if not ch_data.empty:
            metric_value = ch_data[metric].iloc[0]
            if pd.isna(metric_value):
                failed_channels.append(ch)
            values.append(metric_value)
        else:
            missing_channels.append(ch)
            values.append(np.nan)
    
    return np.array(values)


def plot_single_topography(values, info, ax, metric_info, normalize=True, fig=None):
    """Plot single topographical map with optional normalization and missing data handling."""
    # Count data availability
    total_electrodes = len(values)
    valid_mask = ~np.isnan(values)
    n_valid = np.sum(valid_mask)
    n_missing = total_electrodes - n_valid
    
    if n_valid == 0:
        ax.text(0.5, 0.5, f'No valid ISFS data\nfor {metric_info["name"]}', 
                transform=ax.transAxes, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_title(f'{metric_info["name"]}\n(0/{total_electrodes} channels)', fontsize=12)
        return None
    
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    
    if std_val > 0:
        outlier_mask = np.abs(values - mean_val) > 3 * std_val
        cleaned_values = values.copy()
        cleaned_values[outlier_mask] = np.nan
        n_outliers = np.sum(outlier_mask)
    else:
        cleaned_values = values
        n_outliers = 0
    
    n_plotted = np.sum(~np.isnan(cleaned_values))
    mean_value = np.nanmean(cleaned_values)
    
    # Prepare data for plotting
    if normalize:
        if mean_value == 0:
            plot_values = cleaned_values
        else:
            plot_values = cleaned_values / mean_value        
    else:
        plot_values = cleaned_values
    title = f'{metric_info["name"]}\n(Mean = {mean_value:.3f} {metric_info["unit"]})'

    # Add data availability info to title
    if n_missing > 0 or n_outliers > 0:
        availability_info = f'{n_plotted}/{total_electrodes} channels'
        if n_missing > 0:
            availability_info += f' ({n_missing} no ISFS'
            if n_outliers > 0:
                availability_info += f', {n_outliers} outliers)'
            else:
                availability_info += ')'
        elif n_outliers > 0:
            availability_info += f' ({n_outliers} outliers)'
        
        title += f'\n({availability_info})'
    
    # Plot the topography
    vmin = np.nanmin(plot_values)
    vmax = np.nanmax(plot_values)
    
    # Replace NaN values with mean for plotting (so MNE shows the data properly)
    plot_data = plot_values.copy()
    missing_data_mask = np.isnan(plot_values)
    
    if np.any(missing_data_mask):
        plot_data[missing_data_mask] = np.nanmean(plot_values)
    
    # Plot the topography
    im, _ = mne.viz.plot_topomap(plot_data, info, axes=ax, show=False,
                               cmap='RdBu_r', vlim=(vmin, vmax), contours=6)
    
    ax.set_title(title, fontsize=10)
    
    # Don't create colorbar here - it will be handled by the calling function
    # Just return the mean_value and let the caller handle the colorbar
    return mean_value


def create_and_save_topography_figure(subjects, metrics, info, channel_averages, channel_names, normalize=True, output_dir=None):
    """Create figure with all three topographical plots and save to file."""
    # Create figure with subplots, but reserve space for colorbars
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Handle single subject vs multiple subjects
    if len(subjects) == 1:
        subject_name = subjects[0]
        norm_text = "(Normalized by mean across channels)" if normalize else "(Raw values)"
        title = f'Topographical Distribution - Subject {subject_name}\n{norm_text}'
        output_dir = Path(output_dir) if output_dir is not None else Path(subject_name)
        suffix = "normalized" if normalize else "raw"
        filename = f"{subject_name}_topographies_{suffix}.png"
    else:
        norm_text = "(Normalized by mean across channels)" if normalize else "(Raw values)"
        title = f'Topographical Distribution - Average Across {len(subjects)} Subjects\n{norm_text}'
        output_dir = Path(output_dir) if output_dir is not None else Path("group_level_topographies")
        suffix = "normalized" if normalize else "raw"
        filename = f"topographies_average_all_subjects_{suffix}.png"
    
    fig.suptitle(title, fontsize=14)
    
    for i, (metric, metric_info) in enumerate(metrics.items()):
        ax = axes[i]
        
        values = extract_channel_values(channel_averages, channel_names, metric)
        mean_value = plot_single_topography(values, info, ax, metric_info, normalize=normalize, fig=fig)
        
        if mean_value is None:
            continue
            
        if len(ax.images) > 0:
            im = ax.images[0]
            cbar_label = 'Normalized Value\n(/ mean)' if normalize else f'{metric_info["unit"]}'
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=9)
    
    # Use subplots_adjust instead of tight_layout to avoid layout conflicts
    plt.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.05, hspace=0.4)
    
    output_dir.mkdir(exist_ok=True)
    topo_path = output_dir / filename
    plt.savefig(topo_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Topographical plots saved to: {topo_path}\n")


def plot_topographies(subjects, normalize=True, subject_dir_suffix=None, output_dir=None, group=None):
    """Create topographical plots for the three spectral parameters with ISFS failure handling."""
    print(f"{'='*60}")
    print(f"TOPOGRAPHY ANALYSIS - {'Subject ' + subjects[0] if len(subjects) == 1 else f'{len(subjects)} Subjects'}")

    combined_data = load_and_combine_subject_data(subjects, group)
    if combined_data is None:
        return
    
    # Print data summary before filtering
    if 'gaussian_fit_failed' in combined_data.columns:
        total_channels = len(combined_data)
        failed_channels = combined_data[combined_data['gaussian_fit_failed'] == True]
        n_failed = len(failed_channels)
        n_successful = total_channels - n_failed
        
        print(f"📊 Raw Data Summary:")
        print(f"   Successful ISFS detection: {n_successful} channels")
        print(f"   Failed ISFS detection: {n_failed} channels ({failed_channels['channel'].head(5).tolist()}{'...' if n_failed > 5 else ''})")

    channel_averages = filter_eeg_channels(combined_data)
    if channel_averages is None:
        return
    
    info, channel_averages_filtered, channel_names = setup_electrode_info(channel_averages)
    if info is None:
        return
    
    total_electrodes = len(channel_averages)
    valid_positions = len(channel_names)
    print(f"🎨 Creating topographical plots: #channels: {total_electrodes}, #positions: {valid_positions} (Normalization: {'Enabled' if normalize else 'Disabled'})")
    
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'}, 
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    create_and_save_topography_figure(subjects, metrics, info, channel_averages_filtered, channel_names, normalize=normalize, output_dir=output_dir)


def plot_subject_all_channels_violin(subject, output_dir=None):
    """
    Create violin+box+dots plots showing distribution of spectral parameters across all channels for a single subject.
    Each data point represents one channel, showing within-subject spatial variability.
    """
    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    print(f"\nProcessing violin plots for subject {subject}...")    
    subject_data = load_subject_data(subject, output_dir)
    if subject_data is None:
        print(f"  ✗ No data found for subject {subject}")
        return
    
    eeg_data = subject_data.copy()
    if len(eeg_data) == 0:
        print(f"  ✗ No EEG channels found for subject {subject}")
        return
    
    # Create figure with 3 subplots for the 3 metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'All Channels Distribution - Subject {subject}', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = eeg_data[metric].dropna()
        _create_violin_box_dots_subplot(ax, data, info, legend=(i == 0))
    
    plt.tight_layout()
    output_dir = Path(output_dir) / subject if output_dir is not None else Path(subject)
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"{subject}_all_channels_violin.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()    
    print(f"  ✓ Violin plot saved to: {plot_path}")


def load_and_process_all_channels_data(subjects, group=None):
    """
    Load, filter, and average all channels data across subjects.
    
    Returns:
    --------
    subject_averages : pd.DataFrame
        DataFrame with averaged metrics per subject
    total_channels : int
        Total number of channels processed
    valid_subjects : int
        Number of subjects that passed detection rate filter
    """
    all_data = []
    total_channels = 0
    for subject in subjects:
        subject_data = load_subject_data(subject, group)
        if subject_data is None:
            continue
        
        eeg_data = subject_data.copy()
        if len(eeg_data) == 0:
            continue
        
        eeg_data['subject'] = subject  # Add subject column
        all_data.append(eeg_data)
        total_channels += len(eeg_data)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    # Average across channels within each subject
    subject_averages = combined_df.groupby('subject')[['peak_frequency', 'bandwidth', 'auc']].mean().reset_index()
    print(f"✓ Processed {len(subject_averages)} subjects with {total_channels} total channels")
    return subject_averages, total_channels, len(subject_averages)


def plot_group_average_violin(subjects, output_dir):
    """
    Create violin+box+dots plots for group-level analysis (all channels averaged per subject).
    Each data point represents one subject's average across all channels.
    """
    subject_averages, total_channels, valid_subjects = load_and_process_all_channels_data(subjects, output_dir)
    if subject_averages is None:
        return

    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    # Create combined plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Group-Level Distribution (All Channels Averaged Per Subject)\n'
                 f'{valid_subjects} subjects, {total_channels} total channels', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = subject_averages[metric].dropna()
        _create_violin_box_dots_subplot(ax, data, info, legend=(i == 0))
    
    plt.tight_layout()
    
    plot_path = output_dir / "group_average_violin.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Group average violin plot saved to: {plot_path}")


def plot_group_spectral_power(subjects, output_dir, target_channel=None, smoothing_window=5, aggregate_channels=False):
    """
    Plot smoothed mean spectral power for the group.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    target_channel : str
        Specific channel to analyze (used when aggregate_channels=False)
    smoothing_window : int
        Window size for smoothing (default: 5)
    aggregate_channels : bool
        If True, average across all channels per subject
        If False, analyze only the target_channel
    """
    
    if aggregate_channels:
        print(f"\nCollecting spectral data for {len(subjects)} subjects (aggregating across all channels)...")
    else:
        print(f"\nCollecting spectral data for {len(subjects)} subjects (channel: {target_channel})...")
    
    all_spectral_data = []
    subject_data = {}
    for subject in subjects:
        subject_dir = output_dir / subject
        if aggregate_channels:
            # Find all spectral power CSV files for this subject
            channel_data = []
            spectral_files = list(subject_dir.glob(f"{subject}_*_output/{subject}_*_spectral_power.csv"))
            for spectral_file in spectral_files:
                try:
                    df_channel = pd.read_csv(spectral_file)
                    channel_data.append(df_channel)
                except Exception as e:
                    print(f"  ✗ Error loading {spectral_file}: {e}")
            
            if not channel_data:
                print(f"Warning: No valid channel data found for subject {subject}")
                continue
            
            # Average across channels for this subject
            try:
                # Find common frequency range across all channels
                common_freqs = channel_data[0]['frequency'].values
                all_powers = []
                for df_channel in channel_data:
                    # Interpolate to common frequency grid if needed
                    if not np.array_equal(df_channel['frequency'].values, common_freqs):
                        interp_func = interp1d(df_channel['frequency'], df_channel['mean_power'], 
                                              bounds_error=False, fill_value='extrapolate')
                        power = interp_func(common_freqs)
                    else:
                        power = df_channel['mean_power'].values
                    all_powers.append(power)
                
                # Average across channels
                subject_avg_power = np.mean(all_powers, axis=0)
                
                # Create averaged DataFrame for this subject
                df_subject = pd.DataFrame({
                    'frequency': common_freqs,
                    'mean_power': subject_avg_power,
                    'std_power': np.zeros_like(subject_avg_power)  # Not meaningful after averaging
                })
                
                subject_data[subject] = df_subject
                all_spectral_data.append(df_subject)
                print(f"    ✓ Averaged {len(channel_data)} channels for {subject}")
                
            except Exception as e:
                print(f"  ✗ Error averaging channels for subject {subject}: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            # Load single channel data and build subject directory path
            spectral_pattern = f"{subject_dir}/{subject}_{target_channel}_output/{subject}_{target_channel}_spectral_power.csv"
            spectral_file = Path(spectral_pattern)
            if not spectral_file.exists():
                print(f"Warning: No spectral data found for subject {subject} at {spectral_file}")
                continue
                
            try:
                df = pd.read_csv(spectral_file)
                subject_data[subject] = df
                all_spectral_data.append(df)
            except Exception as e:
                print(f"  ✗ Error loading data for subject {subject}: {e}")
    
    if not all_spectral_data:
        print("No valid spectral data found!")
        return
    
    # Use the actual frequency values from the first subject instead of np.arange()
    # This avoids floating-point precision mismatches
    first_subject = list(subject_data.keys())[0]
    common_freqs = subject_data[first_subject]['frequency'].values
    
    # Interpolate all subjects to common frequency grid
    # (Note: if all subjects have identical frequencies, interpolation will just copy values)
    interpolated_powers = []
    for subject, df in subject_data.items():
        # Check if frequencies match exactly (no interpolation needed)
        if np.array_equal(df['frequency'].values, common_freqs):
            interpolated_power = df['mean_power'].values
        else:
            interp_func = interp1d(df['frequency'], df['mean_power'], 
                                  bounds_error=False, fill_value='extrapolate')
            interpolated_power = interp_func(common_freqs)
        interpolated_powers.append(interpolated_power)
    
    # Convert to numpy array and compute group statistics
    power_matrix = np.array(interpolated_powers)
    # Use nanmean/nanstd to ignore NaN values from subjects with missing data
    group_mean = np.nanmean(power_matrix, axis=0)
    group_std = np.nanstd(power_matrix, axis=0)
    
    # Apply smoothing
    if smoothing_window > 1:
        group_mean_smooth = uniform_filter1d(group_mean, size=smoothing_window)
        group_std_smooth = uniform_filter1d(group_std, size=smoothing_window)
        smooth_label = f" (smoothed, window={smoothing_window})"
    else:
        group_mean_smooth = group_mean
        group_std_smooth = group_std
        smooth_label = ""
    
    plt.figure(figsize=(12, 8))
    
    # Main line: group mean
    plt.plot(common_freqs, group_mean_smooth, 'b-', linewidth=2.5, label=f'Group Mean{smooth_label}')
    
    # Standard deviation bands with dotted lines
    plt.plot(common_freqs, group_mean_smooth + group_std_smooth, 'b:', linewidth=1.5, alpha=0.7, label='+1 SD')
    plt.plot(common_freqs, group_mean_smooth - group_std_smooth, 'b:', linewidth=1.5, alpha=0.7, label='-1 SD')
    
    # Shaded area for standard deviation
    plt.fill_between(common_freqs, 
                     group_mean_smooth - group_std_smooth,
                     group_mean_smooth + group_std_smooth,
                     alpha=0.2, color='blue', label='±1 SD region')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power (AU)', fontsize=12)
    
    if aggregate_channels:
        title = f'Group-Level Infraslow Fluctuations of Sigma Power (ISFS)\n' \
                f'Mean across {len(subjects)} subjects, averaged across all electrodes'
        filename = f"group_ISFS_spectral_power_all_channels.png"
    else:
        title = f'Group-Level Infraslow Fluctuations of Sigma Power (ISFS)\n' \
                f'Mean across {len(subjects)} subjects, channel {target_channel}'
        filename = f"group_ISFS_spectral_power_{target_channel}.png"
    
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Group spectral plot saved to: {plot_path}")
    print(f"✓ Analyzed {len(subjects)} subjects")
    print(f"✓ Frequency range: {common_freqs[0]:.4f} - {common_freqs[-1]:.4f} Hz")
    print(f"✓ Smoothing applied: {smoothing_window}-point window" if smoothing_window > 1 else "✓ No smoothing applied")


def collect_peak_frequency_data(subjects, results_dir):
    """
    Collect peak frequency data from all subjects and channels in the results directory.
    
    Returns:
    --------
    pd.DataFrame with columns: subject, channel, peak_frequency
    """
    results_dir = Path(results_dir)
    all_data = []
    for subject in subjects:
        summary_file = results_dir / subject / f"{subject}_all_channels_summary.csv"
        if not summary_file.exists():
            print(f"  ✗ Missing summary file: {summary_file}")
            continue
        
        try:
            df = pd.read_csv(summary_file, comment='#')
            if 'peak_frequency' not in df.columns or 'channel' not in df.columns:
                print(f"  ✗ Missing required columns in {summary_file.name}")
                continue
            
            # Filter out rows where peak_frequency is empty/NaN (failed channels)
            valid_data = df[df['peak_frequency'].notna()].copy()
            if len(valid_data) == 0:
                print(f"  ✗ No valid peak_frequency data in {summary_file.name}")
                continue
            
            # Add subject column and collect data
            valid_data['subject'] = subject
            all_data.append(valid_data[['subject', 'channel', 'peak_frequency']])
            
        except Exception as e:
            print(f"  ✗ Error reading {summary_file.name}: {e}")
    
    if not all_data:
        print("✗ No peak frequency data collected!")
        return None
    
    result_df = pd.concat(all_data, ignore_index=True)
    print(f"✓ Collected {len(result_df)} channel measurements from {result_df['subject'].nunique()} subjects")
    return result_df


def plot_raw_channel_peak_frequency_violin(subjects, dir_path):
    """
    Create half-violin plot for peak frequency across all raw channels (no averaging).
    Each data point represents one individual channel from any subject.
    Shows the distribution of peak frequencies at the raw channel level across the entire dataset.
    """
    df = collect_peak_frequency_data(subjects, dir_path)
    if df is None or len(df) == 0:
        return
    
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    data = df['peak_frequency'].dropna()
    if len(data) == 0:
        print("✗ No valid peak frequency data to plot")
        return
    
    mean_val = data.mean()
    std_val = data.std()
    median_val = data.median()
    
    print(f"\n{'='*60}")
    print(f"Peak Frequency Statistics (Raw Channels):")
    print(f"  N = {len(data)} channels")
    print(f"  Mean = {mean_val:.4f} Hz")
    print(f"  STD = {std_val:.4f} Hz")
    print(f"  Median = {median_val:.4f} Hz")
    print(f"  Min = {data.min():.4f} Hz")
    print(f"  Max = {data.max():.4f} Hz")
    print(f"{'='*60}")
    
    violin = sns.violinplot(y=data, ax=ax, color='#8dd3c7', inner=None, cut=0, linewidth=1.5)
    
    # Modify the violin to show only right half
    for collection in ax.collections:
        if hasattr(collection, 'get_paths'):
            paths = collection.get_paths()
            if len(paths) > 0:
                vertices = paths[0].vertices
                center_x = np.mean(vertices[:, 0])
                mask = vertices[:, 0] >= center_x
                vertices[~mask, 0] = center_x
                collection.set_alpha(0.7)
    
    # Add mean and median lines
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2.5, 
              alpha=0.9, label=f'Mean: {mean_val:.4f} Hz', zorder=5)
    ax.axhline(median_val, color='orange', linestyle='--', linewidth=2, 
              alpha=0.8, label=f'Median: {median_val:.4f} Hz')
    
    # Add boxplot overlay centered
    box_parts = ax.boxplot([data], positions=[0], widths=0.15, 
                            vert=True, patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            medianprops=dict(color='orange', linewidth=2))
    
    # Add individual channel dots with jitter
    np.random.seed(42)
    x_jitter = np.random.normal(0, 0.02, size=len(data))
    ax.scatter(x_jitter, data, alpha=0.4, color='darkblue', s=30, zorder=3, label='Raw Channels')
    
    ax.set_title(f'Peak Frequency Distribution - Raw Channels\nAll Channels Pooled Across {df["subject"].nunique()} Subjects (N={len(data)})', 
                fontsize=16, fontweight='bold')
    ax.set_ylabel('Peak Frequency (Hz)', fontsize=14)
    ax.set_xlim(-0.4, 0.4)
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = f'N = {len(data)}\nSubjects = {df["subject"].nunique()}\nMean = {mean_val:.4f} Hz\nSTD = {std_val:.4f} Hz\nMedian = {median_val:.4f} Hz'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, linewidth=2), 
           fontsize=11, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plot_path = dir_path / "peak_frequency_raw_channels_violin.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Peak frequency raw channels violin plot saved to: {plot_path}")


def main():
    """Run distribution analysis across all subjects."""
    subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    if not subjects:
        print("No subjects found!")
        return
    
    output_dir = Path("new_iso_results")
    output_dir.mkdir(exist_ok=True)
    subjects = [sub for sub in subjects if (output_dir / sub).exists() and sub != "dashboards"]
    subjects, _ = filter_subjects_by_detection_rate(subjects, dir_path=output_dir)
    if not subjects:
        print("No subjects meet the detection rate criteria!")
        return
    
    # 1. Group-level violin: each subject averaged across all channels (subjects as dots)
    plot_group_average_violin(subjects, output_dir)
    
    # 2. Per-subject violin: distribution across all channels within each subject (channels as dots)
    for sub in subjects:
        plot_subject_all_channels_violin(sub, output_dir)
    
    # 3. Single-channel violin: one channel across all subjects (subjects as dots)
    target_channel = "E101"
    df = collect_all_data(subjects, target_channel, output_dir)
    if df is not None:
        plot_single_channel_violin(df, output_dir, target_channel)
    
    # 4. Peak channel-level violin: raw channels pooled across all subjects (channels as dots, no averaging)
    plot_raw_channel_peak_frequency_violin(subjects, output_dir)
    
    # 5. Group spectral power: averaged across all channels and subjects
    plot_group_spectral_power(subjects, output_dir, target_channel=None, smoothing_window=5, aggregate_channels=True)
    
    # 6. Single-channel spectral power: one channel averaged across subjects
    plot_group_spectral_power(subjects, output_dir, target_channel="VREF", smoothing_window=5, aggregate_channels=False)
    
    # 7. Group topographies: spatial maps averaged across all subjects
    plot_topographies(subjects, normalize=True, subject_dir_suffix=None, output_dir=output_dir, group="new_iso_results")
    
    # 8. Per-subject topographies: spatial maps for each individual subject
    for sub in subjects:
        sub_output_dir = Path(f"new_iso_results/{sub}/")
        sub_output_dir.mkdir(exist_ok=True, parents=True)
        plot_topographies([sub], normalize=False, subject_dir_suffix=None, output_dir=sub_output_dir, group="new_iso_results")

if __name__ == "__main__":
    main()