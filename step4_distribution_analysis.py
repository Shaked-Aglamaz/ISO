import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from pathlib import Path
from scipy.interpolate import interp1d
from config import BASE_DIR
from step3_spectral import get_all_subjects
from scipy.ndimage import uniform_filter1d

def load_subject_data(subject_id, dir=None):
    """Load channel summary data for a specific subject."""
    dir = dir if dir is not None else subject_id
    summary_file = f"{dir}/{subject_id}_all_channels_summary.csv"
    
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


def collect_all_data(subjects, target_channel):
    """Collect and combine spectral data from all subjects for target channel."""
    all_data = []
    for subject in subjects:
        subject_data = load_subject_data(subject)
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


def calculate_optimal_bins(data):
    """Calculate optimal histogram bins using Freedman-Diaconis rule."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    
    max_reasonable_bins = min(int(np.sqrt(len(data))) + 5, len(data) // 2)
    n_bins = max(8, min(max_reasonable_bins, int((data.max() - data.min()) / bin_width))) if bin_width > 0 else max_reasonable_bins
    
    return n_bins


def _plot_single_metric_distribution(data, ax, info, include_bins_in_stats=True):
    """Plot single metric distribution on axis."""
    n_bins = calculate_optimal_bins(data)
    sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins, stat='count')
    
    mean_val = data.mean()
    std_val = data.std()
    median_val = data.median()
    
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.3f}')
    
    ax.set_xlabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    stats_base = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
    stats_text = f'{stats_base}\nBins = {n_bins}' if include_bins_in_stats else stats_base
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)


def plot_distributions(df, output_dir, channel_name):
    """Create separate distribution plots for each spectral metric."""
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    for metric, info in metrics.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        data = df[metric].dropna()
        
        _plot_single_metric_distribution(data, ax, info, include_bins_in_stats=True)
        ax.set_title(f'{info["name"]} Distribution - {channel_name} Channel\n(N={len(data)} subjects)', fontsize=14)
        
        plt.tight_layout()
        plot_path = output_dir / f"{channel_name}_{metric}_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"\nAll {len(metrics)} distribution plots saved to: {output_dir}/")


def plot_combined_distributions(df, output_dir, channel_name):
    """Create combined plot with all distributions in subplots."""
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Combined Distribution Analysis - {channel_name} Channel\n(N={len(df)} subjects)', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = df[metric].dropna()
        
        _plot_single_metric_distribution(data, ax, info, include_bins_in_stats=False)
        ax.set_title(f'{info["name"]} Distribution')
    
    plt.tight_layout()
    
    plot_path = output_dir / f"{channel_name}_combined_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined distribution plot saved to: {plot_path}")


def plot_boxplots_with_dots(df, output_dir, channel_name):
    """Create boxplots with individual subject dots overlaid."""
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'Spectral Parameters Distribution - {channel_name} Channel\n(N={len(df)} subjects)', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = df[metric].dropna()
        
        # Create boxplot
        box_plot = ax.boxplot(data, patch_artist=True, widths=0.6)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        # Add individual subject dots
        np.random.seed(42)  # For reproducible jitter
        x_jitter = np.random.normal(1, 0.04, size=len(data))  # Small jitter around x=1
        ax.scatter(x_jitter, data, alpha=0.6, color='red', s=30, zorder=3, label='Subjects')
        
        # Formatting
        ax.set_title(f'{info["name"]}', fontsize=14)
        ax.set_ylabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_xticklabels([''])  # Remove x-axis labels since we only have one box
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    boxplot_path = output_dir / f"{channel_name}_boxplots_with_dots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot with subject dots saved to: {boxplot_path}")


def load_and_combine_subject_data(subjects, subject_dir=None):
    """Load and combine data from all subjects."""
    all_subject_data = []
    for subject in subjects:
        subject_data = load_subject_data(subject, dir=subject_dir)
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


def create_mne_info(channel_names, montage):
    """Create MNE Info object with electrode positions."""
    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    return info


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


def remove_outliers_3std(values, metric_name=""):
    """Remove outliers beyond ±3 standard deviations."""
    original_valid = np.sum(~np.isnan(values))
    
    if original_valid == 0:
        print(f"  {metric_name}: No valid values to process")
        return values
    
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    
    if std_val == 0:
        print(f"  {metric_name}: No variation in data (STD = 0)")
        return values
    
    # Create mask for values within ±3 STD
    outlier_mask = np.abs(values - mean_val) > 3 * std_val
    outliers_count = np.sum(outlier_mask)
    
    # Replace outliers with NaN
    cleaned_values = values.copy()
    cleaned_values[outlier_mask] = np.nan
    
    final_valid = np.sum(~np.isnan(cleaned_values))
    
    if outliers_count > 0:
        print(f"  {metric_name}: Removed {outliers_count} outliers, {final_valid}/{original_valid} channels remaining")
    
    return cleaned_values


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
            cbar_label = f'{metric_info["unit"]} (unnormalized)'
        else:
            plot_values = cleaned_values / mean_value
            cbar_label = 'Normalized Value\n(/ mean)'
        title = f'{metric_info["name"]}\n(Mean = {mean_value:.3f} {metric_info["unit"]})'
    else:
        plot_values = cleaned_values
        title = f'{metric_info["name"]}\n(Mean = {mean_value:.3f} {metric_info["unit"]})'
        cbar_label = f'{metric_info["unit"]}'
    
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


def create_and_save_topography_figure(subjects, metrics, info, channel_averages, channel_names, normalize=True, subject_dir=None):
    """Create figure with all three topographical plots and save to file."""
    # Create figure with subplots, but reserve space for colorbars
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Handle single subject vs multiple subjects
    if len(subjects) == 1:
        subject_name = subjects[0]
        norm_text = "(Normalized by mean across channels)" if normalize else "(Raw values)"
        title = f'Topographical Distribution - Subject {subject_name}\n{norm_text}'
        # Use provided subject_dir if available, otherwise use subject name
        output_dir = Path(subject_dir) if subject_dir is not None else Path(subject_name)
        suffix = "normalized" if normalize else "raw"
        filename = f"{subject_name}_topographies_{suffix}.png"
    else:
        norm_text = "(Normalized by mean across channels)" if normalize else "(Raw values)"
        title = f'Topographical Distribution - Average Across {len(subjects)} Subjects\n{norm_text}'
        output_dir = Path("distribution_analysis")
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
    
    # Save plot
    output_dir.mkdir(exist_ok=True)
    topo_path = output_dir / filename
    plt.savefig(topo_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Topographical plots saved to: {topo_path}")


def plot_topographies(subjects, normalize=True, subject_dir=None):
    """Create topographical plots for the three spectral parameters with ISFS failure handling."""
    print(f"\n{'='*60}")
    print(f"TOPOGRAPHY ANALYSIS - {'Subject ' + subjects[0] if len(subjects) == 1 else f'{len(subjects)} Subjects'}")
    print(f"{'='*60}")
    
    combined_data = load_and_combine_subject_data(subjects, subject_dir=subject_dir)
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
    
    # Print electrode montage summary
    total_electrodes = len(channel_averages)
    valid_positions = len(channel_names)
    print(f"\n📍 Electrode Positioning Summary:")
    print(f"   Channels after filtering: {total_electrodes}")
    print(f"   Channels with known EGI-256 positions: {valid_positions}")
    print(f"   Excluded (no position): {total_electrodes - valid_positions}")
    
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'}, 
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    print(f"\n🎨 Creating topographical plots...")
    print(f"   Normalization: {'Enabled' if normalize else 'Disabled'}")
    
    create_and_save_topography_figure(subjects, metrics, info, channel_averages_filtered, channel_names, normalize=normalize, subject_dir=subject_dir)
    
    print(f"✅ Topography analysis completed")


def print_summary_statistics(df, channel_name):
    """
    Print comprehensive summary statistics for all spectral metrics.
    Includes count, mean, std, quartiles, min/max for each metric.
    """
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS - {channel_name} Channel")
    print(f"{'='*60}")
    
    metrics = ['avg_peak_frequency', 'avg_bandwidth', 'avg_auc']
    
    for metric in metrics:
        data = df[metric].dropna()
        print(f"\n{metric.replace('avg_', '').replace('_', ' ').title()}:")
        print(f"  Min: {data.min():.4f}")
        print(f"  25%: {data.quantile(0.25):.4f}")
        print(f"  Median: {data.median():.4f}")
        print(f"  75%: {data.quantile(0.75):.4f}")
        print(f"  Max: {data.max():.4f}")


def plot_subject_all_channels_distributions(subjects, dir=None):
    """Plot distributions for all channels within each subject."""
    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    for subject in subjects:
        print(f"\nProcessing distributions for subject {subject}...")
        
        # Load subject data
        subject_data = load_subject_data(subject, dir)
        if subject_data is None:
            print(f"  ✗ No data found for subject {subject}")
            continue
        
        eeg_data = subject_data.copy()
        
        if len(eeg_data) == 0:
            print(f"  ✗ No EEG channels found for subject {subject}")
            continue
        
        # Create figure with 3 subplots for the 3 metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'All Channels Distribution - Subject {subject}', fontsize=16)
        
        for i, (metric, info) in enumerate(metrics.items()):
            ax = axes[i]
            data = eeg_data[metric].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{info["name"]}')
                continue
            
            # Calculate optimal number of bins
            n_bins = calculate_optimal_bins(data)
            
            # Create histogram with KDE
            sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins, stat='count')
            
            # Calculate statistics
            mean_val = data.mean()
            std_val = data.std()
            median_val = data.median()
            
            # Add statistical lines
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Mean')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Median')
            
            # Formatting
            ax.set_xlabel(f'{info["name"]} ({info["unit"]})', fontsize=14)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if dir is None:
            dir = Path(subject)
        dir.mkdir(exist_ok=True)
        plot_path = dir / f"{subject}_all_channels_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ All channels distribution plot saved to: {plot_path}")
    
    print(f"\nCompleted all-channels distribution analysis for {len(subjects)} subjects")


def load_and_process_all_channels_data(subjects):
    """Load, filter, and average all channels data across subjects."""
    print(f"\nProcessing data from all channels across {len(subjects)} subjects...")
    
    # Collect all data
    all_data = []
    total_channels = 0
    
    for subject in subjects:
        subject_data = load_subject_data(subject)
        if subject_data is None:
            continue
        
        eeg_data = subject_data.copy()
        
        if len(eeg_data) > 0:
            eeg_data['subject'] = subject  # Add subject column
            all_data.append(eeg_data)
            total_channels += len(eeg_data)
            print(f"  ✓ {subject}: {len(eeg_data)} channels")
    
    if not all_data:
        print("No valid data found for analysis")
        return None, 0, 0
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Average across channels within each subject
    subject_averages = combined_df.groupby('subject')[['avg_peak_frequency', 'avg_bandwidth', 'avg_auc']].mean().reset_index()
    
    print(f"  ✓ Processed {len(subject_averages)} subjects with {total_channels} total channels")
    
    return subject_averages, total_channels, len(subject_averages)


def plot_boxplots_all_channels_all_subjects(subjects, output_dir):
    """Create boxplots for all channels across all subjects."""
    # Use shared data processing function
    subject_averages, total_channels, valid_subjects = load_and_process_all_channels_data(subjects)
    
    if subject_averages is None:
        return

    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'All Channels Averaged Per Subject - Spectral Parameters Distribution\n'
                 f'{len(subjects)} subjects, {total_channels} total channels', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = subject_averages[metric].dropna()
        
        # Create boxplot
        box_plot = ax.boxplot(data, patch_artist=True, widths=0.6)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        # Add individual subject dots
        np.random.seed(42)  # For reproducible jitter
        x_jitter = np.random.normal(1, 0.04, size=len(data))  # Small jitter around x=1
        ax.scatter(x_jitter, data, alpha=0.6, color='red', s=30, zorder=3, label='Subjects')
        
        # Set x-axis limits based on data range with some padding
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        padding = data_range * 0.05  # 5% padding on each side
        ax.set_ylim(data_min - padding, data_max + padding)
        
        # Formatting
        ax.set_title(f'{info["name"]}', fontsize=14)
        ax.set_ylabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_xticklabels([''])  # Remove x-axis labels since we only have one box
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    boxplot_path = output_dir / "all_channels_all_subjects_boxplots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All channels boxplots saved to: {boxplot_path}")


def plot_combined_all_channels_distributions(subjects, output_dir):
    """Plot combined distributions across all channels and subjects."""
    # Use shared data processing function
    subject_averages, total_channels, valid_subjects = load_and_process_all_channels_data(subjects)
    
    if subject_averages is None:
        return

    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    
    # Create combined plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'All Channels Averaged Per Subject - Distribution\n'
                 f'{valid_subjects} subjects, {total_channels} total channels', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = subject_averages[metric].dropna()
        
        # Calculate optimal number of bins
        n_bins = calculate_optimal_bins(data)
        
        # Create histogram with KDE
        sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins, stat='count')
        
        # Calculate statistics
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        # Add statistical lines
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
                  label=f'Median: {median_val:.3f}')
        
        
        # Set x-axis limits based on data range with some padding
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        padding = data_range * 0.05  # 5% padding on each side
        ax.set_xlim(data_min - padding, data_max + padding)
        
        # Formatting
        ax.set_title(f'{info["name"]}', fontsize=14)
        ax.set_xlabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Save combined plot
    plot_path = output_dir / "all_channels_subject_avg_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All channels (subject averages) distribution plot saved to: {plot_path}")


def plot_group_spectral_power(subjects, output_dir, target_channel, smoothing_window=5, aggregate_channels=False):
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
        if aggregate_channels:
            # Load all channels for this subject and average them
            subject_dir = Path(subject)
            channel_data = []
            
            # Find all spectral power CSV files for this subject
            spectral_files = list(subject_dir.glob(f"{subject}_*_output/{subject}_*_spectral_power.csv"))
            
            for spectral_file in spectral_files:
                # Extract channel name from file path
                channel = spectral_file.stem.split('_')[1]  # e.g., EL3004_VREF_spectral_power -> VREF
                
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
                print(f"  ✓ Loaded and averaged {len(channel_data)} channels for subject {subject}")
                
            except Exception as e:
                print(f"  ✗ Error averaging channels for subject {subject}: {e}")
        
        else:
            # Load single channel data
            spectral_pattern = f"{subject}/{subject}_{target_channel}_output/{subject}_{target_channel}_spectral_power.csv"
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
    
    # Find common frequency range
    min_freq = max(df['frequency'].min() for df in all_spectral_data)
    max_freq = min(df['frequency'].max() for df in all_spectral_data)
    
    # Create common frequency grid
    freq_resolution = min(df['frequency'].diff().dropna().min() for df in all_spectral_data)
    common_freqs = np.arange(min_freq, max_freq, freq_resolution)
    
    # Interpolate all subjects to common frequency grid
    interpolated_powers = []
    for subject, df in subject_data.items():
        interp_func = interp1d(df['frequency'], df['mean_power'], 
                              bounds_error=False, fill_value='extrapolate')
        interpolated_power = interp_func(common_freqs)
        interpolated_powers.append(interpolated_power)
    
    # Convert to numpy array and compute group statistics
    power_matrix = np.array(interpolated_powers)
    group_mean = np.mean(power_matrix, axis=0)
    group_std = np.std(power_matrix, axis=0)
    
    # Apply smoothing
    if smoothing_window > 1:
        group_mean_smooth = uniform_filter1d(group_mean, size=smoothing_window)
        group_std_smooth = uniform_filter1d(group_std, size=smoothing_window)
        smooth_label = f" (smoothed, window={smoothing_window})"
    else:
        group_mean_smooth = group_mean
        group_std_smooth = group_std
        smooth_label = ""
    
    # Create the plot
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
    
    # Formatting
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
    
    # Save the plot
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Group spectral plot saved to: {plot_path}")
    print(f"✓ Analyzed {len(subjects)} subjects")
    print(f"✓ Frequency range: {common_freqs[0]:.4f} - {common_freqs[-1]:.4f} Hz")
    print(f"✓ Smoothing applied: {smoothing_window}-point window" if smoothing_window > 1 else "✓ No smoothing applied")


def plot_single_subject_all_channels_spectral_power(subject_id, subject_dir, smoothing_window=5):
    """
    Plot mean spectral power across all channels for a single subject.
    Shows mean ± SD across channels to visualize spatial variability.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., "RD43")
    subject_dir : str or Path
        Directory where subject data is stored and where plot will be saved
    smoothing_window : int
        Window size for smoothing (default: 5)
    """
    print(f"\n{'='*60}")
    print(f"Single Subject Spectral Power Analysis - {subject_id}")
    print(f"{'='*60}")
    
    subject_dir = Path(subject_dir)
    
    # Find all spectral power CSV files for this subject
    spectral_pattern = f"{subject_id}_*_output/{subject_id}_*_spectral_power.csv"
    spectral_files = list(subject_dir.glob(spectral_pattern))
    
    if not spectral_files:
        print(f"✗ No spectral power files found for subject {subject_id}")
        print(f"  Searched in: {subject_dir}/{spectral_pattern}")
        return
    
    print(f"✓ Found {len(spectral_files)} channel files")
    
    # Load all channel data
    channel_data = []
    channel_names = []
    
    for spectral_file in spectral_files:
        # Extract channel name from file path
        channel = spectral_file.stem.split('_')[1]  # e.g., RD43_E1_spectral_power -> E1
        
        try:
            df = pd.read_csv(spectral_file)
            channel_data.append(df)
            channel_names.append(channel)
        except Exception as e:
            print(f"  ✗ Error loading {spectral_file.name}: {e}")
    
    if not channel_data:
        print(f"✗ No valid channel data loaded")
        return
    
    print(f"✓ Successfully loaded {len(channel_data)} channels")
    
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
    
    # Convert to numpy array and compute statistics across channels
    power_matrix = np.array(all_powers)  # Shape: (n_channels, n_frequencies)
    channel_mean = np.mean(power_matrix, axis=0)
    channel_std = np.std(power_matrix, axis=0)
    
    # Apply smoothing
    if smoothing_window > 1:
        channel_mean_smooth = uniform_filter1d(channel_mean, size=smoothing_window)
        channel_std_smooth = uniform_filter1d(channel_std, size=smoothing_window)
        smooth_label = f" (smoothed, window={smoothing_window})"
    else:
        channel_mean_smooth = channel_mean
        channel_std_smooth = channel_std
        smooth_label = ""
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main line: mean across channels
    plt.plot(common_freqs, channel_mean_smooth, 'b-', linewidth=2.5, 
            label=f'Mean across channels{smooth_label}')
    
    # Standard deviation bands with dotted lines
    plt.plot(common_freqs, channel_mean_smooth + channel_std_smooth, 'b:', 
            linewidth=1.5, alpha=0.7, label='+1 SD')
    plt.plot(common_freqs, channel_mean_smooth - channel_std_smooth, 'b:', 
            linewidth=1.5, alpha=0.7, label='-1 SD')
    
    # Shaded area for standard deviation
    plt.fill_between(common_freqs, 
                     channel_mean_smooth - channel_std_smooth,
                     channel_mean_smooth + channel_std_smooth,
                     alpha=0.2, color='blue', label='±1 SD region (spatial variability)')
    
    # Find and mark peak frequency
    peak_idx = np.argmax(channel_mean_smooth)
    peak_freq = common_freqs[peak_idx]
    peak_power = channel_mean_smooth[peak_idx]
    
    # Add vertical dotted line at peak frequency
    plt.axvline(peak_freq, color='red', linestyle=':', linewidth=2, 
                label=f'Peak at {peak_freq:.4f} Hz')
    
    # Formatting
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Relative Power (AU)', fontsize=12)
    plt.title(f'Infraslow Fluctuations of Sigma Power (ISFS) - Subject {subject_id}\n'
              f'Mean across {len(channel_data)} channels', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot to subject directory
    plot_path = subject_dir / f"{subject_id}_all_channels_spectral_power.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Spectral power plot saved to: {plot_path}")
    print(f"✓ Analyzed {len(channel_data)} channels")
    print(f"✓ Frequency range: {common_freqs[0]:.4f} - {common_freqs[-1]:.4f} Hz")
    print(f"✓ Smoothing applied: {smoothing_window}-point window" if smoothing_window > 1 else "✓ No smoothing applied")
    print(f"{'='*60}\n")


def main():
    """Run distribution analysis across all subjects."""
    # subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    # if not subjects:
    #     print("No subjects found!")
    #     return
    subjects = ["RD43"]
    
    subject_dir = Path("RD43_MA_Hann_N2")
    
    # df = collect_all_data(subjects, target_channel)
    # if df is None:
    #     print("No data to analyze!")
    #     return
    
    # plot_distributions(df, output_dir, target_channel)
    # plot_combined_distributions(df, output_dir, target_channel)
    # plot_boxplots_with_dots(df, output_dir, target_channel)
    
    # Topographies with subject_dir
    # plot_topographies(subjects, normalize=True, subject_dir=subject_dir)
    # plot_topographies(subjects, normalize=False, subject_dir=subject_dir)
    
    # Individual subject all-channels distributions
    # plot_subject_all_channels_distributions(subjects, subject_dir)
    
    plot_single_subject_all_channels_spectral_power(subjects[0], subject_dir, smoothing_window=5)
    
    # Combined all-channels distribution across all subjects
    # plot_combined_all_channels_distributions(subjects, output_dir)
    
    # All channels boxplots
    # plot_boxplots_all_channels_all_subjects(subjects, output_dir)
    
    # Single channel plot
    # plot_group_spectral_power(subjects, output_dir, target_channel, aggregate_channels=False)
    
    # All channels aggregated plot
    # plot_group_spectral_power(subjects, output_dir, target_channel, smoothing_window=3, aggregate_channels=True)
    
    # print_summary_statistics(df, target_channel)
    
    # if len(subjects) > 1:
    #     data_path = output_dir / f"{target_channel}_combined_data.csv"
    #     df.to_csv(data_path, index=False)
    #     print(f"\nCombined dataset saved to: {data_path}")

if __name__ == "__main__":
    main()