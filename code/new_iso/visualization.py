"""
Visualization functions for ISFS analysis.
Adapted from step3_spectral.py for the new_iso pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson


def gaussian(x, amplitude, mu, sigma):
    """Gaussian function for fitting."""
    return amplitude * np.exp(-((x - mu) / sigma)**2)


def plot_bout_fft(frequencies, power_spectrum, bout_id, subject_id, channel_name, 
                  output_dir='.'):
    """
    Plot FFT power spectrum for a single bout.
    
    Parameters:
    -----------
    frequencies : array
        Frequency axis (Hz)
    power_spectrum : array
        Power values for this bout
    bout_id : int
        Bout number
    subject_id : str
        Subject identifier
    channel_name : str
        Channel name
    output_dir : str or Path
        Directory to save plot
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(frequencies, power_spectrum, color='#2E86AB', linewidth=2, 
             label=f'Bout {bout_id}')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Relative Power (AU)', fontsize=12)
    plt.title(f'FFT Power Spectrum - Bout {bout_id}\n'
              f'Subject: {subject_id}, Channel: {channel_name}', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"{subject_id}_{channel_name}_bout_{bout_id:02d}_fft.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def plot_mean_spectrum_with_fit(plot_data, subject_id='', channel_name='', output_dir='.'):
    """
    Plot mean spectral power across all bouts with optional Gaussian fit.
    
    Parameters:
    -----------
    plot_data : dict
        Dictionary containing all plot data including:
        - 'frequencies': Frequency axis (Hz)
        - 'mean_power': Mean power across all bouts (baseline-corrected)
        - 'fitted_params': tuple (amplitude, mu, sigma) or None if fit failed
        - 'threshold': ISFS detection threshold or None
        - 'auc': Pre-calculated AUC value
    subject_id : str
        Subject identifier
    channel_name : str
        Channel name
    output_dir : str or Path
        Directory to save plot
    """
    # Extract data from plot_data dictionary
    frequencies = plot_data['frequencies']
    mean_power = plot_data['mean_power']
    fitted_params = plot_data['fitted_params']
    threshold = plot_data['threshold']
    auc = plot_data['auc']
    
    plt.figure(figsize=(10, 7))
    
    # 1. Plot mean power spectrum
    plt.plot(frequencies, mean_power, color='#2E86AB', linewidth=2.5, 
             label='Mean Relative Power (Baseline Corrected)', zorder=2)
    
    # 2. Add Gaussian fit if available
    if fitted_params is not None:
        amplitude, mu, sigma = fitted_params
        
        # Generate smooth Gaussian curve
        x_fit = np.linspace(frequencies[0], frequencies[-1], 500)
        y_fit = gaussian(x_fit, amplitude, mu, sigma)
        
        # Calculate bandwidth boundaries
        x1 = mu - sigma
        x2 = mu + sigma
        bandwidth_height = gaussian(x1, amplitude, mu, sigma)
        
        # Plot Gaussian components
        plt.plot(x_fit, y_fit, '#A23B72', linewidth=2.5, alpha=0.9, 
                label=f'Gaussian Fit (μ={mu:.4f} Hz, σ={sigma:.4f})', zorder=3)
        plt.plot(mu, amplitude, 'o', color='#A23B72', markersize=10, 
                label=f'Peak: {amplitude:.3f} AU', zorder=4)
        
        # Bandwidth line
        plt.hlines(bandwidth_height, x1, x2, colors="#F18F01", linewidth=3, 
                  label=f'Bandwidth: {2*sigma:.4f} Hz', zorder=3)
        
        # Fill ±1σ area with pre-calculated AUC
        mask = (x_fit >= x1) & (x_fit <= x2)
        plt.fill_between(x_fit[mask], y_fit[mask], color='#A23B72', alpha=0.2, 
                        label=f'±1σ Area (AUC={auc:.3f})', zorder=1)
    
    # 3. Add threshold line if provided
    if threshold is not None:
        plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'ISFS Threshold: {threshold:.3f} AU', zorder=1)
    
    # 4. Formatting
    plt.xlabel('Frequency (Hz)', fontsize=13)
    plt.ylabel('Relative Power (AU)', fontsize=13)
    
    fit_status = 'with Gaussian Fit' if fitted_params is not None else '(No Fit)'
    plt.title(f'Mean Spectral Power {fit_status}\n'
              f'Subject: {subject_id}, Channel: {channel_name}', fontsize=14, pad=15)
    
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"{subject_id}_{channel_name}_mean_spectrum.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def plot_all_bouts_overlay(frequencies, all_bout_spectra, mean_power, 
                           subject_id='', channel_name='', output_dir='.'):
    """
    Plot all individual bout spectra overlaid with the mean.
    
    Note: Individual bouts are shown WITHOUT baseline correction to show raw variability.
    The mean is also shown WITHOUT baseline correction to match the individual bouts.
    
    Parameters:
    -----------
    frequencies : array
        Frequency axis (Hz)
    all_bout_spectra : list of arrays
        List containing power spectrum for each bout (without baseline correction)
    mean_power : array
        Mean power across all bouts (without baseline correction)
    subject_id : str
        Subject identifier
    channel_name : str
        Channel name
    output_dir : str or Path
        Directory to save plot
    """
    plt.figure(figsize=(10, 7))
    
    # Plot individual bouts in light gray
    for i, bout_spectrum in enumerate(all_bout_spectra):
        plt.plot(frequencies, bout_spectrum, color='lightgray', linewidth=1, 
                alpha=0.5, zorder=1)
    
    # Plot mean in bold
    plt.plot(frequencies, mean_power, color='#2E86AB', linewidth=3, 
             label=f'Mean (n={len(all_bout_spectra)} bouts)', zorder=2)
    
    # Highlight baseline correction region (0.06 - 0.102 Hz)
    baseline_mask = (frequencies > 0.06) & (frequencies < 0.102)
    if np.any(baseline_mask):
        baseline_mean = np.nanmean(mean_power[baseline_mask])
        plt.axvspan(0.06, 0.102, alpha=0.2, color='orange', 
                   label=f'Baseline region (mean={baseline_mean:.3f})', zorder=0)
        # Add horizontal line showing the baseline value
        plt.axhline(baseline_mean, color='orange', linestyle='--', linewidth=2, 
                   alpha=0.7, zorder=1)
    
    plt.xlabel('Frequency (Hz)', fontsize=13)
    plt.ylabel('Relative Power (AU)', fontsize=13)
    plt.title(f'All Bout Spectra Overlay (No Baseline Correction)\n'
              f'Subject: {subject_id}, Channel: {channel_name}', fontsize=14, pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"{subject_id}_{channel_name}_all_bouts_overlay.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file
