import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def gaussian_func(x, a, b, c):
    """Formula for 'gauss1' in MATLAB: a*exp(-((x-b)/c)^2)"""
    return a * np.exp(-((x - b) / c)**2)


def fit_gaussian(frequencies, power, plot_data):
    # Filter out NaN values from outlier removal
    valid_frequencies = frequencies[~np.isnan(power)]
    valid_power = power[~np.isnan(power)]

    if len(valid_frequencies) < 10:  # Need minimum points for fitting
        plot_data['failure_reason'] = f'Insufficient valid points for fitting ({len(valid_frequencies)} < 10)'
        return (np.nan, np.nan, np.nan, np.nan), plot_data
    
    # Initial guess: [amplitude, mean, sigma]
    p0 = [np.nanmax(valid_power), 0.02, 0.01] 
    popt, _ = curve_fit(gaussian_func, valid_frequencies, valid_power, p0=p0)
    peak_power_fit, peak_freq_fit, sigma_fit = popt
    
    # Generate fit curve on FULL frequency axis (will have values even where original had NaN)
    fit_curve = gaussian_func(frequencies, *popt)
    threshold = np.nanstd(power) * 1.5
    freq_range = [0.0075, 0.04]
    
    # Check if peak power is above threshold
    if peak_power_fit < threshold:
        plot_data['failure_reason'] = f'Peak power below threshold (power={peak_power_fit:.6f}, threshold={threshold:.6f})'
        return (np.nan, np.nan, np.nan, np.nan), plot_data
    
    # Check if peak frequency is within valid range
    if not (freq_range[0] <= peak_freq_fit <= freq_range[1]):
        plot_data['failure_reason'] = f'Peak frequency outside valid range (freq={peak_freq_fit:.6f} Hz, valid range=[{freq_range[0]}, {freq_range[1]}])'
        return (np.nan, np.nan, np.nan, np.nan), plot_data
    
    # Calculate bandwidth
    upper_limit = peak_freq_fit + sigma_fit
    lower_limit = peak_freq_fit - sigma_fit
    bandwidth = upper_limit - lower_limit
    
    # Find actual peaks in the fitted curve
    peaks, _ = find_peaks(fit_curve)
    if len(peaks) == 0:
        plot_data['failure_reason'] = 'No peaks found in fitted curve'
        return (np.nan, np.nan, np.nan, np.nan), plot_data
    
    # Find the highest peak
    best_peak_idx = peaks[np.argmax(fit_curve[peaks])]
    actual_pf = frequencies[best_peak_idx]
    
    auc_mask = (frequencies >= lower_limit) & (frequencies <= upper_limit)
    auc_val = np.trapz(fit_curve[auc_mask])
    
    # Final validation: AUC should be positive
    if auc_val <= 0 or np.isclose(auc_val, 0, atol=1e-8):
        plot_data['failure_reason'] = f'Invalid AUC value ({auc_val:.10f})'
        return (np.nan, np.nan, np.nan, np.nan), plot_data
    
    plot_data['fitted_params'] = (peak_power_fit, peak_freq_fit, sigma_fit)
    plot_data['threshold'] = threshold
    plot_data['auc'] = auc_val
    return (actual_pf, bandwidth, auc_val, peak_power_fit), plot_data
    

def extract_isfs_parameters(power_timecourse, bout_locations, sampling_rate):
    """
    Calculates PF, BW, AUC, PP for the 'relative' condition.
    
    Parameters:
    -----------
    power_timecourse : array
        Amplitude envelope timecourse
    bout_locations : array (2, n_bouts)
        Start and end indices for each bout
    sampling_rate : float
        Sampling frequency
    
    Returns:
    --------
    (pf, bw, auc, pp), plot_data_dict
        Tuple of (parameters, plot_data) where:
        - pf: peak frequency (Hz)
        - bw: bandwidth (Hz)
        - auc: area under curve
        - pp: peak power
        - plot_data_dict: dictionary containing visualization data
    """
    num_bouts = bout_locations.shape[1]
    # 251 points corresponds to roughly 0.5Hz in your linspace
    power_fft_matrix = np.zeros((num_bouts, 251))
    
    # 1. Setup Frequency Axis, linspace(0, sf/2, 50000)
    freq_axis_full = np.linspace(0, sampling_rate / 2, 50000)
    idx_01hz = np.argmin(np.abs(freq_axis_full - 0.1))
    n_bouts_with_outliers = 0
    
    # 2. Process each bout
    for i in range(num_bouts):
        start, end = bout_locations[:, i].astype(int)
        bout_data = power_timecourse[start:end+1]
        
        # FFT logic
        n = len(bout_data)
        freq_bout = np.linspace(0, sampling_rate / 2, (n // 2) + 1)
        
        # Shifted FFT (DC removal)
        bout_shifted = bout_data - np.nanmean(bout_data)
        fft_res = (2 * np.abs(fft(bout_shifted)) / n)**2
        power_bout = fft_res[:len(freq_bout)]
        
        # Interpolate to the master frequency axis
        interp_func = interp1d(freq_bout, power_bout, bounds_error=False, fill_value=0)
        interpolated_power = interp_func(freq_axis_full[:251])
        
        # If any of the first 4 points (0-0.006 Hz) is the max, remove all 4
        if np.nanargmax(interpolated_power[:idx_01hz]) < 4:
            interpolated_power[0:4] = np.nan
            n_bouts_with_outliers += 1
            
        power_fft_matrix[i, :] = interpolated_power

    # 3. Calculate Relative Power, Take only up to 0.1 Hz for fitting
    relative_fft = power_fft_matrix[:, :idx_01hz]
    # Normalize by mean of each bout
    bout_means = np.nanmean(np.abs(relative_fft), axis=1, keepdims=True)
    normalized_fft = relative_fft / bout_means
    
    # Mean across bouts
    mean_spectral_power = np.nanmean(normalized_fft, axis=0)
    frequencies = freq_axis_full[:idx_01hz]
    
    # 4. Shift logic (Baseline subtraction based on 0.06 - 0.102 Hz)
    baseline_mask = (frequencies > 0.06) & (frequencies < 0.102)
    shift_val = np.nanmean(mean_spectral_power[baseline_mask])
    shifted_power = mean_spectral_power - shift_val
    plot_data = {
        'frequencies': frequencies,
        'mean_power': shifted_power,  # Mean with baseline correction
        'fitted_params': None,
        'threshold': None,
        'auc': None,  # Area under curve (computed on successful fit)
        'all_bout_spectra': [normalized_fft[i, :] for i in range(num_bouts)],
        'mean_power_no_baseline': mean_spectral_power,  # Mean without baseline correction
        'freq_axis_full': freq_axis_full[:251],
        'power_fft_matrix': power_fft_matrix[:, :idx_01hz],
        'n_bouts_with_outliers': n_bouts_with_outliers,
        'num_bouts': num_bouts,
        'failure_reason': None  # Tracks why fit failed (None if successful)
    }

    # 5. Gaussian Fitting
    try:
        return fit_gaussian(frequencies, shifted_power, plot_data)
        
    except Exception as e:
        plot_data['failure_reason'] = f'Fitting exception: {str(e)}'
        return (np.nan, np.nan, np.nan, np.nan), plot_data