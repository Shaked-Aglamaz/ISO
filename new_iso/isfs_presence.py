import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import simpson # Modern replacement for trapz

def gaussian_func(x, a, b, c):
    """Formula for 'gauss1' in MATLAB: a*exp(-((x-b)/c)^2)"""
    return a * np.exp(-((x - b) / c)**2)

def extract_isfs_parameters(power_timecourse, bout_locations, sampling_rate):
    """
    Calculates PF, BW, and AUC for the 'relative' condition.
    """
    num_bouts = bout_locations.shape[1]
    # 251 points corresponds to roughly 0.5Hz in your linspace
    power_fft_matrix = np.zeros((num_bouts, 251))
    
    # 1. Setup Frequency Axis
    # linspace(0, sf/2, 50000)
    freq_axis_full = np.linspace(0, sampling_rate / 2, 50000)
    idx_01hz = np.argmin(np.abs(freq_axis_full - 0.1))
    
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
        
        # Outlier removal (0.000 to 0.006 Hz)
        if interpolated_power[0] == np.max(interpolated_power[:idx_01hz]):
            interpolated_power[0:4] = np.nan
            
        power_fft_matrix[i, :] = interpolated_power

    # 3. Calculate Relative Power
    # Take only up to 0.1 Hz for fitting
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
    shifted_spectral_power = mean_spectral_power - shift_val

    # 5. Gaussian Fitting (The 'fit' equivalent)
    # Initial guess: [amplitude, mean, sigma]
    p0 = [np.nanmax(shifted_spectral_power), 0.02, 0.01] 
    
    try:
        # curve_fit handles the 'gauss1' logic
        popt, _ = curve_fit(gaussian_func, frequencies, shifted_spectral_power, p0=p0)
        peak_power_fit, peak_freq_fit, sigma_fit = popt
        
        fit_curve = gaussian_func(frequencies, *popt)
        threshold = np.nanstd(shifted_spectral_power) * 1.5
        freq_range = [0.0075, 0.04]
        
        # Validation
        if peak_power_fit >= threshold and freq_range[0] <= peak_freq_fit <= freq_range[1]:
            # Bandwidth (1 standard deviation in MATLAB 'gauss1' is 'c')
            # Note: MATLAB's 'c' is the denominator in the exp. Python's sigma_fit is same.
            upper_limit = peak_freq_fit + sigma_fit
            lower_limit = peak_freq_fit - sigma_fit
            bandwidth = upper_limit - lower_limit
            
            # Find actual peaks in the fitted curve
            peaks, _ = find_peaks(fit_curve)
            
            if len(peaks) > 0:
                # Find the highest peak
                best_peak_idx = peaks[np.argmax(fit_curve[peaks])]
                actual_pf = frequencies[best_peak_idx]
                
                # Area Under Curve (AUC)
                lb = max(0.0075, lower_limit)
                auc_mask = (frequencies >= lb) & (frequencies <= upper_limit)
                # Use Simpson's rule or Trapezoidal
                auc_val = np.trapz(fit_curve[auc_mask])
                
                return actual_pf, bandwidth, auc_val, peak_power_fit
                
    except Exception as e:
        print(f"Fitting failed: {e}")

    return np.nan, np.nan, np.nan, np.nan