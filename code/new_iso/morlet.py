import numpy as np
from scipy.fft import fft, ifft

def calculate_gabor_wavelet(data, sampling_rate, min_freq, max_freq, freq_res):
    """
    Calculates the complex wavelet transform using a Gabor-Morlet kernel.
    Replicates the logic of f_GaborWavelet.m
    """
    # Ensure data is a 1D array (equivalent to the ' column vector check)
    data = data.flatten()
    
    n_samples = len(data)
    
    # Safety check for empty data
    if n_samples == 0:
        raise ValueError("Cannot perform wavelet transform on empty data array")
    
    half_len = (n_samples // 2) + 1
    cycles = 4
    
    # Create frequency vector: minf : resf : maxf
    frequencies = np.arange(min_freq, max_freq + freq_res, freq_res)
    
    # Pre-calculate FFT of data
    fft_data = fft(data)
    
    # Standardize time vector (2*pi * (0:N-1) / N) * sampling_rate
    time_vector = (2 * np.pi / n_samples) * np.arange(n_samples) * sampling_rate
    time_half = time_vector[:half_len]
    
    # Initialize output matrix (freqs x samples)
    power_timecourse = np.zeros((len(frequencies), n_samples), dtype=complex)
    
    for i, freq in enumerate(frequencies):
        # width_gaussian (sigma) = n / f
        width_gaussian = cycles * (1 / freq)
        
        # Initialize Gaussian window
        gaussian_window = np.zeros(n_samples)
        
        # Formula: exp(-0.5 * (t - 2*pi*f)^2 * sigma^2)
        # Note: realpow(x, 2) in MATLAB is just x**2
        exponent = -0.5 * ((time_half - (2 * np.pi * freq))**2) * (width_gaussian**2)
        gaussian_window[:half_len] = np.exp(exponent)
        
        # Normalization
        # gaussian * sqrt(N) / norm(gaussian, 2)
        norm_val = np.linalg.norm(gaussian_window, ord=2)
        if norm_val != 0:
            gaussian_window = (gaussian_window * np.sqrt(n_samples)) / norm_val
        
        # Frequency domain multiplication and IFFT
        # Result / sqrt(width_gaussian)
        wavelet_filtered = ifft(fft_data * gaussian_window)
        power_timecourse[i, :] = wavelet_filtered / np.sqrt(width_gaussian)
        
    return power_timecourse