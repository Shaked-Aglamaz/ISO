import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yasa
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
mne.set_log_level("error")
mne.viz.set_browser_backend('qt')
mne.set_config('MNE_BROWSER_THEME', 'dark')
face_electrodes = ['E238', 'E234', 'E230', 'E226', 'E225', 'E241', 'E244', 'E248', 'E252', 'E253', 'E242', 'E243', 'E245', 'E246', 'E249', 'E247', 'E250', 'E251', 'E255', 'E254', 
                   'E73', 'E54', 'E37', 'E32', 'E31', 'E18', 'E25', 'E61', 'E46', 'E67', 'E68', 'E239', 'E240', 'E235', 'E231', 'E236', 'E237', 'E232', 'E227', 'E210', 'E219', 'E220', 
                   'E1', 'E10', 'E218', 'E228', 'E233']
neck_electrodes = ['E145', 'E146', 'E147', 'E156', 'E165', 'E174', 'E166', 'E157', 'E148', 'E137', 'E136', 'E135', 'E134', 'E133']
sub_to_hypno_path = {'31': 'HC_V1/EL3006', '26': 'HC_V1/RD43', '9': 'HK5'}


# loading and setting parameters
sub = "26"
path = f"D:/Shaked_data/ISO/{sub}_cropped.fif"
raw_cropped = mne.io.read_raw(path)
low_freq = 13
high_freq = 16
target_channel = 'VREF'

# spindle detection and annotation
spindles = yasa.spindles_detect(raw_cropped, freq_sp=(low_freq, high_freq))
df_spindles = spindles.summary()
target_df = df_spindles[df_spindles['Channel'] == target_channel]
annotations = mne.Annotations(target_df['Start'].values, target_df['Duration'].values, ['Spindle_' + target_channel] * len(target_df))
target_raw = raw_cropped.copy().pick_channels([target_channel])
_ = target_raw.set_annotations(annotations)

# extracting sigma power and envelope
target_raw.load_data()
orig_data = target_raw.get_data()[0] # V
sfreq = target_raw.info['sfreq']
spindle_bandpass = target_raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
bandpass_data, times = spindle_bandpass.get_data(return_times=True)
bandpass_data = bandpass_data[0] # V
analytic_signal = hilbert(bandpass_data)
amplitude_envelope = np.abs(analytic_signal)
amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)

# create a raw with the original channel + sigma power + sigma envelope
combined_data = np.vstack([orig_data, bandpass_data, amplitude_envelope])
ch_names = [target_channel, f'{target_channel}_sigma', f'{target_channel}_sigma_env']
ch_types = ['eeg', 'ecg', 'misc']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_combined = mne.io.RawArray(combined_data, info)
_ = raw_combined.set_meas_date(target_raw.info['meas_date'])
offset_sec = target_raw.first_samp / sfreq
adjusted_annotations = target_raw.annotations.copy()
adjusted_annotations.onset = adjusted_annotations.onset - offset_sec
_ = raw_combined.set_annotations(adjusted_annotations)

def analyze_frequency_content(signal, sfreq, title):
    """Analyze the frequency content of a signal"""
    # Compute FFT
    fft_vals = fft(signal)
    freqs = fftfreq(len(signal), 1/sfreq)
    
    # Get positive frequencies only
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = np.abs(fft_vals[:len(freqs)//2])
    
    # Find dominant frequencies
    peak_indices = find_peaks(pos_fft)[0]
    if len(peak_indices) > 0:
        dominant_freqs = pos_freqs[peak_indices]
        dominant_amps = pos_fft[peak_indices]
        # Sort by amplitude
        sorted_indices = np.argsort(dominant_amps)[::-1]
        top_freqs = dominant_freqs[sorted_indices][:5]
        top_amps = dominant_amps[sorted_indices][:5]
        
        print(f"\n{title} - Top 5 dominant frequencies:")
        for i, (freq, amp) in enumerate(zip(top_freqs, top_amps)):
            print(f"  {i+1}. {freq:.4f} Hz (amplitude: {amp:.2f})")
    
    return pos_freqs, pos_fft

start_time = 75
end_time = 175
X0 = int(start_time * sfreq)
X1 = int(end_time * sfreq)

def work(amplitude_envelope, sfreq, target_channel, raw_combined):

    # Analyze original envelope
    print("\n" + "="*60)
    print("FREQUENCY ANALYSIS OF ENVELOPE")
    print("="*60)
    freqs_orig, fft_orig = analyze_frequency_content(amplitude_envelope, sfreq, "Original envelope")

    # Try different filtering approaches
    print("\n" + "="*60)
    print("FILTERING COMPARISON")
    print("="*60)

    # 1. MNE filter (your original approach)
    h_freq1 = 0.005
    sigma_env_raw = raw_combined.copy().pick_channels([f"{target_channel}_sigma_env"])
    sigma_env_raw.filter(l_freq=None, h_freq=h_freq1, picks="all")
    filtered_env_mne = sigma_env_raw.get_data()[0]

    # 2. More aggressive MNE filter
    h_freq2 = 0.003
    sigma_env_raw_aggressive = raw_combined.copy().pick_channels([f"{target_channel}_sigma_env"])
    sigma_env_raw_aggressive.filter(l_freq=None, h_freq=h_freq2, picks="all")
    filtered_env_aggressive = sigma_env_raw_aggressive.get_data()[0]

    # 3. Gaussian smoothing (alternative approach)
    gaussian_smoothed = gaussian_filter1d(amplitude_envelope, sigma=sfreq*2)  # 2-second smoothing

    # 4. Moving average
    window_size = int(sfreq * 5)  # 5-second window
    moving_avg = np.convolve(amplitude_envelope, np.ones(window_size)/window_size, mode='same')

    # Analyze filtered versions
    freqs_mne, fft_mne = analyze_frequency_content(filtered_env_mne, sfreq, f"MNE filtered ({h_freq1} Hz)")
    freqs_agg, fft_agg = analyze_frequency_content(filtered_env_aggressive, sfreq, f"MNE filtered ({h_freq2} Hz)")
    freqs_gauss, fft_gauss = analyze_frequency_content(gaussian_smoothed, sfreq, "Gaussian smoothed")
    freqs_ma, fft_ma = analyze_frequency_content(moving_avg, sfreq, "Moving average")

    # Calculate differences
    diff_mne = amplitude_envelope - filtered_env_mne
    diff_agg = amplitude_envelope - filtered_env_aggressive
    diff_gauss = amplitude_envelope - gaussian_smoothed
    diff_ma = amplitude_envelope - moving_avg

    print(f"\nRMS differences:")
    print(f"  MNE {h_freq1} Hz: {np.sqrt(np.mean(diff_mne**2)):.6f}")
    print(f"  MNE {h_freq2} Hz: {np.sqrt(np.mean(diff_agg**2)):.6f}")
    print(f"  Gaussian: {np.sqrt(np.mean(diff_gauss**2)):.6f}")
    print(f"  Moving avg: {np.sqrt(np.mean(diff_ma**2)):.6f}")

    # Plot results
    plt.figure(figsize=(20, 12))

    # Plot 1: Time domain comparison
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(X0, X1) / sfreq, amplitude_envelope[X0:X1], label='Original envelope', color='blue', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, filtered_env_mne[X0:X1], label=f'MNE filtered ({h_freq1} Hz)', color='red', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, filtered_env_aggressive[X0:X1], label=f'MNE filtered ({h_freq2} Hz)', color='green', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, gaussian_smoothed[X0:X1], label='Gaussian smoothed', color='orange', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, moving_avg[X0:X1], label='Moving average', color='purple', alpha=0.8)
    plt.legend()
    plt.title('Time Domain Comparison')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # Plot 2: Frequency domain comparison
    plt.subplot(2, 2, 2)
    plt.semilogy(freqs_orig, fft_orig, label='Original', alpha=0.8)
    plt.semilogy(freqs_mne, fft_mne, label=f'MNE {h_freq1} Hz', alpha=0.8)
    plt.semilogy(freqs_agg, fft_agg, label=f'MNE {h_freq2} Hz', alpha=0.8)
    plt.semilogy(freqs_gauss, fft_gauss, label='Gaussian', alpha=0.8)
    plt.semilogy(freqs_ma, fft_ma, label='Moving avg', alpha=0.8)
    plt.legend()
    plt.title('Frequency Domain Comparison')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1)  # Focus on low frequencies
    plt.grid(True, alpha=0.3)

    # Plot 3: Differences
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(X0, X1) / sfreq, diff_mne[X0:X1], label=f'Original - MNE {h_freq1} Hz', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, diff_agg[X0:X1], label=f'Original - MNE {h_freq2} Hz', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, diff_gauss[X0:X1], label='Original - Gaussian', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, diff_ma[X0:X1], label='Original - Moving avg', alpha=0.8)
    plt.legend()
    plt.title('Differences from Original')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Difference")
    plt.grid(True, alpha=0.3)

    # Plot 4: Zoomed view of differences
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(X0, X1) / sfreq, diff_mne[X0:X1], label=f'Original - MNE {h_freq1} Hz', alpha=0.8)
    plt.plot(np.arange(X0, X1) / sfreq, diff_agg[X0:X1], label=f'Original - MNE {h_freq2} Hz', alpha=0.8)
    plt.legend()
    plt.title('Zoomed Differences (MNE filters only)')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Difference")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Define the desired cutoff frequency for infra-slow trends
infra_slow_cutoff = 0.04  # Hz (captures cycles of 25-100 seconds)

# Apply a Butterworth low-pass filter for infra-slow trends (order 4)
b4, a4 = butter(N=4, Wn=infra_slow_cutoff / (sfreq / 2), btype='low')
infra_slow_trend_4 = filtfilt(b4, a4, amplitude_envelope)

# Plot the original envelope and the infra-slow trend for order 4
plt.figure(figsize=(15, 5))

# Plot: Full signal comparison
time_axis = np.arange(len(amplitude_envelope)) / sfreq
plt.plot(time_axis, amplitude_envelope, label='Original envelope', color='blue', alpha=0.5)
plt.plot(time_axis, infra_slow_trend_4, label='Infra-slow trend (order 4)', color='red', linewidth=2)
plt.legend()
plt.title('Infra-Slow Trends of the Envelope (Full Signal)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"output/infra_slow_trend_{infra_slow_cutoff}.png", dpi=300)
plt.show(block=False)  # Non-blocking show

# Save the infra-slow trend for further analysis
output_path_trend_4 = f"output/infra_slow_trend_order4_{infra_slow_cutoff}.npy"
np.save(output_path_trend_4, infra_slow_trend_4)
print(f"Infra-slow trend saved to: {output_path_trend_4}")

# Compute a scaling factor to match the amplitude of the original envelope
scaling_factor = np.std(amplitude_envelope) / np.std(infra_slow_trend_4)
scaled_infra_slow_trend = infra_slow_trend_4 * scaling_factor

# Plot the scaled infra-slow trend and original envelope
plt.figure(figsize=(15, 5))
plt.plot(time_axis, amplitude_envelope, label='Original envelope', color='blue', alpha=0.5)
plt.plot(time_axis, scaled_infra_slow_trend, label='Scaled infra-slow trend', color='red', linewidth=2)
plt.legend()
plt.title('Scaled Infra-Slow Trends of the Envelope')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"output/scaled_infra_slow_trend_{infra_slow_cutoff}.png", dpi=300)
plt.show(block=False)  # Non-blocking show

# Plot the scaled infra-slow trend and original envelope for the range X0 to X1
plt.figure(figsize=(15, 5))
plt.plot(np.arange(X0, X1) / sfreq, amplitude_envelope[X0:X1], label='Original envelope', color='blue', alpha=0.5)
plt.plot(np.arange(X0, X1) / sfreq, scaled_infra_slow_trend[X0:X1], label='Scaled infra-slow trend', color='red', linewidth=2)
plt.legend()
plt.title(f'Scaled Infra-Slow Trends (Zoomed: {start_time}-{end_time} sec)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"output/scaled_infra_slow_trend_zoomed_{infra_slow_cutoff}.png", dpi=300)
plt.show(block=False)  # Non-blocking show

# Save the infra-slow trend for further analysis
output_path_trend = f"output/infra_slow_trend_{infra_slow_cutoff}.npy"
np.save(output_path_trend, infra_slow_trend_4)
print(f"Infra-slow trend saved to: {output_path_trend}")

# Standardize the low-pass filtered envelope
infra_slow_trend_standardized = (infra_slow_trend_4 - np.mean(infra_slow_trend_4)) / np.std(infra_slow_trend_4)

# Find peaks and troughs
peaks, _ = find_peaks(infra_slow_trend_standardized)
troughs, _ = find_peaks(-infra_slow_trend_standardized)

# Find zero-crossings (negative: positive to negative)
zero_crossings = np.where(np.diff(np.sign(infra_slow_trend_standardized)) < 0)[0]

# Identify ISFS segments (25 to 100 seconds apart)
sfreq = raw_combined.info['sfreq']  # Sampling frequency
min_samples = int(25 * sfreq)  # Minimum duration in samples
max_samples = int(100 * sfreq)  # Maximum duration in samples

isfs_segments = []
for i in range(len(zero_crossings) - 1):
    duration = zero_crossings[i + 1] - zero_crossings[i]
    if min_samples <= duration <= max_samples:
        isfs_segments.append((zero_crossings[i], zero_crossings[i + 1]))

# Filter ISFS segments with exactly one trough and one peak between zero-crossings
filtered_isfs_segments = []
for start, end in isfs_segments:
    peaks_between = [p for p in peaks if start < p < end]
    troughs_between = [t for t in troughs if start < t < end]
    if len(peaks_between) == 1 and len(troughs_between) == 1:
        filtered_isfs_segments.append((start, end))

# Create a single figure with two subplots for ISFS detection
plt.figure(figsize=(15, 10))

# Subplot 1: All ISFS segments
plt.subplot(2, 1, 1)
plt.plot(time_axis, infra_slow_trend_standardized, label='Standardized infra-slow trend', color='green', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero line')

# Mark all ISFS segments
for start, end in isfs_segments:
    plt.axvspan(start / sfreq, end / sfreq, color='yellow', alpha=0.3, label='All ISFS' if 'All ISFS' not in plt.gca().get_legend_handles_labels()[1] else None)

# Mark peaks and troughs
plt.scatter(peaks / sfreq, infra_slow_trend_standardized[peaks], color='red', label='Peaks', zorder=5)
plt.scatter(troughs / sfreq, infra_slow_trend_standardized[troughs], color='blue', label='Troughs', zorder=5)

plt.legend()
plt.title('Standardized Infra-Slow Trend with All ISFS Segments')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# Subplot 2: Filtered ISFS segments
plt.subplot(2, 1, 2)
plt.plot(time_axis, infra_slow_trend_standardized, label='Standardized infra-slow trend', color='green', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero line')

# Mark filtered ISFS segments
for start, end in filtered_isfs_segments:
    plt.axvspan(start / sfreq, end / sfreq, color='orange', alpha=0.3, label='Filtered ISFS (1 Trough, 1 Peak)' if 'Filtered ISFS (1 Trough, 1 Peak)' not in plt.gca().get_legend_handles_labels()[1] else None)

# Mark peaks and troughs
plt.scatter(peaks / sfreq, infra_slow_trend_standardized[peaks], color='red', label='Peaks', zorder=5)
plt.scatter(troughs / sfreq, infra_slow_trend_standardized[troughs], color='blue', label='Troughs', zorder=5)

plt.legend()
plt.title('Standardized Infra-Slow Trend with Filtered ISFS Segments (One Trough and One Peak)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# Adjust layout and save the combined figure
plt.tight_layout()
combined_isfs_output_figure_path = f"output/standardized_infra_slow_trend_combined_isfs.png"
plt.savefig(combined_isfs_output_figure_path, dpi=300)
plt.show(block=False)  # Non-blocking show

print(f"Combined figure with ISFS detection saved to: {combined_isfs_output_figure_path}")

# Print filtered ISFS segments
if filtered_isfs_segments:
    print("Filtered ISFS segments identified (start, end in seconds):")
    for start, end in filtered_isfs_segments:
        print(f"  Start: {start / sfreq:.2f}, End: {end / sfreq:.2f}, Duration: {(end - start) / sfreq:.2f} seconds")
else:
    print("No filtered ISFS segments identified.")

# Focus on the first filtered ISFS
if filtered_isfs_segments:
    first_isfs_start, first_isfs_end = filtered_isfs_segments[0]
    positive_zero_crossings = np.where(np.diff(np.sign(infra_slow_trend_standardized)) > 0)[0]
    midpoint_candidates = positive_zero_crossings[(positive_zero_crossings > first_isfs_start) & (positive_zero_crossings < first_isfs_end)]

    if len(midpoint_candidates) > 0:
        first_isfs_midpoint = midpoint_candidates[0]
        first_isfs_trough = troughs[(troughs > first_isfs_start) & (troughs < first_isfs_midpoint)][0]
        first_isfs_peak = peaks[(peaks > first_isfs_midpoint) & (peaks < first_isfs_end)][0]

        # Calculate bin boundaries
        bin_1_2_boundary = int((first_isfs_start + first_isfs_trough) / 2)
        bin_3_4_boundary = int((first_isfs_trough + first_isfs_midpoint) / 2)
        bin_5_6_boundary = int((first_isfs_midpoint + first_isfs_peak) / 2)
        bin_7_8_boundary = int((first_isfs_peak + first_isfs_end) / 2)

        # Define the range for plotting (30 seconds before and after the ISFS)
        plot_start = max(0, first_isfs_start - int(30 * sfreq))
        plot_end = min(len(infra_slow_trend_standardized), first_isfs_end + int(30 * sfreq))
        time_axis = np.arange(plot_start, plot_end) / sfreq
        signal_segment = infra_slow_trend_standardized[plot_start:plot_end]

        # Plot the first filtered ISFS with bins
        plt.figure(figsize=(15, 5))
        plt.plot(time_axis, signal_segment, label='Standardized infra-slow trend', color='green', linewidth=2)
        plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero line')

        # Add vertical lines for bin boundaries (spanning only from x-axis to the signal)
        bin_boundaries = [
            first_isfs_start, bin_1_2_boundary, first_isfs_trough, bin_3_4_boundary,
            first_isfs_midpoint, bin_5_6_boundary, first_isfs_peak, bin_7_8_boundary, first_isfs_end
        ]
        for boundary in bin_boundaries:
            plt.vlines(boundary / sfreq, ymin=0, ymax=infra_slow_trend_standardized[boundary], color='blue', linestyle='--')

        # Mark the peak and trough
        plt.scatter(first_isfs_trough / sfreq, infra_slow_trend_standardized[first_isfs_trough], color='blue', label='Trough', zorder=5)
        plt.scatter(first_isfs_peak / sfreq, infra_slow_trend_standardized[first_isfs_peak], color='red', label='Peak', zorder=5)

        # Annotate bin numbers
        bin_positions = [
            (first_isfs_start, bin_1_2_boundary),
            (bin_1_2_boundary, first_isfs_trough),
            (first_isfs_trough, bin_3_4_boundary),
            (bin_3_4_boundary, first_isfs_midpoint),
            (first_isfs_midpoint, bin_5_6_boundary),
            (bin_5_6_boundary, first_isfs_peak),
            (first_isfs_peak, bin_7_8_boundary),
            (bin_7_8_boundary, first_isfs_end),
        ]
        for i, (start, end) in enumerate(bin_positions, 1):
            y_offset = -0.1 if i <= 4 else 0.1  # First four bins below x-axis, last four above
            plt.text((start + end) / (2 * sfreq), y_offset, str(i), color='red', fontsize=12, ha='center', va='center')

        plt.legend()
        plt.title('First Filtered ISFS with Phase Bins, Peak, and Trough')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        first_isfs_output_figure_path = f"output/first_filtered_isfs_with_bins.png"
        plt.savefig(first_isfs_output_figure_path, dpi=300)
        plt.show(block=False)  # Non-blocking show

        print(f"First filtered ISFS plot saved to: {first_isfs_output_figure_path}")
    else:
        print("No positive zero-crossing found within the first filtered ISFS range.")
else:
    print("No filtered ISFS segments found.")

# Create a raw object with sigma envelope and low-pass filtered envelope
combined_data = np.vstack([amplitude_envelope, infra_slow_trend_4])
ch_names = [f"{target_channel}_sigma_env", f"{target_channel}_infra_slow"]
ch_types = ['misc', 'stim']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_envelope = mne.io.RawArray(combined_data, info)

# Add spindle annotations to the new raw object
raw_envelope.set_annotations(annotations)

# Adjust the scale of both misc and stim channels for plotting
# scalings = {'misc': 1e-2, 'stim': 5.0}  # Increase the scale of the misc channel
# raw_envelope.plot(duration=100) #, scalings=scalings)

# Save the new raw object to a file
output_raw_path = f"output/{sub}_sigma_and_infra_slow_raw.fif"
raw_envelope.save(output_raw_path, overwrite=True)
print(f"Raw file with sigma envelope and infra-slow trend saved to: {output_raw_path}")

# Keep all plots open at the end
plt.show()