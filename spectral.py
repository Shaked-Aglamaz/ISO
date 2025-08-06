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



sub = "26"
path = f"D:/Shaked_data/ISO/{sub}_cropped.fif"
raw_cropped = mne.io.read_raw(path)
low_freq = 13
high_freq = 16
target_channel = 'VREF'

spindles = yasa.spindles_detect(raw_cropped, freq_sp=(low_freq, high_freq))
df_spindles = spindles.summary()
target_df = df_spindles[df_spindles['Channel'] == target_channel]

annotations = mne.Annotations(target_df['Start'].values, target_df['Duration'].values, ['Spindle_' + target_channel] * len(target_df))
target_raw = raw_cropped.copy().pick_channels([target_channel])
_ = target_raw.set_annotations(annotations)
target_raw.load_data()

start_time = 70
duration = 40
sfreq = target_raw.info['sfreq']
start_sample = int(start_time * sfreq)
stop_sample = int((start_time + duration) * sfreq)

data, times = target_raw.get_data(start=start_sample, stop=stop_sample, return_times=True)
data = data[0] * 1e6  # convert from V to µV
spindle_mask = (target_df['Start'] >= start_time) & (target_df['Start'] <= start_time + duration)
curr_spindles = target_df[spindle_mask]

spindle_bandpass = target_raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
data, times = spindle_bandpass.get_data(start=start_sample, stop=stop_sample, return_times=True)
data = data[0] * 1e6  # convert from V to µV
analytic_signal = hilbert(data)
amplitude_envelope = np.abs(analytic_signal)


fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, data, color='black', lw=1, label="Filtered EEG", alpha=0.7)
ax.plot(times, amplitude_envelope, label="Smooth Envelope", color="red", linewidth=2)
for i, (_, row) in enumerate(curr_spindles.iterrows()):
    label = "Spindle" if i == 0 else None  # only label the first
    ax.axvspan(row['Start'], row['Start'] + row['Duration'], color='blue', alpha=0.3, label=label)

ax.set_xlabel("Time (s)")
ax.set_ylabel('Amplitude (µV)')
ax.set_title(f"Envelope of {target_channel} ({low_freq}-{high_freq} Hz)")
ax.set_xlim(times[0], times[-1])
ax.set_ylim(data.min() - 10, data.max() + 10)
xticks = np.arange(start_time, start_time + duration + 1)
ax.set_xticks(xticks)
xtick_labels = [str(tick) if tick % 5 == 0 else '' for tick in xticks]
ax.set_xticklabels(xtick_labels)
plt.legend()
plt.tight_layout()

# Save the first plot
first_plot_path = f"output/{sub}_filtered_eeg_envelope.png"
plt.savefig(first_plot_path, dpi=300)
plt.show(block=False)  # Non-blocking show for the first plot
print(f"First plot saved to: {first_plot_path}")



# entire bout
orig_data = target_raw.get_data()[0] # V
bandpass_data, times = spindle_bandpass.get_data(return_times=True)
bandpass_data = bandpass_data[0] # V
analytic_signal = hilbert(bandpass_data)
amplitude_envelope = np.abs(analytic_signal)
amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)


n_samples = amplitude_envelope.shape[0]
freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)
fft_values = np.fft.rfft(amplitude_envelope)
fft_power = np.abs(fft_values) ** 2
fft_power *= 1e6        # to avoid really small numbers. TODO: does that make sense?
fft_power = fft_power / (n_samples / sfreq) # normalized by length


min_freq, max_freq = 0, 0.1
mask = (freqs >= min_freq) & (freqs <= max_freq)
x = freqs[mask]
y = fft_power[mask]

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Params order: amplitude peak, center, STD
# Initial guess: max(y), mean freq, some small value
p0 = [np.max(y), x[np.argmax(y)], 0.01]
fitted_params, _ = curve_fit(gaussian, x, y, p0=p0)
peak, mu, sigma = fitted_params


x_fit = np.linspace(min_freq, max_freq, 500)
y_fit = gaussian(x_fit, *fitted_params)
x1 = mu - sigma
x2 = mu + sigma
mask = (x_fit >= x1) & (x_fit <= x2)
bandwidth_height = gaussian(x1, *fitted_params)
area = simpson(x=x_fit[mask], y=y_fit[mask])
threshold = 1.5 * np.std(y_fit) # what should be the threshold? 

plt.figure()
plt.plot(x, y, label='FFT Power')
plt.plot(x_fit, y_fit, label=f'Gaussian fit (μ={mu:.3f}, STD={sigma:.3f})')
plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold')
plt.hlines(bandwidth_height, x1, x2, colors="purple", label="Bandwidth")
plt.fill_between(x_fit[mask], y_fit[mask], color='skyblue', alpha=0.5, label='±1 STD Area')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('FFT Power Spectrum with Gaussian Fit')
plt.legend()
plt.tight_layout()

# Save the second plot
second_plot_path = f"output/{sub}_fft_power_gaussian_fit.png"
plt.savefig(second_plot_path, dpi=300)
plt.show(block=False)  # Non-blocking show for the second plot
print(f"Second plot saved to: {second_plot_path}")


plt.show()