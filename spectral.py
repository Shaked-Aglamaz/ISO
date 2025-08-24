import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from pathlib import Path
import textwrap

mne.set_log_level("error")
mne.viz.set_browser_backend('qt')
mne.set_config('MNE_BROWSER_THEME', 'dark')

FACE_ELECTRODES = ['E238', 'E234', 'E230', 'E226', 'E225', 'E241', 'E244', 'E248', 'E252', 'E253', 'E242', 'E243', 'E245', 'E246', 'E249', 'E247', 'E250', 'E251', 'E255', 'E254', 
                   'E73', 'E54', 'E37', 'E32', 'E31', 'E18', 'E25', 'E61', 'E46', 'E67', 'E68', 'E239', 'E240', 'E235', 'E231', 'E236', 'E237', 'E232', 'E227', 'E210', 'E219', 'E220', 
                   'E1', 'E10', 'E218', 'E228', 'E233']
NECK_ELECTRODES = ['E145', 'E146', 'E147', 'E156', 'E165', 'E174', 'E166', 'E157', 'E148', 'E137', 'E136', 'E135', 'E134', 'E133']
SUB_TO_HYPNO_PATH = {'31': 'HC_V1/EL3006', '26': 'HC_V1/RD43', '9': 'HK5'}


class BoutInfo:
    """
    Class to store information about an N2 sleep bout.
    """
    def __init__(self, start_time, end_time, bout_id):
        self.start_time = start_time
        self.end_time = end_time
        self.id = bout_id
        self.duration = end_time - start_time

    def __str__(self):
        return f"Bout {self.id}: {self.start_time:.1f}-{self.end_time:.1f}s ({self.duration:.1f}s)"


class SpindleAnalyzer:
    def __init__(self, subject_id, raw_path, low_freq=13, high_freq=16, target_channels=['VREF'], 
                output_dir=None, min_duration=280):
        self.subject_id = subject_id
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.target_channels = target_channels
        self.output_dir = output_dir if output_dir else f"{subject_id}_output"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.raw_path = raw_path
        self.min_duration = min_duration
        
        self.target_raw = None
        self.spindles_df = None
        self.n2_bouts = []
        self.bout_errors = {}
        self.results_df = None
        
        # For relative spectral power computation
        self._frequencies = []
        self._all_power_spectra = []

    def load_raw_data(self):
        raw_data = mne.io.read_raw(self.raw_path)
        annotation_desc = set(raw_data.annotations.description)
        if "NREM2" not in annotation_desc:
            print(f"ERROR: No 'NREM2' annotations found in the data.")
            print(f"Available annotations: {annotation_desc}")
            exit(1)
        
        if "BAD" not in annotation_desc:
            print("WARNING: No 'BAD' annotations found in the data.")
            print(f"Available annotations: {annotation_desc}")
        
        self.target_raw = raw_data.pick_channels(self.target_channels)
        self.target_raw.load_data()

    def detect_spindles(self, plot_raw=False):
        if self.target_raw is None:
            self.load_raw_data()
            
        spindles = yasa.spindles_detect(self.target_raw, freq_sp=(self.low_freq, self.high_freq))
        self.spindles_df = spindles.summary()
        spindle_annotations = mne.Annotations(
            self.spindles_df['Start'].values, 
            self.spindles_df['Duration'].values, 
            ['Spindle_' + channel for channel in self.spindles_df['Channel']],
            orig_time=self.target_raw.annotations.orig_time
        )
        annotations = self.target_raw.annotations + spindle_annotations
        self.target_raw.set_annotations(annotations)
        if plot_raw:
            self.target_raw.plot()

    def _extract_valid_segments(self, n2_start, n2_end, bad_segments):
        """Extract valid segments from an N2 period, excluding BAD overlaps, and create BoutInfo objects."""
        if len(bad_segments) == 0:
            if (n2_end - n2_start) >= self.min_duration:
                self.n2_bouts.append(BoutInfo(n2_start, n2_end, bout_id=len(self.n2_bouts)))
            return

        # Sort BAD segments and extract valid parts between them
        bad_segments = bad_segments[np.argsort(bad_segments[:, 0])]
        valid_segments = []
        current_pos = n2_start
        
        for bad_start, bad_end in bad_segments:
            if current_pos < bad_start:
                valid_segments.append((current_pos, bad_start))
            current_pos = max(current_pos, bad_end)
        
        if current_pos < n2_end:
            valid_segments.append((current_pos, n2_end))

        for start, end in valid_segments:
            if (end - start) >= self.min_duration:
                self.n2_bouts.append(BoutInfo(start, end, bout_id=len(self.n2_bouts)))

    def extract_n2_bouts(self):
        """Extract valid N2 segments, excluding overlaps with BAD annotations."""
        ann = self.target_raw.annotations
        n2_mask = (ann.description == "NREM2") & (ann.duration >= self.min_duration)
        bad_mask = ann.description == "BAD"
        n2_segments = np.column_stack([ann.onset[n2_mask], ann.onset[n2_mask] + ann.duration[n2_mask]])
        bad_segments = np.column_stack([ann.onset[bad_mask], ann.onset[bad_mask] + ann.duration[bad_mask]])
        
        for n2_start, n2_end in n2_segments:
            # Find overlapping BAD segments
            overlaps = (bad_segments[:, 0] < n2_end) & (bad_segments[:, 1] > n2_start)
            overlapping_bads = bad_segments[overlaps]
            self._extract_valid_segments(n2_start, n2_end, overlapping_bads)
        
        print("")
        print(f"Found {len(self.n2_bouts)} valid N2 bouts (>= {self.min_duration}s)")

    @staticmethod
    def compute_envelope(data):
        """
        Compute amplitude envelope using Hilbert transform and recenter around 0.
        """
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
        return amplitude_envelope
    
    def plot_sigma_envelope(self, times, data, amplitude_envelope, start_time, duration):
        spindle_mask = (self.spindles_df['Start'] >= start_time) & (self.spindles_df['Start'] <= start_time + duration)
        curr_spindles = self.spindles_df[spindle_mask]
        
        _, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, data, color='black', lw=1, label="Sigma", alpha=0.7)
        ax.plot(times, amplitude_envelope, label="Sigma Envelope", color="red", linewidth=2)
        
        for i, (_, row) in enumerate(curr_spindles.iterrows()):
            label = "Spindle" if i == 0 else None
            ax.axvspan(row['Start'], row['Start'] + row['Duration'], color='blue', alpha=0.3, label=label)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f"Envelope of {self.target_channels[0]} ({self.low_freq}-{self.high_freq} Hz)")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(data.min() - 10, data.max() + 10)
        
        xticks = np.arange(start_time, start_time + duration + 1)
        ax.set_xticks(xticks)
        xtick_labels = [str(tick) if tick % 5 == 0 else '' for tick in xticks]
        ax.set_xticklabels(xtick_labels)
        
        plt.legend()
        plt.tight_layout()
        plot_path = Path(self.output_dir) / f"{self.subject_id}_sigma_envelope.png"
        plt.savefig(plot_path, dpi=300)
        plt.show(block=False)
        print(f"Segment plot saved to: {plot_path}")

    @staticmethod
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    def compute_fft_power_spectrum(self, amplitude_envelope, min_freq=0, max_freq=0.1):
        """
        Compute FFT power spectrum of amplitude envelope and calculate relative power.
        Relative power = power at each frequency / mean power across 0-0.1 Hz range.
        """
        sfreq = self.target_raw.info['sfreq']
        n_samples = amplitude_envelope.shape[0]
        freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)
        fft_values = np.fft.rfft(amplitude_envelope)
        fft_power = np.abs(fft_values) ** 2
        fft_power *= 1e6  # to avoid really small numbers
        fft_power = fft_power / (n_samples / sfreq)  # normalized by length
        
        # Extract frequency range of interest
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        x = freqs[mask]
        y = fft_power[mask]
        
        # Calculate relative power: divide by mean power in the 0-0.1 Hz range
        y_relative = y / np.mean(y)
        return x, y_relative

    def fit_gaussian_to_spectrum(self, x, y):
        # Params order: peak, mu, sigma
        p0 = [np.max(y), x[np.argmax(y)], 0.01]
        fitted_params, _ = curve_fit(self.gaussian, x, y, p0=p0)        
        return fitted_params

    def analyze_single_bout(self, bout_info):
        """
        Analyze a single N2 bout: crop raw data, filter to sigma frequencies, compute envelope, FFT, and fit Gaussian.
        Returns bout result and power spectrum data for aggregation.
        """
        # Crop the raw data to bout times first, then filter
        bout_raw = self.target_raw.copy().crop(tmin=bout_info.start_time, tmax=bout_info.end_time)
        bout_sigma = bout_raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)
        bout_data = bout_sigma.get_data()[0]  # Get first (and only) channel
        amplitude_envelope = self.compute_envelope(bout_data)
        x, y = self.compute_fft_power_spectrum(amplitude_envelope)

        fitted_params = None
        try:
            fitted_params = self.fit_gaussian_to_spectrum(x, y)
        except Exception as fit_error:
            self.bout_errors[bout_info] = fit_error
            error_msg = f"  ✗ {bout_info} analysis failed: {fit_error}"
            print(textwrap.fill(error_msg, width=110, subsequent_indent="    "))
        
        bout_result = self.plot_fft_power_spectrum_bout(x, y, fitted_params, bout_info)
        
        # Return both the bout result and the power spectrum data for aggregation
        return bout_result, (x, y)

    def plot_fft_power_spectrum_bout(self, x, y, fitted_params, bout_info, min_freq=0, max_freq=0.1):
        """Plot FFT relative power spectrum for a bout."""
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=f'Relative Power ({self.target_channels[0]})')
        
        bout_result = None
        if fitted_params is not None:
            _, mu, sigma = fitted_params
            x_fit = np.linspace(min_freq, max_freq, 500)
            y_fit = self.gaussian(x_fit, *fitted_params)
            x1 = mu - sigma
            x2 = mu + sigma
            mask = (x_fit >= x1) & (x_fit <= x2)
            bandwidth_height = self.gaussian(x1, *fitted_params)
            area = simpson(x=x_fit[mask], y=y_fit[mask])
            # TODO: think about the threshold
            
            bout_result = {
                'bout_id': bout_info.id,
                'start_time': bout_info.start_time,
                'end_time': bout_info.end_time,
                'duration': bout_info.duration,
                'peak_frequency': mu,
                'bandwidth': 2 * sigma,
                'auc': area
            }
            
            plt.plot(x_fit, y_fit, label=f'Gaussian fit (μ={mu:.3f}, STD={sigma:.3f})')
            plt.hlines(bandwidth_height, x1, x2, colors="purple", label="Bandwidth")
            plt.fill_between(x_fit[mask], y_fit[mask], color='skyblue', alpha=0.5, label='±1 STD Area')
            title_suffix = ""
        else:
            # Failed Gaussian fit - plot FFT only
            title_suffix = " - FAILED"

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Relative Power')
        plt.title(f'FFT Relative Power Spectrum - Bout {bout_info.id} ({bout_info.start_time:.1f}-{bout_info.end_time:.1f}s){title_suffix}\n'
                  f'Duration: {bout_info.duration:.1f}s')
        plt.legend()
        plt.tight_layout()

        plot_path = Path(self.output_dir) / f"{self.subject_id}_bout_{bout_info.id}_fft_power_spectrum.png"
        plt.savefig(plot_path, dpi=300)
        plt.show(block=False)
        return bout_result

    def analyze_segment(self, start_time, duration):
        """For visualization: analyze and plot a specific time segment."""
        
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles()
        
        # Crop the raw data to segment times first, then filter
        end_time = start_time + duration
        segment_raw = self.target_raw.copy().crop(tmin=start_time, tmax=end_time)
        segment_sigma = segment_raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)
        sigma_data, times = segment_sigma.get_data(return_times=True)
        sigma_data = sigma_data[0] * 1e6  # convert from V to µV
        amplitude_envelope = self.compute_envelope(sigma_data)
        self.plot_sigma_envelope(times, sigma_data, amplitude_envelope, start_time, duration)

    def analyze_all_bouts(self, plot_raw=False):
        """
        Analyze all valid N2 bouts in the recording.
        """
        print(f"Starting analysis of all N2 bouts for subject {self.subject_id}...")
        
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles(plot_raw)
        
        if not self.n2_bouts:
            self.extract_n2_bouts()
        
        if not self.n2_bouts:
            print("No valid N2 bouts found!")
            return
        
        bout_results = []
        for bout_info in self.n2_bouts:
            try:
                result, (freqs, power) = self.analyze_single_bout(bout_info)
                if result is not None:
                    bout_results.append(result)
                    self._all_power_spectra.append(power)
                    self._frequencies.append(freqs)
                    print(f"  ✓ {bout_info} analysis succeeded")
            except Exception as e:
                self.bout_errors[bout_info] = e
                print(f"  ✗ {bout_info} analysis failed: {e}")
        
        print(f"Completed analysis of {len(bout_results)}/{len(self.n2_bouts)} bouts\n")
        if bout_results:
            self.results_df = pd.DataFrame(bout_results)
        
    def compute_mean_spectral_power(self):
        """
        Compute the mean spectral power over all bouts.
        Uses interpolation to create a common frequency grid for averaging.
        """
        if not self._frequencies or not self._all_power_spectra:
            print("No power spectra available. Run analyze_all_bouts() first.")
            return None, None, None
        
        # Create common frequency grid with finest resolution for interpolation
        highest_min = max(freqs[0] for freqs in self._frequencies)
        lowest_max = min(freqs[-1] for freqs in self._frequencies)
        finest_resolution = min(freqs[1] - freqs[0] for freqs in self._frequencies)
        common_freqs = np.arange(highest_min, lowest_max, finest_resolution)
        
        # Interpolate all power spectra to common grid
        interpolated_spectra = []
        for freqs, power in zip(self._frequencies, self._all_power_spectra):
            interp_func = interp1d(freqs, power, bounds_error=False, fill_value='extrapolate')
            interpolated_power = interp_func(common_freqs)
            interpolated_spectra.append(interpolated_power)
        
        # Convert to numpy array and compute statistics
        power_matrix = np.array(interpolated_spectra)
        mean_power_spectrum = np.mean(power_matrix, axis=0)
        std_power_spectrum = np.std(power_matrix, axis=0)
        print(f"\nComputed mean spectral power from {len(interpolated_spectra)} bouts successfully.")
        
        return common_freqs, mean_power_spectrum, std_power_spectrum
    
    def plot_mean_spectral_power(self, frequencies=None, mean_power=None, std_power=None):
        """
        Plot the mean spectral power (averaged across all bouts).
        """
        if frequencies is None or mean_power is None:
            frequencies, mean_power, std_power = self.compute_mean_spectral_power()
            
        if frequencies is None:
            print("Cannot plot - no valid spectral data computed.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, mean_power, 'b-', linewidth=2, label='Mean Relative Power')
        
        if std_power is not None:
            plt.fill_between(frequencies, 
                           mean_power - std_power, 
                           mean_power + std_power, 
                           alpha=0.3, color='blue', 
                           label='±1 STD')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Power')
        plt.title(f'Mean Spectral Power - Subject {self.subject_id}\n'
                  f'across all N2 bouts ({self.low_freq}-{self.high_freq} Hz envelope)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = Path(self.output_dir) / f"{self.subject_id}_mean_spectral_power.png"
        plt.savefig(plot_path, dpi=300)
        plt.show(block=False)
        
    def get_summary(self):
        if self.results_df is None:
            print("No analysis results available. Run analyze_all_bouts() first.")
            return
        
        summary_lines = []
        summary_lines.append(f"Analysis Summary for Subject {self.subject_id}")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Target Channels: {self.target_channels}")
        summary_lines.append(f"Frequency Range: {self.low_freq}-{self.high_freq} Hz")
        summary_lines.append(f"Total Spindles Detected: {len(self.spindles_df) if self.spindles_df is not None else 0}")
        summary_lines.append(f"Total N2 Bouts Found: {len(self.n2_bouts)}")
        summary_lines.append(f"Failed Bouts: {len(self.bout_errors)}")
        summary_lines.append("")
        summary_lines.append("Per-Bout Analysis:")
        summary_lines.append(f"{'Bout':<6}{'Duration':<12}{'Peak Freq':<12}{'Bandwidth':<12}{'AUC':<10}")
        summary_lines.append("-" * 60)
        
        for _, row in self.results_df.iterrows():
            summary_lines.append(f"{row['bout_id']:<6}{row['duration']:<12.1f}{row['peak_frequency']:<12.3f}{row['bandwidth']:<12.3f}{row['auc']:<10.3f}")
        
        # Summary statistics using DataFrame
        summary_lines.append("")
        summary_lines.append("Summary Statistics:")
        summary_lines.append("-" * 60)
        summary_lines.append(f"Average bout duration: {self.results_df['duration'].mean():.1f}s (±{self.results_df['duration'].std():.1f})")
        summary_lines.append(f"Average peak frequency: {self.results_df['peak_frequency'].mean():.3f}Hz (±{self.results_df['peak_frequency'].std():.3f})")
        summary_lines.append(f"Average bandwidth: {self.results_df['bandwidth'].mean():.3f}Hz (±{self.results_df['bandwidth'].std():.3f})")
        summary_lines.append(f"Average AUC: {self.results_df['auc'].mean():.3f} (±{self.results_df['auc'].std():.3f})")
        
        for line in summary_lines:
            print(line)
        
        if self.bout_errors:
            summary_lines.append("")
            summary_lines.append("Analysis Errors:")
            summary_lines.append("-" * 60)
            for bout_info, error in self.bout_errors.items():
                error_type = type(error).__name__
                error_msg = str(error)
                summary_lines.append(f"Bout {bout_info.id} ({bout_info.start_time:.1f}-{bout_info.end_time:.1f}s): {error_type} - {error_msg}")
        
        summary_path = Path(self.output_dir) / f"{self.subject_id}_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))

def main():
    sub = "26"
    raw_path = f"D:/Shaked_data/ISO/{sub}_filtered_4_channels.fif"
    analyzer = SpindleAnalyzer(sub, raw_path, low_freq=13, high_freq=16, target_channels=['VREF'])
    
    # Analyze all N2 bouts
    analyzer.analyze_all_bouts()
    analyzer.get_summary()
    analyzer.plot_mean_spectral_power()

    # Optional: segment visualization for a specific time
    # analyzer.analyze_segment(start_time=70, duration=40)
    
    plt.show()


if __name__ == "__main__":
    main()