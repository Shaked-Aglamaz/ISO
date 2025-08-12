import mne
import numpy as np
import matplotlib.pyplot as plt
import yasa
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from pathlib import Path

mne.set_log_level("error")
mne.viz.set_browser_backend('qt')
mne.set_config('MNE_BROWSER_THEME', 'dark')

FACE_ELECTRODES = ['E238', 'E234', 'E230', 'E226', 'E225', 'E241', 'E244', 'E248', 'E252', 'E253', 'E242', 'E243', 'E245', 'E246', 'E249', 'E247', 'E250', 'E251', 'E255', 'E254', 
                   'E73', 'E54', 'E37', 'E32', 'E31', 'E18', 'E25', 'E61', 'E46', 'E67', 'E68', 'E239', 'E240', 'E235', 'E231', 'E236', 'E237', 'E232', 'E227', 'E210', 'E219', 'E220', 
                   'E1', 'E10', 'E218', 'E228', 'E233']
NECK_ELECTRODES = ['E145', 'E146', 'E147', 'E156', 'E165', 'E174', 'E166', 'E157', 'E148', 'E137', 'E136', 'E135', 'E134', 'E133']
SUB_TO_HYPNO_PATH = {'31': 'HC_V1/EL3006', '26': 'HC_V1/RD43', '9': 'HK5'}


class SpindleAnalyzer:
    
    def __init__(self, subject_id, low_freq=13, high_freq=16, target_channels=['VREF'], 
                 data_dir="D:/Shaked_data/ISO", output_dir="output"):
        self.subject_id = subject_id
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.target_channels = target_channels
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        self.target_raw = None
        self.spindles_df = None
        self.analysis_results = None
        
    def load_raw_data(self):
        path = Path(self.data_dir) / f"{self.subject_id}_cropped.fif"
        raw_data = mne.io.read_raw(path)
        self.target_raw = raw_data.pick_channels(self.target_channels)
    
    def detect_spindles(self):
        if self.target_raw is None:
            self.load_raw_data()
            
        spindles = yasa.spindles_detect(self.target_raw, freq_sp=(self.low_freq, self.high_freq))
        self.spindles_df = spindles.summary()
        annotations = mne.Annotations(
            self.spindles_df['Start'].values, 
            self.spindles_df['Duration'].values, 
            ['Spindle_' + channel for channel in self.spindles_df['Channel']]
        )

        self.target_raw.set_annotations(annotations)
        self.target_raw.load_data()
        self.target_raw.plot()
    
    @staticmethod
    def compute_envelope(data):
        """
        Compute amplitude envelope using Hilbert transform.
        """
        analytic_signal = hilbert(data)
        return np.abs(analytic_signal)
    
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
        Compute FFT power spectrum of amplitude envelope
        """
        sfreq = self.target_raw.info['sfreq']
        amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
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
        return x, y

    def fit_gaussian_to_spectrum(self, x, y):
        # Params order: peak, mu, sigma
        p0 = [np.max(y), x[np.argmax(y)], 0.01]
        fitted_params, _ = curve_fit(self.gaussian, x, y, p0=p0)        
        return fitted_params

    def plot_fft_power_spectrum(self, x, y, fitted_params, min_freq=0, max_freq=0.1):
        peak, mu, sigma = fitted_params
        
        x_fit = np.linspace(min_freq, max_freq, 500)
        y_fit = self.gaussian(x_fit, *fitted_params)
        x1 = mu - sigma
        x2 = mu + sigma
        mask = (x_fit >= x1) & (x_fit <= x2)
        bandwidth_height = self.gaussian(x1, *fitted_params)
        area = simpson(x=x_fit[mask], y=y_fit[mask])
        # TODO: think about the threshold
        
        plt.figure()
        plt.plot(x, y, label=f'FFT Power ({self.target_channels[0]})')
        plt.plot(x_fit, y_fit, label=f'Gaussian fit (μ={mu:.3f}, STD={sigma:.3f})')
        plt.hlines(bandwidth_height, x1, x2, colors="purple", label="Bandwidth")
        plt.fill_between(x_fit[mask], y_fit[mask], color='skyblue', alpha=0.5, label='±1 STD Area')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('FFT Power Spectrum with Gaussian Fit')
        plt.legend()
        plt.tight_layout()

        plot_path = Path(self.output_dir) / f"{self.subject_id}_fft_power_gaussian_fit.png"
        plt.savefig(plot_path, dpi=300)
        plt.show(block=False)
        
        results = {'peak': peak, 'mu': mu, 'sigma': sigma, 'area': area, 'bandwidth_height': bandwidth_height}
        self.analysis_results = {
            'subject_id': self.subject_id,
            'plot_path': str(plot_path),
            'spindles_count': len(self.spindles_df),
            'spectral_analysis': results,
            'spindles_df': self.spindles_df
        }

    def analyze_segment(self, start_time, duration):
        """For visualization: analyze and plot a specific time segment."""
        
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles()
        
        sfreq = self.target_raw.info['sfreq']
        start_sample = int(start_time * sfreq)
        stop_sample = int((start_time + duration) * sfreq)
        
        sigma_bandpass = self.target_raw.copy().filter(l_freq=self.low_freq, h_freq=self.high_freq)
        sigma_data, times = sigma_bandpass.get_data(start=start_sample, stop=stop_sample, return_times=True)
        sigma_data = sigma_data[0] * 1e6  # convert from V to µV
        amplitude_envelope = self.compute_envelope(sigma_data)
        self.plot_sigma_envelope(times, sigma_data, amplitude_envelope, start_time, duration)
    
    def analyze(self):
        """Full analysis of the entire recording."""
        
        print(f"Analyzing subject {self.subject_id}...")
        
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles()
        
        spindle_bandpass = self.target_raw.copy().filter(l_freq=self.low_freq, h_freq=self.high_freq)
        bandpass_data, _ = spindle_bandpass.get_data(return_times=True)
        bandpass_data = bandpass_data[0]  # V
        amplitude_envelope = self.compute_envelope(bandpass_data)
        x, y = self.compute_fft_power_spectrum(amplitude_envelope)
        fitted_params = self.fit_gaussian_to_spectrum(x, y)
        self.plot_fft_power_spectrum(x, y, fitted_params)
    
    def get_summary(self):
        if self.analysis_results is None:
            print("No analysis results available. Run analyze() first.")
            return
            
        print(f"\nAnalysis Summary for Subject {self.subject_id}")
        print(f"{'='*50}")
        print(f"Target Channels: {self.target_channels}")
        print(f"Frequency Range: {self.low_freq}-{self.high_freq} Hz")
        print(f"Spindles Detected: {self.analysis_results['spindles_count']}")
        print(f"Spectral Peak Frequency: {self.analysis_results['spectral_analysis']['mu']:.3f} Hz")
        print(f"Spectral Bandwidth (σ): {self.analysis_results['spectral_analysis']['sigma']:.3f} Hz")
        print(f"Peak Power: {self.analysis_results['spectral_analysis']['peak']:.3f}")
        print(f"Area under Curve: {self.analysis_results['spectral_analysis']['area']:.3f}")
        print(f"FFT plot saved to: {self.analysis_results['plot_path']}")


def main():
    analyzer = SpindleAnalyzer(subject_id="26", low_freq=13, high_freq=16, target_channels=['VREF'], output_dir="output_1208")
    
    # Full analysis
    analyzer.analyze()
    analyzer.get_summary()
    
    # Segment visualization
    analyzer.analyze_segment(start_time=70, duration=40)
    
    plt.show()


if __name__ == "__main__":
    main()