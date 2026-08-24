import mne
import numpy as np

mne.set_log_level('error')

CASES = [
    ('EL3011',
     r'I:\Shaked\ISO_data\control_clean\EL3011\EL3011_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif',
     r'I:\Shaked\ISO_data\scoring\young_control\EL3011.txt'),
    ('MCI27',
     r'I:\Shaked\ISO_data\elderly_control_clean\MCI27\MCI27_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif',
     r'I:\Shaked\ISO_data\scoring\elderly_control\MCI27_hypno.txt'),
    ('YS0',
     r'I:\Shaked\ISO_data\MCI_clean\YS0\YS0_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif',
     r'I:\Shaked\ISO_data\scoring\MCI\YS0_hypno.txt'),
]

for sub, fif, hypno in CASES:
    raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
    sec = raw.n_times / raw.info['sfreq']
    with open(hypno, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() != '']
    n = len(lines)
    uniq = sorted(set(lines))
    epoch_sec = sec / n
    print(f"{sub}: dur={sec:.1f}s ({sec/60:.1f}min)  hypno_lines={n}  sec_per_line~{epoch_sec:.2f}  uniq={uniq[:10]}")
