"""
Raw-data figure showing sleep spindles.

Single broadband EEG trace over a chosen window, with the YASA-detected spindles
of that channel shaded. Detections come from the table written by the spindle
inspection notebook (code/tmp_spindles_EL3023.ipynb).

    python code/make_spindle_figure.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'code'))
from utils.utils import find_subject_fif_file

SUBJECT = 'EL3023'
GROUP_DIR = Path('I:/Shaked/ISO_data/control_clean')
CHANNEL = 'E90'
T_START, T_END = 1090.0, 1120.0

SPINDLE_CSV = REPO / 'results' / 'spindle_examples' / f'{SUBJECT}_spindles.csv'
OUT_DIR = REPO / 'results' / 'spindle_examples'

TRACE_COLOR = '#111111'
SPINDLE_COLOR = '#f0a04b'
FIGSIZE = (13, 3.2)

# Each "minus press" widens the y range by 5/4, i.e. flattens the trace by 4/5 -
# the same factor mne_qt_browser applies to its scale factor per '-' keystroke.
MINUS_PRESSES = 2


def load_window(subject_dir, channel, t_start, t_end):
    """Broadband data for one channel over [t_start, t_end], in microvolts."""
    fif_path = find_subject_fif_file(subject_dir)
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose='error')
    raw.pick([channel])
    sfreq = raw.info['sfreq']
    start, stop = int(t_start * sfreq), int(t_end * sfreq)
    data, times = raw[0, start:stop]
    return times, data[0] * 1e6, sfreq


def main(channel=CHANNEL, t_start=T_START, t_end=T_END, minus_presses=MINUS_PRESSES):
    times, trace, sfreq = load_window(GROUP_DIR / SUBJECT, channel, t_start, t_end)

    spindles = pd.read_csv(SPINDLE_CSV)
    window = spindles[(spindles['Channel'] == channel) &
                      (spindles['Start'] < t_end) & (spindles['End'] > t_start)]
    print(f'{len(window)} spindles in {t_start:.0f}-{t_end:.0f} s on {channel}')
    for _, sp in window.iterrows():
        print(f"  {sp['Start']:.2f}-{sp['End']:.2f} s  dur={sp['Duration']:.2f}s  "
              f"amp={sp['Amplitude']:.1f}uV  f={sp['Frequency']:.1f}Hz")

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11,
                         'axes.labelsize': 12, 'axes.titlesize': 12})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for _, sp in window.iterrows():
        ax.axvspan(sp['Start'], sp['End'], color=SPINDLE_COLOR, alpha=0.45, lw=0, zorder=1)
    ax.plot(times, trace, color=TRACE_COLOR, lw=0.6, zorder=2)

    ylim = np.ceil(np.abs(trace).max() / 10) * 10 * (1.25 ** minus_presses)
    ytick_step = 10 if ylim <= 35 else 20 if ylim <= 75 else 25
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (\u00b5V)')
    ax.set_xticks(np.arange(t_start, t_end + 1, 5))
    ax.set_yticks(np.arange(-(ylim // ytick_step) * ytick_step,
                           (ylim // ytick_step) * ytick_step + 1, ytick_step))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3)
    ax.legend(handles=[Patch(facecolor=SPINDLE_COLOR, alpha=0.45, label='Detected spindle')],
              loc='upper right', frameon=False, handlelength=1.4)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f'{SUBJECT}_{channel}_{int(t_start)}-{int(t_end)}s_spindles'
    for ext in ('png', 'pdf'):
        out = OUT_DIR / f'{stem}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    # optional CLI: python code/make_spindle_figure.py E90 1090 1120 [minus_presses]
    argv = sys.argv[1:]
    main(argv[0] if len(argv) > 0 else CHANNEL,
         float(argv[1]) if len(argv) > 1 else T_START,
         float(argv[2]) if len(argv) > 2 else T_END,
         int(argv[3]) if len(argv) > 3 else MINUS_PRESSES)
