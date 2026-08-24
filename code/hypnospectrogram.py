"""Generate hypnogram+spectrogram figures with sleepeegpy.SpectralPipe.

Repurposed to probe the four <20%-ISFS-detection excluded young controls:
for each subject, plot one channel where ISFS WAS detected and one (adjacent,
central) where it was NOT, to inspect whether the failures look like a
processing/signal problem rather than localized bad channels.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from sleepeegpy.pipeline import SpectralPipe

mne.set_log_level("error")

OUT_ROOT = Path("results/new_iso_results/a_excluded_V2")

# Detected/undetected channel pairs (adjacent, near vertex) from the V2 detection run.
CASES = [
    dict(
        sub="EL3017",
        fif=r"I:\Shaked\ISO_data\control_clean\a_excluded\EL3017\EL3017_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif",
        hypno=r"I:\Shaked\ISO_data\scoring\young_control\EL3017.txt",
        hypno_freq=1.0,
        letter=False,
        detected="E144",
        undetected="E132",
    ),
    dict(
        sub="EL3018",
        fif=r"I:\Shaked\ISO_data\control_clean\a_excluded\EL3018\EL3018_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif",
        hypno=r"I:\Shaked\ISO_data\scoring\young_control\EL3018.txt",
        hypno_freq=1.0,
        letter=False,
        detected="E90",
        undetected="E81",
    ),
    dict(
        sub="EL3021",
        fif=r"I:\Shaked\ISO_data\control_clean\a_excluded\EL3021\EL3021_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif",
        hypno=r"I:\Shaked\ISO_data\scoring\young_control\EL3021.txt",
        hypno_freq=1.0,
        letter=False,
        detected="E90",
        undetected="E81",
    ),
    dict(
        sub="EL3033",
        fif=r"I:\Shaked\ISO_data\control_clean\a_excluded\EL3033\EL3033_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif",
        hypno=r"I:\Shaked\ISO_data\scoring\young_control\EL3033.txt",
        hypno_freq=1.0,
        letter=False,
        detected="E81",
        undetected="E90",
    ),
]

LETTER_TO_INT = {"W": 0, "1": 1, "2": 2, "3": 3, "R": 4, "?": -1}


def load_hypno(path: str, letter: bool) -> np.ndarray:
    with open(path) as f:
        items = [ln.strip() for ln in f if ln.strip()]
    if letter:
        return np.array([LETTER_TO_INT[v] for v in items], dtype=int)
    return np.array([int(v) for v in items], dtype=int)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        sub = c["sub"]
        print(f"=== {sub} ===")
        hypno = load_hypno(c["hypno"], c["letter"])

        # Align hypno length to the recording: pad any trailing unscored stretch
        # (EL3033's scoring ends ~37 min before the recording, tail is artifact-only).
        raw_probe = mne.io.read_raw_fif(c["fif"], preload=False, verbose="error")
        target_len = int(round(raw_probe.n_times / raw_probe.info["sfreq"] * c["hypno_freq"]))
        if len(hypno) < target_len:
            pad = np.full(target_len - len(hypno), -1, dtype=int)  # -1 = unscored/art
            hypno = np.concatenate([hypno, pad])
            print(f"  padded hypno {len(hypno) - len(pad)} -> {len(hypno)} (trailing unscored)")
        elif len(hypno) > target_len:
            hypno = hypno[:target_len]

        sub_dir = OUT_ROOT / sub
        sub_dir.mkdir(parents=True, exist_ok=True)

        pipe = SpectralPipe(
            path_to_eeg=c["fif"],
            output_dir=sub_dir,
            hypno=hypno,
            hypno_freq=c["hypno_freq"],
        )
        print(f"  fif n_times={pipe.mne_raw.n_times}  sf={pipe.sf}  hypno_up={pipe.hypno_up.shape}")

        for kind in ("detected", "undetected"):
            pick = c[kind]
            plt.close("all")
            pipe.plot_hypnospectrogram(
                picks=(pick,),
                win_sec=10,
                freq_range=(0, 25),
                cmap="Spectral_r",
                overlap=True,
                save=False,
            )
            fig = plt.gcf()
            fig.suptitle(f"{sub} — {pick} (ISFS {kind})", y=1.02)
            out_path = OUT_ROOT / f"{sub}_{pick}_{kind}_hypnospectrogram.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
