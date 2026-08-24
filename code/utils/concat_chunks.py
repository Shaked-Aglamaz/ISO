"""
Concatenate channel-chunk FIF files into a single FIF per subject.

Loads all chunk files for a subject, concatenates along the channel axis,
saves the combined file, and deletes the chunk files.
"""

import gc
import os
import sys
import time

import mne
import numpy as np


OUTPUT_DIR = r"G:\Shaked_MCI_raw"

SUBJECTS = ["KS5", "SC5", "SM004", "SM006", "SM0016", "SM0017", "SM0018", "SM0019", "SM0020"]


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_chunk_files(subject_id):
    """Return sorted list of chunk FIF paths for a subject."""
    subject_dir = os.path.join(OUTPUT_DIR, subject_id)
    if not os.path.isdir(subject_dir):
        return []
    files = sorted(
        f for f in os.listdir(subject_dir)
        if f.startswith(f"{subject_id}_chunk_") and f.endswith("_raw.fif")
    )
    return [os.path.join(subject_dir, f) for f in files]


def concat_subject(subject_id):
    """Concatenate all chunk files for one subject into a single FIF."""
    chunk_paths = get_chunk_files(subject_id)
    if not chunk_paths:
        print(f"  {subject_id}: no chunk files found, skipping")
        return False

    print(f"  {subject_id}: {len(chunk_paths)} chunks")

    # Load all chunks (preload=True so we can access data arrays)
    raws = []
    all_ch_names = []
    for path in chunk_paths:
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        raws.append(raw)
        all_ch_names.extend(raw.ch_names)
        print(f"    Loaded {os.path.basename(path)}: {len(raw.ch_names)} ch")

    n_channels = len(all_ch_names)
    out_name = f"{subject_id}_{n_channels}-head-ch_resample250_raw.fif"
    out_path = os.path.join(OUTPUT_DIR, subject_id, out_name)
    if os.path.exists(out_path):
        print(f"  {subject_id}: combined file already exists, skipping")
        return True

    # Verify all chunks have the same duration and sfreq
    sfreqs = [r.info["sfreq"] for r in raws]
    n_times = [r.n_times for r in raws]
    if len(set(sfreqs)) != 1:
        raise ValueError(f"Mismatched sfreq across chunks: {sfreqs}")
    if len(set(n_times)) != 1:
        raise ValueError(f"Mismatched n_times across chunks: {n_times}")

    # Stack channel data into a single array
    data = np.concatenate([r.get_data() for r in raws], axis=0)

    # Build combined info
    sfreq = sfreqs[0]
    ch_types = []
    for r in raws:
        ch_types.extend([mne.channel_type(r.info, i) for i in range(len(r.ch_names))])
    info = mne.create_info(ch_names=all_ch_names, sfreq=sfreq, ch_types=ch_types)

    # Combine montages from all chunks into a single DigMontage
    ch_pos = {}
    lpa, rpa, nasion = None, None, None
    for r in raws:
        montage = r.get_montage()
        if montage:
            pos = montage.get_positions()
            ch_pos.update(pos["ch_pos"])
            if lpa is None:
                lpa = pos["lpa"]
                rpa = pos["rpa"]
                nasion = pos["nasion"]

    # Free chunk data
    for r in raws:
        del r
    del raws
    gc.collect()

    # Create combined Raw and set montage
    combined = mne.io.RawArray(data, info, verbose=False)
    if ch_pos:
        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos, lpa=lpa, rpa=rpa, nasion=nasion
        )
        combined.set_montage(montage, verbose=False)
    combined.save(out_path, overwrite=True, verbose=False)
    del combined, data
    gc.collect()

    print(f"    Saved: {out_path}")
    return True


def delete_chunks(subject_id):
    """Delete chunk files after successful concatenation."""
    chunk_paths = get_chunk_files(subject_id)
    for path in chunk_paths:
        os.remove(path)
    print(f"    Deleted {len(chunk_paths)} chunk files")


if __name__ == "__main__":
    keep_chunks = "--keep-chunks" in sys.argv

    print("=" * 70)
    print("Concatenating chunk FIF files per subject")
    print("=" * 70)

    succeeded, failed, skipped = [], [], []
    total_t0 = time.time()

    for subject_id in SUBJECTS:
        try:
            t0 = time.time()
            result = concat_subject(subject_id)
            if result:
                if not keep_chunks:
                    delete_chunks(subject_id)
                elapsed = time.time() - t0
                print(f"    Done in {format_duration(elapsed)}")
                succeeded.append(subject_id)
            else:
                skipped.append(subject_id)
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append(subject_id)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE in {format_duration(total_elapsed)}")
    print(f"  Succeeded: {', '.join(succeeded) or 'none'}")
    if skipped:
        print(f"  Skipped:   {', '.join(skipped)}")
    if failed:
        print(f"  Failed:    {', '.join(failed)}")
    print("=" * 70)
