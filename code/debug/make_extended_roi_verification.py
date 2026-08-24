"""Regenerate the 256-ch ROI verification image using the extended ROI list
from config (single color, all 36 electrodes)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mne

from utils.config import EAR_ELECTRODES, EXTENDED_CENTRAL_PARIETAL_ROI, FACE_ELECTRODES, NECK_ELECTRODES
from debug.roi_128_to_256_mapping import plot_roi_topomap

EXCLUDED_256 = set(FACE_ELECTRODES + NECK_ELECTRODES + EAR_ELECTRODES)

montage = mne.channels.make_standard_montage('EGI_256')
pos = montage.get_positions()['ch_pos']
all_256 = [ch for ch in montage.ch_names if ch in pos and ch not in EXCLUDED_256]

roi = EXTENDED_CENTRAL_PARIETAL_ROI
out_path = os.path.join(os.path.dirname(__file__), 'roi_256_verification.png')

plot_roi_topomap(
    montage, roi, all_256,
    f'256-ch ROI ({len(roi)} electrodes)',
    out_path,
    ring_channels=None,
)
