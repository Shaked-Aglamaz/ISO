"""
Configuration file for EEG processing pipeline
Contains shared constants and settings used across multiple scripts
"""

# Base directory - change this path when moving between computers
BASE_DIR = "I:/Shaked/ISO_data"

# Electrode definitions for exclusion during processing
FACE_ELECTRODES = ['E238', 'E234', 'E230', 'E226', 'E225', 'E241', 'E244', 'E248', 'E252', 'E253', 'E242', 'E243', 'E245', 'E246', 'E249', 'E247', 'E250', 'E251', 'E255', 'E254', 
                   'E73', 'E54', 'E37', 'E32', 'E31', 'E18', 'E25', 'E61', 'E46', 'E67', 'E68', 'E239', 'E240', 'E235', 'E231', 'E236', 'E237', 'E232', 'E227', 'E210', 'E219', 'E220', 
                   'E1', 'E10', 'E218', 'E228', 'E233']

NECK_ELECTRODES = ['E145', 'E146', 'E147', 'E156', 'E165', 'E174', 'E166', 'E157', 'E148', 'E137', 'E136', 'E135', 'E134', 'E133']

EAR_ELECTRODES = ["E256", "E82", "E91", "E92", "E102", "E103", "E111", "E112", "E120", "E121",
                  "E229", "E217", "E216", "E209", "E208", "E200", "E199", "E188", "E187", "E175"]

# Central-parietal ROI (256-ch EGI system)
# Reverse-engineered from 128-ch YA_AUC.png hotspot via spatial mapping
# See code/debug/roi_128_to_256_mapping.py for derivation
CENTRAL_PARIETAL_ROI = [
    'E89', 'E130', 'E90', 'E100', 'E129', 'E101', 'E80', 'E131',
    'E88', 'E142', 'E79', 'E143', 'E81', 'E110', 'E128', 'E99',
    'E141', 'E119', 'E78', 'E154'
]

# Extended ROI: core ROI + one surrounding ring of electrodes
EXTENDED_CENTRAL_PARIETAL_ROI = CENTRAL_PARIETAL_ROI + [
    'E87', 'E153', 'E53', 'E144', 'E45', 'E132', 'E60', 'E155',
    'E118', 'E127', 'E109', 'E140', 'E98', 'E152', 'E77', 'E163'
]