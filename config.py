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