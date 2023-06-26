import sys
import os
import pickle

import numpy as np

sys.path.append('..')
from env_config import DATA_PATH
from features import get_astromer_features

# Read reduced lightcurves
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_lc_reduced'
with open(os.path.join(DATA_PATH, file_path), 'rb') as file:
    ztf_x_sdss_reduced = pickle.load(file)

# Get features
sys.stdout = open(os.devnull, 'w')
features = get_astromer_features(ztf_x_sdss_reduced)
sys.stdout = sys.__stdout__

# Save features
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_astromer.npy'
np.save(os.path.join(DATA_PATH, file_path), features)
