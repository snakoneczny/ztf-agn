import sys
import os
import pickle

import numpy as np
import tensorflow as tf

sys.path.append('..')
from env_config import DATA_PATH
from features import get_astromer_features


output_file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_astromer_retrained_QSO.npy'

# Run without GPU, probably due to batch size problem it's faster
tf.config.set_visible_devices([], 'GPU')

# Read reduced lightcurves
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_lc_reduced'
with open(os.path.join(DATA_PATH, file_path), 'rb') as file:
    ztf_x_sdss_reduced = pickle.load(file)

# Get features
features = get_astromer_features(ztf_x_sdss_reduced, retrained='QSO')

# Save features
np.save(os.path.join(DATA_PATH, output_file_path), features)
