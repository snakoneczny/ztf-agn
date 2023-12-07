import sys
import os
import h5py
import pickle

import numpy as np
import dask.array as da

sys.path.append('..')
from env_config import DATA_PATH
from features import get_astromer_embedding


# Read reduced lightcurves
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_lc_reduced'
with open(os.path.join(DATA_PATH, file_path), 'rb') as file:
    ztf_x_sdss_reduced = pickle.load(file)

# Get attention vectors
sys.stdout = open(os.devnull, 'w')
attention_vectors = get_astromer_embedding(ztf_x_sdss_reduced)
sys.stdout = sys.__stdout__

# file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_astromer-att.h5'
# da.to_hdf5(os.path.join(DATA_PATH, file_path), '/data', att)

# with h5py.File(os.path.join(DATA_PATH, file_path), 'w') as hf:
#     hf.create_dataset('astromer attention',  data=attention_vectors)

# Save attention vectors
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_g__features_astromer-att.npy'
np.save(os.path.join(DATA_PATH, file_path), attention_vectors)
