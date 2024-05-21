import sys
import os
import pickle
import gc

import numpy as np
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH
from sdss import read_sdss
from light_curves import get_longests
from ztf import ZTF_DATES


# Input parameters
date = ZTF_DATES['DR 20']

# Make a list of all chunk files
chunk_size = 10000
file_name = 'ZTF_x_SDSS/chunks_{}/ztf_{}_x_specObj-dr18__{}-{}'
file_path = os.path.join(DATA_PATH, file_name)

# Find paths to all chunks
i = 0
files = []
while True:
    path_full = file_path.format(date, date, i * chunk_size, (i+1) * chunk_size)
    if os.path.exists(path_full):
        files.append(path_full)
        i += 1
    else:
        break

# Read and merge chunks into one list
data_ztf = []
for file_path in tqdm(files):
    with open(file_path, 'rb') as file:
        data_ztf.extend(pickle.load(file))
data_ztf = np.array(data_ztf)
print('Chunks merged: {}'.format(len(data_ztf)))

# Read the corresponding SDSS file
data_sdss, data_features = read_sdss(dr=18, clean=True, return_cross_matches=True)

# Get singles
data_ztf, data_sdss, data_features = get_longests(data_ztf, data_sdss, data_features)
gc.collect()
print('Longests: g {}, r {}'.format(data_sdss['g'].shape[0], data_sdss['r'].shape[0]))

# Save data
for filter in ['g', 'r']:
    # ZTF
    file_name = 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__longests_filter_{}'.format(date, date, filter)
    file_path = os.path.join(DATA_PATH, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data_ztf[filter], file)
    print('Light curves saved to: ' + file_path)

    # SDSS
    file_name = 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_x_ztf_{}__longests_filter_{}'.format(date, date, filter)
    file_path = os.path.join(DATA_PATH, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data_sdss[filter], file)
    print('SDSS saved to: ' + file_path)

    # Features
    file_name = 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_PWG_x_ztf_{}__longests_filter_{}'.format(date, date, filter)
    file_path = os.path.join(DATA_PATH, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data_features[filter], file)
    print('Features saved to: ' + file_path)
