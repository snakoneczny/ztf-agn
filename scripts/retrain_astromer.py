import sys
import os
import pickle
import argparse
import random

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import load_numpy

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', help='ZTF filter, either g or r', required=True)
args = parser.parse_args()
filter = args.filter

output_weight_path = 'outputs/models/astromer_{}'.format(filter)

batch_size = 32

# Read ZTF x SDSS lightcurves subset with available features
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_{}__features_lc-reduced'.format(filter)
with open(os.path.join(DATA_PATH, file_path), 'rb') as file:
    ztf_x_sdss_reduced = pickle.load(file)

# Read SDSS x ZTF subset with available features
file_path = 'ZTF_x_SDSS/specObj-dr18_x_ztf_20210401__singles_filter_{}__features'
fp = file_path.format(filter)
with open(os.path.join(DATA_PATH, fp), 'rb') as file:
    sdss_x_ztf_features = pickle.load(file)

# Get only subset of AGNs
# sdss_x_ztf_features = sdss_x_ztf_features.loc[sdss_x_ztf_features['CLASS'] == 'QSO']
# indices = sdss_x_ztf_features.index
# ztf_x_sdss_reduced = np.array(ztf_x_sdss_reduced)[indices.tolist()]

# Change shape to feed a neural network
random.seed(1257)
X = [np.array([np.array([lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]], dtype='object') for i in
               (range(len(lc_dict['mjd'])) if len(lc_dict['mjd']) <= 200 else sorted(random.sample(range(len(lc_dict['mjd'])), 200)))],
              dtype='object') for lc_dict in tqdm(ztf_x_sdss_reduced, desc='Input matrix')]

class_dict = {
    'GALAXY': 0,
    'QSO': 1,
    'STAR': 2,
}
y = sdss_x_ztf_features['CLASS'].apply(lambda x: class_dict[x]).to_numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_batches = load_numpy(
    X_train, labels=y_train, batch_size=batch_size, shuffle=True, sampling=True, max_obs=200,
    msk_frac=0.5, rnd_frac=0.1, same_frac=0.1, repeat=1,
)

validation_batches = load_numpy(
    X_val, labels=y_val, batch_size=batch_size, shuffle=True, sampling=True, max_obs=200,
    msk_frac=0.5, rnd_frac=0.1, same_frac=0.1, repeat=1,
)

astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')

astromer.fit(
    train_batches, validation_batches, epochs=1000, patience=10, lr=1e-3, verbose=0,
    project_path=os.path.join(PROJECT_PATH, output_weight_path),
)
