import sys
import os
import argparse

from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import load_numpy

sys.path.append('..')
from env_config import PROJECT_PATH
from ztf import ZTF_DATES
from ml import read_train_matrices


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', help='ZTF filter, either g or r', required=True)
args = parser.parse_args()
filter = args.filter

# Training params
ztf_date = ZTF_DATES['DR 20']
batch_size = 32

output_weight_path = 'outputs/models/astromer__ztf_{}__band_{}'.format(ztf_date, filter)

# # Read the train data
# if ztf_date == '20210401':
#     ztf_x_sdss, sdss_x_ztf, _ = \
#         get_train_data(ztf_date=ztf_date, filter=args.filter, data_subsets=['ZTF'], return_features=True)
# else:
#     ztf_x_sdss, sdss_x_ztf = \
#         get_train_data(ztf_date=ztf_date, filter=args.filter, data_subsets=None, return_features=False)

# # Change shape to feed the neural network
# X_train, X_val, _, y_train, y_val, _ = get_train_matrices(ztf_x_sdss, sdss_x_ztf)

# Read the already saved matrices
X_train, X_val, _, y_train, y_val, _ = read_train_matrices(ztf_date, args.filter)

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
    train_batches, validation_batches, epochs=20, patience=5, lr=1e-3, verbose=0,
    project_path=os.path.join(PROJECT_PATH, output_weight_path),
)
