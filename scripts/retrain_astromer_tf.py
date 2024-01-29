import sys
import os
import pickle
import argparse
import random

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH
from features import FEATURES_DICT
from astromer_models import build_model


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', required=True, help='ZTF filter, either g or r')
parser.add_argument('-p', '--ps', dest='is_ps', action='store_true', help='Flag wether use cross-match with PS')
parser.add_argument('-w', '--wise', dest='is_wise', action='store_true', help='Flag wether use cross-match with WISE')
parser.add_argument('-g', '--gaia', dest='is_gaia', action='store_true', help='Flag wether use cross-match with GAIA')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as a suffix to the experiment name')
parser.add_argument('--timespan', dest='timespan', type=int, help='timespan, used for sampling tests')
parser.add_argument('--frac_n_obs', dest='frac_n_obs', type=float, help='fraction of median number of observations')
parser.add_argument('--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()

# Flag for reduced data processing
is_test = args.is_test

# Training params
filter = args.filter

data_subsets = ['ZTF']  # ZTF, PS, WISE
if args.is_ps:
    data_subsets.append('PS')
if args.is_wise:
    data_subsets.append('WISE')
if args.is_gaia:
    data_subsets.append('GAIA')

feature_labels = []  # ZTF, PS, WISE
data_label = '_'.join(data_subsets)
feature_label = 'ftrs_' + '_'.join(feature_labels)

batch_size = 32
epochs = 1000 if not is_test else 8
early_stopping = 5 if args.filter == 'g' else 10
model_type = 'FC'

experiment_name = '{}-band__{}__astromer_FC-1024-512-256'.format(filter, data_label)
# experiment_name = '{}__LSTM-256-256'.format(data_label)
# experiment_name = '{}__CNN-16-32-64-64_FC-1024-512-256'.format(data_label)

if len(feature_labels) > 0:
    experiment_name += '__{}'.format(feature_label)
if args.tag:
    experiment_name += '__{}'.format(args.tag)

# Read ZTF x SDSS lightcurves with available features
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_{}__features_lc-reduced'.format(filter)
with open(os.path.join(DATA_PATH, file_path), 'rb') as file:
    ztf_x_sdss_reduced = pickle.load(file)

# Read ZTF x SDSS subset with available features
with open(os.path.join(DATA_PATH, 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_{}__features'.format(filter)),
          'rb') as file:
    ztf_x_sdss_features = pickle.load(file)

# Read SDSS x ZTF subset with available features
file_path = 'ZTF_x_SDSS/specObj-dr18_x_ztf_20210401__singles_filter_{}__features'
fp = file_path.format(filter)
with open(os.path.join(DATA_PATH, fp), 'rb') as file:
    sdss_x_ztf_features = pickle.load(file)

# Take subset with features
features_list = np.concatenate([FEATURES_DICT[label] for label in data_subsets])
ztf_x_sdss_features = ztf_x_sdss_features.dropna(subset=features_list)
indices = ztf_x_sdss_features.index
ztf_x_sdss_features = ztf_x_sdss_features.reset_index(drop=True)
ztf_x_sdss_reduced = np.array(ztf_x_sdss_reduced)[indices.tolist()]
sdss_x_ztf_features = sdss_x_ztf_features.loc[indices].reset_index(drop=True)

# Get limited subset if testing
to_process = ztf_x_sdss_reduced[:1000] if is_test else ztf_x_sdss_reduced

# Get sampling test
if args.timespan:    

    # Subset of minimal timespan
    minimal_timespan = 1000
    idx_timespan = [i for i in range(len(to_process)) if to_process[i]['mjd'][-1] - to_process[i]['mjd'][0] >= minimal_timespan]
    to_process = to_process[idx_timespan]
    sdss_x_ztf_features = sdss_x_ztf_features.loc[idx_timespan].reset_index(drop=True)

    # Truncate lightcurves to the exact timespan
    for i in range(len(to_process)):
        mjds = to_process[i]['mjd']
        idx_start = 0
        while (mjds[-1] - mjds[idx_start]) > args.timespan:
            idx_start += 1
        for dict_key in ['mjd', 'mag', 'magerr']:
            to_process[i][dict_key] = to_process[i][dict_key][idx_start:]

    # Find median number of observations
    n_obs_arr = [len(lc_dict['mjd']) for lc_dict in to_process]
    n_obs_median = np.median(n_obs_arr)

    # Subsample minimal number of observations
    idx_median = [i for i in range(len(to_process)) if len(to_process[i]['mjd']) >= n_obs_median]
    to_process = to_process[idx_median]
    sdss_x_ztf_features = sdss_x_ztf_features.loc[idx_median].reset_index(drop=True)

    # Sample a fraction of observations
    n_obs_goal = int(args.frac_n_obs * n_obs_median)
    random.seed(7235)
    for i in range(len(to_process)):
        idx_goal = sorted(random.sample(range(len(to_process[i]['mjd'])), n_obs_goal))
        for dict_key in ['mjd', 'mag', 'magerr']:
            to_process[i][dict_key] = to_process[i][dict_key][idx_goal]
        
    # Add a tag to the experiment name
    experiment_name += '__timespan={}_p-nobs={}_nobs={}'.format(
        args.timespan, int(args.frac_n_obs * 100), n_obs_goal)

# Change shape to feed a neural network and sample random 200 observations
random.seed(1257)
X = [np.array([np.array([lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]], dtype='object') for i in
               (range(len(lc_dict['mjd'])) if len(lc_dict['mjd']) <= 200 else sorted(random.sample(range(len(lc_dict['mjd'])), 200)))],
              dtype='object') for lc_dict in tqdm(to_process)]

# ztf_x_sdss_features, feature_sets = add_colors(ztf_x_sdss_features)
# X_features = ztf_x_sdss_features[np.concatenate([feature_sets[label] for label in feature_labels])]

class_dict = {
    'GALAXY': 0,
    'QSO': 1,
    'STAR': 2,
}
y = sdss_x_ztf_features['CLASS'].apply(lambda x: class_dict[x]).to_list()

if is_test:
    y = y[:1000]
    # X_features = X_features[:1000]

# X_train, X_val, X_f_train, X_f_val, y_train, y_val = train_test_split(
#     X, X_features, y, test_size=0.33, random_state=42
# )
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Standardize features
# scaler = MinMaxScaler()
# X_f_train = scaler.fit_transform(X_f_train)
# X_f_val = scaler.transform(X_f_val)

# Load weights finetuned to our data
astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')
astromer.load_weights(os.path.join(PROJECT_PATH, 'outputs/models/astromer_{}'.format(filter)))

# Create TF model
astromer_encoder = astromer.model.get_layer('encoder')
classifier = build_model(model_type, encoder=astromer_encoder, n_classes=len(np.unique(y)), maxlen=astromer.maxlen,
                         train_astromer=True)  # n_features=X_f_train.shape[1]

train_batches = make_pretraining(
    X_train, labels=y_train, n_classes=3, batch_size=batch_size, shuffle=False,
    sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
)  # features=X_f_train, num_cls=3

validation_batches = make_pretraining(
    X_val, labels=y_val, n_classes=3, batch_size=batch_size, shuffle=False,
    sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
)  # features=X_f_val, num_cls=3

# Make callbacks
callbacks = [EarlyStopping(patience=early_stopping)]
if not is_test:
    log_dir = os.path.join(PROJECT_PATH, 'outputs/tensorboard/{}'.format(experiment_name))
    callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

history = classifier.fit(
    train_batches, validation_data=validation_batches, epochs=epochs, callbacks=callbacks,
)

# Iterate train and validation batches, train needed for ensembling
for label, batches, y_true in [('train', train_batches, y_train), ('val', validation_batches, y_val)]:

    # Make preds
    y_pred = classifier.predict(batches)
    y_class = np.argmax(y_pred, 1)

    # Save preds
    df = pd.DataFrame(y_pred, columns=['GALAXY', 'QSO', 'STAR'])
    num_to_class_dict = {v: k for k, v in class_dict.items()}
    df['y_pred'] = [num_to_class_dict[y] for y in y_class]
    df['y_true'] = [num_to_class_dict[y] for y in y_true]

    outputs_file_path = os.path.join(PROJECT_PATH, 'outputs/preds/{}__{}.csv'.format(experiment_name, label))
    df.to_csv(outputs_file_path, index=False)
    print('Predictions {} saved to: {}'.format(label, outputs_file_path))
