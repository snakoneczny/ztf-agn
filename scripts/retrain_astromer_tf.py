import sys
import os
import argparse
import random

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('..')
from env_config import PROJECT_PATH
from ml import get_train_data
from light_curves import subsample_light_curves
from astromer import build_model


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', required=True, help='ZTF filter, either g or r')
parser.add_argument('-p', '--ps', dest='is_ps', action='store_true', help='Flag wether use cross-match with PS')
parser.add_argument('-w', '--wise', dest='is_wise', action='store_true', help='Flag wether use cross-match with WISE')
parser.add_argument('-g', '--gaia', dest='is_gaia', action='store_true', help='Flag wether use cross-match with GAIA')
parser.add_argument('-s', '--save', dest='with_save', action='store_true', help='Save the model')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as a suffix to the experiment name')
parser.add_argument('--timespan', dest='timespan', type=int, help='timespan, used for sampling tests')
parser.add_argument('--frac_n_obs', dest='frac_n_obs', type=float, help='fraction of median number of observations')
parser.add_argument('--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()

# Data
ztf_date = '20230821'  # '20210401', '20230821'

# Astromer pretraining
path_astromer = 'outputs/models/astromer__ztf_{}__band_{}'.format(ztf_date, args.filter)

# Training params
batch_size = 32
epochs = 1000 if not args.is_test else 3
early_stopping = 15 if args.filter == 'g' else 20
minimum_timespan = {
    '20210401': 1000,
    '20230821': 1800,
}
model_type = 'FC'
test_size = 10000

# Create a data subsets label
data_subsets = ['ZTF']
if args.is_ps:
    data_subsets.append('PS')
if args.is_wise:
    data_subsets.append('WISE')
if args.is_gaia:
    data_subsets.append('GAIA')
data_label = '_'.join(data_subsets)

# Create an experiment name
experiment_name = 'ZTF_{}__band_{}__xmatch_{}__astromer_FC-1024-512-256'.format(ztf_date, args.filter, data_label)
if args.tag:
    experiment_name += '__{}'.format(args.tag)

# Read the train data
return_features = True if ztf_date == '20210401' else False
if ztf_date == '20210401':
    ztf_x_sdss_reduced, sdss_x_ztf_features, ztf_x_sdss_features = \
        get_train_data(ztf_date=ztf_date, filter=args.filter, data_subsets=data_subsets, return_features=True)
else:
    ztf_x_sdss_reduced, sdss_x_ztf_features = \
        get_train_data(ztf_date=ztf_date, filter=args.filter, data_subsets=data_subsets, return_features=False)

# Get limited subset if testing
to_process = ztf_x_sdss_reduced[:test_size] if args.is_test else ztf_x_sdss_reduced

# Make sampling experiment
if args.timespan:
    to_process, sdss_x_ztf_features, n_obs_subsampled = subsample_light_curves(
        to_process, sdss_x_ztf_features, minimum_timespan=minimum_timespan[ztf_date],
        timespan=args.timespan, frac_n_obs=args.frac_n_obs)

    # Add a tag to the experiment name
    experiment_name += '__timespan={}_p-nobs={}_nobs={}'.format(
        args.timespan, int(args.frac_n_obs * 100), n_obs_subsampled)

# Change shape to feed a neural network and sample random 200 observations
random.seed(1257)
X = [np.array([np.array([lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]], dtype='object') for i in
               (range(len(lc_dict['mjd'])) if len(lc_dict['mjd']) <= 200 else sorted(random.sample(range(len(lc_dict['mjd'])), 200)))],
              dtype='object') for lc_dict in tqdm(to_process)]

class_dict = {
    'GALAXY': 0,
    'QSO': 1,
    'STAR': 2,
}
y = sdss_x_ztf_features['CLASS'].apply(lambda x: class_dict[x]).to_list()

if args.is_test:
    y = y[:test_size]
    sdss_x_ztf_features = sdss_x_ztf_features.head(test_size)

idx_train = np.where(sdss_x_ztf_features['is_test'] == False)[0]
idx_val = np.where(sdss_x_ztf_features['is_test'] == True)[0]
X_train = [X[i] for i in idx_train]
X_val =   [X[i] for i in idx_val]
y_train = [y[i] for i in idx_train]
y_val =   [y[i] for i in idx_val]

# Load weights finetuned to our data
astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')
astromer.load_weights(os.path.join(PROJECT_PATH, path_astromer))

# Create TF model
astromer_encoder = astromer.model.get_layer('encoder')
classifier = build_model(model_type, encoder=astromer_encoder, n_classes=len(np.unique(y)), maxlen=astromer.maxlen,
                         train_astromer=True)

# Shuffle training data and keep the suffling index
np.random.seed(546)
index_rnd = np.random.permutation(len(X_train))
index_org = np.argsort(index_rnd)
X_train = [X_train[i] for i in index_rnd]
y_train = [y_train[i] for i in index_rnd]

train_batches = make_pretraining(
    X_train, labels=y_train, n_classes=3, batch_size=batch_size, shuffle=False,
    sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
)

validation_batches = make_pretraining(
    X_val, labels=y_val, n_classes=3, batch_size=batch_size, shuffle=False,
    sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
)

# Make callbacks
callbacks = [EarlyStopping(patience=early_stopping)]
if not args.is_test:
    log_dir = os.path.join(PROJECT_PATH, 'outputs/tensorboard/{}'.format(experiment_name))
    callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
if args.with_save:
    file_name = 'outputs/models/ZTF_{}/{}'.format(ztf_date, experiment_name)
    model_file_path = os.path.join(PROJECT_PATH, file_name)
    callbacks.append(ModelCheckpoint(filepath=model_file_path, save_weights_only=True, save_best_only=True, verbose=1))

# Fit the model
history = classifier.fit(
    train_batches, validation_data=validation_batches, epochs=epochs, callbacks=callbacks,
)

# Iterate train and validation batches, train needed for ensembling
for label, batches, y_true in [('train', train_batches, y_train), ('val', validation_batches, y_val)]:
    # Make preds
    y_pred = classifier.predict(batches)
    # If train then undo the shuffling
    if label == 'train':
        y_pred = [y_pred[i] for i in index_org]
        y_true = [y_true[i] for i in index_org]
    # Get class labels
    y_class = np.argmax(y_pred, 1)

    print(label)
    print('accuracy: ', np.round(accuracy_score(y_true, y_class), 2))
    print('f1: ', np.round(f1_score(y_true, y_class, average=None), 2))

    # Save preds
    if not args.is_test:
        df = pd.DataFrame(y_pred, columns=['GALAXY', 'QSO', 'STAR'])
        num_to_class_dict = {v: k for k, v in class_dict.items()}
        df['y_pred'] = [num_to_class_dict[y] for y in y_class]
        df['y_true'] = [num_to_class_dict[y] for y in y_true]

        file_name = 'outputs/preds/ZTF_{}/{}__{}.csv'.format(ztf_date, experiment_name, label)
        outputs_file_path = os.path.join(PROJECT_PATH, file_name)
        df.to_csv(outputs_file_path, index=False)
        print('Predictions {} saved to: {}'.format(label, outputs_file_path))
