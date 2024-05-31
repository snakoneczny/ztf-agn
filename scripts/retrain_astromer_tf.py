import sys
import os
import argparse
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('..')
from env_config import PROJECT_PATH
from ml import get_train_matrices, read_train_matrices
from astromer import build_model
from ztf import ZTF_DATES


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', required=True, help='ZTF filter, either g or r')
parser.add_argument('-s', '--save', dest='with_save', action='store_true', help='Save the training history, model and predictions')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as a suffix to the experiment name')
parser.add_argument('--timespan', dest='timespan', type=int, help='timespan, used for sampling tests')
parser.add_argument('--frac_n_obs', dest='frac_n_obs', type=float, help='fraction of median number of observations')
parser.add_argument('--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()

# Data
ztf_date = ZTF_DATES['DR 20']

# Astromer pretraining
path_astromer = 'outputs/models/astromer__ztf_{}__band_{}'.format(ztf_date, args.filter)

# Training params
batch_size = 32

epochs = {
    'g': 15,
    'r': 20,
}[args.filter]

early_stopping = {
    'g': 5,
    'r': 10,
}[args.filter]

minimum_timespan = {
    '20210401': 1000,
    '20230821': 1800,
    '20240117': 1800,
}

# Test params
test_size = 10000
epochs = 3 if args.is_test else epochs

# Create an experiment name
experiment_name = 'ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256'.format(ztf_date, args.filter)
if args.tag:
    experiment_name += '__{}'.format(args.tag)

if args.timespan:
    # Read data and subsample light curves according to the input parameters
    X_train, X_val, X_test, y_train, y_val, y_test, n_obs_subsampled = get_train_matrices(
        ztf_date, args.filter, minimum_timespan=minimum_timespan[ztf_date],
        timespan=args.timespan, frac_n_obs=args.frac_n_obs,
    )
    
    # Add a tag to the experiment name
    experiment_name += '__timespan={}_p-nobs={}_nobs={}'.format(
        args.timespan, int(args.frac_n_obs * 100), n_obs_subsampled)

else:
    # Read the already saved matrices
    X_train, X_val, X_test, y_train, y_val, y_test = read_train_matrices(ztf_date, args.filter)

# Limit data size for testing
if args.is_test:
    X_train, X_val, X_test, y_train, y_val, y_test = \
        X_train[:test_size], X_val[:test_size], X_test[:test_size], \
        y_train[:test_size], y_val[:test_size], y_test[:test_size]

# Shuffle the training data and keep the shuffling index
np.random.seed(546)
index_rnd = np.random.permutation(len(X_train))
index_org = np.argsort(index_rnd)
X_train = [X_train[i] for i in index_rnd]
y_train = [y_train[i] for i in index_rnd]

# Collect any data left from the data processing
gc.collect()

# Make batches
batches_dict = {}
for X, y, label in [
    (X_train, y_train, 'train'),
    (X_val, y_val, 'val'),
    (X_test, y_test, 'test'),
]:
    batches_dict[label] = make_pretraining(
        X, labels=y, n_classes=3, batch_size=batch_size, shuffle=False,
        sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
    )

# Load weights fine tuned to our data
astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')
if path_astromer:
    astromer.load_weights(os.path.join(PROJECT_PATH, path_astromer))

# Create a TF model
astromer_encoder = astromer.model.get_layer('encoder')
classifier = build_model(astromer_encoder, n_classes=3, maxlen=astromer.maxlen, train_astromer=True,
                         lr=1e-3)

# Make callbacks
callbacks = [EarlyStopping(patience=early_stopping)]
if args.with_save:
    # Model Checkpoint
    model_file_path = os.path.join(PROJECT_PATH, 'outputs/models/ZTF_{}/{}'.format(ztf_date, experiment_name))
    callbacks.append(ModelCheckpoint(filepath=model_file_path, save_weights_only=True, save_best_only=True, verbose=1))
    # Tensorboard
    log_dir = os.path.join(PROJECT_PATH, 'outputs/tensorboard/{}'.format(experiment_name))
    callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch < 3:
        return 1e-3
    elif epoch < 8:
        return 5 * 1e-4
    else:
        return 1e-4
callbacks.append(LearningRateScheduler(scheduler))

# Fit the model
history = classifier.fit(
    batches_dict['train'], validation_data=batches_dict['val'],
    epochs=epochs, callbacks=callbacks,
)

# Iterate train and test batches, train needed for ensembling
for label, y_true in [('train', y_train), ('val', y_val), ('test', y_test)]:
    batches = batches_dict[label]

    # Make preds
    y_pred = classifier.predict(batches)

    # If train then undo the shuffling
    if label == 'train':
        y_pred = [y_pred[i] for i in index_org]
        y_true = [y_true[i] for i in index_org]
        
    # Get the class labels
    y_class = np.argmax(y_pred, 1)

    # Print simple statistics
    print(label)
    print('accuracy: ', np.round(accuracy_score(y_true, y_class), 2))
    print('f1: ', np.round(f1_score(y_true, y_class, average=None), 2))

    # Save preds
    if args.with_save:
        class_dict = {
            'GALAXY': 0,
            'QSO': 1,
            'STAR': 2,
        }
        
        df = pd.DataFrame(y_pred, columns=['GALAXY', 'QSO', 'STAR'])
        num_to_class_dict = {v: k for k, v in class_dict.items()}
        df['y_pred'] = [num_to_class_dict[y] for y in y_class]
        df['y_true'] = [num_to_class_dict[y] for y in y_true]

        outputs_file_name = 'outputs/preds/ZTF_{}/{}__{}.csv'.format(ztf_date, experiment_name, label)
        outputs_file_path = os.path.join(PROJECT_PATH, outputs_file_name)
        Path(os.path.dirname(outputs_file_path)).mkdir(parents=True, exist_ok=True)
        df.to_csv(outputs_file_path, index=False)
        print('Predictions {} saved to: {}'.format(label, outputs_file_path))
