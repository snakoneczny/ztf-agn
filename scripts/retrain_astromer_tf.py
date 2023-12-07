import sys
import os
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH
from features import FEATURES_DICT, add_colors
from astromer_preprocessing import load_numpy

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', help='ZTF filter, either g or r', required=True)
parser.add_argument('-p', '--ps', dest='is_ps', help='Flag wether use cross-match with PS', action='store_true')
parser.add_argument('-w', '--wise', dest='is_wise', help='Flag wether use cross-match with WISE', action='store_true')
parser.add_argument('-g', '--gaia', dest='is_gaia', help='Flag wether use cross-match with GAIA', action='store_true')
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as a suffix to the experiment name')
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
early_stopping = 20
model_type = 'FC'

experiment_name = '{}-band__{}__astromer_FC-1024-512-256'.format(filter, data_label)
# experiment_name = '{}__LSTM-256-256'.format(data_label)
# experiment_name = '{}__CNN-16-32-64-64_FC-1024-512-256'.format(data_label)

if len(feature_labels) > 0:
    experiment_name += '__{}'.format(feature_label)
if args.tag:
    experiment_name += '__{}'.format(args.tag)


def get_convo_layers(placeholder, encoder=None, n_classes=3, maxlen=200):
    # Without astromer
    # x = tf.concat([placeholder['times'], placeholder['input']], 2)

    # With astromer
    # x = encoder(placeholder)
    # x = tf.reshape(x, [-1, maxlen, encoder.output.shape[-1]])
    # x = LayerNormalization()(x)

    x = Conv1D(16, 3, activation='relu')(placeholder['input'])
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)

    x = Dense(n_classes, name='output')(x)

    return x


def get_lstm_layers(placeholder, encoder=None, n_classes=3, maxlen=200):
    mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))
    mask = tf.squeeze(mask, axis=-1)

    # Without astromer
    x = tf.concat([placeholder['times'], placeholder['input']], 2)

    # With astromer
    # x = encoder(placeholder)
    # x = tf.reshape(x, [-1, maxlen, encoder.output.shape[-1]])
    # x = LayerNormalization()(x)

    dropout = 0.3
    x = LSTM(256, return_sequences=True, dropout=dropout, name='LSTM_0')(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(256, return_sequences=False, dropout=dropout, name='LSTM_1')(x, mask=mask)
    x = LayerNormalization()(x)

    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = LayerNormalization()(x)

    x = Dense(n_classes, name='output')(x)

    return x


def get_fc_layers(placeholder, encoder=None, n_classes=3):
    mask = 1. - placeholder['mask_in']
    x = encoder(placeholder, training=False)  # training flag here controls the dropout
    x = x * mask

    x = tf.reduce_sum(x, 1) / tf.reduce_sum(mask, 1)
    # x = tf.concat([
    #         tf.reduce_sum(x, 1) / tf.reduce_sum(mask, 1),
    #         tf.reduce_max(x, 1),
    # ], 1)

    # x = tf.concat([x, placeholder['features']], 1)    
    # x = LayerNormalization()(x)

    x = Dense(1024, activation='relu')(x)
    # x = LayerNormalization()(x)
    x = Dense(512, activation='relu')(x)
    # x = LayerNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    
    x = Dense(n_classes, name='output')(x)

    return x


def build_model(model_type, encoder=None, n_classes=3, maxlen=200, n_features=None, train_astromer=True, lr=1e-3):
    serie = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    # features = Input(shape=n_features, batch_size=None, name='features')

    placeholder = {'input': serie, 'mask_in': mask, 'times': times}  # , 'features': features}

    encoder.trainable = train_astromer

    if model_type == 'LSTM':
        x = get_lstm_layers(placeholder, encoder, n_classes, maxlen)
    elif model_type == 'CNN':
        x = get_convo_layers(placeholder, encoder, n_classes, maxlen)
    else:
        x = get_fc_layers(placeholder, encoder, n_classes)

    classifier = Model(inputs=placeholder, outputs=x, name='FCATT')
    classifier.compile(
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        optimizer=Adam(lr)
    )

    return classifier


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

# Change shape to feed a neural network
to_process = ztf_x_sdss_reduced[:1000] if is_test else ztf_x_sdss_reduced
X = [np.array([np.array([lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]], dtype='object') for i in
               range(len(lc_dict['mjd']))], dtype='object') for lc_dict in tqdm(to_process)]

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
