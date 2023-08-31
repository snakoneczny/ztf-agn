import sys
import os
import pickle

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, LSTM
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH
from features import FEATURE_SETS

# Training params
filter = 'g'
batch_size = 32
epochs = 1000
early_stopping = 10
features_subset = ['ZTF', 'PS']

data_label = '_'.join(features_subset)
outputs_file_path = os.path.join(PROJECT_PATH, 'outputs/preds/{}__astromer_LSTM-512_FC-1024-512-256.csv'.format(data_label))


def build_model(encoder, n_classes, maxlen, train_astromer=True, lr=1e-3):
    serie = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input': serie, 'mask_in': mask, 'times': times}

    encoder.trainable = train_astromer

    # Fully connected
    # mask = 1. - placeholder['mask_in']
    # x = encoder(placeholder, training=False)
    # x = x * mask
    # x = tf.reduce_sum(x, 1) / tf.reduce_sum(mask, 1)

    # x = Dense(2048, activation='relu')(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = LayerNormalization()(x)
    # x = Dense(n_classes)(x)

    # LSTM
    mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))
    mask = tf.squeeze(mask, axis=-1)

    x = encoder(placeholder)
    x = tf.reshape(x, [-1, maxlen, encoder.output.shape[-1]])
    x = LayerNormalization()(x)

    # x = tf.concat([placeholder['times'], placeholder['input']], 2)

    dropout = 0.1
    x = LSTM(512, return_sequences=False, dropout=dropout, name='RNN_0')(x, mask=mask)
    x = LayerNormalization()(x)
    
    # x = LSTM(256, return_sequences=False, dropout=dropout, name='RNN_1')(x, mask=mask)
    # x = LayerNormalization()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    
    x = Dense(n_classes, name='FCN')(x)

    classifier = Model(inputs=placeholder, outputs=x, name='FCATT')
    classifier.compile(loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'], optimizer=Adam(lr))

    return classifier


# Read ZTF x SDSS lighcurves with available features
file_path = 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_{}__features_lc_reduced'.format(filter)
with open(os.path.join(DATA_PATH, file_path), 'rb') as file:
    ztf_x_sdss_reduced = pickle.load(file)

# Read ZTF x SDSS subset with available features
with open(os.path.join(DATA_PATH, 'ZTF_x_SDSS/ztf_20210401_x_specObj-dr18__singles_filter_{}__features'.format(filter)), 'rb') as file:
    ztf_x_sdss_features = pickle.load(file)

# Read SDSS x ZTF subset with available features
file_path = 'ZTF_x_SDSS/specObj-dr18_x_ztf_20210401__singles_filter_{}__features'
fp = file_path.format(filter)
with open(os.path.join(DATA_PATH, fp), 'rb') as file:
    sdss_x_ztf_features = pickle.load(file)

# Take subset with features
max_features = np.concatenate([FEATURE_SETS[label] for label in features_subset])
ztf_x_sdss_features = ztf_x_sdss_features.dropna(subset=max_features)
indices = ztf_x_sdss_features.index
ztf_x_sdss_features = ztf_x_sdss_features.reset_index(drop=True)
ztf_x_sdss_reduced = np.array(ztf_x_sdss_reduced)[indices.tolist()]
sdss_x_ztf_features = sdss_x_ztf_features.loc[indices].reset_index(drop=True)

# Change shape to feed a neural network
X = [np.array([np.array([lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]], dtype='object') for i in
               range(len(lc_dict['mjd']))], dtype='object') for lc_dict in tqdm(ztf_x_sdss_reduced)]

class_dict = {
    'GALAXY': 0,
    'QSO': 1,
    'STAR': 2,
}
y = sdss_x_ztf_features['CLASS'].apply(lambda x: class_dict[x]).to_list()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Load weights finetuned to our data
astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')
astromer.load_weights(os.path.join(PROJECT_PATH, 'outputs/models/astromer_g'))

# Create TF model
astromer_encoder = astromer.model.get_layer('encoder')
classifier = build_model(
    astromer_encoder, len(np.unique(y)), maxlen=astromer.maxlen, train_astromer=True,
)

train_batches = make_pretraining(
    X_train, labels=y_train, n_classes=3, batch_size=batch_size, shuffle=True,
    sampling=False, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
)

validation_batches = make_pretraining(
    X_val, labels=y_val, n_classes=3, batch_size=batch_size, shuffle=False,
    sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
)

history = classifier.fit(
    train_batches, validation_data=validation_batches, epochs=epochs,
    callbacks=[EarlyStopping(patience=early_stopping)],
)

# Make preds
y_pred = classifier.predict(validation_batches)
y_class = np.argmax(y_pred, 1)

# Save preds
df = pd.DataFrame(y_pred, columns=['GALAXY', 'QSO', 'STAR'])
num_to_class_dict = {v: k for k, v in class_dict.items()}
df['y_pred'] = [num_to_class_dict[y] for y in y_class]
df['y_true'] = [num_to_class_dict[y] for y in y_val]
df.to_csv(outputs_file_path, index=False)
print('Predictions saved to: {}'.format(outputs_file_path))
