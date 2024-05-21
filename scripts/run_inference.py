import sys
import os
import glob
import gc
import random
import argparse

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
from scipy.special import softmax

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH
from light_curves import preprocess_ztf_light_curves
from astromer import build_model
from ztf import ZTF_DATES


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', required=True, help='ZTF filter')
parser.add_argument('-t', '--test', dest='is_test', help='Test run flag', action='store_true')
args = parser.parse_args()
filter = args.filter

date = ZTF_DATES['DR 20']

# Get a list of all the downloaded fields
input_regex = 'ZTF/ZTF_{}/fields/ZTF_{}__field_*__{}-band'.format(date, date, filter)
input_paths = sorted(glob.glob(os.path.join(DATA_PATH, input_regex)))

# Get names of the desired output files
output_paths = [os.path.join(DATA_PATH, 'ZTF/ZTF_{}/catalog/{}.csv'.format(date, os.path.basename(input_file)))
                    for input_file in input_paths]

# Get not processed files
idx = np.where([not os.path.exists(file) for file in output_paths])[0]
output_paths = np.array(output_paths)[idx]
input_paths = np.array(input_paths)[idx]

# Read the model weights
astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')
astromer_encoder = astromer.model.get_layer('encoder')
classifier = build_model(astromer_encoder, n_classes=3, maxlen=astromer.maxlen, train_astromer=False)
path_astromer = 'outputs/models/ZTF_{}/ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256'.format(date, date, filter)
classifier.load_weights(os.path.join(PROJECT_PATH, path_astromer))

for input_path, output_path in tqdm(list(zip(input_paths, output_paths)), 'Fields'):
    # Read the input data
    with open(input_path, 'rb') as file:
        data = pickle.load(file)
    print('Input file: {}, size: {}'.format(os.path.basename(input_path), len(data)))
    
    if args.is_test:
        data = data[:1000]
        
    # Process the lightcurves
    data = [lc_dict for lc_dict in data if len(lc_dict['mjd']) > 20]
    data = preprocess_ztf_light_curves(data)

    # Make X
    random.seed(1257)
    X = [np.array([np.array([lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]], dtype='object') for i in
                (range(len(lc_dict['mjd'])) if len(lc_dict['mjd']) <= 200 else sorted(random.sample(range(len(lc_dict['mjd'])), 200)))],
                dtype='object') for lc_dict in tqdm(data, 'X matrix')]

    # Make batches
    batch_size = 64
    batches = make_pretraining(
        X, labels=None, n_classes=3, batch_size=batch_size, shuffle=False,
        sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
    )
    
    # Collect data from the previous iteration
    gc.collect()

    # Make preds
    y_pred = classifier.predict(batches)
    y_pred = softmax(y_pred, axis=1)
    y_class = np.argmax(y_pred, 1)
    print('Predictions: ', np.unique(y_class, return_counts=True))

    # Make a results dataframe
    df = pd.DataFrame()
    for column in ['id', 'ra', 'dec']:
        if column in data[0].keys():
            df[column] = [lc_dict[column] for lc_dict in data]
    df['CLASS'] = y_class
    for i, c in enumerate(['GALAXY', 'QSO', 'STAR']):
        df[c] = y_pred[:, i]
            
    # Save predictions
    if not args.is_test:
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        df.to_csv(output_path, index=False)
        print('Predictions saved to: {}'.format(output_path))
