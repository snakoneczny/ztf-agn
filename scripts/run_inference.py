import sys
import os
import glob
import gc
import random
import argparse
import lzma

import pickle
import numpy as np
from tqdm import tqdm
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import make_pretraining
from scipy.special import softmax

sys.path.append('..')
from env_config import STORAGE_PATH, PROJECT_PATH
from utils import read_fits_to_pandas, save_fits
from light_curves import preprocess_ztf_light_curves
from astromer import build_model
from ztf import ZTF_DATES
from ml import subsample_matrix_row


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', required=True, help='ZTF filter')
parser.add_argument('-t', '--test', dest='is_test', help='Test run flag', action='store_true')
args = parser.parse_args()
filter = args.filter

# Input parameters
date = ZTF_DATES['DR 20']
test_size = 10000

# Get a list of all the downloaded fields
input_regex = 'ZTF/ZTF_{}/fields/ZTF_{}__field_*__{}-band*.xz'.format(date, date, filter)
input_paths = sorted(glob.glob(os.path.join(STORAGE_PATH, input_regex)))

# Get names of the desired output files
output_paths = [os.path.join(STORAGE_PATH, 'ZTF/ZTF_{}/catalog/{}.fits'.format(date, os.path.basename(input_file).split('.')[0]))
                    for input_file in input_paths]

# Get not processed files
idx = np.where([not os.path.exists(file) for file in output_paths])[0]
print('Processing {} out of {}'.format(len(idx), len(input_paths)))
output_paths = np.array(output_paths)[idx]
input_paths = np.array(input_paths)[idx]

# Read the model weights
astromer = SingleBandEncoder()
astromer = astromer.from_pretraining('ztfg')
astromer_encoder = astromer.model.get_layer('encoder')
classifier = build_model(astromer_encoder, n_outputs=3, maxlen=astromer.maxlen, train_astromer=False)
path_astromer = 'outputs/models/ZTF_{}/ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256'.format(date, date, filter)
classifier.load_weights(os.path.join(PROJECT_PATH, path_astromer))

for input_path, output_path in tqdm(list(zip(input_paths, output_paths)), 'Fields'):    
    # Read the input data
    with lzma.open(input_path, 'rb') as file:
        data = pickle.load(file)
    df = read_fits_to_pandas(input_path.split('.')[0] + '.fits')
    print('Input file: {}, size: {}, {}'.format(os.path.basename(input_path), len(data), df.shape))

    # Only run a test    
    if args.is_test:
        data = data[:test_size]
        df = df.head(test_size)

    # Make dictionaries
    data = [{'mjd': row[0], 'mag': row[1], 'magerr': row[2]} for row in data]

    # Process the lightcurves
    data, df = preprocess_ztf_light_curves(data, df, with_multiprocessing=False)

    # Collect data after these steps
    gc.collect()

    # For fields with number of observations lower than 20
    if len(data) > 0:
        # Make X
        random.seed(1257)
        X = [subsample_matrix_row(lc_dict) for lc_dict in tqdm(data, 'Input matrix')]

        # Make batches
        batch_size = 64
        batches = make_pretraining(
            X, labels=None, n_classes=3, batch_size=batch_size, shuffle=False,
            sampling=True, max_obs=200, msk_frac=0., rnd_frac=0., same_frac=0., repeat=1,
        )
        
        # Collect data after these steps
        gc.collect()
        
        # Make preds
        y_pred = classifier.predict(batches)
        y_pred = softmax(y_pred, axis=1)
        y_class = np.argmax(y_pred, 1)
        print('Predictions: ', np.unique(y_class, return_counts=True))

        # Add stuff to the DF
        df['mag'] = [np.median(lc_dict['mag']) for lc_dict in data]
        class_dict = {
            0: 'GALAXY',
            1: 'QSO',
            2: 'STAR',
        }
        df['CLASS'] = [class_dict[val] for val in y_class]
        for i, c in enumerate(['GALAXY', 'QSO', 'STAR']):
            df[c] = y_pred[:, i]

    # Save predictions
    if not args.is_test:
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        save_fits(df, output_path, overwrite=True, with_print=False)
    print('Predictions saved to: {}'.format(output_path))
