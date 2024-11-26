import sys
import os
import pickle
import gc
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from env_config import PROJECT_PATH, STORAGE_PATH
from features import add_colors, get_features
from ml import get_file_name


# Input params
tasks = ['clf', 'z']
feature_keys = {
    'clf': ['ZTF', 'AstrmClf', 'WISE'],
    'z': ['ZTF', 'WISE'],
}

# Input config
chunk_size = 100000
input_regex = 'ZTF/ZTF_20240117/catalog_v2_wise_chunks/*'
ztf_file_path = 'ZTF/ZTF_20240117/catalog_ZTF_20240117_g-band__v2.csv'
output_file = 'ZTF/ZTF_20240117/catalog_v2_xgb_chunks/catalog_ZTF_20240117_g-band__v2_XGB__{}.csv'

# Read ZTF data
file_path = os.path.join(STORAGE_PATH, ztf_file_path)
columns = ['mag median', 'GALAXY', 'QSO', 'STAR']
df_ztf = pd.read_csv(file_path, usecols=columns)
df_ztf.rename(columns={'GALAXY': 'astrm_galaxy', 'QSO': 'astrm_qso', 'STAR': 'astrm_star'}, inplace=True)

# Load models for classification and redshifts
models = {}
for task in tasks:
    model_file_name = get_file_name('model', task, '20240117', 'g', '_'.join(feature_keys[task]),
                                    mag_limit=False, cross_features=False,
                                    qso_vs_rest=False, n_obs_limit=False)
    file_path = os.path.join(PROJECT_PATH, model_file_name)
    with open(file_path, 'rb') as file:
        models[task] = pickle.load(file)
    print('Model read: {}'.format(file_path))

# For each WISE chunk
input_paths = sorted(glob.glob(os.path.join(STORAGE_PATH, input_regex)))
for input_path in tqdm(input_paths, 'Running inference on WISE chunks'):
    chunk_start = int(input_path.split('__')[-1].split('.')[0])
    file_path = os.path.join(STORAGE_PATH, output_file.format(chunk_start))

    if not os.path.exists(file_path):
        # Add ZTF features to a WISE chunk
        df = pd.read_csv(input_path)
        df = pd.concat([df, df_ztf[chunk_start:chunk_start + chunk_size].reset_index(drop=True)], axis=1)

        # For classification and redshift tasks
        df, features_dict = add_colors(df, is_cross_features=False)
        results = pd.DataFrame()
        for task in ['clf', 'z']:
            # Make X matrix
            features = get_features(features_dict, feature_keys[task], is_cross_features=False)
            X = df[features].values    
            
            # Run inference
            if task == 'z':
                results['z_pred'] = models[task].predict(X)
            else:
                y_pred_proba = models[task].predict_proba(X)
                y_pred_cls = np.argmax(y_pred_proba, axis=1)
                
                results['y_pred'] = y_pred_cls
                results[['y_galaxy', 'y_qso', 'y_star']] = y_pred_proba

        # Save chunk files
        results.to_csv(file_path, index=False)
        gc.collect()
