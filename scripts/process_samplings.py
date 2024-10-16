import sys
import os
import gc
import glob
import json

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('..')
from env_config import PROJECT_PATH
from ztf import ZTF_DATES, LIMITING_MAGS
from ml import get_train_matrices

# Input params and paths
ztf_date = ZTF_DATES['DR 20']
filter_name = 'g'

# Get all sampling files
exp_name = 'ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256'.format(ztf_date, filter_name)
file_pattern = '{}__timespan=*_p-nobs=*_nobs=*__test.csv'.format(exp_name)
file_list = sorted(glob.glob(os.path.join(PROJECT_PATH, 'outputs/preds/ZTF_20240117', file_pattern)))

# Get not processed files
to_process = []
for file_name in file_list:
    f_name = os.path.basename(file_name)
    splitted = f_name.split('__')[-2].split('_')
    timespan = int(splitted[0].split('=')[-1])
    p_obs = int(splitted[1].split('=')[-1])
    n_obs = int(splitted[2].split('=')[-1])
    
    output_file = 'ZTF_{}/ZTF_{}__band_g__xmatch_ZTF__astromer_FC-1024-512-256__timespan={}_p-nobs={}_nobs={}.json'
    output_file = output_file.format(ztf_date, ztf_date, timespan, p_obs, n_obs)
    output_file = os.path.join(PROJECT_PATH, 'outputs/sampling', output_file)
    if not os.path.exists(output_file):
        to_process.append([file_name, output_file, timespan, p_obs, n_obs])

# Read experiments
for file_in, file_out, timespan, p_obs, n_obs in tqdm(to_process):
    print('Processing: ' + file_in)
    to_save = {}

    # Add stats
    to_save['timespan'] = timespan
    to_save['p_obs'] = p_obs
    to_save['n_obs'] = n_obs

    # Read data and subsample light curves according to the input parameters
    _, _, X_test, _, _, _, _, _, _, _ = get_train_matrices(
        ztf_date, filter_name, minimum_timespan=1800,
        timespan=timespan, frac_n_obs=p_obs/100.0,
    )
    gc.collect()

    # Read the preds
    df_preds = {
        'all': pd.read_csv(file_in)
    }
    if len(df_preds['all']) != len(X_test):
        print('Wrong shapes')
        print(to_save)
        print('Preds length', len(df_preds['all']))
        print('Test data length', len(X_test))
        continue

    # Apply the limiting magnitudes
    lc_test = [[lc_row[1] for lc_row in lc_matrix] for lc_matrix in X_test]
    mag_test = [np.median(lc) for lc in lc_test]
    is_limiting_mag = np.array(mag_test) < LIMITING_MAGS[filter_name]
    df_preds['limited'] = df_preds['all'].loc[is_limiting_mag]
    df_preds['faint'] = df_preds['all'].loc[~is_limiting_mag]
    gc.collect()

    # Calculate the scores
    for key in df_preds:
        to_save['n_obj {}'.format(key)] = df_preds[key].shape[0]
        to_save['accuracy {}'.format(key)] = accuracy_score(df_preds[key]['y_true'], df_preds[key]['y_pred'])
        to_save['QSO F1 {}'.format(key)] = f1_score(df_preds[key]['y_true'], df_preds[key]['y_pred'], average=None)[1]

    # Calculate the mean cadence of observations and its standard deviation
    mjd_test = [[lc_row[0] for lc_row in lc_matrix] for lc_matrix in X_test]
    cadences = [mjds[j+1] - mjds[j] for mjds in mjd_test for j in range(len(mjds) - 1)]
    to_save['mean cadence'] = np.mean(cadences)
    to_save['std cadence'] = np.std(cadences)

    if len(cadences) > 1:
        p = np.percentile(cadences, [16, 50, 84])
        to_save['median cadence'], to_save['minus sigma cadence'], to_save['plus sigma cadence'] = \
            p[1], p[1] - p[0], p[2] - p[1]

    # Save the results
    with open(file_out, 'w') as f:
        json.dump(to_save, f)
    print('File saved: {}'.format(file_out))
