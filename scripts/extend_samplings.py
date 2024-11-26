import sys
import os
import gc
import glob

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

sys.path.append('..')
from env_config import PROJECT_PATH
from ztf import ZTF_DATES
from ml import get_train_matrices


# Input params and paths
ztf_date = ZTF_DATES['DR 20']
filter_name = 'g'

# Get all sampling files
exp_name = 'ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256'.format(ztf_date, filter_name)
file_pattern = '{}__timespan=1800_p-nobs=*_nobs=*__test.csv'.format(exp_name)
file_list = sorted(glob.glob(os.path.join(PROJECT_PATH, 'outputs/preds/ZTF_20240117', file_pattern)))

# Get files to process
to_process = []
for file_name in file_list:
    f_name = os.path.basename(file_name)
    splitted = f_name.split('__')[-2].split('_')
    timespan = int(splitted[0].split('=')[-1])
    p_obs = int(splitted[1].split('=')[-1])
    n_obs = int(splitted[2].split('=')[-1])
    
    splitted = file_name.split('.')
    output_file = splitted[0] + '__extended.' + splitted[1]
    if not os.path.exists(output_file):
        to_process.append([file_name, output_file, timespan, p_obs, n_obs])
        
# Read experiments
for file_in, file_out, timespan, p_obs, n_obs in tqdm(to_process):
    print('Processing ' + file_in)

    # Read data and subsample light curves according to the input parameters
    _, _, X_test, _, _, _, _, _, _, _ = get_train_matrices(
        ztf_date, filter_name, minimum_timespan=1800,
        timespan=timespan, frac_n_obs=p_obs/100.0,
    )
    gc.collect()

    # Read the preds
    df_preds = pd.read_csv(file_in)

    # Make sure the samples are the same
    if len(df_preds) != len(X_test):
        print('Wrong shapes')
        print('Preds length', len(df_preds['all']))
        print('Test data length', len(X_test))
        continue

    # Merge with magnitudes
    lc_test = [[lc_row[1] for lc_row in lc_matrix] for lc_matrix in X_test]
    mag_test = [np.median(lc) for lc in lc_test]
    df_preds['median mag'] = mag_test

    # Save the predictions with magnitudes
    with open(file_out, 'w') as f:
        df_preds.to_csv(f, index=False)
    print('File saved: {}'.format(file_out))
