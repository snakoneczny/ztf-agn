import sys
import os
import gc

import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from env_config import STORAGE_PATH
from ztf import ZTF_DATES
from catalog import find_duplicates

pd.set_option('mode.chained_assignment', None)


# Data params
ztf_date = ZTF_DATES['DR 20']
filter = 'g'

# Input params in degrees
stripe_width = 1
delta = 0.001

# Read ZTF preds
path_ztf = 'ZTF/ZTF_{}/catalog_ZTF_{}_{}-band__v2.csv'.format(ztf_date, ztf_date, filter)
path_ztf = os.path.join(STORAGE_PATH, path_ztf)
cols = ['ra', 'dec', 'n_obs']
data = pd.read_csv(path_ztf, usecols=cols)

# Iterate the stripes
for x in tqdm(range(0, 360, stripe_width)):
    
    # Take a subset
    subset = data.loc[(data['ra'] > x - delta) & (data['ra'] < x + stripe_width + delta)]
    
    # Find duplicates within a subset
    subset = find_duplicates(subset, with_tqdm=False)
    
    # Save a chunk
    path_output = 'ZTF/ZTF_{}/catalog_v2_dups/ZTF_{}_{}-band__v2__dups_ra-{}.csv'.format(ztf_date, ztf_date, filter, x)
    path_output = os.path.join(STORAGE_PATH, path_output)
    subset['is_duplicate'].to_csv(path_output, index=True)
    
    n_dups = subset['is_duplicate'].sum()
    print('RA={}\tn={}\tn_dups={}\t({:.1f}%)'.format(x, len(subset), n_dups, n_dups/len(subset) * 100))
    
    gc.collect()
