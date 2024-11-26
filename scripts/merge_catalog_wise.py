import sys
import os
import gc
import glob

import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from env_config import STORAGE_PATH


# Input params
chunk_size = 100000

# Paths
paths = {
    # 'wise': {
    #     'input': 'ZTF/ZTF_20240117/catalog_v2_wise_chunks/catalog_ZTF_20240117_g-band__v2_WISE__{}.csv',
    #     'output': 'ZTF/ZTF_20240117/catalog_ZTF_20240117_g-band__v2_WISE.csv',
    # },
    'xgb': {
        'input': 'ZTF/ZTF_20240117/catalog_v2_xgb_chunks/catalog_ZTF_20240117_g-band__v2_XGB__{}.csv',
        'output': 'ZTF/ZTF_20240117/catalog_ZTF_20240117_g-band__v2_XGB.csv',
    },
}

# Get all chunk ids/starts
regex = 'ZTF/ZTF_20240117/catalog_v2_wise_chunks/catalog_ZTF_20240117_g-band__v2_WISE__*.csv'
regex = os.path.join(STORAGE_PATH, regex)
paths_chunks = glob.glob(os.path.join(STORAGE_PATH, regex))
chunk_ids = sorted([int(x.split('__')[-1].split('.')[0]) for x in paths_chunks])

# Iterate chunks
for chunk_start in tqdm(chunk_ids, 'Merging WISE and XGB chunks'):

    # Get proper paths
    for x in paths:
        input_path = os.path.join(STORAGE_PATH, paths[x]['input'].format(chunk_start))
        output_path = os.path.join(STORAGE_PATH, paths[x]['output'])

        # Read data
        data = pd.read_csv(input_path)
        
        # Save
        if chunk_start == 0:
            data.to_csv(output_path, index=False, header=True)
        else:        
            data.to_csv(output_path, index=False, header=False, mode='a')
    
    gc.collect()
