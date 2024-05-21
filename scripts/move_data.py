import sys
import os
import pickle
import json
import lzma
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH, STORAGE_PATH
from utils import save_fits
from ztf import ZTF_DATES


ztf_date = ZTF_DATES['DR 20']
filter_name = 'g'

# Get the list of fields
with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields = json.load(file)
fields = {x: y for x, y in fields.items() if y != 0} 

# Scan for files
to_process, n_obj = [], []
for field in fields:
    file_name = 'ZTF/ZTF_{}/fields/ZTF_{}__field_{}__{}-band'.format(
        ztf_date, ztf_date, field, filter_name)
    input_file_path = os.path.join(DATA_PATH, file_name)    
    output_file_path = os.path.join(STORAGE_PATH, file_name)
    if os.path.exists(input_file_path) and not os.path.exists(output_file_path + '.xz'):
        
        # Skip broken files
        if input_file_path not in [
            '/home/sjnakoneczny/data/ZTF/ZTF_20240117/fields/ZTF_20240117__field_334__g-band',
        ]:
            to_process.append((input_file_path, output_file_path))
            n_obj.append(fields[field])

# Progress
all_data = sum(n_obj)
done_data = 0

with tqdm('Compressing data', total=all_data) as pbar:
    for i, (input_file_path, output_file_path) in enumerate(to_process):
        print('Processing {}'.format(input_file_path))
        gc.collect()
        
        # Load data
        with open(input_file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Check if ra, dec present
        if 'ra' in data[0]:
        
            # Extract DF and array
            df = pd.DataFrame(data, columns=['id', 'ra', 'dec', 'n obs'])
            df['n obs'] = [len(lc_dict['mjd']) for lc_dict in data]
            data = [np.array([lc_dict['mjd'], lc_dict['mag'], lc_dict['magerr']]) for lc_dict in data]

            # Save both
            save_fits(df, output_file_path + '.fits', overwrite=True, with_print=False)
            with lzma.open(output_file_path + '.xz', 'wb') as f:
                pickle.dump(data, f)

        # Update the progress
        pbar.update(n_obj[i])
