import sys
import os
import pickle
import lzma
import gc

import numpy as np
from tqdm import tqdm
import json

sys.path.append('..')
from env_config import DATA_PATH, STORAGE_PATH
from utils import read_fits_to_pandas
from ztf import ZTF_DATES


ztf_date = ZTF_DATES['DR 20']
filter_name = 'g'
output_file = os.path.join(STORAGE_PATH, 'ZTF/ZTF_{}/fields_progress.txt'.format(ztf_date))
start_field = 334

with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields = json.load(file)
fields = {x: y for x, y in fields.items() if y != 0 and int(x) >= start_field}

info = []
for field in tqdm(fields):
    gc.collect()

    # Define the data paths
    data_file_name = 'ZTF/ZTF_{}/fields/ZTF_{}__field_{}__{}-band'.format(
                            ztf_date, ztf_date, field, filter_name)
    data_file_name = os.path.join(STORAGE_PATH, data_file_name)

    ids_file_name = 'ZTF/ZTF_{}/field_IDs/ZTF_{}__field_{}__{}-band__IDs.npy'.format(
                        ztf_date, ztf_date, field, filter_name)
    ids_file_name = os.path.join(DATA_PATH, ids_file_name)

    # Check if present
    if not os.path.exists(data_file_name + '.xz'):
        info.append('Field {} data not present'.format(field))
    else:
        # Read all data
        with open(ids_file_name, 'rb') as file:
            ids = np.load(file)
        df = read_fits_to_pandas(data_file_name + '.fits')
        with lzma.open(data_file_name + '.xz', 'rb') as f:
            data = pickle.load(f)

        # Check if ra and dec present
        if not ('ra' in df and 'dec' in df):
            info.append('Field {} position not present'.format(field))
        else:
            # Check if ok
            n_ids = len(ids)
            n_rows = len(df)
            n_data = len(data)
            is_ok = n_ids == n_rows == n_data
            info.append('Field {}\tis ok {}\tIDs {}\tDF {}\tdata {}'.format(field, is_ok, n_ids, n_rows, n_data))

    # Write stats
    with open(output_file, 'w') as f:
        for line in info:
            f.write(f'{line}\n')
