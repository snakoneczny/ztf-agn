import sys
import os
import pickle
import lzma
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.append('..')
from env_config import DATA_PATH, STORAGE_PATH
from utils import read_fits_to_pandas
from ztf import ZTF_DATES

ztf_date = ZTF_DATES['DR 20']
filter_name = 'g'
output_file = os.path.join(STORAGE_PATH, 'ZTF/ZTF_{}/fields_progress.txt'.format(ztf_date))

with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields = json.load(file)
fields = {x: y for x, y in fields.items() if y != 0}

fields = [field for field in fields][348:]
for field in tqdm(fields):
    gc.collect()

    # Define data paths
    data_file_name = 'ZTF/ZTF_{}/fields/ZTF_{}__field_{}__{}-band'.format(
                            ztf_date, ztf_date, field, filter_name)
    data_file_name = os.path.join(STORAGE_PATH, data_file_name)

    ids_file_name = 'ZTF/ZTF_{}/field_IDs/ZTF_{}__field_{}__{}-band__IDs.npy'.format(
                        ztf_date, ztf_date, field, filter_name)
    ids_file_name = os.path.join(DATA_PATH, ids_file_name)

    # Check if present
    if not os.path.exists(data_file_name + '.xz') and not os.path.exists(data_file_name + '__chunk_0.xz'):
        # Check if IDs were downloaded
        ids_file_name = 'ZTF/ZTF_{}/field_IDs/ZTF_{}__field_{}__{}-band__IDs.npy'.format(
                ztf_date, ztf_date, field, filter_name)
        ids_file_name = os.path.join(DATA_PATH, ids_file_name)
        if os.path.exists(ids_file_name):
            with open(ids_file_name, 'rb') as file:
                ids = np.load(file)
            print('Field {} not present, {} IDs present'.format(field, len(ids)))    
        else:
            print('Field {} not present, no IDs downloaded'.format(field))

    else:
        # Read IDs
        with open(ids_file_name, 'rb') as file:
            ids = np.load(file)

        # Read chunked fields
        if os.path.exists(data_file_name + '__chunk_0.xz'):
            chunk_id = 0
            df = []
            n_data = 0
            while os.path.exists(data_file_name + '__chunk_{}.xz'.format(chunk_id)):
                df.append(read_fits_to_pandas(data_file_name + '__chunk_{}.fits'.format(chunk_id)))
                with lzma.open(data_file_name + '__chunk_{}.xz'.format(chunk_id), 'rb') as f:
                    tmp_data = pickle.load(f)
                n_data += len(tmp_data)
                chunk_id += 1
            df = pd.concat(df)

        # Read non chunked fields
        else:
            df = read_fits_to_pandas(data_file_name + '.fits')
            with lzma.open(data_file_name + '.xz', 'rb') as f:
                data = pickle.load(f)
            n_data = len(data)

        # Check if ra and dec present
        if not ('ra' in df and 'dec' in df):
            print('Field {} position not present'.format(field))

        # Check if ok
        n_ids = len(ids)
        n_rows = len(df)
        # n_data = len(data)
        is_ok = n_ids == n_rows == n_data
        if not is_ok:
            print('Fields {} not ok'.format(field))
