import sys
import os
import pickle
import lzma
import gc

import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH, STORAGE_PATH
from credentials import KOWALSKI_USERNAME, KOWALSKI_PASSWORD
from ztf import ZTF_DATES, get_ztf_light_curves
from utils import save_fits


date = ZTF_DATES['DR 20']
filter_name = 'g'

# Read number of objects in the fields
with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields_dict = json.load(file)
fields_dict = {int(k): int(v) for k, v in fields_dict.items()}
fields = [k for k in fields_dict if fields_dict[k] > 0]

# Get only the test fields
# test_fields = TEST_FIELDS
# fields = [k for k in fields if k in test_fields]

# Scan for files
to_process, n_obj = [], []
for field in fields:
    file_name = 'ZTF/ZTF_{}/fields/ZTF_{}__field_{}__{}-band.xz'.format(
        date, date, field, filter_name)
    file_path = os.path.join(STORAGE_PATH, file_name)
    if not os.path.exists(file_path):
        to_process.append(field)
        n_obj.append(fields_dict[field])

# Progress
all_data = sum(n_obj)
done_data = 0

with tqdm('Downloading data', total=all_data) as pbar:
    for i, field in enumerate(to_process):
        output_file_name = 'ZTF/ZTF_{}/fields/ZTF_{}__field_{}__{}-band'.format(
            date, date, field, filter_name)
        output_file_name = os.path.join(STORAGE_PATH, output_file_name)

        ids_file_name = 'ZTF/ZTF_{}/field_IDs/ZTF_{}__field_{}__{}-band__IDs.npy'.format(
                date, date, field, filter_name)
        ids_file_name = os.path.join(DATA_PATH, ids_file_name)

        if not os.path.exists(output_file_name + '.xz') and os.path.exists(ids_file_name):
            # Read the IDs
            with open(ids_file_name, 'rb') as file:
                ids = np.load(file)

            # Split the IDs in chunks not too large to zip the results
            chunk_size = 2000000
            ids_chunked = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

            # Get the data
            for j, ids in enumerate(ids_chunked):
                if len(ids) > 1:
                    output_chunk_file_name = output_file_name + '__chunk_{}'.format(j)
                else:
                    output_chunk_file_name = output_file_name
                
                # Check if a chunk file exists
                if not os.path.exists(output_chunk_file_name + '.xz'):
                    print('Processing filter: {}, field: {}, chunk {}'.format(filter_name, field, j))
                    
                    kowalski = Kowalski(
                        username=KOWALSKI_USERNAME,
                        password=KOWALSKI_PASSWORD,
                        host='melman.caltech.edu',
                        timeout=99999999,
                    )
                    df, data = get_ztf_light_curves(ids, date, kowalski)
                    kowalski.close()
                    gc.collect()

                    # Save both
                    save_fits(df, output_chunk_file_name + '.fits', overwrite=True, with_print=False)
                    with lzma.open(output_chunk_file_name + '.xz', 'wb') as f:
                        pickle.dump(data, f)
                    gc.collect()
                    
                    # Print the progress
                    print('Original IDs: {}, light curves: {}, saved to: {}'.format(
                        len(ids), len(data), output_chunk_file_name))

            # Update the progress
            pbar.update(n_obj[i])
