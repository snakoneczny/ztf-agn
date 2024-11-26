import sys
import os
import gc
import glob

import numpy as np
from tqdm import tqdm

sys.path.append('..')
from env_config import STORAGE_PATH
from utils import read_fits_to_pandas
from ztf import ZTF_DATES


# Input params
ztf_date = ZTF_DATES['DR 20']
filter = 'g'

# Output path
output_path = 'ZTF/ZTF_{}/catalog_ZTF_{}_{}-band__v2.csv'.format(ztf_date, ztf_date, filter)
output_path = os.path.join(STORAGE_PATH, output_path)

# Get a list of all the fields in catalog
input_regex = 'ZTF/ZTF_{}/catalog_v2/ZTF_{}__field_*__{}-band*.fits'.format(ztf_date, ztf_date, filter)
input_paths = sorted(glob.glob(os.path.join(STORAGE_PATH, input_regex)))

# Get fields and chunks
fields, chunks = [], []
for i in range(len(input_paths)):
    splitted = input_paths[i].split('__')

    # Chunked
    if splitted[-2] == 'g-band':
        field = int(splitted[-3].split('_')[1])
        chunk = int(splitted[-1].split('.')[-2].split('_')[1])
    else:
        field = int(splitted[-2].split('_')[1])
        chunk = 0

    fields.append(field)
    chunks.append(chunk)

# Sort input paths
tmp = zip(input_paths, fields, chunks)
sorted_inputs = sorted(tmp, key=lambda x: (x[1], x[2]))

for i, (input_path, field, _) in tqdm(list(enumerate(sorted_inputs)), 'Merging catalog on disk'):
    data = read_fits_to_pandas(input_path)
    data['field'] = field
    data['field'] = data['field'].astype(np.int32)

    # Write first field overwriting previous file and starting with a header
    if i == 0:
        data.to_csv(output_path, header=True, index=False)
    else:
        data.to_csv(output_path, header=False, index=False, mode='a')

    gc.collect()
