import sys
import os
import glob
import gc

import pandas as pd
from astropy.table import Table, vstack
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH
from utils import read_fits_to_pandas, save_fits


catalogs = ['AllWISE']  # 'PS1_DR1', 'Gaia_EDR3'

# Read the chunks
for catalog in catalogs:
    # Get input paths sorted
    input_regex = '{}/chunks/{}__*_*.fits'.format(catalog, catalog)
    input_paths = sorted(glob.glob(os.path.join(DATA_PATH, input_regex)))    
    chunk_starts = [int(input_path.split('__')[-1].split('_')[0]) for input_path in input_paths]
    input_paths = [input_path for _, input_path in sorted(zip(chunk_starts, input_paths))]

    # Read and merge data
    # data = read_fits_to_pandas(input_paths[0])
    # for input_path in tqdm(input_paths[1:], '{} chunks'.format(catalog)):
    #     data = pd.concat([data, read_fits_to_pandas(input_path)], ignore_index=True, axis=0)
    #     # data = vstack([data, Table.read(input_path, format='fits')])
    #     gc.collect()
    data = read_fits_to_pandas(input_paths[0])
    for input_path in tqdm(input_paths[1:]):
        # TODO: pick the 4 bands detections
        data = pd.concat([data, read_fits_to_pandas(input_path)])
        gc.collect()

    # Save
    file_path = os.path.join(DATA_PATH, '{}/{}.fits'.format(catalog, catalog))
    save_fits(data, file_path, overwrite=True, with_print=True)
    # data.write(file_path, overwrite=True)
    print('Saved {}'.format(file_path))
    gc.collect()
