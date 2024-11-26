import sys
import os
import glob
import gc

from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH
from utils import read_fits_to_pandas


catalog = 'AllWISE'  # 'PS1_DR1', 'Gaia_EDR3'
columns = ['ra', 'dec', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']

output_path = '{}/{}_reduced.csv'.format(catalog, catalog)
output_path = os.path.join(DATA_PATH, output_path)

# Get input paths sorted
input_regex = '{}/chunks/{}__*_*.fits'.format(catalog, catalog)
input_paths = sorted(glob.glob(os.path.join(DATA_PATH, input_regex)))    
chunk_starts = [int(input_path.split('__')[-1].split('_')[0]) for input_path in input_paths]
input_paths = [input_path for _, input_path in sorted(zip(chunk_starts, input_paths))]

# Write first field overwriting previous file and starting with a header
data = read_fits_to_pandas(input_paths[0], columns=columns)
data.to_csv(output_path, header=True, index=False)
for input_path in tqdm(input_paths[1:], 'Merging a catalog on disk'):
    data = read_fits_to_pandas(input_path, columns=columns)
    data.to_csv(output_path, mode='a', header=False, index=False)
    gc.collect()
