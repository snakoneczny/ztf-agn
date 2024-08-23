import argparse
import sys
import os
import gc
import glob

from tqdm import tqdm
import json

sys.path.append('..')
from env_config import STORAGE_PATH
from ztf import ZTF_DATES
from utils import read_fits_to_pandas, save_fits
from catalog import find_duplicates


# Command line params
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', dest='is_test', help='Test run flag', action='store_true')
args = parser.parse_args()

# Predefined input params
ztf_date = ZTF_DATES['DR 20']
filter = 'g'
test_size = 10000

# Get a list of all the fields in catalog
input_regex = 'ZTF/ZTF_{}/catalog/ZTF_{}__field_*__{}-band.fits'.format(ztf_date, ztf_date, filter)
input_paths = sorted(glob.glob(os.path.join(STORAGE_PATH, input_regex)))

# Get names of the desired output files
output_paths = [os.path.join(STORAGE_PATH, 'ZTF/ZTF_{}/catalog_v2/{}'.format(ztf_date, os.path.basename(input_file)))
                    for input_file in input_paths]

# Get not processed files
# idx = np.where([not os.path.exists(file) for file in output_paths])[0]
# print('Processing {} out of {}'.format(len(idx), len(input_paths)))
# output_paths = np.array(output_paths)[idx]
# input_paths = np.array(input_paths)[idx]

# Get a total number of objects
with open(os.path.join(STORAGE_PATH, 'ZTF/ZTF_{}/catalog_field_counts.json'.format(ztf_date)), 'r') as file: 
    field_counts = json.load(file)
field_counts = {int(k): int(v) for k, v in field_counts.items()}
field_ids = [int(input_path.split('__')[-2].split('_')[1]) for input_path in input_paths]
counts_all = sum([field_counts[field_id] for field_id in field_ids])

# Take a test subset
to_process = list(zip(input_paths, output_paths, field_ids))
if args.is_test:
    to_process = to_process[:1]

with tqdm('Running catalog v2', total=counts_all) as pbar:
    for input_path, output_path, field_id in to_process:

        # Read the field file
        data = read_fits_to_pandas(input_path)
        if len(data) == 0:
            continue
        gc.collect()

        # Limit data if test
        if args.is_test:
            data = data[:test_size]
        gc.collect()

        # Rename columns
        data = data.rename(columns={
            'mag': 'mag_median',
            'n obs': 'n_obs'
        })

        # Get duplicates
        if 'is_duplicate' not in data.columns:
            data = find_duplicates(data)
            gc.collect()

        # Get WISE cross-match

        # Run inference

        # Save catalog v2
        if not args.is_test:
            output_folder = os.path.dirname(output_path)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            save_fits(data, output_path, overwrite=True, with_print=False)

        # Update the progress
        pbar.update(field_counts[field_id])
