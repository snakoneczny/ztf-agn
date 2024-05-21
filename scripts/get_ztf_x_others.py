import argparse
import sys
import os
import json
import gc

from tqdm import tqdm
from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH, STORAGE_PATH
from credentials import KOWALSKI_USERNAME, KOWALSKI_PASSWORD
from ztf import ZTF_DATES, get_cross_matches
from utils import read_fits_to_pandas, save_fits


# Read command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()


# Define run parameters
date = ZTF_DATES['DR 20']
filter_name = 'g'
catalogs = ['PS1_DR1', 'AllWISE', 'Gaia_EDR3']
n = 100 if args.is_test else None

# Read number of objects in the fields
with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields_dict = json.load(file)
fields_dict = {int(k): int(v) for k, v in fields_dict.items()}
fields = [k for k in fields_dict if fields_dict[k] > 0]

# Get a chunk of fields
# fields = fields[500:]

# Scan for files
to_process, n_obj = [], []
for field in fields:
    file_name = 'ZTF/ZTF_{}/fields/ZTF_{}__field_{}__{}-band'.format(
        date, date, field, filter_name)
    file_path = os.path.join(STORAGE_PATH, file_name)
    input_file_path = file_path + '.fits'
    output_file_path = file_path + '__PWG.fits'
    if os.path.exists(input_file_path) and not os.path.exists(output_file_path):    
        to_process.append((input_file_path, output_file_path))
        n_obj.append(fields_dict[field])

# Progress
all_data = sum(n_obj)
done_data = 0

with tqdm('Downloading data', total=all_data) as pbar:

    for i, (input_file_path, output_file_path) in enumerate(to_process):
        print('Processing filter: {}, file: {}'.format(filter_name, input_file_path))
        gc.collect()
        
        # Read the coordinates
        with open(input_file_path, 'rb') as file:
            input_data = read_fits_to_pandas(file)
        coordinates = list(zip(input_data['ra'], input_data['dec']))

        # Get the cross matches
        kowalski = Kowalski(
            username=KOWALSKI_USERNAME,
            password=KOWALSKI_PASSWORD,
            host='gloria.caltech.edu',
            timeout=99999999,
        )
        data = get_cross_matches(coordinates, catalogs, kowalski, radius=1.0)
        kowalski.close()

        # Save the results
        save_fits(data, output_file_path, overwrite=True, with_print=False)

        # Update the progress
        pbar.update(n_obj[i])
        print('Input shape: {}, resulting shape: {}, saved to: {}'.format(
            input_data.shape, data.shape, output_file_path))
