import sys
import os
import pickle
import argparse
import gc

from penquins import Kowalski
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH
from credentials import KOWALSKI_USERNAME, KOWALSKI_PASSWORD
from sdss import read_sdss
from ztf import ZTF_DATES, get_ztf_matches


# Read command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()

# Params
date = ZTF_DATES['DR 16']

# Define run parameters
file_path = os.path.join(DATA_PATH, 'ZTF_x_SDSS/chunks_{}/ztf_{}_x_specObj-dr18__{}-{}')
chunk_size = 10000 if not args.is_test else 10
test_size = 6

# Read SDSS data for cross-match
sdss_data = read_sdss(dr=18, clean=True)

# Limit data size for testing
if args.is_test:
    sdss_data = sdss_data[:test_size]

# Make coordinate chunks
coordinates = list(zip(sdss_data['PLUG_RA'], sdss_data['PLUG_DEC']))
coordinate_chunks = [coordinates[i:i + chunk_size] for i in range(0, len(coordinates), chunk_size)]

# Run each chunk and save a chunk file
for i, coordinates in tqdm(list(enumerate(coordinate_chunks))):
    file_path_chunk = file_path.format(date, date, i * chunk_size, (i+1) * chunk_size)

    # Check if file exists already
    if not os.path.exists(file_path_chunk):
        kowalski = Kowalski(
            username=KOWALSKI_USERNAME,
            password=KOWALSKI_PASSWORD,
            host='gloria.caltech.edu',  # melman
            timeout=99999999,
        )

        data = get_ztf_matches(coordinates, date, kowalski, radius=1.0)

        kowalski.close()
        gc.collect()

        with open(file_path_chunk, 'wb') as file:
            pickle.dump(data, file)
        print('Chunk {}/{} saved to: {}'.format(i + 1, len(coordinate_chunks), file_path_chunk))

    else:
        print('Chunk {}/{} already exists, path: {}'.format(i + 1, len(coordinate_chunks), file_path_chunk))
