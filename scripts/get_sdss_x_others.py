import sys
import os
import argparse
import gc

from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH
from credentials import KOWALSKI_USERNAME, KOWALSKI_PASSWORD
from sdss import read_sdss
from ztf import get_cross_matches
from utils import save_fits


# Read command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()


# Define run parameters
file_path = os.path.join(DATA_PATH, 'SDSS/specObj-dr18__PWG.fits')
catalogs = ['PS1_DR1', 'AllWISE', 'Gaia_EDR3']
n = 100 if args.is_test else None

# Read SDSS data for cross-match
sdss = read_sdss(dr=18, clean=True, n=n)

# Make coordinates
coordinates = list(zip(sdss['PLUG_RA'], sdss['PLUG_DEC']))

kowalski = Kowalski(
    username=KOWALSKI_USERNAME,
    password=KOWALSKI_PASSWORD,
    host='gloria.caltech.edu',
    timeout=99999999,
)

data = get_cross_matches(coordinates, catalogs, kowalski, radius=1.0)
print('Resulting data shape: {}'.format(data.shape))

kowalski.close()
gc.collect()

save_fits(data, file_path, overwrite=True)
