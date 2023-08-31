import sys
import os
import pickle

from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH
from data import read_sdss
from ztf import get_ztf_kowalski

sdss_data = read_sdss(dr=18, clean=True)

kowalski = Kowalski(
    username='nakonecz',
    password='%p!Sn8YVP12h',
    host='gloria.caltech.edu',
)

token = kowalski.authenticate()

coordinates = list(zip(sdss_data['PLUG_RA'], sdss_data['PLUG_DEC']))

data = get_ztf_kowalski(coordinates, token, radius=1.0)

with open(os.path.join(DATA_PATH, 'SDSS_x_ZTF/specObj-dr18_x_ztf_20210401'), 'wb') as file:
    pickle.dump(data, file)
