import sys
import os
import pickle

from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH
from ztf import get_ztf_features


to_process = [
    'ZTF_x_SDSS/ZTF_20230821/ztf_20230821_x_specObj-dr18__singles__filter_g__reduced',
    'ZTF_x_SDSS/ZTF_20230821/ztf_20230821_x_specObj-dr18__singles__filter_r__reduced',
]

kowalski = Kowalski(
    username='nakonecz',
    password='%p!Sn8YVP12h',
    host='gloria.caltech.edu',
    timeout=99999999,
)

token = kowalski.authenticate()

for file_path in to_process:
    print('Processing file: {}'.format(file_path))

    file_path = os.path.join(DATA_PATH, file_path)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    ids = [obj['id'] for obj in data]

    features = get_ztf_features(ids, token)

    file_path += '__features'
    with open(file_path, 'wb') as file:
        pickle.dump(features, file)
    print('Saved to: ' + file_path)
