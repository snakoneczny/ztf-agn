import sys
import os
import pickle

import numpy as np
from penquins import Kowalski
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH
from ztf import ZTF_FILTER_NAMES, get_ztf_light_curves


fields = [296, 297, 423, 424, 487, 488, 562, 563, 682, 683, 699, 700, 717, 718, 777, 778, 841, 842, 852, 853]

for filter in [1, 2]:
    filter_name = ZTF_FILTER_NAMES[filter]

    for field in tqdm(fields):
        output_file_name = 'ZTF/ZTF_20240117/fields/ZTF_20240117__field_{}__{}-band'.format(field, filter_name)
        output_file_name = os.path.join(DATA_PATH, output_file_name)
        
        if not os.path.exists(output_file_name):
            print('Processing filter: {}, field: {}'.format(filter_name, field))

            kowalski = Kowalski(
                username='nakonecz',
                password='%p!Sn8YVP12h',
                host='melman.caltech.edu',
                timeout=99999999,
            )

            # Read the IDs
            file_name = 'ZTF/ZTF_20240117/field_IDs/ZTF_20240117__field_{}__{}-band__IDs.npy'.format(field, filter_name)
            file_name = os.path.join(DATA_PATH, file_name)
            with open(file_name, 'rb') as file:
                ids = np.load(file)

            data = get_ztf_light_curves(ids, kowalski)

            with open(output_file_name, 'wb') as file:
                pickle.dump(data, file)
            print('Original IDs: {}, light curves: {}, saved to: {}'.format(len(ids), len(data), output_file_name))
