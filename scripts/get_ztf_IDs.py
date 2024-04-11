import sys
import os

import numpy as np
from penquins import Kowalski
from tqdm import tqdm
import json

sys.path.append('..')
from env_config import DATA_PATH
from ztf import ZTF_FILTER_NAMES, get_ztf_field_ids


with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields = json.load(file)
fields = [int(k) for k in fields.keys()]

for filter in [2]:
    filter_name = ZTF_FILTER_NAMES[filter]

    for field in tqdm(fields):
        file_name = 'ZTF/ZTF_20240117/field_IDs/ZTF_20240117__field_{}__{}-band__IDs.npy'.format(field, filter_name)
        file_name = os.path.join(DATA_PATH, file_name)

        if not os.path.exists(file_name):
            kowalski = Kowalski(
                username='nakonecz',
                password='%p!Sn8YVP12h',
                host='melman.caltech.edu',
                timeout=99999999,
            )

            data = get_ztf_field_ids(field, kowalski, filter=filter, verbose=0)
            print('Filter {}, field {}: {}'.format(filter_name, field, len(data)))

            if len(data) > 0:
                with open(file_name, 'wb') as file:
                    np.save(file, data)
                print('Data saved to: ' + file_name)
