import sys
import os

import numpy as np
import json
from tqdm import tqdm
from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH
from credentials import KOWALSKI_USERNAME, KOWALSKI_PASSWORD
from ztf import ZTF_FILTER_NAMES, ZTF_DATES, get_catalog_data


date = ZTF_DATES['DR 20']

with open(os.path.join(DATA_PATH, 'ZTF/DR19_field_counts.json'), 'r') as file:
    fields = json.load(file)
fields = [int(k) for k in fields if fields[k] > 0]

for filter in [1]:
    filter_name = ZTF_FILTER_NAMES[filter]

    for field in tqdm(fields):
        file_name = 'ZTF/ZTF_{}/field_IDs/ZTF_{}__field_{}__{}-band__IDs.npy'.format(date, date, field, filter_name)
        file_name = os.path.join(DATA_PATH, file_name)

        if not os.path.exists(file_name):
            print('Downloading field', field)
            
            kowalski = Kowalski(
                username=KOWALSKI_USERNAME,
                password=KOWALSKI_PASSWORD,
                host='melman.caltech.edu',
                timeout=99999999,
            )
            
            catalog = 'ZTF_sources_{}'.format(date)
            data = get_catalog_data(catalog, kowalski, field_id=field, filter=filter, verbose=0)
            print('Filter {}, field {}: {}'.format(filter_name, field, len(data)))

            if len(data) > 0:
                with open(file_name, 'wb') as file:
                    np.save(file, data)
                print('Data saved to: ' + file_name)
