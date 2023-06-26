import sys
import os
import pickle

from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH
from data import ZTF_FILTER_NAMES, get_ztf_kowalski_fields


field_ids = [296, 297, 423, 424, 487, 488, 562, 563, 682, 683, 699, 700, 717, 718, 777, 778, 841, 842, 852, 853]

to_process = [
    (1, 10000),
    (2, 10000),
    # (3, 1000),
]

kowalski = Kowalski(
    username='nakonecz',
    password='%p!Sn8YVP12h',
    host='gloria.caltech.edu',
    timeout=99999999,
)

token = kowalski.authenticate()

for filter, limit_per_field in to_process:
    filter_name = ZTF_FILTER_NAMES[filter]
    print('Processing filter: {}'.format(filter_name))

    data = get_ztf_kowalski_fields(field_ids, token, filter=filter, limit_per_field=limit_per_field)

    file_name = os.path.join(DATA_PATH, 'ZTF/ZTF_20210401__fields_Roestel_2M_{}'.format(filter_name))
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
