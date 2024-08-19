import sys
import os
import gc
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from penquins import Kowalski

sys.path.append('..')
from env_config import DATA_PATH
from credentials import KOWALSKI_USERNAME, KOWALSKI_PASSWORD
from ztf import CATALOGS_DICT, get_catalog_data
from utils import save_fits


catalogs = [
    'PS1_DR1',
    # 'AllWISE',
    # 'Gaia_EDR3',
]
chunk_size = 1e6

for catalog in tqdm(catalogs):
    keep_going = True
    i = 0

    while keep_going:
        chunk_start, chunk_end = int(i * chunk_size), int((i + 1) * chunk_size)
        print('Processing {} - {}'.format(chunk_start, chunk_end))

        file_name = '{}/chunks/{}__{}_{}.fits'.format(catalog, catalog, chunk_start, chunk_end)
        file_name = os.path.join(DATA_PATH, file_name)

        if not os.path.exists(file_name):
            kowalski = Kowalski(
                username=KOWALSKI_USERNAME,
                password=KOWALSKI_PASSWORD,
                host='melman.caltech.edu',
                timeout=99999999,
            )

            # Download data and measure time
            start = datetime.now()
            data = get_catalog_data(
                catalog, kowalski, chunk_start=chunk_start, chunk_end=chunk_end, verbose=0,
            )
            delta = datetime.now() - start
            print('Time taken: {}'.format(delta))
            print('Data downloaded {}'.format(len(data)))

            kowalski.close()
            gc.collect()

            if len(data) > 0:
                df = pd.DataFrame.from_records(data, columns=CATALOGS_DICT[catalog])
                save_fits(df, file_name, overwrite=True, with_print=True)
                gc.collect()

            keep_going = (len(data) == chunk_size)       
        i += 1
