import os

from env_config import DATA_PATH
from utils import read_fits_to_pandas


COLUMNS_SDSS = ['PLUG_RA', 'PLUG_DEC', 'CLASS', 'SUBCLASS', 'Z', 'Z_ERR', 'ZWARNING']  # 'OBJID'


def read_sdss(dr=18, clean=True, columns=None, n=None):
    filepath = os.path.join(DATA_PATH, 'SDSS/specObj-dr{}.fits'.format(dr))
    columns = columns if columns is not None else COLUMNS_SDSS
    data = read_fits_to_pandas(filepath, columns=columns, n=n)
    if clean:
        data = clean_sdss(data)
    return data


def clean_sdss(data, with_print=True):
    if with_print:
        print('Original SDSS: {}'.format(data.shape[0]))
    data_cleaned = data.loc[data['ZWARNING'].isin([0, 16])].reset_index(drop=True)
    if with_print:
        print('Cleaning SDSS: {}'.format(data_cleaned.shape[0]))
    return data_cleaned
