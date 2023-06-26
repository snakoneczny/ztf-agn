import os
import pickle

from utils import read_fits_to_pandas

ZTF_FILTER_NAMES = {1: 'g', 2: 'r', 3: 'i'}

COLUMNS_SDSS = ['PLUG_RA', 'PLUG_DEC', 'CLASS', 'SUBCLASS', 'Z', 'Z_ERR', 'ZWARNING']


def read_ztf_fields(filter):
    with open(os.path.join(DATA_PATH, 'ZTF/ZTF_20210401__fields_Roestel_2M_{}'.format(filter)), 'rb') as file:
        data = pickle.load(file)
    return data


def read_single_matches():
    file_path = os.path.join(DATA_PATH, 'SDSS_x_ZTF/specObj-dr18_x_ztf_20210401__ztf__single-matches')
    with open(file_path, 'rb') as file:
        ztf = pickle.load(file)

    file_path = os.path.join(DATA_PATH, 'SDSS_x_ZTF/specObj-dr18_x_ztf_20210401__sdss__single-matches')
    with open(file_path, 'rb') as file:
        sdss = pickle.load(file)

    return ztf, sdss


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
