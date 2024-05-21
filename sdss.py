import os

from astropy.table import Table
from astropy import table as astro_table

from env_config import DATA_PATH
from utils import read_fits_to_pandas


COLUMNS_SDSS = ['PLUG_RA', 'PLUG_DEC', 'CLASS', 'SUBCLASS', 'Z', 'Z_ERR', 'ZWARNING']  # 'OBJID'


def read_sdss(dr=18, clean=True, columns=None, n=None, return_cross_matches=False):
    file_path = os.path.join(DATA_PATH, 'SDSS/specObj-dr{}.fits'.format(dr))
    columns = columns if columns is not None else COLUMNS_SDSS
    data = read_fits_to_pandas(file_path, columns=columns, n=n)
    if clean:
        data = clean_sdss(data)

    if return_cross_matches:
        file_path = os.path.join(DATA_PATH, 'SDSS/specObj-dr18__PWG.fits')
        features = read_fits_to_pandas(file_path)
        return data, features
    else:
        return data


def clean_sdss(data, with_print=True):
    if with_print:
        print('Original SDSS: {}'.format(data.shape[0]))
    data_cleaned = data.loc[data['ZWARNING'].isin([0, 16])].reset_index(drop=True)
    if with_print:
        print('Cleaning SDSS: {}'.format(data_cleaned.shape[0]))
    return data_cleaned


def remove_duplicates():
    # Read original SDSS file
    file_path = os.path.join(DATA_PATH, 'SDSS/specObj-dr18__DUPLICATES.fits')
    sdss_data = Table.read(file_path, format='fits')
    print(len(sdss_data))

    # Remove duplicates
    sdss_data_nodups = astro_table.unique(sdss_data, keys=['PLUG_RA', 'PLUG_DEC'])
    len(sdss_data_nodups)

    # Write a new file
    file_path = os.path.join(DATA_PATH, 'SDSS/specObj-dr18.fits')
    sdss_data_nodups.write(file_path)
