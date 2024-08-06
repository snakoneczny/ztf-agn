import logging

from astropy.table import Table

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def pretty_print(x):
    pretty_dict = {
        'Z': 'redshift',
        'n obs': 'number of observation epochs',
        'mag median': 'median magnitude',
        'mean cadence': 'mean cadence',
    }
    if x in pretty_dict:
        x = pretty_dict[x]    
    if 'cadence' in x or 'timespan' in x:
        x = ' '.join(x.split('_')) + ' [days]'
    return x


def pretty_print_features(x):
    pretty_dict = {
        'ZTF + AstrmClf': 'ZTF',
        'ZTF + AstrmClf + PS': 'ZTF + PS',
        'Astrm': 'Astromer',
        'PS': 'Pan-STARRS',
    }
    if x in pretty_dict:
        return pretty_dict[x]
    else:
        return x


def save_fits(data, file_path, overwrite=False, with_print=True):
    # if with_print:
    #     print('Constructing an astropy table')
    astropy_table = Table.from_pandas(data)
    astropy_table.write(file_path, overwrite=overwrite)
    if with_print:
        print('Astropy table saved to {}'.format(file_path))


def read_fits_to_pandas(filepath, columns=None, n=None):
    table = Table.read(filepath, format='fits')

    # Get first n rows if limit specified
    if n:
        table = table[0:n]

    # Get proper columns into a pandas data frame
    if columns:
        table = table[columns]
    table = table.to_pandas()

    # Astropy table assumes strings are byte arrays
    for col in ['ID', 'ID_1', 'CLASS', 'SUBCLASS', 'CLASS_PHOTO', 'id1']:
        if col in table and hasattr(table.loc[0, col], 'decode'):
            table.loc[:, col] = table[col].apply(lambda x: x.decode('UTF-8').strip())

    # Change type to work with it as with a bit map
    # if 'IMAFLAGS_ISO' in table:
    #     table.loc[:, 'IMAFLAGS_ISO'] = table['IMAFLAGS_ISO'].astype(int)

    return table
