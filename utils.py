import logging

from astropy.table import Table

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def pretty_print(x, band=None):
    if x == 'mag median':
        return 'ZTF ${}$-band median magnitude'.format(band)
    elif x == 'mag err mean':
        return 'ZTF ${}$-band mean magnitude error'.format(band)
    pretty_dict = {
        'Z': 'redshift',
        'n obs': 'number of observation epochs',
        'n_obs': 'number of observation epochs',
    }
    if x in pretty_dict:
        x = pretty_dict[x]    
    if 'cadence' in x or 'timespan' in x:
        x = ' '.join(x.split('_')) + ' [days]'
    return x


def pretty_print_feature_sets(x):
    pretty_dict = {
        'ZTF + AstrmClf': 'ZTF',
        'ZTF + AstrmClf + PS': 'ZTF + PS',
        'ZTF + AstrmClf + WISE': 'ZTF + WISE',
        'Astrm': 'Astromer',
        'PS': 'Pan-STARRS',
        'GAIA': 'Gaia',
    }
    if x in pretty_dict:
        return pretty_dict[x]
    else:
        return x


def pretty_print_features(x, ztf_band=None):
    if '__minus__' in x:
        features = x.split('__minus__')
        return '${}-{}$'.format(pretty_print_feature(features[0], ztf_band), pretty_print_feature(features[1], ztf_band))
    elif 'astrm' in x:
        cls_name = x[6:]
        cls_name = 'QSO' if cls_name == 'qso' else cls_name
        return '$p_\mathrm{{lc}}(\mathrm{{{}}})$'.format(cls_name)
    elif x == 'mag median':
        return '${}_\mathrm{{{}}}$'.format(ztf_band, 'ZTF')
    elif 'pmdec' in x or 'pmra' in x:
        measurement = x.split('__')[1][2:]
        return '$PM_\mathrm{' + measurement + ', Gaia}$'
    else:
        survey_name, feature_name = x.split('__')
        if '-' in feature_name:
            feature_name_a, feature_name_b = feature_name.split('-')
            feature_name_a = pretty_print_feature('{}__{}'.format(survey_name, feature_name_a), ztf_band)
            feature_name_b = pretty_print_feature('{}__{}'.format(survey_name, feature_name_b), ztf_band)
            return('${}-{}$'.format(feature_name_a, feature_name_b))
        else:
            return '${}$'.format(pretty_print_feature('{}__{}'.format(survey_name, feature_name), ztf_band))


def pretty_print_feature(feature_name, ztf_band=None):
    if feature_name == 'mag median':
        survey_name, filter_name = 'ZTF', ztf_band
    else:
        survey_name, filter_name = feature_name.split('__')
        survey_name = {
            'PS1_DR1': 'PS',
            'AllWISE': 'WISE',
            'Gaia_EDR3': 'Gaia',
        }[survey_name]
        if survey_name == 'PS':
            filter_name = filter_name[0]
        elif survey_name == 'WISE':
            filter_name = filter_name[:2].capitalize()
        elif survey_name == 'Gaia':
            filter_name = filter_name[5:6]
    return '{}_\mathrm{{{}}}'.format(filter_name, survey_name)


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
