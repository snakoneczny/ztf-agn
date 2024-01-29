import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from astropy.table import Table
from astropy import time
from datetime import timedelta

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def pretty_print(x):
    pretty_dict = {
        'n_obs': 'number of observation epochs',
        'n_obs_200': 'number of observation epochs (200)',
        'mag_median': 'median magnitude',
        'mean_cadence': 'mean cadence',
    }
    if x in pretty_dict:
        x = pretty_dict[x]    
    if 'cadence' in x or 'timespan' in x:
        x = ' '.join(x.split('_')) + ' [days]'
    return x


def average_nights(lc_dict):
    df = pd.DataFrame.from_dict(dict((k, lc_dict[k]) for k in ('mjd', 'mag', 'magerr')))

    df['date'] = time.Time(lc_dict['mjd'], format='mjd').datetime
    df['date'] += timedelta(hours=12)
    df['date'] = df.date.dt.floor('d')

    grouped = df.groupby(df.date.dt.floor('d'))[['mjd', 'mag', 'magerr']].mean()

    for col in ['mjd', 'mag', 'magerr']:
        lc_dict[col] = grouped[col].to_numpy()

    return lc_dict


def get_filter(ztf_data, filter):
    return [get_filter_record(record, filter) for record in ztf_data]


def get_filter_record(record, filter):
    record_new = {}
    for col in ['id', 'mjd', 'mag', 'magerr']:
        record_new[col] = record['{}_{}'.format(col, filter)]
    record_new['filter'] = filter
    return record_new


def get_first_matches(data):
    data = [get_first_match(matches) for matches in data]
    return data


def get_first_match(matches):
    for key in matches:
        matches[key] = matches[key][0]
    return matches


def get_data_stats(data):
    filter = data[0]['filter']
    title = 'filter: {}'.format(filter)

    # Number of observations
    n_obs = np.array([len(obj['mag']) for obj in data])

    plt.figure()
    sns.histplot(n_obs)
    plt.xlabel('number of observation epochs')
    plt.title(title)

    plt.figure()
    sns.ecdfplot(n_obs)
    plt.xlabel('number of observation epochs')
    plt.title(title)
    plt.show()

    print('Minimum number of observations: number of objects')
    size = len(n_obs)
    for min_obs in np.arange(0, 11):
        cut_obs = sum(n_obs >= min_obs)
        print('{}: {} ({:.1f}%)'.format(min_obs, cut_obs, cut_obs / size * 100))

    # Magnitude
    # color_palette = get_cubehelix_palette(len(columns))
    # plt.figure()
    # for i, column in enumerate(columns):
    #     sns.distplot(data[column], label=get_plot_text(column), kde=False, rug=False, norm_hist=True,
    #                  color=color_palette[i],
    #                  hist_kws={'alpha': 1.0, 'histtype': 'step', 'linewidth': 1.5, 'linestyle': get_line_style(i)})
    # plt.xlabel('probability')
    # plt.ylabel('normalized counts per bin')
    # plt.legend(framealpha=1.0)

    magnitudes = np.array([np.mean(obj['mag']) for obj in data])
    plt.figure()
    bins = np.histogram(magnitudes[~np.isnan(magnitudes)], 100)[1]
    for min_obs in [0, 2, 5, 10, 20]:
        idx = np.where(n_obs >= min_obs)
        sns.histplot(magnitudes[idx], bins=bins, element='step', fill=False, label='min obs: {}'.format(min_obs))
    plt.xlabel('magnitude {}'.format(filter))
    plt.title(title)
    plt.legend()
    plt.show()


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
