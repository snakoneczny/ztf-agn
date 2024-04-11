import random
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from astropy import time
from tqdm.notebook import tqdm


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


def add_lc_stats(data):
    for i in tqdm(range(len(data))):
        data[i]['mag median'] = np.median(data[i]['mag'])
        data[i]['mag err mean'] = np.mean(data[i]['magerr'])

        mjds = data[i]['mjd']
        if len(mjds) > 0:
            data[i]['n obs'] = len(mjds)
            data[i]['timespan'] = mjds[-1] - mjds[0]

        if len(mjds) > 1:
            cadences = [mjds[j + 1] - mjds[j] for j in range(len(mjds) - 1)]
            data[i]['cadence mean'] = np.mean(cadences)
            data[i]['cadence median'] = np.median(cadences)

    return data


def preprocess_ztf_light_curves(data):
    original_size = len(data)
    print('Original data size: {}'.format(original_size))

    # Remove deep drilling
    data = [average_nights(lc_dict) for lc_dict in tqdm(data) if len(lc_dict['mjd']) > 20]
    print('Removing deep drilling at n_obs > 20: {} ({:.2f}%)'.format(len(data), len(data) / original_size * 100))

    # Get at least 21 observations
    data = [lc_dict for lc_dict in data if len(lc_dict['mjd']) > 20]
    print('Number of observations higher than 20: {} ({:.2f}%)'.format(len(data), len(data) / original_size * 100))

    return data


def average_nights(lc_dict):
    df = pd.DataFrame.from_dict(dict((k, lc_dict[k]) for k in ('mjd', 'mag', 'magerr')))

    df['date'] = time.Time(lc_dict['mjd'], format='mjd').datetime
    df['date'] += timedelta(hours=12)
    df['date'] = df.date.dt.floor('d')

    grouped = df.groupby(df.date.dt.floor('d'))[['mjd', 'mag', 'magerr']].mean()

    for col in ['mjd', 'mag', 'magerr']:
        lc_dict[col] = grouped[col].to_numpy()

    return lc_dict


def limit_date(data, date):
    return [limit_date_on_lc(lc_dict, date=date) for lc_dict in tqdm(data)]


def limit_date_on_lc(lc_dict, date):
    mjds = lc_dict['mjd']
    idx_last = 0
    while idx_last < len(mjds) and mjds[idx_last] < date:
        idx_last += 1
    for dict_key in ['mjd', 'mag', 'magerr']:
        lc_dict[dict_key] = lc_dict[dict_key][:idx_last]
    return lc_dict


def subsample_light_curves(light_curves, features_df=None, minimum_timespan=1000, timespan=1000, frac_n_obs=1.0):
    # Subset of the minimal timespan
    idx_timespan = [i for i in range(len(light_curves)) if light_curves[i]['mjd'][-1] - light_curves[i]['mjd'][0] >= minimum_timespan]
    lc_sampled = deepcopy(light_curves[idx_timespan])
    if features_df is not None:
        features_df = features_df.loc[idx_timespan].reset_index(drop=True)

    # Truncate lightcurves to the exact timespan
    for i in range(len(lc_sampled)):
        mjds = lc_sampled[i]['mjd']
        idx_start = 0
        while (mjds[-1] - mjds[idx_start]) > timespan:
            idx_start += 1
        for dict_key in ['mjd', 'mag', 'magerr']:
            lc_sampled[i][dict_key] = lc_sampled[i][dict_key][idx_start:]

    # Find median number of observations
    n_obs_arr = [len(lc_dict['mjd']) for lc_dict in lc_sampled]
    n_obs_median = np.median(n_obs_arr)

    # Subsample minimal number of observations
    idx_median = [i for i in range(len(lc_sampled)) if len(lc_sampled[i]['mjd']) >= n_obs_median]
    lc_sampled = lc_sampled[idx_median]
    if features_df is not None:
        features_df = features_df.loc[idx_median].reset_index(drop=True)

    # Sample a fraction of observations
    n_obs_goal = int(frac_n_obs * n_obs_median)
    random.seed(7235)
    for i in range(len(lc_sampled)):
        idx_goal = sorted(random.sample(range(len(lc_sampled[i]['mjd'])), n_obs_goal))
        for dict_key in ['mjd', 'mag', 'magerr']:
            lc_sampled[i][dict_key] = lc_sampled[i][dict_key][idx_goal]

    return lc_sampled, features_df, n_obs_goal


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
