import random
from copy import deepcopy
from datetime import timedelta
from multiprocessing import Pool
import gc

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from astropy import time
import astropy.units as u
from astropy.coordinates import SkyCoord


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
    for i in tqdm(range(len(data)), 'Adding light curve statistics'):
        data[i]['mag median'] = np.median(data[i]['mag'])
        data[i]['mag err mean'] = np.mean(data[i]['magerr'])

        mjds = data[i]['mjd']
        if len(mjds) > 0:
            data[i]['n obs'] = len(mjds)
            data[i]['timespan'] = mjds[-1] - mjds[0]

        if len(mjds) > 1:
            intervals = [mjds[j + 1] - mjds[j] for j in range(len(mjds) - 1)]
            data[i]['cadence mean'] = np.mean(intervals)

            percentiles = np.percentile(intervals, [16, 50, 84])
            diff = np.diff(percentiles)
            data[i]['cadence median'] = percentiles[1]
            data[i]['cadence plus sigma'] = diff[1]
            data[i]['cadence minus sigma'] = diff[0]

    return data


def preprocess_ztf_light_curves(data, data_sdss=None, data_features=None, min_n_obs=20, with_multiprocessing=True):
    original_size = len(data)
    print('Preprocessing input data size: {}'.format(original_size))

    # Get at least 21 observations
    n_obs = [len(lc_dict['mjd']) for lc_dict in data]
    idx = np.array(n_obs) > min_n_obs
    data = np.array(data)[np.where(idx)]
    if data_sdss is not None:
        data_sdss = data_sdss.loc[idx].reset_index(drop=True)
    if data_features is not None:
        data_features = data_features.loc[idx].reset_index(drop=True)
    gc.collect()

    # Remove deep drilling
    if with_multiprocessing:
        with Pool(24) as p:
            data = p.map(average_nights, data)
    else:
            data = [average_nights(lc_dict) for lc_dict in tqdm(data, 'Averaging nights')]
    gc.collect()

    # Get at least 21 observations after the deep drilling
    n_obs = [len(lc_dict['mjd']) for lc_dict in data]
    idx = np.array(n_obs) > min_n_obs
    data = np.array(data)[np.where(idx)]
    if data_sdss is not None:
        data_sdss = data_sdss.loc[idx].reset_index(drop=True)
    if data_features is not None:
        data_features = data_features.loc[idx].reset_index(drop=True)
    print('Deep drilling and n_obs > 20: {} ({:.2f}%)'.format(len(data), len(data) / original_size * 100))
    gc.collect()

    if data_sdss is not None and data_features is not None:
        return data, data_sdss, data_features
    elif data_sdss is not None:
        return data, data_sdss
    else:
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
    return [limit_date_on_lc(lc_dict, date=date) for lc_dict in data]


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


def get_longests(data_ztf, data_sdss, data_features=None):
    data_ztf_dict = {'g': [], 'r': []}
    indices_sdss_dict = {'g': [], 'r': []}
    
    for i in range(len(data_ztf)):
        for filter in ['g', 'r']:
            cross_match_dict = data_ztf[i]
            
            # Check if a cross-match is present in the given filter
            if len(cross_match_dict['mjd_{}'.format(filter)]) > 0:
                
                # Remove too large distances, unknown error
                # distances = [math.sqrt((data_ztf[i]['ra_{}'.format(filter)][j] - data_sdss.loc[i, 'PLUG_RA']) ** 2 +
                #             (data_ztf[i]['dec_{}'.format(filter)][j] - data_sdss.loc[i, 'PLUG_DEC']) ** 2)
                #                 for j in range(len(data_ztf[i]['ra_{}'.format(filter)]))]
                pos_sdss = SkyCoord(data_sdss.loc[i, 'PLUG_RA'] * u.degree, data_sdss.loc[i, 'PLUG_DEC'] * u.degree)
                distances = []
                for j in range(len(data_ztf[i]['ra_{}'.format(filter)])):
                    pos_ztf = SkyCoord(
                        data_ztf[i]['ra_{}'.format(filter)][j] * u.degree,
                        data_ztf[i]['dec_{}'.format(filter)][j] * u.degree,
                    )
                    distances.append(pos_sdss.separation(pos_ztf).arcsecond)
                
                idx_ok = np.where(np.array(distances) < 1)[0]

                # If at least one good cross-match present
                if len(idx_ok) > 0:
                    # Add a proper index to the SDSS data frame
                    indices_sdss_dict[filter].append(i)

                    # Find the longest light curve within the accepted distances
                    mjds_arr = cross_match_dict['mjd_{}'.format(filter)]
                    n_obs = [len(mjds_arr[j]) for j in idx_ok]
                    idx_max = idx_ok[np.argmax(n_obs)]

                    # Add the longest light curve to the resulting data
                    new_row = {}
                    for key in ['id', 'ra', 'dec']:
                        new_row[key] = cross_match_dict['{}_{}'.format(key, filter)]
                    for key in ['mjd', 'mag', 'magerr']:
                        new_row[key] = cross_match_dict['{}_{}'.format(key, filter)][idx_max]
                    data_ztf_dict[filter].append(new_row)

    # Extract rows from the SDSS data frame
    data_sdss_dict = {}
    for filter in ['g', 'r']:
        data_sdss_dict[filter] = data_sdss.loc[indices_sdss_dict[filter]].reset_index(drop=True)

    if data_features is not None:
        data_features_dict = {}
        for filter in ['g', 'r']:
            data_features_dict[filter] = data_features.loc[indices_sdss_dict[filter]].reset_index(drop=True)

        return data_ztf_dict, data_sdss_dict, data_features_dict

    else:
        return data_ztf_dict, data_sdss_dict


def get_singles(data_ztf, data_sdss):
    # Singles
    idx_singles = [i for i in range(len(data_ztf)) if (len(data_ztf[i]['id_g']) == 1 and len(data_ztf[i]['id_r']) == 1 and len(data_ztf[i]['mjd_g'][0]) > 0 and len(data_ztf[i]['mjd_r'][0]) > 0)]

    data_ztf_singles = dict([(filter, get_filter(data_ztf[idx_singles], filter)) for filter in ['g', 'r']])
    data_sdss_singles = data_sdss.loc[idx_singles].reset_index(drop=True)

    # Get rid of a left over dimension
    for filter in ['g', 'r']:
        for i in range(len(data_ztf_singles[filter])):
            for key in ['id', 'ra', 'dec', 'mjd', 'mag', 'magerr']:
                if key in data_ztf_singles[filter][i]:
                    data_ztf_singles[filter][i][key] = data_ztf_singles[filter][i][key][0]

    return data_ztf_singles, data_sdss_singles


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
