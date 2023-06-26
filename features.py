import os
import copy
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from ASTROMER.models import SingleBandEncoder

from env_config import PROJECT_PATH


FEATURE_SETS = {
    'ZTF': [
        'ad', 'ccd', 'chi2red', 'dec',
        'f1_BIC', 'f1_a', 'f1_amp', 'f1_b', 'f1_phi0', 'f1_power',
        'f1_relamp1', 'f1_relamp2', 'f1_relamp3', 'f1_relamp4', 'f1_relphi1',
        'f1_relphi2', 'f1_relphi3', 'f1_relphi4', 'field', 'i60r', 'i70r',
        'i80r', 'i90r', 'inv_vonneumannratio', 'iqr',
        'median', 'median_abs_dev', 'n', 'n_ztf_alerts', 'norm_excess_var',
        'norm_peak_to_peak_amp', 'pdot', 'period', 'quad', 'ra', 'roms',
        'significance', 'skew', 'smallkurt', 'stetson_j', 'stetson_k', 'sw',
        'welch_i', 'wmean', 'wstd',
        # 'dmdt',
        # 'features_present',  'mean_ztf_alert_braai',
    ],
    'astromer': pd.read_csv(os.path.join(PROJECT_PATH, 'outputs/feature_importance/RF_astromer.csv')).loc[:256,
                'feature'].values,
    'WISE': [
        'AllWISE__w1mpro', 'AllWISE__w2mpro', 'AllWISE__w3mpro', 'AllWISE__w4mpro',
        # 'AllWISE__ph_qual',
        # 'AllWISE___id',
        # 'AllWISE__w1sigmpro', 'AllWISE__w2sigmpro', 'AllWISE__w3sigmpro', 'AllWISE__w4sigmpro',
    ],
    'PS': [
        'PS1_DR1__gMeanPSFMag', 'PS1_DR1__rMeanPSFMag', 'PS1_DR1__iMeanPSFMag', 'PS1_DR1__zMeanPSFMag',
        # 'PS1_DR1__qualityFlag',
        # 'PS1_DR1__yMeanPSFMag', 'PS1_DR1__yMeanPSFMagErr',
        # 'PS1_DR1__gMeanPSFMagErr', 'PS1_DR1__rMeanPSFMagErr', 'PS1_DR1__iMeanPSFMagErr', 'PS1_DR1__zMeanPSFMagErr',
    ],
    'GAIA': [
        'Gaia_EDR3__phot_g_mean_mag',
        'Gaia_EDR3__parallax',
        'Gaia_EDR3__pmra', 'Gaia_EDR3__pmdec',
        'Gaia_EDR3__phot_bp_mean_mag', 'Gaia_EDR3__phot_rp_mean_mag', 'Gaia_EDR3__phot_bp_rp_excess_factor',
        # 'Gaia_EDR3__parallax_error', 'Gaia_EDR3__pmra_error', 'Gaia_EDR3__pmdec_error',
        # 'Gaia_EDR3__astrometric_excess_noise',
    ],
}


def get_features(data, with_variability=False):
    features = defaultdict(list)
    percentiles = [0, 2, 16, 50, 84, 98, 100] if with_variability else [50]

    for i in tqdm(range(data.shape[0])):
        for mag in ['g', 'r', 'i']:

            # Percentiles
            for p in percentiles:
                features['percentile_{}_{}'.format(p, mag)].append(np.percentile(data[i]['mag_{}'.format(mag)][0], p))

        # Colors for each percentile
        colors = [('g', 'r'), ('r', 'i'), ('g', 'i')]
        for mag_a, mag_b in colors:
            for p in percentiles:
                m_a = features['percentile_{}_{}'.format(p, mag_a)][i]
                m_b = features['percentile_{}_{}'.format(p, mag_b)][i]
                features['color_{}_{}-{}'.format(p, mag_a, mag_b)].append(m_a - m_b)
                features['ratio_{}_{}/{}'.format(p, mag_a, mag_b)].append(m_a / m_b)

    return pd.DataFrame(features)


def get_astromer_features(ztf_data):
    percentiles = [0, 2, 16, 50, 84, 98, 100]

    model = SingleBandEncoder()
    model = model.from_pretraining('ztfg')

    train_vectors = [np.array([[ztf_data[i]['mjd'][j], ztf_data[i]['mag'][j], ztf_data[i]['magerr'][j]] for j in
                               range(len(ztf_data[i]['mjd']))]) for i in tqdm(range(len(ztf_data)))]

    train_vectors = [vector[-199:] for vector in train_vectors]

    # Split it on my own and make a progress bar
    chunk_size = 10
    chunks = [train_vectors[i:i + chunk_size] for i in range(0, len(train_vectors), chunk_size)]

    # attention_vectors = da.zeros(shape=(321929, 200, 256), chunks=(100, 200, 256), dtype=np.float64)
    features = []
    # attention_vectors = np.empty((len(ztf_data), 200, 256), dtype=np.float64)
    # i = 0
    for chunk in tqdm(chunks):
        tmp = model.encode(chunk, batch_size=chunk_size, concatenate=True)
        assert np.all([tmp[j].shape[0] == chunk[j].shape[0] for j in range(len(chunk))])
        f = [np.percentile(row, percentiles, axis=0).flatten(order='F') for row in tmp]
        features.extend(f)
        # for j in range(len(tmp)):
        # Dimensions: light curves, observations in a light curve, number of attention layers
        # attention_vectors[i+j, :tmp[j].shape[0], :] = tmp[j]
        # i += len(tmp)

    return features


def add_colors(data):
    new_feature_sets = copy.deepcopy(FEATURE_SETS)
    for set_name in ['PS', 'WISE']:
        new_feature_names = []
        cols = FEATURE_SETS[set_name]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                feature_name = '{}__{}-{}'.format(cols[i].split('__')[0], cols[i].split('__')[1],
                                                  cols[j].split('__')[1])
                data.loc[:, feature_name] = data.loc[:, cols[i]] - data.loc[:, cols[j]]
                new_feature_names.append(feature_name)
        new_feature_sets[set_name].extend(new_feature_names)
    return data, new_feature_sets
