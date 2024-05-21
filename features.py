import os
import copy

import numpy as np
from tqdm import tqdm
from ASTROMER.models import SingleBandEncoder

from env_config import PROJECT_PATH

FEATURES_DICT = {
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
    'astromer_all': np.concatenate((
        ['astromer_{}_p{}'.format(att, p) for att in range(256) for p in [0, 2, 16, 50, 84, 98, 100]],
        ['astromer_{}_mean'.format(att) for att in range(256)],
        ['astromer_{}_sum'.format(att) for att in range(256)],
    )),
    'astromer_min-max':
        ['astromer_{}_p{}'.format(att, p) for att in range(256) for p in [0, 2, 98, 100]],
    # 'astromer_top_clf': pd.read_csv(
    #     os.path.join(PROJECT_PATH, 'outputs/feature_importance/astromer_retrained-encoder_RF.csv')).loc[:256,
    #             'feature'].values,
    'AstrmClf': ['astrm_galaxy', 'astrm_qso', 'astrm_star'],
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


def add_colors(data):
    new_feature_sets = copy.deepcopy(FEATURES_DICT)
    for set_name in ['PS', 'WISE']:
        new_feature_names = []
        cols = FEATURES_DICT[set_name]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                feature_name = '{}__{}-{}'.format(cols[i].split('__')[0], cols[i].split('__')[1],
                                                  cols[j].split('__')[1])
                data.loc[:, feature_name] = data.loc[:, cols[i]] - data.loc[:, cols[j]]
                new_feature_names.append(feature_name)
        new_feature_sets[set_name].extend(new_feature_names)
    return data, new_feature_sets


def get_astromer_features(ztf_data, retrained=None):
    percentiles = [0, 2, 16, 50, 84, 98, 100]

    model = SingleBandEncoder()
    model = model.from_pretraining('ztfg')

    if retrained:
        if retrained == 'QSO':
            model.load_weights(os.path.join(PROJECT_PATH, 'outputs/models/astromer_g__QSO'))
        else:
            model.load_weights(os.path.join(PROJECT_PATH, 'outputs/models/astromer_g'))

    train_vectors = [np.array([[ztf_data[i]['mjd'][j], ztf_data[i]['mag'][j], ztf_data[i]['magerr'][j]] for j in
                               range(len(ztf_data[i]['mjd']))]) for i in tqdm(range(len(ztf_data)))]

    train_vectors = [vector[-199:] for vector in train_vectors]

    # Split it on my own and make a progress bar
    chunk_size = 10
    chunks = [train_vectors[i:i + chunk_size] for i in range(0, len(train_vectors), chunk_size)]

    features = []
    for chunk in tqdm(chunks):
        tmp = model.encode(chunk, batch_size=chunk_size, concatenate=True)
        assert np.all([tmp[j].shape[0] == chunk[j].shape[0] for j in range(len(chunk))])

        # Add percentiles
        f = [np.percentile(row, percentiles, axis=0).flatten(order='F') for row in tmp]

        # Add sum and mean
        for i in range(len(f)):
            f[i] = np.concatenate((f[i], np.mean(tmp[i], axis=0), np.sum(tmp[i], axis=0)))

        features.extend(f)

    return features
