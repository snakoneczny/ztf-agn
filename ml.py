import os
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import json
from tqdm.autonotebook import tqdm
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score

from env_config import DATA_PATH, PROJECT_PATH
from features import FEATURES_DICT, get_features
from light_curves import subsample_light_curves
from ztf import LIMITING_MAGS


FILE_NAMES = {
    'clf': {
        'preds': 'outputs/preds/ZTF_{}/ZTF_{}_band_{}__xmatch_{}__XGB.csv',
        'features': 'outputs/feature_importance/ZTF_{}/ZTF_{}_band_{}__xmatch_{}__XGB.json',
        'model': 'outputs/models/ZTF_{}/ZTF_{}_band_{}__features_{}__XGB.pickle',
    },
    'z': {
        'preds': 'outputs/preds/ZTF_{}/ZTF_{}_band_{}__xmatch_{}__XGB_z.csv',
        'features': 'outputs/feature_importance/ZTF_{}/ZTF_{}_band_{}__xmatch_{}__XGB_z.json',
        'model': 'outputs/models/ZTF_{}/ZTF_{}_band_{}__features_{}__XGB_z.pickle',        
    },
}


def get_file_name(output, problem, ztf_date, filter, data_label, cross_features=False, mag_limit=False, qso_vs_rest=False,
                  n_obs_limit=False):
    file_name = FILE_NAMES[problem][output]
    file_name, extension = file_name.split('.')
    if mag_limit:
        file_name += '_mag-limit'        
    if n_obs_limit:
        file_name += '_min-n-obs-100'
    if cross_features:
        file_name += '_cross-features'
    if qso_vs_rest:
        file_name += '_qso-vs-rest'
    file_name = file_name + '.' + extension
    file_name = file_name.format(ztf_date, ztf_date, filter, data_label)
    return file_name


def run_experiments(data_labels, feature_labels, master_df, features_dict, filter, ztf_date, is_redshift,
                    is_mag_limit, is_cross_features, is_qso_vs_rest, is_n_obs_limit):
    # Add Astromer as features
    if 'ZTF' in data_labels:
        file_name = 'outputs/preds/ZTF_{}/ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256'.format(
            ztf_date, ztf_date, filter)
        if is_redshift:
            file_name += '__redshift'
        file_name += '__{}.csv'
        for split_label in ['train', 'val', 'test']:
            file_path = os.path.join(PROJECT_PATH, file_name.format(split_label))
            if os.path.exists(file_path):
                df_preds = pd.read_csv(file_path)
                if is_redshift:
                    master_df.loc[master_df['train_split'] == split_label, 'astrm_z_qso'] = df_preds['y_pred'].to_list()
                else:
                    for class_label in ['GALAXY', 'QSO', 'STAR']:
                        master_df.loc[master_df['train_split'] == split_label, 'astrm_{}'.format(class_label.lower())] = df_preds[class_label].to_list()

    # Take all data together to reduce the input data frame
    data_features = np.concatenate([FEATURES_DICT[label] for label in data_labels])
    master_df = master_df.dropna(subset=data_features).reset_index(drop=True)

    # Limit the magnitudes
    if is_mag_limit:
        master_df = limit_mags(master_df, filter, data_labels)

    # Limit number of observations
    if is_n_obs_limit:
        master_df = master_df.loc[master_df['n obs'] >= 100]

    # Make one split for all experiments
    df_dict, y_dict = {}, {}
    split_labels = ['train', 'val', 'test']
    cls_dict = {'GALAXY': 0, 'QSO': 1, 'STAR': 0} if is_qso_vs_rest else {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
    for label in split_labels:
        df_dict[label] = master_df.loc[master_df['train_split'] == label]
        if is_redshift:
            y_dict[label] = df_dict[label]['Z']
        else:
            y_dict[label] = df_dict[label]['CLASS']
            y_dict[label] = [cls_dict[x] for x in y_dict[label]]

    # Add things which are common to all feature set experiments
    results = df_dict['test'][[
        'CLASS', 'Z', 'mag median', 'mag err mean', 'n obs', 'timespan',
        'cadence mean', 'cadence median', 'cadence plus sigma', 'cadence minus sigma',
        'PS1_DR1__gMeanPSFMag', 'PS1_DR1__rMeanPSFMag', 'PS1_DR1__iMeanPSFMag', 'PS1_DR1__zMeanPSFMag',
        'AllWISE__w1mpro', 'AllWISE__w2mpro', 'Gaia_EDR3__phot_g_mean_mag'
    ]]

    # Placeholder for classifiers
    models = {}

    # Iterate over feature sets
    for feature_sets in tqdm(feature_labels):
        # Extract features
        features = get_features(features_dict, feature_sets, is_cross_features)
        X_dict = {}
        for label in split_labels:
            X_dict[label] = df_dict[label][features].values

        # Train a model
        if is_redshift:
            model = XGBRegressor(
                max_depth=9, learning_rate=0.1, gamma=0, min_child_weight=1, colsample_bytree=0.9, subsample=0.8,
                scale_pos_weight=2, reg_alpha=0, reg_lambda=1, n_estimators=100000, objective='reg:squarederror',
                booster='gbtree', max_delta_step=0, colsample_bylevel=1, base_score=0.5, random_state=18235, missing=np.nan,
                verbosity=0, n_jobs=36, early_stopping_rounds=20,
            )
        else:
            objective = 'binary:logistic' if is_qso_vs_rest else 'multi:softmax'
            model = XGBClassifier(
                max_depth=9, learning_rate=0.1, gamma=0, min_child_weight=1, colsample_bytree=0.9, subsample=0.8,
                scale_pos_weight=2, reg_alpha=0, reg_lambda=1, n_estimators=100000, objective=objective,
                booster='gbtree', max_delta_step=0, colsample_bylevel=1, base_score=0.5, random_state=18235, missing=np.nan,
                verbosity=0, n_jobs=36, early_stopping_rounds=20,
            )

        # Fit the model
        model.fit(X_dict['train'], y_dict['train'], eval_set=[(X_dict['val'], y_dict['val'])], verbose=False)

        # Save the model and features
        features_label = ' + '.join(feature_sets)
        models[features_label] = model
        
        # Make classification and save the results
        if is_redshift:
            z_pred = model.predict(X_dict['test'])
            results['z_pred {}'.format(features_label)] = z_pred
        else:
            y_pred_proba = model.predict_proba(X_dict['test'])
            y_pred_cls = np.argmax(y_pred_proba, axis=1)
            cls_dict = {0: 'GALAXY/STAR', 1: 'QSO'} if is_qso_vs_rest else {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}
            y_pred_cls = [cls_dict[x] for x in y_pred_cls]

            results['y_pred {}'.format(features_label)] = y_pred_cls
            if is_qso_vs_rest:
                results['y_galaxy_star {}'.format(features_label)] = y_pred_proba[:, 0]
                results['y_qso {}'.format(features_label)] = y_pred_proba[:, 1]
            else:
                results['y_galaxy {}'.format(features_label)] = y_pred_proba[:, 0]
                results['y_qso {}'.format(features_label)] = y_pred_proba[:, 1]
                results['y_star {}'.format(features_label)] = y_pred_proba[:, 2]

        # Print basic stats, QSO F1 and 3-class accuracy
        cls_dict = {0: 'GALAXY/STAR', 1: 'QSO'} if is_qso_vs_rest else {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}
        tmp = [cls_dict[x] for x in y_dict['test']]
        qso_f1 = f1_score(
            tmp, y_pred_cls, average=None, zero_division=0, labels=['GALAXY', 'QSO', 'STAR'],
        )[1]
        accuracy = accuracy_score(tmp, y_pred_cls)
        print('Data: {}, features: {}'.format(data_labels, feature_sets))
        print('QSO F1: {:.5f}, accuracy: {:.5f}'.format(qso_f1, accuracy))

    return results, models


def limit_mags(data, filter, data_labels):
    if 'ZTF' in data_labels:
        data = data.loc[(data['mag median'] < LIMITING_MAGS[filter])]

    if 'PS' in data_labels:
        for filter_name in ['g', 'r', 'i', 'z']:
            col = 'PS1_DR1__{}MeanPSFMag'.format(filter_name)
            data = data.loc[data[col] < LIMITING_MAGS[col]]

    if 'WISE' in data_labels:
        for filter_name in ['w1', 'w2']:
            col = 'AllWISE__{}mpro'.format(filter_name)
            data = data.loc[data[col] < LIMITING_MAGS[col]]

    if 'GAIA' in data_labels:
        for filter_name in ['g']:
            col = 'Gaia_EDR3__phot_{}_mean_mag'.format(filter_name)
            data = data.loc[data[col] < LIMITING_MAGS[col]]

    return data


def load_results(data_compositions, ztf_date, filters, mag_limit=False, cross_features=False):
    # Resulting data structures
    results_dict = {}
    feature_importance_dict = {}

    for task in ['clf', 'z']:
        results_dict[task] = {}
        feature_importance_dict[task] = {}

        for filter in filters:
            results_dict[task][filter] = {}
            feature_importance_dict[task][filter] = {}

            for data_composition in data_compositions:
                data_label = '_'.join(data_composition)

                # Read predictions
                file_name = get_file_name('preds', task, ztf_date, filter, data_label,
                                          cross_features=cross_features, mag_limit=mag_limit)                
                file_path = os.path.join(PROJECT_PATH, file_name)
                if os.path.exists(file_path):
                    results_dict[task][filter][data_label] = pd.read_csv(file_path)
                    print('Loaded results: {}'.format(file_path))

                # Read feature importances
                file_name = get_file_name('features', task, ztf_date, filter, data_label,
                                          cross_features=cross_features, mag_limit=mag_limit)                
                file_path = os.path.join(PROJECT_PATH, file_name)
                if os.path.exists(file_path):
                    with open(file_path) as file:
                        feature_importance_dict[task][filter][data_label] = json.load(file)
                    print('Loaded feature importances: {}'.format(file_path))

                # Check if Astromer present and add as well
                file_name = 'outputs/preds/ZTF_{}/ZTF_{}__band_{}__xmatch_{}__astromer_FC-1024-512-256'
                if task == 'z':
                    file_name += '__redshift'
                file_name += '__test.csv'
                file_name = file_name.format(ztf_date, ztf_date, filter, data_label)
                file_path = os.path.join(PROJECT_PATH, file_name)
                if os.path.exists(file_path) and data_label in results_dict[task][filter]:
                    df_preds = pd.read_csv(file_path)
                    col_pred = 'z_pred Astrm' if task == 'z' else 'y_pred Astrm'
                    results_dict[task][filter][data_label][col_pred] = df_preds['y_pred']
                    print('Loaded Astromer: {}'.format(file_path))

        # TODO: Refactor
        for filter in filters:
            for key in results_dict[task][filter]:
                col_true_a = 'CLASS' if task == 'clf' else 'Z'
                col_true_b = 'z_true' if task == 'z' else 'y_true'
                results_dict[task][filter][key][col_true_b] = results_dict[task][filter][key][col_true_a]

    return results_dict['clf'], results_dict['z'], feature_importance_dict['clf'], feature_importance_dict['z']


def read_train_matrices(ztf_date, filter_name):
    file_name = 'ZTF_x_SDSS/ZTF_{}/matrices/ZTF_{}_filter_{}.pickle'.format(ztf_date, ztf_date, filter_name)
    with open(os.path.join(DATA_PATH, file_name), 'rb') as file:
        data = pickle.load(file)
    return data['X']['train'], data['X']['val'], data['X']['test'], data['y']['train'], data['y']['val'], data['y']['test'], data['z']['train'], data['z']['val'], data['z']['test']


def save_train_matrices(ztf_date, filter_name, X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test):
    data = {
        'X': {
            'train': X_train,
            'val': X_val,
            'test': X_test,
        },
        'y': {
            'train': y_train,
            'val': y_val,
            'test': y_test,
        },
        'z': {
            'train': z_train,
            'val': z_val,
            'test': z_test,
        },
    }
    file_name = 'ZTF_x_SDSS/ZTF_{}/matrices/ZTF_{}_filter_{}.pickle'.format(ztf_date, ztf_date, filter_name)
    with open(os.path.join(DATA_PATH, file_name), 'wb') as file:
        pickle.dump(data, file)


def get_train_matrices(ztf_date, filter, minimum_timespan=None, timespan=None, frac_n_obs=None):
    # Read the train data
    ztf_x_sdss, sdss_x_ztf = \
        get_train_data(ztf_date=ztf_date, filter=filter, data_subsets=['ZTF'], return_features=False)
    print('Data read: {}'.format(len(ztf_x_sdss)))
    
    # Make a sampling experiment
    n_obs_subsampled = None
    if timespan:
        ztf_x_sdss, sdss_x_ztf, n_obs_subsampled = subsample_light_curves(
            ztf_x_sdss, sdss_x_ztf, minimum_timespan=minimum_timespan,
            timespan=timespan, frac_n_obs=frac_n_obs)
    print('Subsampled data: {}'.format(len(ztf_x_sdss)))

    # Change shape to feed a neural network and sample random 200 observations
    X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = make_train_test_split(ztf_x_sdss, sdss_x_ztf)

    if timespan:
        return X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test, n_obs_subsampled
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test


def make_train_test_split(ztf_x_sdss, sdss_x_ztf, with_multiprocessing=False):
    random.seed(1257)
    if with_multiprocessing:
        print('Building the X matrix')
        with Pool(24) as p:
            X = p.map(subsample_matrix_row, ztf_x_sdss)
    else:
        X = [subsample_matrix_row(lc_dict) for lc_dict in tqdm(ztf_x_sdss, 'Input matrix')]

    class_dict = {
        'GALAXY': 0,
        'QSO': 1,
        'STAR': 2,
    }
    y = sdss_x_ztf['CLASS'].apply(lambda x: class_dict[x]).to_list()
    z = sdss_x_ztf['Z'].to_list()

    idx_train = np.where(sdss_x_ztf['train_split'] == 'train')[0]
    idx_val = np.where(sdss_x_ztf['train_split'] == 'val')[0]
    idx_test = np.where(sdss_x_ztf['train_split'] == 'test')[0]
    X_train = [X[i] for i in idx_train]
    X_val = [X[i] for i in idx_val]
    X_test = [X[i] for i in idx_test]
    y_train = [y[i] for i in idx_train]
    y_val = [y[i] for i in idx_val]
    y_test = [y[i] for i in idx_test]
    z_train = [z[i] for i in idx_train]
    z_val = [z[i] for i in idx_val]
    z_test = [z[i] for i in idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test


def subsample_matrix_row(lc_dict):
    idx = range(len(lc_dict['mjd'])) if len(lc_dict['mjd']) <= 200 else sorted(random.sample(range(len(lc_dict['mjd'])), 200))
    return np.array([[lc_dict['mjd'][i], lc_dict['mag'][i], lc_dict['magerr'][i]] for i in idx])


def get_train_data(ztf_date='20210401', filter='g', data_subsets=None, return_light_curves=True, return_features=False):
    if ztf_date == '20210401':
        paths = {
            'sdss_x_ztf': 'ZTF_x_SDSS/ZTF_20210401/specObj-dr18_x_ztf_20210401__singles_filter_{}__features'.format(filter),
            'ztf_x_sdss_lc': 'ZTF_x_SDSS/ZTF_20210401/ztf_20210401_x_specObj-dr18__singles_filter_{}__features_lc-reduced'.format(filter),
            'ztf_x_sdss_features': 'ZTF_x_SDSS/ZTF_20210401/ztf_20210401_x_specObj-dr18__singles_filter_{}__features'.format(filter),
        }
    elif ztf_date == '20230821':
        paths = {
            'sdss_x_ztf': 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_x_ztf_{}__singles__filter_{}__reduced'.format(ztf_date, ztf_date, filter),
            'ztf_x_sdss_lc': 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__singles__filter_{}__reduced'.format(ztf_date, ztf_date, filter),
        }
    else:
        paths = {
            'sdss_x_ztf': 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_x_ztf_{}__longests_filter_{}__reduced'.format(ztf_date, ztf_date, filter),
            'ztf_x_sdss_lc': 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__longests_filter_{}__reduced'.format(ztf_date, ztf_date, filter),
            'sdss_pwg_x_ztf': 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_PWG_x_ztf_{}__longests_filter_{}__reduced'.format(ztf_date, ztf_date, filter),
        }

    # Read ZTF x SDSS lightcurves
    if return_light_curves:
        with open(os.path.join(DATA_PATH, paths['ztf_x_sdss_lc']), 'rb') as file:
            ztf_x_sdss = pickle.load(file)
    else:
        ztf_x_sdss = None

    # Read SDSS x ZTF subset
    with open(os.path.join(DATA_PATH, paths['sdss_x_ztf']), 'rb') as file:
        sdss_x_ztf = pickle.load(file).reset_index(drop=True)

    # Read PWG features
    if 'sdss_pwg_x_ztf' in paths:
        with open(os.path.join(DATA_PATH, paths['sdss_pwg_x_ztf']), 'rb') as file:
            sdss_pwg_x_ztf = pickle.load(file).reset_index(drop=True)

        # Concat SDSS with the features
        sdss_x_ztf = pd.concat([sdss_x_ztf, sdss_pwg_x_ztf], axis=1)

    # Make always the same train test split
    idx_train, idx_test = train_test_split(sdss_x_ztf.index, test_size=0.15, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=42)
    
    sdss_x_ztf.loc[idx_train, 'train_split'] = 'train'
    sdss_x_ztf.loc[idx_val, 'train_split'] = 'val'
    sdss_x_ztf.loc[idx_test, 'train_split'] = 'test'
    
    # Take a subset with features
    if ztf_date == '20210401' and return_features:
        # Read ZTF x SDSS subset with available features
        with open(os.path.join(DATA_PATH, paths['ztf_x_sdss_features']), 'rb') as file:
            ztf_x_sdss_features = pickle.load(file)
        ztf_x_sdss_features = ztf_x_sdss_features.reset_index(drop=True)

        if data_subsets is not None:
            features_list = np.concatenate([FEATURES_DICT[label] for label in data_subsets])
            ztf_x_sdss_features = ztf_x_sdss_features.dropna(subset=features_list)
            indices = ztf_x_sdss_features.index
            ztf_x_sdss_features = ztf_x_sdss_features.reset_index(drop=True)
            ztf_x_sdss = np.array(ztf_x_sdss)[indices.tolist()]
            sdss_x_ztf = sdss_x_ztf.loc[indices].reset_index(drop=True)

        return ztf_x_sdss, sdss_x_ztf, ztf_x_sdss_features

    else:
        return ztf_x_sdss, sdss_x_ztf


def get_embedding(data, perplexity=30.0, n_iter=5000):
    X = MinMaxScaler().fit_transform(data)
    X_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12.0, learning_rate=200.0,
                      n_iter=n_iter, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                      init='random', verbose=0, random_state=8364, method='barnes_hut', angle=0.5,
                      n_jobs=48).fit_transform(X)
    return X_embedded
