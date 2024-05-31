import os
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import json
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from env_config import DATA_PATH, PROJECT_PATH
from features import FEATURES_DICT
from light_curves import subsample_light_curves


def run_experiments(feature_labels, master_df, features_dict, filter, ztf_date):
    data_labels = [label for label in feature_labels if label != 'AstrmClf']

    # Define experiments concatenating all features
    to_process = [feature_labels]
    # And single feature sets if comparing only two surveys
    if len(feature_labels) == 2:
        to_process = [[feature_label] for feature_label in feature_labels] + to_process

    # Add Astromer as features
    if 'AstrmClf' in feature_labels:
        for split_label in ['train', 'val', 'test']:
            file_name = 'outputs/preds/ZTF_{}/ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256__{}.csv'.format(
                ztf_date, ztf_date, filter, split_label)
            file_path = os.path.join(PROJECT_PATH, file_name)
            if os.path.exists(file_path):
                df_preds = pd.read_csv(file_path)
                for class_label in ['GALAXY', 'QSO', 'STAR']:                
                    master_df.loc[master_df['train_split'] == split_label, 'astrm_{}'.format(class_label.lower())] = df_preds[class_label].to_list()
    
    # Take all features together to reduce the input data frame
    if len(data_labels) > 0:
        max_features = np.concatenate([FEATURES_DICT[label] for label in data_labels])
        master_df = master_df.dropna(subset=max_features).reset_index(drop=True)

    # Make one split for all experiments
    df_train = master_df.loc[master_df['train_split'] == 'train']
    df_test = master_df.loc[master_df['train_split'] == 'test']
    y_train = df_train['CLASS']

    # Add things which are common to all feature set experiments
    results = df_test[['CLASS', 'Z', 'mag median', 'mag err mean', 'n obs', 'timespan',
                 'cadence mean', 'cadence median', 'cadence plus sigma', 'cadence minus sigma']]

    # Placeholder for classifiers
    classifiers = {}

    # Iterate over feature sets
    for feature_sets in tqdm(to_process):
        # Extract features
        features = np.concatenate([features_dict[feature_set] for feature_set in feature_sets])
        X_train = df_train[features]
        X_test = df_test[features]

        # Make classification
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', random_state=491237, n_jobs=24, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Save results
        features_label = ' + '.join(feature_sets)
        results['y_pred {}'.format(features_label)] = y_pred
        classifiers[features_label] = clf

    return results, classifiers


def load_results(data_compositions, ztf_date, filters):
    # Resulting data structures
    results_dict = {}
    feature_importance_dict = {}

    for filter in filters:
        results_dict[filter] = {}
        feature_importance_dict[filter] = {}

        for data_composition in data_compositions:
            data_label = '_'.join(data_composition)

            # Read predictions
            file_name = 'ZTF_{}_band_{}__{}__RF'.format(ztf_date, filter, data_label)
            file_path = os.path.join(PROJECT_PATH, 'outputs/preds/ZTF_{}'.format(ztf_date), file_name + '__test.csv')
            results_dict[filter][data_label] = pd.read_csv(file_path)
            print('Loaded results: {}'.format(file_path))

            # Read feature importances
            file_path = os.path.join(PROJECT_PATH, 'outputs/feature_importance/ZTF_{}'.format(ztf_date), file_name + '.json')
            with open(file_path) as file:
                feature_importance_dict[filter][data_label] = json.load(file)
            print('Loaded feature importances: {}'.format(file_path))

            # Check if Astromer present and add as well
            file_name = 'outputs/preds/ZTF_{}/ZTF_{}__band_{}__xmatch_{}__astromer_FC-1024-512-256'.format(ztf_date, ztf_date, filter, data_label)
            file_name += '__test.csv'
            file_path = os.path.join(PROJECT_PATH, file_name)
            if os.path.exists(file_path):
                df_preds = pd.read_csv(file_path)
                prediction_label = 'y_pred Astrm'
                results_dict[filter][data_label][prediction_label] = df_preds['y_pred']
                print('Loaded Astromer: {}'.format(file_path))
    
    # TODO: Refactor
    for filter in filters:
        for key in results_dict[filter]:
            results_dict[filter][key]['y_true'] = results_dict[filter][key]['CLASS']

    return results_dict, feature_importance_dict


def read_train_matrices(ztf_date, filter_name):
    path = os.path.join(DATA_PATH, 'ZTF_x_SDSS/ZTF_20240117/matrices/ZTF_{}_filter_{}__{}_{}.pickle')
    data = {'X': {}, 'y': {}}
    for label_matrix in data:
        for label_split in ['train', 'val', 'test']:
            with open(path.format(ztf_date, filter_name, label_matrix, label_split), 'rb') as file:
                data[label_matrix][label_split] = pickle.load(file)
    return data['X']['train'], data['X']['val'], data['X']['test'], data['y']['train'], data['y']['val'], data['y']['test']


def get_train_matrices(ztf_date, filter, minimum_timespan=None, timespan=None, frac_n_obs=None):
    # Read the train data
    ztf_x_sdss, sdss_x_ztf = \
        get_train_data(ztf_date=ztf_date, filter=filter, data_subsets=['ZTF'], return_features=False)

    # Make a sampling experiment
    n_obs_subsampled = None
    if timespan:
        ztf_x_sdss, sdss_x_ztf, n_obs_subsampled = subsample_light_curves(
            ztf_x_sdss, sdss_x_ztf, minimum_timespan=minimum_timespan,
            timespan=timespan, frac_n_obs=frac_n_obs)

    # Change shape to feed a neural network and sample random 200 observations
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_test_split(ztf_x_sdss, sdss_x_ztf)

    if timespan:
        return X_train, X_val, X_test, y_train, y_val, y_test, n_obs_subsampled
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


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

    idx_train = np.where(sdss_x_ztf['train_split'] == 'train')[0]
    idx_val = np.where(sdss_x_ztf['train_split'] == 'val')[0]
    idx_test = np.where(sdss_x_ztf['train_split'] == 'test')[0]
    X_train = [X[i] for i in idx_train]
    X_val = [X[i] for i in idx_val]
    X_test = [X[i] for i in idx_test]
    y_train = [y[i] for i in idx_train]
    y_val = [y[i] for i in idx_val]
    y_test = [y[i] for i in idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test


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
