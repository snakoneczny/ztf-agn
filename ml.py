import os
import pickle

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from env_config import DATA_PATH, PROJECT_PATH
from features import FEATURES_DICT


def run_experiments(feature_labels, master_df, sdss_df, features_dict, filter=None):
    data_labels = [label for label in feature_labels if label != 'AstrmClf']
    if data_labels[0] != 'ZTF':
        data_labels.insert(0, 'ZTF')

    # Define experiments concatenating all features
    to_process = [feature_labels]
    # And single feature sets if comparing only two surveys
    if len(feature_labels) == 2:
        to_process = [[feature_label] for feature_label in feature_labels] + to_process

    # Make always the same train test split on the master data frame
    idx_train, idx_test = train_test_split(master_df.index, test_size=0.33, random_state=42)
    master_df.loc[idx_test, 'is_test'] = True
    master_df.loc[idx_train, 'is_test'] = False

    # Pick only ZTF subset, required to add Astromer data
    max_features = FEATURES_DICT['ZTF']
    reduced_df = master_df.dropna(subset=max_features)
    indices = reduced_df.index
    reduced_df = reduced_df.reset_index(drop=True)
    reduced_sdss = sdss_df.loc[indices].reset_index(drop=True)

    # Add astromer classification features to the train and test data frames
    if 'AstrmClf' in feature_labels:
        for is_test, df_label in [(True, 'val'), (False, 'train')]:
            ztf_date = '20210401'
            file_name = 'outputs/preds/ZTF_{}/ZTF_{}__band_{}__xmatch_ZTF__astromer_FC-1024-512-256__{}.csv'.format(
                ztf_date, ztf_date, filter, df_label)
            file_path = os.path.join(PROJECT_PATH, file_name)
            if os.path.exists(file_path):
                df_preds = pd.read_csv(file_path)
                for class_label in ['GALAXY', 'QSO', 'STAR']:                
                    reduced_df.loc[reduced_df['is_test'] == is_test, 'astrm_{}'.format(class_label.lower())] = df_preds[class_label].to_list()
    
    # Take all features together to reduce input data frames
    max_features = np.concatenate([FEATURES_DICT[label] for label in data_labels])
    reduced_df = reduced_df.dropna(subset=max_features)
    indices = reduced_df.index
    reduced_df = reduced_df.reset_index(drop=True)
    reduced_sdss = reduced_sdss.loc[indices].reset_index(drop=True)

    # Make one split for all experiments
    idx_train = reduced_df['is_test'] == False
    idx_test = reduced_df['is_test'] == True
    df_train = reduced_df.loc[idx_train]
    df_test = reduced_df.loc[idx_test]
    sdss_train = reduced_sdss.loc[idx_train]
    sdss_test = reduced_sdss.loc[idx_test]
    y_train = sdss_train['CLASS']
    y_test = sdss_test['CLASS']

    # Add things which are common to all feature set experiments
    results = pd.DataFrame()
    results['y_true'] = y_test.to_numpy()
    results['redshift'] = sdss_test['Z'].to_numpy()
    results['mag_median'] = df_test['median'].to_numpy()
    for col in ([
        'n_obs',  # 'n_obs_200',
        'timespan',  # 'timespan_200',
        'cadence_mean',  # 'cadence_mean_200',
        'cadence_median',  # 'cadence_median_200',
        'cadence_std',  # 'cadence_std_200'
    ]):
        results[col] = df_test[col].to_numpy()

    # Placeholder for classifiers
    classifiers = {}

    # Iterate over feature sets
    for feature_sets in tqdm(to_process):
        # Extract features
        features = np.concatenate([features_dict[feature_set] for feature_set in feature_sets])
        X_train = df_train[features]
        X_test = df_test[features]

        # Make classification
        clf = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=491237, n_jobs=48, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Save results
        features_label = ' + '.join(feature_sets)
        results['y_pred {}'.format(features_label)] = y_pred
        classifiers[features_label] = clf

    return results, classifiers


def get_train_data(ztf_date='20210401', filter='g', data_subsets=None, return_features=True):
    if ztf_date == '20210401':
        paths = {
            'sdss_x_ztf': 'ZTF_x_SDSS/ZTF_20210401/specObj-dr18_x_ztf_20210401__singles_filter_{}__features'.format(filter),
            'ztf_x_sdss_lc': 'ZTF_x_SDSS/ZTF_20210401/ztf_20210401_x_specObj-dr18__singles_filter_{}__features_lc-reduced'.format(filter),
            'ztf_x_sdss_features': 'ZTF_x_SDSS/ZTF_20210401/ztf_20210401_x_specObj-dr18__singles_filter_{}__features'.format(filter),
        }
    else:
        paths = {
            'sdss_x_ztf': 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_x_ztf_{}__singles__filter_{}__reduced'.format(ztf_date, ztf_date, filter),
            'ztf_x_sdss_lc': 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__singles__filter_{}__reduced'.format(ztf_date, ztf_date, filter),
            'ztf_x_sdss_features': None,
        }
    
    # Read ZTF x SDSS lightcurves with available features
    with open(os.path.join(DATA_PATH, paths['ztf_x_sdss_lc']), 'rb') as file:
        ztf_x_sdss_reduced = pickle.load(file)

    # Read SDSS x ZTF subset with available features
    with open(os.path.join(DATA_PATH, paths['sdss_x_ztf']), 'rb') as file:
        sdss_x_ztf_features = pickle.load(file)
    sdss_x_ztf_features = sdss_x_ztf_features.reset_index(drop=True)

    # Make always the same train test split on the data
    idx_train, idx_test = train_test_split(sdss_x_ztf_features.index, test_size=0.33, random_state=42)
    sdss_x_ztf_features.loc[idx_test, 'is_test'] = True
    sdss_x_ztf_features.loc[idx_train, 'is_test'] = False

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
            ztf_x_sdss_reduced = np.array(ztf_x_sdss_reduced)[indices.tolist()]
            sdss_x_ztf_features = sdss_x_ztf_features.loc[indices].reset_index(drop=True)

        return ztf_x_sdss_reduced, sdss_x_ztf_features, ztf_x_sdss_features
    else:
        return ztf_x_sdss_reduced, sdss_x_ztf_features


def get_embedding(data, perplexity=30.0, n_iter=5000):
    X = MinMaxScaler().fit_transform(data)
    X_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12.0, learning_rate=200.0,
                      n_iter=n_iter, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                      init='random', verbose=0, random_state=8364, method='barnes_hut', angle=0.5,
                      n_jobs=48).fit_transform(X)
    return X_embedded
