import os

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from env_config import PROJECT_PATH
from features import FEATURES_DICT


def run_experiments(feature_labels, master_df, sdss_df, features_dict, filter=None):
    data_labels = [label for label in feature_labels if label != 'AstrmClf']
    if data_labels[0] != 'ZTF':
        data_labels.insert(0, 'ZTF')

    # Define experiments, single for each survey, and one concatenating all features
    to_process = [[feature_label] for feature_label in feature_labels]

    # Add pairs
    if len(feature_labels) > 1:
        for i in range(len(feature_labels) - 1):
            for j in range(i + 1, len(feature_labels)):
                to_process.append([feature_labels[i], feature_labels[j]])

    # Add all
    if len(feature_labels) > 2:
        to_process.append(feature_labels)

    # Take all features together to make a subset
    max_features = np.concatenate([FEATURES_DICT[label] for label in data_labels])

    # Generate indices of common rows for all experiments
    reduced_df = master_df.dropna(subset=max_features)
    indices = reduced_df.index

    # Reduce input data frames
    reduced_df = reduced_df.reset_index(drop=True)
    reduced_sdss = sdss_df.loc[indices].reset_index(drop=True)

    # Make one split for all experiments
    df_train, df_test, sdss_train, sdss_test = train_test_split(
        reduced_df, reduced_sdss, test_size=0.33, random_state=42
    )
    y_train = sdss_train['CLASS']
    y_test = sdss_test['CLASS']

    # TODO: use train/test split indices to add those features to the main dataframe, somewhere earlier, in a function
    # Add astromer classification features to the train and test data frames
    if 'AstrmClf' in feature_labels:
        for df_label, df_exp in [('train', df_train), ('val', df_test)]:
            data_label = '_'.join(data_labels)
            file_name = 'outputs/preds/{}-band__{}__astromer_FC-1024-512-256__{}.csv'.format(filter, data_label, df_label)
            file_path = os.path.join(PROJECT_PATH, file_name)
            if os.path.exists(file_path):
                df_preds = pd.read_csv(file_path)
                df_exp['astrm_galaxy'] = df_preds['GALAXY'].to_list()
                df_exp['astrm_qso'] = df_preds['QSO'].to_list()
                df_exp['astrm_star'] = df_preds['STAR'].to_list()

    # Add things which are common to all feature set experiments
    results = pd.DataFrame()
    results['y_true'] = y_test.to_numpy()
    results['redshift'] = sdss_test['Z'].to_numpy()
    results['mag_median'] = df_test['median'].to_numpy()
    for col in ([
        'n_obs', 'n_obs_200',
        'timespan', 'timespan_200',
        'cadence_mean', 'cadence_mean_200',
        'cadence_median', 'cadence_median_200',
        'cadence_std', 'cadence_std_200'
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
        clf = RandomForestClassifier(
            n_estimators=500, criterion='gini', random_state=491237, n_jobs=48, verbose=0
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Save results
        features_label = ' + '.join(feature_sets)
        results['y_pred {}'.format(features_label)] = y_pred
        classifiers[features_label] = clf

    return results, classifiers


def get_embedding(data, perplexity=30.0, n_iter=5000):
    X = MinMaxScaler().fit_transform(data)
    X_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12.0, learning_rate=200.0,
                      n_iter=n_iter, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                      init='random', verbose=0, random_state=8364, method='barnes_hut', angle=0.5,
                      n_jobs=48).fit_transform(X)
    return X_embedded
