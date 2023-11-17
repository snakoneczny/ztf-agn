import os

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from env_config import PROJECT_PATH
from features import FEATURE_SETS


def run_experiments(data_labels, master_df, sdss_df, feature_sets):
    reduced_data_labels = [label for label in data_labels if label != 'astrm_clf']
    # Define experiments, single for each survey, and one concatenating all features
    to_process = [[data_label] for data_label in data_labels]

    # TODO: delete
    to_process.append(['astrm_clf', 'PS'])
    to_process.append(['ZTF', 'astrm_clf'])

    if len(data_labels) > 1:
        to_process.append(data_labels)

    # Take all features together to make a subset
    max_features = np.concatenate([FEATURE_SETS[label] for label in reduced_data_labels])

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
    for df_label, df_exp in [('train', df_train), ('val', df_test)]:
        data_label = '_'.join(reduced_data_labels)
        file_name = 'outputs/preds/g-band__{}__astromer_FC-1024-512-256__non-shuffle__{}.csv'.format(data_label, df_label)
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

    # Placeholder for classifiers
    classifiers = {}

    # Iterate over feature sets
    for feature_names in tqdm(to_process):
        # Extract features
        features = np.concatenate([feature_sets[feature_name] for feature_name in feature_names])
        X_train = df_train[features]
        X_test = df_test[features]

        # Make classification
        clf = RandomForestClassifier(
            n_estimators=500, criterion='gini', random_state=491237, n_jobs=48, verbose=0
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Save results
        features_label = ' + '.join(feature_names)
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
