import sys
import os
import pickle
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH
from ztf import ZTF_DATES
from ml import get_train_data, run_experiments, get_file_name
from features import add_colors, get_features


parser = argparse.ArgumentParser()
parser.add_argument('-z',   '--redshift',   dest='is_redshift',         action='store_true')
parser.add_argument('-c',   '--cross',      dest='is_cross_features',   action='store_true')
parser.add_argument('-q',   '--qso',        dest='is_qso_vs_rest',      action='store_true')
parser.add_argument('-o',   '--obs',        dest='is_n_obs_limit',      action='store_true')
parser.add_argument('-s',   '--save',       dest='with_save',           action='store_true')
parser.add_argument('-t',   '--test',       dest='is_test',             action='store_true')
args = parser.parse_args()

problem = 'z' if args.is_redshift else 'clf'

# Run arguemtns
ztf_date = ZTF_DATES['DR 20']
filters = ['g']
mag_limits = [False]

# Data subset, feature set
data_compositions = [
    # Final catalog
    (['ZTF'], [
        ['ZTF', 'AstrmClf'],
        ['ZTF', 'AstrmClf', 'PS'],
        ['ZTF', 'AstrmClf', 'WISE'],
        ['ZTF', 'AstrmClf', 'GAIA'],
        ['ZTF', 'AstrmClf', 'PS', 'WISE'],
        ['ZTF', 'AstrmClf', 'PS', 'GAIA'],
        ['ZTF', 'AstrmClf', 'WISE', 'GAIA'],
        ['ZTF', 'AstrmClf', 'PS', 'WISE', 'GAIA'],
        # ['PS', 'WISE', 'GAIA'],
    ]),
    # Cross match comparisons
    # (['ZTF', 'PS'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['PS'],
    #     ['ZTF', 'AstrmClf', 'PS'],
    # ]),
    # (['ZTF', 'WISE'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['WISE'],
    #     ['ZTF', 'AstrmClf', 'WISE'],
    # ]),
    # (['ZTF', 'GAIA'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['GAIA'],
    #     ['ZTF', 'AstrmClf', 'GAIA'],
    # ]),
    # All together, feature importance
    # (['ZTF', 'PS', 'GAIA'], [
    #     ['ZTF', 'AstrmClf', 'PS', 'GAIA'],
    # ]),
    # (['ZTF', 'PS', 'WISE', 'GAIA'], [
    #     ['ZTF', 'AstrmClf', 'PS', 'WISE', 'GAIA'],
    # ]),
    # 
]
# Redshift
if args.is_redshift:
    data_compositions = [
        # Final catalog
        (['ZTF'], [
            ['ZTF', 'WISE'],
        ]),
        (['ZTF', 'WISE'], [
            ['ZTF', 'WISE'],
        ]),
    ]

# Read the train data
data = {}
for filter in filters:
    _, data[filter] = \
        get_train_data(ztf_date=ztf_date, filter=filter, return_light_curves=False)

# Add lightcurve stats
for filter in filters:
    file_name = 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__longests_filter_{}_reduced__stats'.format(
        ztf_date, ztf_date, filter)
    path = os.path.join(DATA_PATH, file_name)
    with open(path, 'rb') as file:
        df = pickle.load(file)
        data[filter] = pd.concat([data[filter], df], axis=1)

# Add colors
for filter in filters:
    data[filter], features_dict = add_colors(data[filter], args.is_cross_features)

# Pick only QSOs if redshift
if args.is_redshift:
    for filter in filters:
        data[filter] = data[filter].loc[data[filter]['CLASS'] == 'QSO']

# Iterate filters and feature compositions
for is_mag_limit in mag_limits:
    for filter in filters:
        for data_composition in tqdm(data_compositions):
            data_labels = data_composition[0]
            data_label = '_'.join(data_labels)
            features_labels = data_composition[1]

            # Run the expeirments for given data composition and almost all feature combinations
            preds, classifiers = run_experiments(
                data_labels, features_labels, data[filter], features_dict, filter, ztf_date, args.is_redshift,
                is_mag_limit, args.is_cross_features, args.is_qso_vs_rest, args.is_n_obs_limit,
            )

            if args.with_save:
                # Save predictions
                preds_file_name = get_file_name('preds', problem, ztf_date, filter, data_label,
                                                mag_limit=is_mag_limit, cross_features=args.is_cross_features,
                                                qso_vs_rest=args.is_qso_vs_rest, n_obs_limit=args.is_n_obs_limit)
                file_path = os.path.join(PROJECT_PATH, preds_file_name)
                preds.to_csv(file_path, index=False)
                print('Preds saved to\t{}'.format(file_path))

                # Save feature importances
                feature_importances = {}
                for key, classifier in classifiers.items():
                    # Extract feature importances
                    feature_labels_tmp = key.split(' + ')
                    
                    importances = classifier.get_booster().get_score(importance_type='gain')
                    importances = list(importances.values())
                    importances = list(np.array(importances) / sum(importances))
                    
                    feature_importances[key] = {
                        'features': get_features(features_dict, feature_labels_tmp, args.is_cross_features),
                        'importances': importances,
                    }
                
                feature_file_name = get_file_name('features', problem, ztf_date, filter, data_label,
                                                  mag_limit=is_mag_limit, cross_features=args.is_cross_features,
                                                  qso_vs_rest=args.is_qso_vs_rest, n_obs_limit=args.is_n_obs_limit)
                file_path = os.path.join(PROJECT_PATH, feature_file_name)
                out_file = open(file_path, 'w')
                json.dump(feature_importances, out_file, indent=4)
                out_file.close()
                print('Feature importances saved to\t{}'.format(file_path))
                
                # Save models
                if args.with_save:
                    if data_label == 'ZTF':
                        if args.is_redshift:
                            feature_labels_list = [
                                ['ZTF', 'WISE'],
                            ]
                        else:
                            feature_labels_list = [
                                ['ZTF', 'AstrmClf'],
                                ['ZTF', 'AstrmClf', 'PS'],
                                ['ZTF', 'AstrmClf', 'WISE'],
                                ['ZTF', 'AstrmClf', 'GAIA'],
                                ['ZTF', 'AstrmClf', 'PS', 'WISE'],
                                ['ZTF', 'AstrmClf', 'PS', 'GAIA'],
                                ['ZTF', 'AstrmClf', 'WISE', 'GAIA'],
                                ['ZTF', 'AstrmClf', 'PS', 'WISE', 'GAIA'],
                            ]
                        for feature_labels in feature_labels_list:
                            feature_label = ' + '.join(feature_labels)
                            if feature_label in classifiers:
                                model_file_name = get_file_name('model', problem, ztf_date, filter, '_'.join(feature_labels),
                                                                mag_limit=is_mag_limit, cross_features=args.is_cross_features,
                                                                qso_vs_rest=args.is_qso_vs_rest, n_obs_limit=args.is_n_obs_limit)
                                file_path = os.path.join(PROJECT_PATH, model_file_name)
                                with open(file_path, 'wb') as file:
                                    pickle.dump(classifiers[feature_label], file)
                                print('Model saved to\t{}'.format(file_path))
