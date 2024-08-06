import sys
import os
import pickle
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from env_config import DATA_PATH, PROJECT_PATH
from ztf import ZTF_DATES
from ml import FILE_NAMES, get_train_data, run_experiments
from features import add_colors


# Run arguemtns
ztf_date = ZTF_DATES['DR 20']
filters = ['g', 'r']

# Data subset, feature set
data_compositions = [
    # Final catalog
    # (['ZTF'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['ZTF', 'AstrmClf', 'PS'],
    #     ['ZTF', 'AstrmClf', 'WISE'],
    #     ['ZTF', 'AstrmClf', 'GAIA'],
    #     ['ZTF', 'AstrmClf', 'PS', 'WISE'],
    #     ['ZTF', 'AstrmClf', 'PS', 'GAIA'],
    #     ['ZTF', 'AstrmClf', 'PS', 'WISE', 'GAIA'],
    # ]),
    # Cross match comparisons
    # (['ZTF', 'PS'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['PS'],
    #     ['ZTF', 'AstrmClf', 'PS'],
    # ]),
    (['ZTF', 'WISE'], [
        ['ZTF', 'AstrmClf'],
        ['WISE'],
        ['ZTF', 'AstrmClf', 'WISE'],
    ]),
    # (['ZTF', 'GAIA'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['GAIA'],
    #     ['ZTF', 'AstrmClf', 'GAIA'],
    # ]),
    # # All together, feature importance
    # (['ZTF', 'PS', 'WISE', 'GAIA'], [
    #     ['ZTF', 'AstrmClf'],
    #     ['PS', 'WISE', 'GAIA'],
    #     ['ZTF', 'AstrmClf', 'PS', 'WISE', 'GAIA'],
    # ]),
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

# Get PS and WISE colors
for filter in filters:
    data[filter], features_dict = add_colors(data[filter])

# Iterate filters and feature compositions
for filter in filters:
    for data_composition in tqdm(data_compositions):
        data_labels = data_composition[0]
        data_label = '_'.join(data_labels)
        features_labels = data_composition[1]

        # Run the expeirments for given data composition and almost all feature combinations
        preds, classifiers = run_experiments(
            data_labels, features_labels, data[filter], features_dict, filter, ztf_date,
        )
        
        # Save results
        file_name = FILE_NAMES['preds'].format(ztf_date, ztf_date, filter, data_label)
        file_path = os.path.join(PROJECT_PATH, file_name)
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
                'features': np.concatenate([features_dict[feature_label] for feature_label in feature_labels_tmp]).tolist(),
                'importances': importances,
            }
        
        # Save feature importances
        file_path = os.path.join(PROJECT_PATH, FILE_NAMES['features'].format(ztf_date, ztf_date, filter, data_label))
        out_file = open(file_path, 'w')
        json.dump(feature_importances, out_file, indent=4)
        out_file.close()
        print('Feature importances saved to\t{}'.format(file_path))
        
        # Save a model
        if data_label == 'ZTF':
            for feature_labels in [
                ['ZTF', 'AstrmClf'],
                ['ZTF', 'AstrmClf', 'PS'],
                ['ZTF', 'AstrmClf', 'WISE'],
                ['ZTF', 'AstrmClf', 'GAIA'],
                ['ZTF', 'AstrmClf', 'PS', 'WISE']
            ]:
                file_name = FILE_NAMES['model'].format(ztf_date, ztf_date, filter, '_'.join(feature_labels))                
                file_path = os.path.join(PROJECT_PATH, file_name)
                with open(file_path, 'wb') as file:
                    pickle.dump(classifiers[' + '.join(feature_labels)], file)
                print('Model saved to\t{}'.format(file_path))
