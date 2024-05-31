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
from ml import get_train_data, run_experiments
from features import add_colors


# Run arguemtns
ztf_date = ZTF_DATES['DR 20']
filters = ['g', 'r']

feature_compositions = [
    # Baseline
    ['AstrmClf'],
    # Cross match comparisons
    ['AstrmClf', 'PS'],
    ['AstrmClf', 'WISE'],
    ['AstrmClf', 'GAIA'],
    # Road to the best ensemble classification
    ['AstrmClf', 'PS', 'WISE'],
    ['AstrmClf', 'PS', 'GAIA'],
    ['AstrmClf', 'PS', 'WISE', 'GAIA'],
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
    for feature_labels in tqdm(feature_compositions):
        data_labels = [label for label in feature_labels if label != 'AstrmClf']
        if 'ZTF' not in data_labels:
            data_labels = ['ZTF'] + data_labels
        data_label = '_'.join(data_labels)
        
        # Run the expeirments for given data composition and almost all feature combinations
        preds, classifiers = run_experiments(
            feature_labels, data[filter], features_dict, filter, ztf_date,
        )
        
        # Save results
        file_name = 'ZTF_{}_band_{}__{}__RF'.format(ztf_date, filter, data_label)
        file_path = os.path.join(PROJECT_PATH, 'outputs/preds/ZTF_{}'.format(ztf_date), file_name + '__test.csv')
        preds.to_csv(file_path, index=False)
        print('Preds saved to\t{}'.format(file_path))
        
        # Save feature importances
        feature_importances = {}
        for key, classifier in classifiers.items():
            # Extract feature importances
            feature_labels_tmp = key.split(' + ')
            feature_importances[key] = {
                'features': np.concatenate([features_dict[feature_label] for feature_label in feature_labels_tmp]).tolist(),
                'importances': classifier.feature_importances_.tolist(),
            }
        
        # Save feature importances
        file_path = os.path.join(PROJECT_PATH, 'outputs/feature_importance/ZTF_{}'.format(ztf_date), file_name + '.json')
        out_file = open(file_path, 'w')
        json.dump(feature_importances, out_file, indent=4)
        out_file.close()
        print('Feature importances saved to\t{}'.format(file_path))
        
        # Save a model
        feature_label = '_'.join(feature_labels)
        file_name = 'ZTF_{}_band_{}__RF_{}'.format(ztf_date, filter, feature_label)
        if feature_label in ['AstrmClf_PS', 'AstrmClf_PS_WISE']:
            file_path = os.path.join(PROJECT_PATH, 'outputs/models/ZTF_{}'.format(ztf_date), file_name + '.pickle')
            with open(file_path, 'wb') as file:
                pickle.dump(classifiers[' + '.join(feature_labels)], file)
            print('Model saved to\t{}'.format(file_path))
