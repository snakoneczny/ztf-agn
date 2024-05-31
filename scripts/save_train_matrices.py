import sys
import os
import pickle
import gc

sys.path.append('..')
from env_config import DATA_PATH
from ztf import ZTF_DATES
from ml import get_train_data, make_train_test_split


ztf_date = ZTF_DATES['DR 20']

for filter_name in ['g', 'r']:
    # Read the train data
    data_subsets = ['ZTF']
    ztf_x_sdss, sdss_x_ztf = \
        get_train_data(ztf_date=ztf_date, filter=filter_name, data_subsets=data_subsets, return_features=False)

    # Change shape to feed a neural network and sample random 200 observations
    data = {'X': {}, 'y': {}}
    data['X']['train'], data['X']['val'], data['X']['test'], data['y']['train'], data['y']['val'], data['y']['test'] = \
        make_train_test_split(ztf_x_sdss, sdss_x_ztf, with_multiprocessing=False)
    gc.collect()

    # Save
    path = os.path.join(DATA_PATH, 'ZTF_x_SDSS/ZTF_20240117/matrices/ZTF_{}_filter_{}__{}_{}.pickle')
    for data_label in data:
        for split_label in ['train', 'val', 'test']:
            path_formatted = path.format(ztf_date, filter_name, data_label, split_label)
            print('Saving {}'.format(path_formatted))
            with open(path_formatted, 'wb') as file:
                pickle.dump(data[data_label][split_label], file)
