import sys
import os
import pickle
import gc
import argparse

sys.path.append('..')
from env_config import DATA_PATH
from light_curves import preprocess_ztf_light_curves
from ztf import ZTF_DATES


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filter', dest='filter', help='ZTF filter, either g or r', required=True)
args = parser.parse_args()
filter = args.filter

# Input parameters
date = ZTF_DATES['DR 20']

# Read data
# ZTF
file_name = 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__longests_filter_{}'.format(date, date, filter)
file_path = os.path.join(DATA_PATH, file_name)
with open(file_path, 'rb') as file:
    data_ztf = pickle.load(file)

# SDSS
file_name = 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_x_ztf_{}__longests_filter_{}'.format(date, date, filter)
file_path = os.path.join(DATA_PATH, file_name)
with open(file_path, 'rb') as file:
    data_sdss = pickle.load(file).reset_index(drop=True)

# Features
file_name = 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_PWG_x_ztf_{}__longests_filter_{}'.format(date, date, filter)
file_path = os.path.join(DATA_PATH, file_name)
with open(file_path, 'rb') as file:
    data_features = pickle.load(file).reset_index(drop=True)

# Reduce deep drilling and get number of observations higher than 20
data_ztf, data_sdss, data_features = \
    preprocess_ztf_light_curves(data_ztf, data_sdss, data_features, with_multiprocessing=False)
gc.collect()

# Save the resulting lists and data frames
# ZTF
file_name = 'ZTF_x_SDSS/ZTF_{}/ztf_{}_x_specObj-dr18__longests_filter_{}__reduced'.format(date, date, filter)
file_path = os.path.join(DATA_PATH, file_name)
with open(file_path, 'wb') as file:
    pickle.dump(data_ztf, file)
print('Light curves saved to: ' + file_path)

# SDSS
file_name = 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_x_ztf_{}__longests_filter_{}__reduced'.format(date, date, filter)
file_path = os.path.join(DATA_PATH, file_name)
with open(file_path, 'wb') as file:
    pickle.dump(data_sdss, file)
print('SDSS saved to: ' + file_path)

# Features
file_name = 'ZTF_x_SDSS/ZTF_{}/specObj-dr18_PWG_x_ztf_{}__longests_filter_{}__reduced'.format(date, date, filter)
file_path = os.path.join(DATA_PATH, file_name)
with open(file_path, 'wb') as file:
    pickle.dump(data_features, file)
print('Features saved to: ' + file_path)
