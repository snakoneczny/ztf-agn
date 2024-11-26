import sys
import os
import gc
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord

sys.path.append('..')
from env_config import DATA_PATH, STORAGE_PATH
from utils import read_fits_to_pandas


# Define a ZTF chunk size
chunk_size = 100000

# Define a file to save the results to
output_path = 'ZTF/ZTF_20240117/catalog_v2_wise_chunks/catalog_ZTF_20240117_g-band__v2_WISE__{}.csv'
output_path = os.path.join(STORAGE_PATH, output_path)

# Read ZTF coords
ztf_path = 'ZTF/ZTF_20240117/catalog_ZTF_20240117_g-band__v2_coords.fits'
ztf_path = os.path.join(STORAGE_PATH, ztf_path)
df_ztf = read_fits_to_pandas(ztf_path)

# Get all file names and chunks to process
to_process = []
for i in range(0, len(df_ztf), chunk_size):
    tmp_path = output_path.format(i)
    if not os.path.exists(tmp_path):
        to_process.append((i, tmp_path))

# Read WISE features
wise_path = 'AllWISE/AllWISE_reduced.fits'
wise_path = os.path.join(DATA_PATH, wise_path)
df_wise = read_fits_to_pandas(wise_path)

# Build the WISE positions once
positions_wise = SkyCoord(
    ra=df_wise['ra'].to_numpy() * u.degree,
    dec=df_wise['dec'].to_numpy() * u.degree,
)

# Define the column names
cols_wise = ['w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']
cols_to_save = ['AllWISE__w1mpro', 'AllWISE__w2mpro', 'AllWISE__w3mpro', 'AllWISE__w4mpro']

# Iterate the ZTF catalog in chunks
for i, output_path in tqdm(to_process, 'Processing chunks'):
    chunk = df_ztf[i:i + chunk_size]
    
    # Build the chunk position and get the matches
    positions_chunk = SkyCoord(
        ra=chunk['ra'].to_numpy() * u.degree,
        dec=chunk['dec'].to_numpy() * u.degree,
    )
    idx_chunk, idx_wise, sep_2d, dist_3d = positions_wise.search_around_sky(positions_chunk, 1 * u.arcsec)

    # Initiate a DF with the right size
    to_save = pd.DataFrame()
    for col_name in ['distance'] + cols_to_save:
        to_save[col_name] = [np.nan] * len(chunk)
    
    # Iterate through all matches
    for j in range(len(idx_chunk)):
        min_distance = to_save.loc[idx_chunk[j], 'distance']
        cand_distance = sep_2d[j].radian
        if math.isnan(min_distance) or min_distance > cand_distance:
            to_save.loc[idx_chunk[j], 'distance'] = cand_distance
            to_save.loc[idx_chunk[j], cols_to_save] = df_wise.loc[idx_wise[j], cols_wise].to_list()
    
    # Save chunk results
    to_save.to_csv(output_path, columns=cols_to_save, header=True, index=False)
    
    gc.collect()
