import astropy.units as u
from astropy.coordinates import SkyCoord

from tqdm import tqdm


def find_duplicates(data):
    positions = SkyCoord(
        ra=data['ra'].to_numpy() * u.degree,
        dec=data['dec'].to_numpy() * u.degree,
    )
    idx_1, idx_2, sep_2d, dist_3d = positions.search_around_sky(positions, 1 * u.arcsec)

    n_obs = data['n_obs']
    data['is_duplicate'] = False
    for i in tqdm(range(len(idx_1))):
        i_1 = idx_1[i]
        i_2 = idx_2[i]
        n_obs_1 = n_obs.loc[i_1]
        n_obs_2 = n_obs.loc[i_2]
        if (n_obs_2 > n_obs_1) or ((n_obs_1 == n_obs_2) and (i_1 > i_2)):
            data.loc[i_1, 'is_duplicate'] = True

    return data
