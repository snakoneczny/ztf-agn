import astropy.units as u
from astropy.coordinates import SkyCoord

def find_duplicates(data):
    positions = SkyCoord(
        ra=data['ra'].to_numpy() * u.degree,
        dec=data['dec'].to_numpy() * u.degree,
    )
    idx_1, idx_2, sep_2d, dist_3d = positions.search_around_sky(positions, 1 * u.arcsec)

    data['is_duplicate'] = False
    checked_pairs = [[]] * len(data)
    for idx_a, idx_b in zip(idx_1, idx_2):
        if idx_a != idx_b and idx_a not in checked_pairs[idx_b]:
            if data.loc[idx_a, 'n_obs'] >= data.loc[idx_b, 'n_obs']:
                data.loc[idx_b, 'is_duplicate'] = True
            else:
                data.loc[idx_a, 'is_duplicate'] = True
            checked_pairs[idx_a].append(idx_b)

    return data
