from collections import defaultdict
from multiprocessing import Pool
import gc
import math

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from penquins import Kowalski


ZTF_FILTER_NAMES = {1: 'g', 2: 'r', 3: 'i'}

TEST_FIELDS = [296, 297, 423, 424, 487, 488, 562, 563, 682, 683, 699, 700, 717, 718, 777, 778, 841, 842, 852, 853]

LIMITING_MAGS = {
    'g': 20.8,
    'r': 20.6,
    'i': 19.9,
    'PS1_DR1__gMeanPSFMag': 22.0,
    'PS1_DR1__rMeanPSFMag': 21.8,
    'PS1_DR1__iMeanPSFMag': 21.5,
    'PS1_DR1__zMeanPSFMag': 20.9,
    'AllWISE__w1mpro': 17.1,
    'AllWISE__w2mpro': 15.7,
    'Gaia_EDR3__phot_g_mean_mag': 21,
}

ZTF_DATES = {
    'DR 5': '20210401',
    'DR 16': '20230821',
    'DR 20': '20240117',
}

ZTF_LAST_DATES = {
    'DR 5': 59243.22367999982,
    'DR 16': 60133.4643600001,
    'DR 20': 60247.40577000007,
}

CATALOGS_DICT = {
    'ZTF': {
        '_id': 1,
        'filter': 1,
        'data.ra': 1,
        'data.dec': 1,
        'data.catflags': 1,
        'data.hjd': 1,
        'data.mag': 1,
        'data.magerr': 1,
    },
    'PS1_DR1': {
        '_id': 1,
        'raMean': 1,
        'decMean': 1,
        'qualityFlag': 1,
        'gMeanPSFMag': 1,
        'rMeanPSFMag': 1,
        'iMeanPSFMag': 1,
        'zMeanPSFMag': 1,
        'yMeanPSFMag': 1,
        'gMeanPSFMagErr': 1,
        'rMeanPSFMagErr': 1,
        'iMeanPSFMagErr': 1,
        'zMeanPSFMagErr': 1,
        'yMeanPSFMagErr': 1,
    },
    'AllWISE': {
        '_id': 1,
        'ra': 1,
        'dec': 1,
        'ph_qual': 1,
        'w1mpro': 1,
        'w2mpro': 1,
        'w3mpro': 1,
        'w4mpro': 1,
        'w1sigmpro': 1,
        'w2sigmpro': 1,
        'w3sigmpro': 1,
        'w4sigmpro': 1,
    },
    'Gaia_EDR3': {
        '_id': 1,
        'ra': 1,
        'dec': 1,
        'phot_g_mean_mag': 1,
        'phot_bp_mean_mag': 1,
        'phot_rp_mean_mag': 1,
        'phot_bp_rp_excess_factor': 1,
        'astrometric_excess_noise': 1,
        'parallax': 1,
        'parallax_error': 1,
        'pmra': 1,
        'pmra_error': 1,
        'pmdec': 1,
        'pmdec_error': 1,
    },
}


def get_ztf_light_curves(ids, date, kowalski):
    chunk_size = 10000
    n_threads = 10
    n_batch_queries = n_threads

    # Make chunked queries
    queries = [get_light_curves_query(ids[i:i + chunk_size], date) for i in range(0, len(ids), chunk_size)]
    # Make batched queries
    queries = [queries[i:i + n_batch_queries] for i in range(0, len(queries), n_batch_queries)]

    to_return = []
    for query in tqdm(queries):
        responses = kowalski.query(queries=query, use_batch_query=True, max_n_threads=10)        
        
        data = np.concatenate([response.get('data') for response in responses.get('default')])
        data = sorted(data, key=lambda d: d['_id'])

        # with Pool(24) as p:
        #     data = p.map(process_ztf_record, data)
        data = [process_ztf_record(lc_dict) for lc_dict in data]

        to_return.extend(data)
        gc.collect()

    # Extract DF and array        
    df = pd.DataFrame(to_return, columns=['id', 'ra', 'dec', 'n obs'])
    df['n obs'] = [len(lc_dict['mjd']) for lc_dict in to_return]
    to_return = [np.array([lc_dict['mjd'], lc_dict['mag'], lc_dict['magerr']]) for lc_dict in to_return]
    
    return df, to_return


# Return only IDs for a ZTF catalog, or all columns from the projection dict for other surveys
def get_catalog_data(catalog, kowalski, chunk_start=None, chunk_end=None, field_id=None, filter=None, verbose=0):
    limit = 100000
    n_threads = 10

    to_return = []
    still_working = True
    next_batch = 0 if chunk_start is None else chunk_start    
    
    while still_working:
        gc.collect()
        
        # Get responses
        queries = [get_find_query(catalog, limit, next_batch + i * limit, field_id=field_id, filter=filter) for i in range(n_threads)]
        responses = kowalski.query(queries=queries, use_batch_query=True, max_n_threads=n_threads)
        
        # Concatenate
        data_arr = [response.get('data') for response in responses['default']]
        data_arr = [elem if elem is not None else [] for elem in data_arr]
        data = np.concatenate(data_arr)
        
        # Sort
        ids = [obj['_id'] for obj in data]
        if field_id is not None:
            data = sorted(ids)
        else:
            data = [x for _, x in sorted(zip(ids, data))]

        # Add
        to_return.extend(data)
        
        # Move next
        next_batch += limit * n_threads
        still_working = (len(data) == limit * n_threads)
        if chunk_start is not None:
            still_working &= (next_batch < chunk_end)

        if verbose:
            print('Processed: {}'.format(next_batch))

    return to_return


def get_ztf_features(ids, date, token, get_classification=False):
    chunk_size = 1000
    to_process = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    kowalski = Kowalski(token=token, host='gloria.caltech.edu', protocol='https', port=443, timeout=99999999)

    data = [get_ztf_features_data(ids_chunk, date, kowalski, get_classification) for ids_chunk in tqdm(to_process)]
    data = np.concatenate(data)

    return pd.DataFrame.from_records(data)


def get_ztf_features_data(id_list, date, kowalski, get_classification):
    query = get_classification_query(id_list, date) if get_classification else get_features_query(id_list)
    response = kowalski.query(query=query)
    data = response.get('default').get('data')
    return data


def get_cross_matches(coordinates, catalogs, kowalski, radius=1.0):
    records = [get_cross_match_record(ra, dec, catalogs, kowalski, radius=radius) for ra, dec in tqdm(coordinates, 'Running near queries')]
    columns = ['{}__{}'.format(catalog, k) for catalog in catalogs for k in CATALOGS_DICT[catalog]['projection'] if k != '_id']
    print('Constructing a data frame')
    df = pd.DataFrame.from_records(records, columns=columns) 
    return df


def get_cross_match_record(ra, dec, catalogs, kowalski, radius=1.0):
    if not math.isnan(ra) and not math.isnan(dec):
        q = get_near_query(ra, dec, radius, catalogs)
        response = kowalski.query(query=q)
        response = response.get('default').get('data')
        record = {'{}__{}'.format(catalog, k): v for catalog in catalogs if len(response[catalog]['query_coords']) > 0 for k, v in response[catalog]['query_coords'][0].items()}
        return record
    else:
        return {}


def get_ztf_matches(coordinates, date, kowalski, radius=1.):
    original_coord_len = len(coordinates)
    filter_names = {1: 'g', 2: 'r', 3: 'i'}

    # Make chunks
    chunk_size = 10000
    coordinates = [coordinates[i:i + chunk_size] for i in range(0, len(coordinates), chunk_size)]

    results = []
    for coords in coordinates:
        q = [get_cone_query(ra, dec, radius, ztf_date=date) for ra, dec in coords]
        responses = kowalski.query(queries=q, use_batch_query=True, max_n_threads=10)
        responses = responses.get('default')

        # Sort the responses according to the original coordinate values
        resulting_coords = [list(r['data']['ZTF_sources_20230821'].keys())[0] for r in responses]
        processed_coords = []
        for x in resulting_coords:
            splitted = x[1:-1].split(', ')
            coord = [float(v.replace('_', '.')) for v in splitted]
            processed_coords.append((coord[0], coord[1]))
        idx_sort = [coords.index(x) for x in processed_coords]
        responses = [x for _, x in sorted(zip(idx_sort, responses))]

        for r in responses:
            data = r['data']['ZTF_sources_20230821']
            row = defaultdict(list)

            for pos in data.keys():  # that's always just one element, why?                
                for match in data[pos]:
                    filter = filter_names[match['filter']]
                    df = clean_ztf_record(match)

                    row['id_{}'.format(filter)].append(match['_id'])
                    row['ra_{}'.format(filter)].append(df['ra'].mean())
                    row['dec_{}'.format(filter)].append(df['dec'].mean())
                    row['mjd_{}'.format(filter)].append(df['hjd'].to_numpy())
                    row['mag_{}'.format(filter)].append(df['mag'].to_numpy())
                    row['magerr_{}'.format(filter)].append(df['magerr'].to_numpy())

            results.append(row)

    # Assert we got all the records
    assert len(results) == original_coord_len

    return results


def process_ztf_record(ztf_record):
    row = {}
    row['id'] = ztf_record['_id']
    row['filter'] = ZTF_FILTER_NAMES[ztf_record['filter']]
    row['field'] = ztf_record['field']

    df = clean_ztf_record(ztf_record)
    row['ra'] = df['ra'].mean()
    row['dec'] = df['dec'].mean()
    row['mjd'] = df['hjd'].to_numpy()
    row['mag'] = df['mag'].to_numpy()
    row['magerr'] = df['magerr'].to_numpy()

    return row


def clean_ztf_record(ztf_record):
    df = pd.DataFrame(ztf_record['data'])
    df = df.loc[df['catflags'] < 32768, ['ra', 'dec', 'hjd', 'mag', 'magerr']].dropna(axis=0)
    df['hjd'] -= 2400000.5
    df = df.sort_values(by='hjd')
    return df


def get_light_curves_query(ids, date):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_sources_{}'.format(date),
            'filter': {
                '_id': {'$in': list(map(int, ids))}
            },
            'projection': {
                '_id': 1,
                'field': 1,
                'filter': 1,
                'data.ra': 1,
                'data.dec': 1,
                'data.catflags': 1,
                'data.hjd': 1,
                'data.mag': 1,
                'data.magerr': 1,
            },
        },
        'kwargs': {
            'max_time_ms': 5 * 60000,  # 5 minutes
        },
    }
    return query


def get_find_query(catalog, limit, skip, field_id=None, filter=None):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': catalog,
            'filter': {},
            'projection': {
                '_id': 1,
            },
        },
        'kwargs': {
            'max_time_ms': 2 * 60 * 60000,  # 2 hours
            'limit': limit,
            'skip': skip,
        },
    }
    if catalog[:3] != 'ZTF':
        query['query']['projection'] = CATALOGS_DICT[catalog]
    if field_id is not None:
        query['query']['filter'] = {
            'field': {'$eq': field_id},
            'filter': {'$eq': filter},
        }
    return query


def get_features_query(id_list, date):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_source_features_DR16',
            'filter': {'_id': {'$in': id_list}},
        },
    }
    return query


def get_classification_query(id_list, date):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_source_classifications_DR16',
            'filter': {'_id': {'$in': id_list}},
            'projection': {
                '_id': 1,
                'puls_xgb': 1,
                'bis_xgb': 1,
                'agn_xgb': 1,
                'yso_xgb': 1,
                'puls_dnn': 1,
                'bis_dnn': 1,
                'agn_dnn': 1,
                'yso_dnn': 1,
            },
        },
    }

    return query


def get_near_query(ra, dec, radius, catalogs):
    return {
        'query_type': 'near',
        'query': {
            'max_distance': radius,
            'distance_units': 'arcsec',
            'radec': {'query_coords': [ra, dec]},
            'catalogs': {
                catalog: {'projection': CATALOGS_DICT[catalog]} for catalog in catalogs
            },
        },
        'kwargs': {
            'max_time_ms': 60000,  # 1 minute
            'limit': 1,
        },
    }


def get_cone_query(ra, dec, radius, ztf_date=None, catalogs=None):
    return {
        'query_type': 'cone_search',
        'query': {
            'object_coordinates': {
                'radec': f'[({ra},{dec})]',
                'cone_search_radius': f'{radius}',
                'cone_search_unit': 'arcsec'
            },
            'catalogs': {
                'ZTF_sources_{}'.format(ztf_date): {
                    'projection': CATALOGS_DICT['ZTF']
                }
            },
        }
    }
