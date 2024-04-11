from collections import defaultdict
from multiprocessing import Pool
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm
from penquins import Kowalski

from data import ZTF_FILTER_NAMES


ZTF_FILTER_NAMES = {1: 'g', 2: 'r', 3: 'i'}


def get_ztf_field_ids(field_id, kowalski, filter=1, verbose=0):
    limit = 20000
    n_threads = 10
    n_batch_queries = n_threads

    to_return = []
    still_working = True
    next_batch = 0
    while still_working:
        queries = [get_field_ids_query(field_id, limit, next_batch + j * limit, filter) for j in range(n_batch_queries)]
        responses = kowalski.query(queries=queries, use_batch_query=True, max_n_threads=n_threads)
        data_arr = [response.get('data') for response in responses.get('default')]
        data = sorted([obj['_id'] for obj in np.concatenate(data_arr)])

        to_return.extend(data)
        still_working = len(data_arr[-1]) > 0
        next_batch += limit * n_batch_queries

        if verbose:
            print('Processed: {}'.format(next_batch))

    return to_return


def get_ztf_light_curves(ids, kowalski):
    chunk_size = 10000
    n_threads = 10
    n_batch_queries = n_threads

    # Make chunked queries
    queries = [get_light_curves_query(ids[i:i + chunk_size]) for i in range(0, len(ids), chunk_size)]
    # Make batched queries
    queries = [queries[i:i + n_batch_queries] for i in range(0, len(queries), n_batch_queries)]

    to_return = []
    for query in tqdm(queries):
        responses = kowalski.query(queries=query, use_batch_query=True, max_n_threads=10)
        data = np.concatenate([response.get('data') for response in responses.get('default')])
        data = sorted(data, key=lambda d: d['_id'])

        with Pool(48) as p:
            data = p.map(process_ztf_record, data)

        to_return.extend(data)

        gc.collect()

    return to_return


def get_ztf_features(ids, token, get_classification=False):
    chunk_size = 1000
    to_process = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    kowalski = Kowalski(token=token, host='gloria.caltech.edu', protocol='https', port=443, timeout=99999999)

    data = [get_ztf_features_data(ids_chunk, kowalski, get_classification) for ids_chunk in tqdm(to_process)]
    data = np.concatenate(data)

    return pd.DataFrame.from_records(data)


def get_ztf_features_data(id_list, kowalski, get_classification):
    query = get_classification_query(id_list) if get_classification else get_features_query(id_list)
    response = kowalski.query(query=query)
    data = response.get('default').get('data')
    return data


def get_ztf_matches(coordinates, kowalski, radius=1.):
    original_coord_len = len(coordinates)
    filter_names = {1: 'g', 2: 'r', 3: 'i'}

    # Make chunks
    chunk_size = 10000
    coordinates = [coordinates[i:i + chunk_size] for i in range(0, len(coordinates), chunk_size)]

    results = []
    for coords in coordinates:
        q = [get_cone_query(ra, dec, radius) for ra, dec in coords]
        responses = kowalski.query(queries=q, use_batch_query=True, max_n_threads=10)
        responses = responses.get('default')

        for r in responses:
            data = r['data']['ZTF_sources_20240117']
            row = defaultdict(list)

            for pos in data.keys():  # that's always just one element, why?                
                for match in data[pos]:
                    id = match['_id']
                    filter = filter_names[match['filter']]
                    df = clean_ztf_record(match)

                    row['id_{}'.format(filter)].append(id)
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

    for x in ['field', 'ra', 'dec']:
        if x in ztf_record:
            row[x] = ztf_record[x]    

    df = clean_ztf_record(ztf_record)
    row['mjd'] = df['hjd'].to_numpy()
    row['mag'] = df['mag'].to_numpy()
    row['magerr'] = df['magerr'].to_numpy()

    return row


def clean_ztf_record(ztf_record):
    df = pd.DataFrame(ztf_record['data'])
    df = df.loc[df['catflags'] < 32768, ['hjd', 'mag', 'magerr']].dropna(axis=0)
    df['hjd'] -= 2400000.5
    df = df.sort_values(by='hjd')
    return df


def get_light_curves_query(ids):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_sources_20240117',
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
            'max_time_ms': 5 * 60 * 1000,  # 5 minute
        },
    }
    return query


def get_field_ids_query(field_id, limit, skip, filter):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_sources_20240117',
            'filter': {
                'field': {'$eq': field_id},
                'filter': {'$eq': filter},
            },
            'projection': {
                '_id': 1,
            },
        },
        'kwargs': {
            'max_time_ms': 60000,  # 1 minute
            'limit': limit,
            'skip': skip,
        },
    }
    return query


def get_features_query(id_list):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_source_features_DR16',
            'filter': {'_id': {'$in': id_list}},
        },
    }
    return query


def get_classification_query(id_list):
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


def get_cone_query(ra, dec, radius):
    query = {
        'query_type': 'cone_search',
        'query': {
            'object_coordinates': {
                'radec': f'[({ra},{dec})]',
                'cone_search_radius': f'{radius}',
                'cone_search_unit': 'arcsec'
            },
            'catalogs': {
                'ZTF_sources_20240117': {
                    'filter': {},
                    'projection': {
                        '_id': 1,
                        'filter': 1,
                        'data.ra': 1,
                        'data.dec': 1,
                        'data.catflags': 1,
                        'data.hjd': 1,
                        'data.mag': 1,
                        'data.magerr': 1,
                    }
                }
            }
        }
    }
    return query
