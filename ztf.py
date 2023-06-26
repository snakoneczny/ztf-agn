from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
from ztfquery import lightcurve
from penquins import Kowalski


def get_ztf_features(ids, token, get_classification=False):
    chunk_size = 1000
    to_process = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    kowalski = Kowalski(token=token, host='gloria.caltech.edu', protocol='https', port=443, timeout=99999999)

    data = [get_ztf_features_data(ids_chunk, kowalski, get_classification) for ids_chunk in tqdm(to_process)]
    data = np.concatenate(data)

    return pd.DataFrame.from_records(data)


def get_ztf_features_data(id_list, kowalski, get_classification):
    query = get_classification_query(id_list) if get_classification else get_features_query
    response = kowalski.query(query=query)
    data = response.get('data')
    return data


def get_ztf_fields(field_ids, token, filter=1, limit_per_field=100000):
    chunk_size = 10
    field_ids = [field_ids[i:i + chunk_size] for i in range(0, len(field_ids), chunk_size)]

    kowalski = Kowalski(token=token, host='gloria.caltech.edu', protocol='https', port=443, timeout=99999999)

    data = [get_ztf_fields_data(field_ids_chunk, kowalski, filter=filter, limit_per_field=limit_per_field) for
            field_ids_chunk in tqdm(field_ids)]
    data = np.concatenate(data)

    kowalski.close()

    results = [process_ztf_record(ztf_record) for ztf_record in tqdm(data)]

    return results


def get_ztf_fields_data(field_ids, kowalski, filter=1, limit_per_field=100000):
    queries = [get_field_query(field_id, filter, limit_per_field) for field_id in field_ids]
    responses = kowalski.batch_query(queries=queries, n_treads=10)
    data = [response.get('data') for response in responses]
    return np.concatenate(data)


def get_ztf_matches(coordinates, token, radius=1.):
    filter_names = {1: 'g', 2: 'r', 3: 'i'}

    # Make chunks
    chunk_size = 1000
    coordinates = [coordinates[i:i + chunk_size] for i in range(0, len(coordinates), chunk_size)]

    results = []
    for coords in tqdm(coordinates):
        k = Kowalski(token=token, host='gloria.caltech.edu', protocol='https', port=443)
        q = [get_cone_query(ra, dec, radius) for ra, dec in coords]
        responses = k.batch_query(queries=q, n_treads=10)

        for r in responses:
            data = r['data']['ZTF_sources_20210401']
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
        k.close()
    return results


def process_ztf_record(ztf_record):
    id = ztf_record['_id']
    filter = ZTF_FILTER_NAMES[ztf_record['filter']]
    df = clean_ztf_record(ztf_record)

    row = {}
    row['id'] = id
    row['filter'] = filter
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


def get_classification_query(id_list):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_source_classifications_DR5',
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


def get_features_query(id_list):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_source_features_DR5',
            'filter': {'_id': {'$in': id_list}},
        },
    }
    return query


def get_field_query(field_id, filter, limit):
    query = {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_sources_20210401',
            'filter': {
                'field': {'$eq': field_id},
                'filter': {'$eq': filter},
            },
            'projection': {
                '_id': 1,
                'filter': 1,
                'data.catflags': 1,
                'data.hjd': 1,
                'data.mag': 1,
                'data.magerr': 1,
            },
        },
        'kwargs': {
            'limit': limit,
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
                'ZTF_sources_20210401': {
                    'filter': {},
                    'projection': {
                        '_id': 1,
                        'filter': 1,
                        'data.catflags': 1,
                        'data.hjd': 1,
                        'data.mag': 1,
                        'data.magerr': 1,
                        # 'data.expid': 1,
                        # 'data.ra': 1,
                        # 'data.dec': 1,
                        # 'data.programid': 1,
                        # 'data.chi': 1,
                    }
                }
            }
        }
    }
    return query


def get_ztf_ipac(ra_arr, dec_arr, radius=1):
    query = lightcurve.LCQuery()
    results = []
    for ra, dec in tqdm_notebook(list(zip(ra_arr, dec_arr))):
        response = query.from_position(ra, dec, radius)
        data = response.data

        # Check if object exists in ZTF
        if 'catflags' not in data:
            results.append(None)
        else:
            row = {}
            data = data.loc[(data['catflags'] < 32768)]

            for filter in ['zg', 'zr', 'zi']:
                # Get lightcurves for a given filter
                data_filter = data.loc[data['filtercode'] == filter].reset_index(drop=True)

                if data_filter.shape[0] > 0:
                    # Get only the first object
                    id = data_filter['oid'][0]
                    data_filter = data_filter.loc[data_filter['oid'] == id]

                    # Get columns
                    row['oid_{}'.format(filter)] = id
                    row['mjd_{}'.format(filter)] = data_filter['mjd'].to_numpy()
                    row['mag_{}'.format(filter)] = data_filter['mag'].to_numpy()
                    row['magerr_{}'.format(filter)] = data_filter['magerr'].to_numpy()

            results.append(row)

    return results
