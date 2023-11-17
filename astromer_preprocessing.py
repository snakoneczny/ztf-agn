import tensorflow as tf
import logging
import os

from ASTROMER.core.masking import get_masked, set_random, get_padding_mask
from ASTROMER.core.utils import standardize

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def adjust_fn(func, *arguments):
    def wrap(*args, **kwargs):
        result = func(*args, *arguments)
        return result

    return wrap


def deserialize(sample):
    """
    Read a serialized sample and convert it to tensor
    Context and sequence features should match with the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    context_features = {'label': tf.io.FixedLenFeature([], dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([], dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
        serialized=sample,
        context_features=context_features,
        sequence_features=sequence_features
    )

    input_dict = dict()
    input_dict['lcid'] = tf.cast(context['id'], tf.string)
    input_dict['length'] = tf.cast(context['length'], tf.int32)
    input_dict['label'] = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input'] = sequence
    return input_dict


def sample_lc(sample, max_obs, binary=True):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    if binary:
        input_dict = deserialize(sample)
    else:
        input_dict = sample

    sequence = input_dict['input']

    serie_len = tf.shape(sequence)[0]

    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len - max_obs + 1,
                                  dtype=tf.int32)

        sequence = tf.slice(sequence, [pivot, 0], [max_obs, -1])
    else:
        sequence = tf.slice(sequence, [0, 0], [serie_len, -1])

    input_dict['sequence'] = sequence
    return sequence, input_dict['label'], input_dict['lcid'], input_dict['features']


def get_window(sequence, length, pivot, max_obs):
    pivot = tf.minimum(length - max_obs, pivot)
    pivot = tf.maximum(0, pivot)
    end = tf.minimum(length, max_obs)

    sliced = tf.slice(sequence, [pivot, 0], [end, -1])
    return sliced


def get_windows(sample, max_obs, binary=True):
    if binary:
        input_dict = deserialize(sample)
    else:
        input_dict = sample

    sequence = input_dict['input']
    rest = input_dict['length'] % max_obs

    pivots = tf.tile([max_obs], [tf.cast(input_dict['length'] / max_obs, tf.int32)])
    pivots = tf.concat([[0], pivots], 0)
    pivots = tf.math.cumsum(pivots)

    splits = tf.map_fn(lambda x: get_window(sequence,
                                            input_dict['length'],
                                            x,
                                            max_obs), pivots,
                       infer_shape=False,
                       fn_output_signature=(tf.float32))

    # aqui falta retornar labels y oids
    y = tf.tile([input_dict['label']], [len(splits)])
    ids = tf.tile([input_dict['lcid']], [len(splits)])

    return splits, y, ids


def mask_sample(x, y, i, f, msk_prob, rnd_prob, same_prob, max_obs):
    '''
    Pretraining formater
    '''
    x = standardize(x, return_mean=False)

    seq_time = tf.slice(x, [0, 0], [-1, 1])
    seq_magn = tf.slice(x, [0, 1], [-1, 1])
    seq_errs = tf.slice(x, [0, 2], [-1, 1])

    # Save the true values
    orig_magn = seq_magn

    # [MASK] values
    mask_out = get_masked(seq_magn, msk_prob)

    if msk_prob > 0.:
        # [MASK] -> Same values
        seq_magn, mask_in = set_random(seq_magn,
                                       mask_out,
                                       seq_magn,
                                       same_prob,
                                       name='set_same')

        # [MASK] -> Random value
        seq_magn, mask_in = set_random(seq_magn,
                                       mask_in,
                                       tf.random.shuffle(seq_magn),
                                       rnd_prob,
                                       name='set_random')
    else:
        mask_in = tf.zeros_like(mask_out)

    time_steps = tf.shape(seq_magn)[0]

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out = tf.concat([mask_out, 1 - mask_fill], 0)
        mask_in = tf.concat([mask_in, mask_fill], 0)
        seq_magn = tf.concat([seq_magn, 1 - mask_fill], 0)
        seq_time = tf.concat([seq_time, 1 - mask_fill], 0)
        orig_magn = tf.concat([orig_magn, 1 - mask_fill], 0)

    input_dict = dict()
    input_dict['output'] = orig_magn
    input_dict['input'] = seq_magn
    input_dict['times'] = seq_time
    input_dict['mask_out'] = mask_out
    input_dict['mask_in'] = mask_in
    input_dict['length'] = time_steps
    input_dict['label'] = y
    input_dict['id'] = i
    input_dict['features'] = f

    return input_dict


def format_label(input_dict, num_cls):
    x = {
        'input': input_dict['input'],
        'times': input_dict['times'],
        'mask_in': input_dict['mask_in'],
        'features': input_dict['features'],
    }
    y = tf.one_hot(input_dict['label'], num_cls)
    return x, y


def pretraining_records(source, batch_size, max_obs=100, msk_frac=0.2,
                        rnd_frac=0.1, same_frac=0.1, sampling=False,
                        shuffle=False, n_classes=-1, repeat=1):
    """
    Pretraining data loader.
    This method build the ASTROMER input format.
    ASTROMER format is based on the BERT masking strategy.

    Args:
        source (string): Record folder
        batch_size (int): Batch size
        no_shuffle (bool): Do not shuffle training and validation dataset
        max_obs (int): Max. number of observation per serie
        msk_frac (float): fraction of values to be predicted ([MASK])
        rnd_frac (float): fraction of [MASKED] values to replace with random values
        same_frac (float): fraction of [MASKED] values to replace with true values

    Returns:
        Tensorflow Dataset: Iterator withg preprocessed batches
    """
    rec_paths = []
    for folder in os.listdir(source):
        if folder.endswith('.csv'):
            continue
        for x in os.listdir(os.path.join(source, folder)):
            rec_paths.append(os.path.join(source, folder, x))

    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs)
    else:
        fn_0 = adjust_fn(get_windows, max_obs)

    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    dataset = tf.data.TFRecordDataset(rec_paths)
    dataset = dataset.repeat(repeat)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(fn_0)

    if not sampling:
        dataset = dataset.flat_map(lambda x, y, i: tf.data.Dataset.from_tensor_slices((x, y, i)))

    dataset = dataset.map(fn_1)

    if n_classes != -1:
        print('[INFO] Processing labels')
        fn_2 = adjust_fn(format_label, n_classes)
        dataset = dataset.map(fn_2)

    dataset = dataset.padded_batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def adjust_fn_clf(func, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, max_obs)
        return result

    return wrap


def create_generator(list_of_arrays, labels=None, ids=None, features=None):
    if ids is None:
        ids = list(range(len(list_of_arrays)))
    if labels is None:
        labels = list(range(len(list_of_arrays)))

    for i, j, k, f in zip(list_of_arrays, labels, ids, features):
        yield {'input': i,
               'label': int(j),
               'lcid': str(k),
               'length': int(i.shape[0]),
               'features': f}


def load_numpy(samples,
               ids=None,
               labels=None,
               features=None,
               batch_size=1,
               shuffle=False,
               sampling=False,
               max_obs=100,
               msk_frac=0.,
               rnd_frac=0.,
               same_frac=0.,
               repeat=1,
               num_cls=-1):
    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs, False)
    else:
        fn_0 = adjust_fn(get_windows, max_obs, False)

    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    n_features = features.shape[1]
    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples, labels, ids, features),
                                             output_types={'input': tf.float32,
                                                           'label': tf.int32,
                                                           'lcid': tf.string,
                                                           'length': tf.int32,
                                                           'features': tf.float32},
                                             output_shapes={'input': (None, 3),
                                                            'label': (),
                                                            'lcid': (),
                                                            'length': (),
                                                            'features': n_features})
    dataset = dataset.repeat(repeat)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(fn_0)
    if not sampling:
        if features is not None:
            dataset = dataset.flat_map(lambda x, y, i, f: tf.data.Dataset.from_tensor_slices((x, y, i, f)))
        else:
            dataset = dataset.flat_map(lambda x, y, i: tf.data.Dataset.from_tensor_slices((x, y, i)))
    dataset = dataset.map(fn_1)

    if labels is not None and num_cls != -1:
        dataset = dataset.map(lambda x: format_label(x, num_cls))

    dataset = dataset.padded_batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
