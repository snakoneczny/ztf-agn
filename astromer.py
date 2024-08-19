import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

from ASTROMER.core.data import adjust_fn, sample_lc, get_windows, mask_sample


def build_model(encoder, n_outputs=3, maxlen=200, train_astromer=True, lr=1e-3, is_regression=False):
    model_type = 'regression' if is_regression else 'classification'
    
    serie = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input': serie, 'mask_in': mask, 'times': times}

    encoder.trainable = train_astromer
    
    x = get_fc_layers(placeholder, encoder, n_outputs)
    
    model = Model(inputs=placeholder, outputs=x, name='FCATT')
    params = {
        'classification': {
            'loss': CategoricalCrossentropy(from_logits=True),
            'metrics': ['accuracy'],
        },
        'regression': {
            'loss': MeanSquaredError(),
            'metrics': ['mean_squared_error', 'mean_absolute_error'],
        },
    }
    model.compile(optimizer=Adam(lr), **params[model_type])

    return model


def get_fc_layers(placeholder, encoder=None, n_outputs=3):
    mask = 1. - placeholder['mask_in']
    x = encoder(placeholder, training=False)  # training flag here controls the dropout
    x = x * mask

    x = tf.reduce_sum(x, 1) / tf.reduce_sum(mask, 1)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    
    x = Dense(n_outputs, name='output')(x)

    return x


def load_numpy(samples,
               ids=None,
               labels=None,
               batch_size=1,
               shuffle=False,
               sampling=False,
               max_obs=100,
               msk_frac=0.,
               rnd_frac=0.,
               same_frac=0.,
               repeat=1,
               num_cls=-1,
               is_redshift=False):
    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs, False)
    else:
        fn_0 = adjust_fn(get_windows, max_obs, False)

    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    y_type = tf.float32 if is_redshift else tf.int32
    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples, labels, ids, is_redshift),
                                         output_types = {'input': tf.float32,
                                                        'label': y_type,
                                                        'lcid': tf.string,
                                                        'length': tf.int32},
                                         output_shapes = {'input': (None, 3),
                                                        'label': (),
                                                        'lcid': (),
                                                        'length': ()})
    dataset = dataset.repeat(repeat)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(fn_0)
    if not sampling:
        dataset = dataset.flat_map(lambda x, y, i: tf.data.Dataset.from_tensor_slices((x, y, i)))
    dataset = dataset.map(fn_1)

    if labels is not None and num_cls != -1:
        dataset = dataset.map(lambda x: format_label(x, num_cls, is_redshift))

    dataset = dataset.padded_batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_generator(list_of_arrays, labels=None, ids=None, is_redshift=False):
    if ids is None:
        ids = list(range(len(list_of_arrays)))
    if labels is None:
        labels = list(range(len(list_of_arrays)))

    for i, j, k in zip(list_of_arrays, labels, ids):
        if not is_redshift:
            yield {
                'input': i,
                'label': int(j),
                'lcid': str(k),
                'length': int(i.shape[0]),
            }
        else:
            yield {
                'input': i,
                'label': float(j),
                'lcid': str(k),
                'length': int(i.shape[0]),
            }


def format_label(input_dict, num_cls, is_redshift):
    x = {
        'input': input_dict['input'],
        'times': input_dict['times'],
        'mask_in': input_dict['mask_in'],
    }
    if is_redshift:
        y = input_dict['label']
    else:
        y = tf.one_hot(input_dict['label'], num_cls)
    return x, y
