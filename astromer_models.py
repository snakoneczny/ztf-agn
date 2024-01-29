import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def build_model(model_type, encoder=None, n_classes=3, maxlen=200, n_features=None, train_astromer=True, lr=1e-3):
    serie = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    # features = Input(shape=n_features, batch_size=None, name='features')

    placeholder = {'input': serie, 'mask_in': mask, 'times': times}  # , 'features': features}

    encoder.trainable = train_astromer

    if model_type == 'LSTM':
        x = get_lstm_layers(placeholder, encoder, n_classes, maxlen)
    elif model_type == 'CNN':
        x = get_convo_layers(placeholder, encoder, n_classes, maxlen)
    else:
        x = get_fc_layers(placeholder, encoder, n_classes)

    classifier = Model(inputs=placeholder, outputs=x, name='FCATT')
    classifier.compile(
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        optimizer=Adam(lr)
    )

    return classifier


def get_fc_layers(placeholder, encoder=None, n_classes=3):
    mask = 1. - placeholder['mask_in']
    x = encoder(placeholder, training=False)  # training flag here controls the dropout
    x = x * mask

    x = tf.reduce_sum(x, 1) / tf.reduce_sum(mask, 1)
    # x = tf.concat([
    #         tf.reduce_sum(x, 1) / tf.reduce_sum(mask, 1),
    #         tf.reduce_max(x, 1),
    # ], 1)

    # x = tf.concat([x, placeholder['features']], 1)    
    # x = LayerNormalization()(x)

    x = Dense(1024, activation='relu')(x)
    # x = LayerNormalization()(x)
    x = Dense(512, activation='relu')(x)
    # x = LayerNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    
    x = Dense(n_classes, name='output')(x)

    return x


def get_lstm_layers(placeholder, encoder=None, n_classes=3, maxlen=200):
    mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))
    mask = tf.squeeze(mask, axis=-1)

    # Without astromer
    x = tf.concat([placeholder['times'], placeholder['input']], 2)

    # With astromer
    # x = encoder(placeholder)
    # x = tf.reshape(x, [-1, maxlen, encoder.output.shape[-1]])
    # x = LayerNormalization()(x)

    dropout = 0.3
    x = LSTM(256, return_sequences=True, dropout=dropout, name='LSTM_0')(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(256, return_sequences=False, dropout=dropout, name='LSTM_1')(x, mask=mask)
    x = LayerNormalization()(x)

    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = LayerNormalization()(x)

    x = Dense(n_classes, name='output')(x)

    return x


def get_convo_layers(placeholder, encoder=None, n_classes=3, maxlen=200):
    # Without astromer
    # x = tf.concat([placeholder['times'], placeholder['input']], 2)

    # With astromer
    # x = encoder(placeholder)
    # x = tf.reshape(x, [-1, maxlen, encoder.output.shape[-1]])
    # x = LayerNormalization()(x)

    x = Conv1D(16, 3, activation='relu')(placeholder['input'])
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LayerNormalization()(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)

    x = Dense(n_classes, name='output')(x)

    return x
