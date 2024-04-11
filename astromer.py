import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def build_model(encoder, n_classes=3, maxlen=200, train_astromer=True, lr=1e-3):
    serie = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input': serie, 'mask_in': mask, 'times': times}

    encoder.trainable = train_astromer
    
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

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    
    x = Dense(n_classes, name='output')(x)

    return x
