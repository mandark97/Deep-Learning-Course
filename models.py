import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

IMAGE_SIZE = 224


@tf.function
def f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 2 * (K.sum(y_true * y_pred) + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def simple_model(opts={}):
    head = Sequential()
    head.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(32, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(64, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    average_pool = tf.keras.Sequential()
    average_pool.add(layers.AveragePooling2D())
    average_pool.add(layers.Flatten())
    average_pool.add(layers.Dense(2, activation='softmax'))

    model = tf.keras.Sequential([
        head,
        average_pool
    ])

    model.compile(optimizer=Adam(**opts),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    return model


def densenet_model(*args, **kwargs):
    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        #         loss=focal_loss,
        loss='binary_crossentropy',
        optimizer=Adam(*args, **kwargs),
        metrics=['accuracy', f1])

    return model
