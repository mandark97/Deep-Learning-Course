import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import backend as K


BATCH_SIZE = 64
IMAGE_SIZE = 224

img_gen = ImageDataGenerator(validation_split=0.15)


def model():
    head = tf.keras.Sequential()
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

    standard_model = tf.keras.Sequential([
        head,
        average_pool
    ])
    return standard_model


standard_model = model()

standard_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='binary_crossentropy',
                       metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, average=None)])
standard_model.fit(img_gen.flow_from_directory(directory='data/data/',
                                               class_mode='binary', batch_size=32, shuffle=False), epochs=7,
                   verbose=2)
