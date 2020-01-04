import keras
import numpy as np
from keras import Sequential
from keras import backend as K
from keras import layers
from keras.applications import DenseNet121, ResNet101
from keras.optimizers import Adam

IMAGE_SIZE = 224


def densenet_model(learning_rate, metrics, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    else:
        output_bias = 'zeros'

    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics)

    return model


def resnet101_model(learning_rate, metrics, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    else:
        output_bias = 'zeros'

    resnet101 = ResNet101(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3))
    model = Sequential()
    model.add(resnet101)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics)

    return model
