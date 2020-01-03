import keras
from keras import Sequential
from keras import backend as K
from keras import layers
from keras.applications import DenseNet121, ResNet101
from keras.optimizers import Adam
import numpy as np

IMAGE_SIZE = 224


# @tf.function
# def f1(y_true, y_pred):
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
#     return 2 * (K.sum(y_true * y_pred) + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]


# def simple_model(opts={}):
#     head = Sequential()
#     head.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
#     head.add(layers.BatchNormalization())
#     head.add(layers.Activation('relu'))
#     head.add(layers.MaxPooling2D(pool_size=(2, 2)))

#     head.add(layers.Conv2D(32, (3, 3)))
#     head.add(layers.BatchNormalization())
#     head.add(layers.Activation('relu'))
#     head.add(layers.MaxPooling2D(pool_size=(2, 2)))

#     head.add(layers.Conv2D(64, (3, 3)))
#     head.add(layers.BatchNormalization())
#     head.add(layers.Activation('relu'))
#     head.add(layers.MaxPooling2D(pool_size=(2, 2)))

#     average_pool = keras.Sequential()
#     average_pool.add(layers.AveragePooling2D())
#     average_pool.add(layers.Flatten())
#     average_pool.add(layers.Dense(2, activation='softmax'))

#     model = keras.Sequential([
#         head,
#         average_pool
#     ])

#     model.compile(optimizer=Adam(**opts),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', f1])
#     return model


def densenet_model(learning_rate, output_bias=None, metrics=METRICS):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics)

    return model


def resnet101_model(learning_rate, output_bias=None,  metrics=METRICS):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    resnet101 = ResNet101(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3))
    model = Sequential()
    model.add(resnet101)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics)

    return model
