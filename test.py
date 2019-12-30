import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

from utils import evaluate_model, load_dataset

IMAGE_SIZE = 224
train_dataset, val_dataset, test_dataset = load_dataset()


def simple_model():
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

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, average=None)])
    return model


def densenet_model():
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
        optimizer=Adam(lr=0.001),
        metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, average=None)])

    return model


model = densenet_model()

# callbacks = [tf.keras.callbacks.TensorBoard(
#     log_dir='./log/model', update_freq='batch')]

model.fit(train_dataset, epochs=5,
          validation_data=val_dataset, verbose=2)

score = model.evaluate(test_dataset)
print(score)
model.save("densenet121_v1")
evaluate_model(model, "densenet121_v1")
