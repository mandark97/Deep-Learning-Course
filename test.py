import numpy as np
import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    return parts[-2] == np.array(['0', '1'])


def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, label


list_ds = tf.data.Dataset.list_files("data/*/*")
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

DATASET_SIZE = 17000

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = labeled_ds.shuffle(5000)
train_dataset = full_dataset.take(train_size).batch(32)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size).batch(32)
test_dataset = test_dataset.take(test_size).batch(32)
import pdb; pdb.set_trace()
IMAGE_SIZE = 224

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

standard_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
callbacks = [tf.keras.callbacks.TensorBoard(
    log_dir='./log/standard_model', update_freq='batch')]

labeled_ds
standard_model.fit(train_dataset, steps_per_epoch=100, epochs=7,
                   validation_data=val_dataset, validation_steps=10, verbose=2, callbacks=callbacks)

score = standard_model.evaluate(test_dataset)
print(score)


test_y = standard_model.predict(test_dataset, batch_size=32)
