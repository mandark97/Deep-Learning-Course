import pandas as pd
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_SIZE = 17000


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


def load_dataset():
    list_ds = tf.data.Dataset.list_files("data/data/*/*")
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    full_dataset = labeled_ds.shuffle(5000)
    train_dataset = full_dataset.take(train_size).batch(32)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size).batch(32)
    test_dataset = test_dataset.take(test_size).batch(32)

    return train_dataset, val_dataset, test_dataset


def process_test_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, [True, False]


def evaluate_model(model, model_name):
    list_ds = tf.data.Dataset.list_files("data/test/*")
    test_ds = list_ds.map(process_test_path, num_parallel_calls=AUTOTUNE)
    y = model.predict(test_ds.batch(32))
    ans_df = pd.DataFrame({'id': [f"{'0'* (6-len(str(x)))}{x}" for x in range(17001, 22150)],
                           'class': np.argmax(y, axis=1)})
    ans_df.to_csv(f'{model_name}.csv', index=False)
