import functools
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_SIZE = 17000


# https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize(
            [img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(224, 224))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(
        shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    return int(parts[-2])


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
    dataset = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    augmentations = [flip, zoom, rotate]
    for f in augmentations:
        dataset = dataset.map(lambda x, y: tf.cond(tf.random.uniform(
            [], 0, 1) > 0.75, lambda: (f(x), y), lambda: (x, y)), num_parallel_calls=4)

    dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))

    return dataset


# https://github.com/fenwickslab/fenwicks
def crossval_ds(dataset, n_folds: int, val_fold_idx: int, training: bool = True) -> tf.data.Dataset:
    """
    Partition a given `tf.data` dataset into training and validation sets, according to k-fold cross validation
    requirements.

    :param dataset: A given `tf.data` dataset containing the whole dataset.
    :param n_folds: Number of cross validation folds.
    :param val_fold_idx: Fold ID for validation set, in cross validation.
    :param training: Whether to return training or validation data.
    :return: either training or validation dataset.
    """
    if training:
        trn_shards = itertools.chain(
            range(val_fold_idx), range(val_fold_idx + 1, n_folds))

        def update_func(ds, i): return ds.concatenate(
            dataset.shard(n_folds, i))
        dataset = functools.reduce(
            update_func, trn_shards, dataset.shard(n_folds, next(trn_shards)))
    else:
        dataset = dataset.shard(n_folds, val_fold_idx)
    return dataset


def process_test_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, [True, False]


# wtf tensorflow
def get_labels(dataset):
    return [y.numpy() for x, y in dataset]


def class_weight(dataset):
    y = get_labels(dataset)
    counter = Counter(y)
    total = len(y)
    weight_for_0 = (1 / counter[0])*(total)/2.0
    weight_for_1 = (1 / counter[1])*(total)/2.0
    return {0: weight_for_0, 1: weight_for_1}, np.log([counter[1]/counter[0]])
