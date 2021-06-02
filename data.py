import os
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2


def load_data(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    total_size = len(images)
    validation_size = int(split * total_size)
    test_size = int(split * total_size)
    train_x, valid_x = train_test_split(images, test_size=validation_size, random_state=48)
    train_y, valid_y = train_test_split(masks, test_size=validation_size, random_state=48)
    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=48)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=48)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path):
    path = path.decode()
    k = cv2.imread(path, cv2.IMREAD_COLOR)
    k = cv2.resize(k, (256, 256))
    k = k / 255
    return k


def read_mask(path):
    path = path.decode()
    k = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    k = cv2.resize(k, (256, 256))
    k = k / 255
    k = np.expand_dims(k, axis=-1)
    return k


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset



