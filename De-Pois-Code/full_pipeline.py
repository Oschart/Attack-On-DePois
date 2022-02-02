import tensorflow as tf
from tensorflow import keras

import numpy as np

from depois_model import DePoisModel
from depois_attack import DePoisAttack


def preprocess_data(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    return x_train, x_test

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = preprocess_data(x_train, x_test)

depois_model = DePoisModel((x_test, y_test))

depois_attack = DePoisAttack()
depois_attack.wb_attack(depois_model, (x_test, y_test), 0.1)