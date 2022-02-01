import pickle as pickle
#from keras.optimizers import Adam
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend import sign
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Embedding, Flatten, Input, Reshape, ZeroPadding2D,
                          multiply)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import _Merge
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.framework.ops import (disable_eager_execution,
                                             enable_eager_execution)
from tqdm import tqdm

from main import *
from main import load_data
from mimic_model_construction import *

disable_eager_execution()
import math
import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


class DePoisModel():
    def __init__(self, D_trust):
        self.load_models()
        self.dec_bound = self.compute_decision_bound(D_trust[0], D_trust[1])


    def load_models(self):
        self.load_critic()

    def load_critic(self):
        epochs = 10000
        batch_size = 32
        sample_interval = 100
        wgan = CWGANGP(epochs, batch_size, sample_interval)
        wgan.critic.load_weights(f'data/weights/discriminator_CWGANGP_{epochs}')
        self.critic = wgan.critic

    def load_classifier(self):
        self.classifier = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(10),
            ],
            name="classifier",
        )
        self.classifier.load_weights(f"weights/mnist_classifier")
        
    def compute_decision_bound(self, X_t, y_t):
        validity = self.critic.predict([X_t, y_t]).flatten()
        z_score = np.mean(validity) - np.std(validity)
        return z_score

    def predict(self, X, y):
        y_pred = self.classifier.predict(X)
        is_poisoned_idx = self.check_poisoned(X, y_pred)
        # Deactivate the classifier label for poisoned data
        y_pred[is_poisoned_idx] = -1
        return y_pred

    def check_poisoned(self, X, y):
        validity = self.critic.predict([X, y]).flatten()
        is_poisoned_idx = validity <= self.dec_bound
        #is_valid = np.ones(validity.shape[0])
        #is_valid[is_poisoned_idx] = 0
        return is_poisoned_idx
