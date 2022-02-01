import keras.backend as K
import os
import math
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
from mnist_classifier import MNISTClassifier

disable_eager_execution()


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
        wgan.critic.load_weights(
            f'data/weights/discriminator_CWGANGP_{epochs}')
        self.critic = wgan.critic

    def load_classifier(self):
        self.classifier = MNISTClassifier(load=True).classifier

    def compute_decision_bound(self, X_t, y_t):
        validity = self.critic.predict([X_t, y_t]).flatten()
        z_score = np.mean(validity) - np.std(validity)
        return z_score

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        is_poisoned_idx = self.check_poisoned(X, y_pred)
        # Deactivate the classifier label for poisoned data
        y_pred[is_poisoned_idx] = -1
        return y_pred

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)

        # Compute accuracy scores
        P = metrics.precision_score(y_true, y_pred.astype(float))
        R = metrics.recall_score(y_true, y_pred.astype(float))
        F1 = (2 * P * R) / (P + R)
        acc = metrics.accuracy_score(
            label_poisoned_real, label_poisoned_fake.astype(float))

        # Combine the the stats
        stats = dict(P=P, R=R, F1=F1, acc=acc)
        return stats

    def check_poisoned(self, X, y):
        validity = self.critic.predict([X, y]).flatten()
        is_poisoned_idx = validity <= self.dec_bound
        return is_poisoned_idx
