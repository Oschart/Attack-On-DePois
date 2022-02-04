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

from mimic_model_construction import *
from mnist_classifier import MNISTClassifier

disable_eager_execution()

K.set_image_data_format('channels_first')


class DePoisModel():
    def __init__(self, D_trust):
        self.load_models()
        self.dec_bound = self.compute_decision_bound(D_trust[0], D_trust[1])

    def load_models(self):
        self.load_critic()
        self.load_classifier()

        self.load_shadow_critic()
        self.load_shadow_classifier()

    def load_critic(self):
        epochs = 10000
        batch_size = 32
        sample_interval = 100
        wgan = CWGANGP(epochs, batch_size, sample_interval)
        wgan.critic.load_weights(
            f'weights/critic/discriminator_CWGANGP_{epochs}')
        self.critic = wgan.critic

    def load_classifier(self):
        self.classifier = MNISTClassifier(load=True)

    def load_shadow_critic(self):
        epochs = 10000
        batch_size = 32
        sample_interval = 100
        wgan = CWGANGP(epochs, batch_size, sample_interval)
        wgan.critic.load_weights(
            f'weights/shadow_critic/shadow_critic')
        self.shadow_critic = wgan.critic

    def load_shadow_classifier(self):
        self.shadow_classifier = MNISTClassifier(load=True, load_pth='weights/shadow_classifier/shadow_classifier')

    def compute_decision_bound(self, X_t, y_t):
        validity = self.critic.predict([X_t, y_t]).flatten()
        z_score = np.mean(validity) - np.std(validity)
        return z_score

    def predict(self, X, y=None):
        if y is not None:
            y_critic = y.copy().astype(np.int64)
            y_pred = np.argmax(self.classifier.predict(X), axis=1)
        else:
            y_critic = np.argmax(self.classifier.predict(X), axis=1)
            y_pred = y_critic.copy()

        is_poisoned_idx = self.check_poisoned(X, y_critic)
        # Deactivate the classifier label for poisoned data
        y_pred[is_poisoned_idx] = -1
        return y_pred

    def evaluate(self,x_test, x_test_adv_cr_cl, y_test, eps):
        x_test_adv_cr_cl = x_test_adv_cr_cl.reshape((x_test_adv_cr_cl.shape[0], 28, 28, 1))
        y_pred = self.predict(x_test_adv_cr_cl, y=None)
        
        is_valid_idx = y_pred != -1
        is_poi_idx = 1 - is_valid_idx
        # critic_acc = poinsoned_detected/all_poisoned
        critic_acc = np.sum(is_poi_idx)/len(x_test_adv_cr_cl)
        # cls_acc = poisone_fooled&correctly classified/poinson_fooled
        depois_acc = (np.sum(np.logical_and(is_valid_idx, y_pred == y_test)) + np.sum(is_poi_idx))/len(x_test_adv_cr_cl)
        stats = {"critic_acc":critic_acc, "depois_acc":depois_acc}
        return stats

    def check_poisoned(self, X, y):
        validity = self.critic.predict([X, y])
        is_poisoned_idx = validity <= self.dec_bound
        return is_poisoned_idx.squeeze()
