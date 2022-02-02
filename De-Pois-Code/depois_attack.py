import keras.backend as K
import os
import math
import pickle as pkl
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

enable_eager_execution()
#disable_eager_execution()


class DePoisAttack():
	def __init__(self):
		self.class_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def create_adv_pattern(self, model, img, label, model_type='critic'):
		img = tf.cast(img, dtype=tf.float32)
		label = tf.cast(label, dtype=tf.float32)
		with tf.GradientTape() as tape:
			tape.watch(img)
			if model_type == 'critic':
				prediction = model([img, label])
				loss = self.wasserstein_loss(1, prediction)
			else:
				prediction = model.predict(img)
				loss = self.class_loss(label, prediction)
			
		# Get the gradients of the loss w.r.t to the input image.
		gradient = tape.gradient(loss, img)
		# Get the sign of the gradients to create the perturbation
		signed_grad = tf.sign(gradient)

		return signed_grad.numpy()

	def craft_adv_dataset(self, model, D_src, eps, model_type='critic', name='base_adv'):
		adv_dataset_pth = f"data/adversarial/{name}_{model_type}_{eps}.pkl"
		if os.path.isfile(adv_dataset_pth):
			adv_dataset = pkl.load(open(adv_dataset_pth, "rb"))
			return adv_dataset
		
		X_src, y_src = D_src
		X_adv = []
		for img, label in tqdm(zip(X_src, y_src)):
			img = np.expand_dims(img, axis=0)
			label = np.expand_dims(label, axis=0)

			perturbations = self.create_adv_pattern(model, img, label, model_type)
			adv_x = img + perturbations*eps
			adv_x = np.squeeze(np.clip(adv_x, -1, 1))
			X_adv.append(adv_x)
		
		adv_dataset = [X_adv, y_src]
		pkl.dump(adv_dataset, open(adv_dataset_pth, "wb"))
		return adv_dataset

	def clone_model(self, src_model, clone_model):
		return

	def wb_attack(self, depois_model, D_src, eps):
		adv_dataset_cr = self.craft_adv_dataset(depois_model.critic, D_src, eps, model_type='critic')
		adv_dataset_cr_cl = self.craft_adv_dataset(depois_model.classifier, adv_dataset_cr, eps, model_type='classifier')

	
	def bb_attack(self, depois_model, X_src):
		return