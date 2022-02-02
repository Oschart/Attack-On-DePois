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

from critic_distiller import CriticDistiller

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
				prediction = model.classifier(img)
				loss = self.class_loss(label, prediction)
			
		# Get the gradients of the loss w.r.t to the input image.
		gradient = tape.gradient(loss, img)
		# Get the sign of the gradients to create the perturbation
		signed_grad = tf.sign(gradient)

		return signed_grad.numpy()

	def craft_adv_dataset(self, model, D_src, eps, model_type='critic', name='adv_ds', critic_first=True):
		os.makedirs('data/adversarial', exist_ok=True)
		adv_dataset_pth = f"data/adversarial/{name}_{model_type}_{eps}_{critic_first}.pkl"
		if os.path.isfile(adv_dataset_pth):
			adv_dataset = pkl.load(open(adv_dataset_pth, "rb"))
			return np.array(adv_dataset[0]), np.array(adv_dataset[1])
		
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
		return np.array(X_adv), y_src

	def clone_critic_model(self, data):

		# Initialize and compile distiller
		distiller = CriticDistiller()
		distiller.compile(
			optimizer=keras.optimizers.Adam(),
			distillation_loss_fn=keras.losses.KLDivergence(),
			alpha=0.1,
			temperature=1,
		)

		(x_train, y_train), (x_test, y_test) = data

		# Distill teacher to student
		distiller.fit(x_train, y_train, batch_size=2084, epochs=10)

		return distiller

	def wb_attack(self, depois_model, D_src, eps, critic_first=True):
		if critic_first:
			adv1 = self.craft_adv_dataset(depois_model.critic, D_src, eps, model_type='critic', critic_first=critic_first)
			adv11 = self.craft_adv_dataset(depois_model.classifier, adv1, eps, model_type='classifier', critic_first=critic_first)
		else:
			adv1 = self.craft_adv_dataset(depois_model.classifier, D_src, eps, model_type='classifier', critic_first=critic_first)
			adv11= self.craft_adv_dataset(depois_model.critic, adv1, eps, model_type='critic', critic_first=critic_first)
		return adv11

	
	def bb_attack(self, depois_model, X_src):
		return