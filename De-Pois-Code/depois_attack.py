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
from classifier_distiller import ClassifierDistiller
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
			if 'critic' in model_type:
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

	def craft_adv_dataset(self, model, D_src, eps, model_type='critic', attack_mode='CL_only'):
		os.makedirs('data/adversarial', exist_ok=True)
		adv_dataset_pth = f"data/adversarial/adv_ds_{model_type}_{eps}_{attack_mode}.pkl"
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

	def clone_critic(self, x_train, y_train):
		if not (os.path.exists("weights/shadow_critic")):
			# Initialize and compile distiller
			critic_distiller = CriticDistiller()
			critic_distiller.distill(x_train, y_train)
		
	def clone_classifier(self, x_train, y_train):
		if not (os.path.exists("weights/shadow_classifier/shadow_classifier")):
			# Initialize and compile distiller
			classifier_distiller = ClassifierDistiller()
			classifier_distiller.distill(x_train, y_train)

	def wb_attack_classifier(self, depois_model, D_src, eps):
		adv1 = self.craft_adv_dataset(depois_model.classifier, D_src, eps, model_type='classifier')
		return adv1

	def wb_attack(self, depois_model, D_src, eps, attack_mode='CL_only'):
		if attack_mode == 'CR_then_CL':
			adv_critic = self.craft_adv_dataset(depois_model.critic, D_src, eps, model_type='critic', attack_mode=attack_mode)
			adv_data = self.craft_adv_dataset(depois_model.classifier, adv_critic, eps, model_type='classifier', attack_mode=attack_mode)
		elif attack_mode == 'CL_then_CR':
			adv_classifier = self.craft_adv_dataset(depois_model.classifier, D_src, eps, model_type='classifier', attack_mode=attack_mode)
			adv_data = self.craft_adv_dataset(depois_model.critic, adv_classifier, eps, model_type='critic', attack_mode=attack_mode)
		elif attack_mode == 'CR_only':
			reuse_name = 'CR_then_CL'
			adv_data = self.craft_adv_dataset(depois_model.critic, D_src, eps, model_type='critic', attack_mode=reuse_name)
		else:
			reuse_name = 'CL_then_CR'
			adv_data = self.craft_adv_dataset(depois_model.classifier, D_src, eps, model_type='classifier', attack_mode=reuse_name)

		return adv_data
	
	def bb_attack(self, depois_model, D_src, eps, attack_mode='CL_only'):
		if attack_mode == 'CR_then_CL':
			adv_critic = self.craft_adv_dataset(depois_model.shadow_critic, D_src, eps, model_type='shadow_critic', attack_mode=attack_mode)
			adv_data = self.craft_adv_dataset(depois_model.shadow_classifier, adv_critic, eps, model_type='shadow_classifier', attack_mode=attack_mode)
		elif attack_mode == 'CL_then_CR':
			adv_classifier = self.craft_adv_dataset(depois_model.shadow_classifier, D_src, eps, model_type='shadow_classifier', attack_mode=attack_mode)
			adv_data = self.craft_adv_dataset(depois_model.shadow_critic, adv_classifier, eps, model_type='shadow_critic', attack_mode=attack_mode)
		elif attack_mode == 'CR_only':
			reuse_name = 'CR_then_CL'
			adv_data = self.craft_adv_dataset(depois_model.shadow_critic, D_src, eps, model_type='shadow_critic', attack_mode=reuse_name)
		else:
			reuse_name = 'CL_then_CR'
			adv_data = self.craft_adv_dataset(depois_model.shadow_classifier, D_src, eps, model_type='shadow_classifier', attack_mode=reuse_name)

		return adv_data