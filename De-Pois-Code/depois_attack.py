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

disable_eager_execution()


class DePoisAttack():
	def __init__(self):
		self.class_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


	def create_adv_pattern(self, model, img, label, model_type='critic'):
		img = tf.cast(img, dtype=tf.float32)
		label = tf.cast(label, dtype=tf.float32)
		with tf.GradientTape() as tape:
			tape.watch(img)
			if model_type == 'critic':
				prediction = model([img, label])
				loss = model.wasserstein_loss(1, prediction)
			else:
				prediction = model(img)
				loss = self.class_loss(label, prediction)
			
		# Get the gradients of the loss w.r.t to the input image.
		gradient = tape.gradient(loss, img)
		# Get the sign of the gradients to create the perturbation
		signed_grad = tf.sign(gradient)

		return signed_grad.numpy()

	def craft_adv_dataset(self, model, D_src, model_type='critic', name='base_adv'):
		adv_dataset_pth = f"data/adversarial/{name}.pkl"
		if os.path.isfile(adv_dataset_pth):
			adv_dataset = pkl.load(open(adv_dataset_pth, "rb"))
			return adv_dataset
		
		X_src, y_src = D_src
		epsilons = [0.001, 0.01, 0.1, 0.5]
		adv_dataset = {eps: [] for eps in epsilons}
		for img, label in tqdm(zip(X_src, y_src)):
			img = np.expand_dims(img, axis=0)
			label = np.expand_dims(label, axis=0)

			perturbations = self.create_adv_pattern(model, img, label, model_type)
			for eps in epsilons:
				adv_x = img + perturbations*eps
				adv_x = np.squeeze(np.clip(adv_x, -1, 1))
	  	adv_dataset[eps].append((adv_x, label))
		return adv_dataset

	def clone_model(self, src_model, clone_model):
		return

	def wb_attack(self, depois_model, X_src):
		return
	
	def bb_attack(self, depois_model, X_src):
		return