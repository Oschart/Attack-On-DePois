# %%
#def loss_object(labels, validity):
from keras.backend import sign
import tensorflow as tf
from mimic_model_construction import *
from main import load_data
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
enable_eager_execution()

batch_size = 32
sample_interval = 100
epochs = 200



(x_train, y_train), (x_test, y_test) = load_data()

wgan = CWGANGP(epochs, batch_size, sample_interval)
wgan.critic.trainable = True
#wgan.critic.load_weights('data/weights/discriminator_weights')

def create_adversarial_pattern( img, label):
  img = tf.cast(img, dtype=tf.float32)
  label = tf.cast(label, dtype=tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(img)
    #tape.watch(x)
    prediction = wgan.critic([img, label])
    loss = wgan.wasserstein_loss(label, prediction)
    #loss = x*x
    #loss = tf.convert_to_tensor(5, dtype=tf.float32)
    
    # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, img)
      # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)

  return signed_grad.numpy()


epsilons = [0, 0.01, 0.1, 0.15]
adv_imgs_dict = {eps: [] for eps in epsilons}

for img, label in zip(x_test, y_test):
  img = np.expand_dims(img, axis=0)
  label = np.expand_dims(label, axis=0)

  perturbations = create_adversarial_pattern(img, label)
  for eps in epsilons:
    adv_x = img + perturbations*eps
    #adv_x = tf.clip_by_value(adv_x, -1, 1)
    adv_x = np.squeeze(np.clip(adv_x, -1, 1))
    # adv_img_array = tf.make_ndarray(adv_x)
    adv_imgs_dict[eps].append((adv_x, label))
  
os.makedirs('dataset/adversarial', exist_ok=True)
with open('dataset/adversarial/adversarial_x_test.pkl', 'wb') as f:
    pickle.dump(adv_imgs_dict, f)
disable_eager_execution()