# %%
#def loss_object(labels, validity):
from keras.backend import sign
import tensorflow as tf
from mimic_model_construction import *
from main import load_data
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from tqdm import tqdm
from main import *

enable_eager_execution()

batch_size = 32
sample_interval = 100
epochs = 10000

(x_train, y_train), (x_test, y_test) = load_data()

wgan = CWGANGP(epochs, batch_size, sample_interval)
wgan.critic.trainable = True
wgan.critic.load_weights(f'data/weights/discriminator_CWGANGP_{epochs}')
print("Loaded critic weights!!")
adv_dataset_pth = 'dataset/adversarial_x_test_1.pkl'
def create_adversarial_pattern( img, label):
  img = tf.cast(img, dtype=tf.float32)
  label = tf.cast(label, dtype=tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(img)
    #tape.watch(x)
    prediction = wgan.critic([img, label])
    loss = wgan.wasserstein_loss(1, prediction)
    #loss = x*x
    #loss = tf.convert_to_tensor(5, dtype=tf.float32)
    
    # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, img)
      # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)

  return signed_grad.numpy()


epsilons = [0.001, 0.01, 0.1, 0.5]
adv_imgs_dict = {eps: [] for eps in epsilons}

if not os.path.isfile(adv_dataset_pth):
  for img, label in tqdm(zip(x_test, y_test)):
    img = np.expand_dims(img, axis=0)
    label = np.expand_dims(label, axis=0)

    perturbations = create_adversarial_pattern(img, label)
    for eps in epsilons:
      adv_x = img + perturbations*eps
      #adv_x = tf.clip_by_value(adv_x, -1, 1)
      adv_x = np.squeeze(np.clip(adv_x, -1, 1))
      # adv_img_array = tf.make_ndarray(adv_x)
      adv_imgs_dict[eps].append((adv_x, label))
 
  os.makedirs('dataset', exist_ok=True)

  with open(adv_dataset_pth, 'wb') as f:
    pickle.dump(adv_imgs_dict, f)
else:
  with open(adv_dataset_pth, 'rb') as f:
    adv_imgs_dict = pickle.load(f)

def evaluate_critic(critic, adv_imgs_dict):
  for eps in adv_imgs_dict:
    eps_data = adv_imgs_dict[eps]
    validity = []
    poisoned_x_data = np.array([img_ for img_, _ in eps_data])
    poisoned_y_data = np.array([label for _, label in eps_data])
    poisoned_x_data = np.expand_dims(poisoned_x_data, axis=3)
    x_poisoned_raw_test = np.concatenate((x_train[0:len(poisoned_x_data)],poisoned_x_data))   
    y_poisoned_raw_test = np.concatenate((y_train[0:len(poisoned_x_data)],np.squeeze(poisoned_y_data)))
    validity = critic.predict([x_poisoned_raw_test, y_poisoned_raw_test]).flatten()
    F1, P, R, acc, acc_adv, acc_real = count_score(validity, poison_number=len(poisoned_x_data))
    print(f"Eps= {eps}, Critic Avdersarial Accuracy={acc_adv}, Critical Accuracy on Real= {acc_real}")

#disable_eager_execution()

evaluate_critic(wgan.critic, adv_imgs_dict)

# %%
