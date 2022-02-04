# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:19:13 2020

@author: zxx
"""
from __future__ import print_function, division

from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import RMSprop
#from keras.optimizers import Adam
from functools import partial
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()
import keras.backend as K
import os
import matplotlib.pyplot as plt

import math
import numpy as np


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        batch_size = 32
#        global batch_size
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class CWGANGP():
    def __init__(self, epochs=100, batch_size=32, sample_interval=50):
        global fake_img
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.nclasses = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.losslog = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        
        # Generate image based of noise (fake sample) and add label to the input 
        label = Input(shape=(1,))
        fake_img = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_img, label])
        valid = self.critic([real_img, label])

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_img, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, label, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # add label to the input
        label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen, label])
        # Discriminator determines validity
        valid = self.critic([img, label])
        # Defines generator model
        self.generator_model = Model([z_gen, label], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        
        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.nclasses, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)   
    

    
    def build_critic(self):

        model = Sequential()

        model.add(Dense(1024, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1))
        #model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.nclasses, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)
    
    
    def train(self, will_load_model):

        # Load the dataset

        X_train = np.load("data/saved_TrueAndGeneratorData.npy")  
        y_train = np.load("data/saved_TrueAndGeneratorLabel.npy") 

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 0.5) / 0.5
        #X_train = np.expand_dims(X_train, axis=1)

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        
        if will_load_model:
            self.generator.load_weights(f'data/weights/generator_CWGANGP_{self.epochs}')
            self.critic.load_weights(f'data/weights/discriminator_CWGANGP_{self.epochs}')
            print('The %d trained CWGANGP model loaded!'%self.epochs)
            #print('There is not %d trained CWGANGP model!'%self.epochs)
        else:
            for epoch in range(self.epochs):
                for _ in range(self.n_critic):
    
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
    
                    # Select a random batch of images
                    idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                    imgs, labels = X_train[idx], y_train[idx]

                    # Sample generator input
                    noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                    # Train the critic

    
                    d_loss = self.critic_model.train_on_batch([imgs, labels, noise], [valid, fake, dummy])
    
                # ---------------------
                #  Train Generator
                # ---------------------
                sampled_labels = np.random.randint(0, self.nclasses, self.batch_size).reshape(-1, 1)
                g_loss = self.generator_model.train_on_batch([noise, sampled_labels], valid)
    
                # Plot the progress
                
                self.losslog.append([d_loss[0], g_loss])
                
                # If at save interval => save generated image samples
                if (epoch+1) % self.sample_interval == 0:
                    print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                    self.sample_images(epoch)
                    self.generator.save_weights('data/weights/generator_CWGANGP_%d'%(epoch+1), overwrite=True)
                    self.critic.save_weights('data/weights/discriminator_CWGANGP_%d'%(epoch+1), overwrite=True)

            np.save('data/loss.npy',self.losslog)
            self.generator.save_weights('data/generator_CWGANGP_%d'%self.epochs, overwrite=True)
            self.critic.save_weights('data/discriminator_CWGANGP_%d'%self.epochs, overwrite=True)
            print('save the CWGANGP model')



    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array(list(range(r))*c).reshape(-1, 1)
        
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * (gen_imgs + 1)
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        os.makedirs('images_cwgan_gp', exist_ok=True)
        fig.savefig("images_cwgan_gp/mnist_%d.png" % (epoch))
        plt.close()
        
    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[1:3]
        print(shape)
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[:, :, 0]
        return image
    
    def generate_images(self, label):
        self.generator.load_weights('generator')
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict([noise, np.array(label).reshape(-1,1)])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1
        print(gen_imgs.shape)
        plt.imshow(gen_imgs[0,0,:,:], cmap='gray')

    def discriminate_img(self, img, label):
        self.critic.load_weights('data/weights/discriminator_CWGANGP_%d'%self.epochs)
        validity = self.critic.predict([img, label])
        return validity
    
    def savemodel(self,epoch):
        self.generator.save_weights('data/weights/generator_%d'%epoch, overwrite=True)
        self.critic.save_weights('data/weights/discriminator_%d'%epoch, overwrite=True)




       
    
    
    
    
    
    
