
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from distiller import Distiller
import os

class MNISTClassifier():
    def __init__(self, load=False, load_pth='classifier/mnist_classifier') -> None:
        if load:
            self.classifier =  keras.models.load_model(load_pth)
        else:
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
            # Train classifier as usual
            self.classifier.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()],
            )
    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Normalize data
        x_train = x_train.astype("float32") / 255.0
        x_train = np.reshape(x_train, (-1, 28, 28, 1))

        x_test = x_test.astype("float32") / 255.0
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
        return (x_train, y_train), (x_test, y_test)
    def train(self):
        (x_train, y_train), (x_test, y_test) = self.load_dataset()
        # Train and evaluate classifier on data.
        self.classifier.fit(x_train, y_train, epochs=5)
    def evaluate(self, x_test, y_test):
        return self.classifier.evaluate(x_test, y_test)
    def save(self, save_pth=None):
        if save_pth is None:
            os.makedirs('classifier', exist_ok=True)
            self.classifier.save('classifier/mnist_classifier')
        else:
            self.classifier.save(save_pth)
