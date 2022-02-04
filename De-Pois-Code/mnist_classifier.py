# %%
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
K.set_image_data_format('channels_first')

class MNISTClassifier():
    def __init__(self, load=False, load_pth='weights/mnist_classifier') -> None:
        self.model_pth = load_pth
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
    def preprocess_data(self, x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_test = (x_test.astype(np.float32) - 127.5) / 127.5
        return x_train, x_test

    def load_dataset(self):

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = self.preprocess_data(x_train, x_test)
        return (x_train, y_train), (x_test, y_test)

    def train(self, epochs=5):
        (x_train, y_train), (_, _) = self.load_dataset()
        # Train and evaluate classifier on data.
        self.classifier.fit(x_train, y_train, epochs=epochs, batch_size=32)
    def evaluate(self, x_test, y_test):
        return self.classifier.evaluate(x_test, y_test)
    def save(self, save_pth=None):
        if save_pth is None:
            self.classifier.save(self.model_pth)
        else:
            self.classifier.save(save_pth)
    def predict(self, x_test):
        return self.classifier.predict(x_test)