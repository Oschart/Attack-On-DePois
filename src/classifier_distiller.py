from distutils.command.build import build
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers import  Embedding
from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.backend import sign
import tensorflow as tf
from mimic_model_construction import *
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from tqdm import tqdm
from mnist_classifier import MNISTClassifier
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
enable_eager_execution()


class ClassifierDistiller(keras.Model):
    def __init__(self):
        super(ClassifierDistiller, self).__init__()
        self.teacher = self.build_teacher_classifier()
        self.student = self.build_student_classifier()

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(ClassifierDistiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def build_teacher_classifier(self):

        return MNISTClassifier(load=True, load_pth='weights/mnist_classifier').classifier

    def build_student_classifier(self):

        return keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(10),
            ],
            name="student",
        )


    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def distill(self, x_train, y_train):

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )
        self.fit(x_train, y_train, batch_size=32, epochs=1)
        self.student.save('weights/shadow_classifier/shadow_classifier')





