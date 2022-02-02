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
from main import load_data
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from tqdm import tqdm
from main import *
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
enable_eager_execution()


class CriticDistiller(keras.Model):
    def __init__(self):
        super(CriticDistiller, self).__init__()
        self.teacher = self.build_teacher_critic()
        self.student = self.build_student_critic()

    def compile(
        self,
        optimizer,
        distillation_loss_fn,
        alpha=0.1,
        temperature=1,
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
        super(CriticDistiller, self).compile(optimizer=optimizer)
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def build_student_critic(self):
        img_shape = (28, 28, 1)
        nclasses = 10
        model = Sequential()
        model.add(Dense(1024, input_dim=np.prod(img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1))
        model.summary()

        img = Input(shape=img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(nclasses, np.prod(img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)


    def build_teacher_critic(self):
        batch_size = 32
        sample_interval = 100
        epochs = 10000

        wgan = CWGANGP(epochs, batch_size, sample_interval)
        wgan.critic.trainable = True
        wgan.critic.load_weights(f'data/weights/discriminator_CWGANGP_{epochs}')
        print("Loaded critic weights!!")
        return wgan.critic

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher([x, y], training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student([x, y], training=True)

            # Compute losses
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions[:,0] / self.temperature),
                tf.nn.softmax(student_predictions[:,0] / self.temperature),
            )
            loss =  distillation_loss


        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"distillation_loss": distillation_loss}
        )
        return results





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
from main import load_data
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from tqdm import tqdm
from main import *
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
enable_eager_execution()


class ClassifierDistiller(keras.Model):
    def __init__(self, student, teacher):
        super(ClassifierDistiller, self).__init__()
        self.teacher = teacher
        self.student = student

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

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
    def distill(self, data):
        self.compile(
            optimizer=keras.optimizers.Adam(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=1,
        )
        (x_train, y_train), (x_test, y_test) = data
        self.fit(x_train, y_train, batch_size=2084, epochs=10)
