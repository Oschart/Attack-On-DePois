from skimage.io import imshow
import numpy as np
from depois_model import *

GP_ds =  np.load('dataset/GP_mnist.npz')

GP_x_test, GP_y_test = GP_ds['X'], GP_ds['Y']
GP_x_test = GP_x_test.reshape(GP_x_test.shape[0], 5, 28, 28, 1)
GP_y_test = GP_y_test.reshape(GP_y_test.shape[0], 5) - 1

# D trusted 
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test, _ = preprocess_data(x_test, x_test)

depois_model = DePoisModel((x_test, y_test))
stats = {}
for budget_idx in range(5):
    X = GP_x_test[:, budget_idx, :, :, :]*(255)
    y = GP_y_test[:, budget_idx]
    X, _ = preprocess_data(X, X)
    is_poisoned_idx = depois_model.check_poisoned(X,y)
    acc = np.sum(is_poisoned_idx)/len(X)
    stats[budget_idx] = acc
print(stats)