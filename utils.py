import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import os
import shutil
import cv2
from google.colab.patches import cv2_imshow

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


# Description: This file is used to check the image
img = x_train[6] * 255
cv2.imshow(img)


yhat = model.predict(x_test)
y_predicted = np.argmax(yhat, axis=1)
y_true = np.argmax(y_test, axis=1)
test_result = pd.DataFrame({'predicted': y_predicted, 'true': y_true})
test_result['correct'] = (test_result['predicted'] == test_result['true'])
summary = test_result.groupby(['true'])['correct'].agg(
    correct='sum', total='count')
summary['correction_rate'] = summary['correct'] / summary['total'] * 100


confusion_matrix(test_result['true'], test_result['predicted'])
