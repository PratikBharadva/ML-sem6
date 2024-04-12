import tensorflow.keras
keras.backend.backend()
#keras.backend.image_dim_ordering()
# using theano has backend
'''K = keras.backend.backend()
if K=='tensorflow':
    keras.backend.set_image_dim_ordering('tf')
else:
    keras.backend.set_image_dim_ordering('th')
'''
from matplotlib import pyplot as plt
 
import numpy as np
np.random.seed(2017)
from keras import backend as K
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.preprocessing import sequence
from keras import backend as K
from keras.utils.visualize_util import plot
from IPython.display import SVG, display
from keras.utils.visualize_util import model_to_dot, plot
img_rows, img_cols = 28, 28
nb_classes = 10
nb_filters = 5 # the number of filters
nb_pool = 2 # window size of pooling
nb_conv = 3 # window or kernel size of filter
nb_epoch = 5
 
# image dimension based on backend. ‘th’ = theano and ‘tf’ = tensorflow
if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
# data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
 
# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
Convolution2D(nb_filters, nb_conv, nb_conv, input_shape=input_shape),
Activation('relu'),
Convolution2D(nb_filters, nb_conv, nb_conv),
Activation('relu'),
MaxPooling2D(pool_size=(nb_pool, nb_pool)),
Dropout(0.25),
Flatten(),
]
classification_layers = [
Dense(128),
Activation('relu'),
Dropout(0.5),
Dense(nb_classes),
Activation('softmax')
]
# create complete model
model = Sequential(feature_layers + classification_layers)
# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
Convolution2D(nb_filters, nb_conv, nb_conv, input_shape=input_shape),
Activation('relu'),
Convolution2D(nb_filters, nb_conv, nb_conv),
Activation('relu'),
MaxPooling2D(pool_size=(nb_pool, nb_pool)),
Dropout(0.25),
Flatten(),
] 
classification_layers = [
Dense(128),
Activation('relu'),
Dropout(0.5),
Dense(nb_classes),
Activation('softmax')
]
# create complete model
model = Sequential(feature_layers + classification_layers)
print(model.summary()) 