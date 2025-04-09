import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

X_train = pd.read_csv('input.csv')
Y_train = pd.read_csv('labels.csv')
X_test = pd.read_csv('input_test.csv')
Y_test = pd.read_csv('labels_test.csv')

print("Shape of X train:", X_train.shape)
print("Shape of Y train:", Y_train.shape)
print("Shape of X test:", X_test.shape)
print("Shape of Y test:", Y_test.shape)

# X_train = X_train.reshape(len(X_train), 100,100,3)
# Y_train = Y_train.reshape(len(Y_train), 1)
# X_test = X_test.reshape(len(X_test), 100,100,3)
# Y_test = Y_test.reshape(len(Y_test), 1)

# X_train = X_train/255.0
# X_test = X_test/255.0   

# index = random.randint(0,len(X_train))
# plt.imshow(X_train[index, :])
# plt.show()

# cnn = Sequential([
#     #cnn
#     Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(100,100,3)),
#     MaxPooling2D((2,2)),

#     Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
#     MaxPooling2D((2,2)),

#     #dense
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
# cnn.compile(optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy'])
# cnn.fit(X_train, Y_train, epochs=5, batch_size=64)
# cnn.evaluate(X_test,Y_test)

# index2 = random.randint(0,len(Y_test))
# plt.imshow(X_test[index2, :])
# plt.show()

# Y_pred = cnn.predict(X_test[index2,:].reshape(1,100,100,3))
# c = Y_pred>0.5
# if(c==0):
#     print("It's a Dog")
# else:
#     print("It's a cat")
