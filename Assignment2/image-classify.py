import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
(X_train, y_train),(X_test, y_test) = datasets.cifar10.load_data()
print(X_train.shape)
print(X_test.shape)
plt.imshow(X_train[0])
#plt.show()

X_train = X_train/255
X_test = X_test/255

cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    #dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=10)
cnn.evaluate(X_test,y_test)
y_pred = cnn.predict(X_test)
#print(y_pred[:5])
y_classes = [np.argmax(element) for element in y_pred]
#print(y_classes[:5])
# plt.imshow(X_train[3])
# plt.show()
print(classes[y_classes[3]])