import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

classes = ["tshirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add channel dimension for grayscale images
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# plt.imshow(X_train[0])
# plt.show()

X_train = X_train/255
X_test = X_test/255
input_shape=(28,28,1)
cnn = models.Sequential([
    Input(shape=input_shape),
    #cnn
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    #dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=5)
cnn.evaluate(X_test,y_test)
y_pred = cnn.predict(X_test)
#print(y_pred[:5])
y_classes = [np.argmax(element) for element in y_pred]
#print(y_classes[:5])
plt.imshow(X_test[0])
plt.show()
print(classes[y_classes[0]])