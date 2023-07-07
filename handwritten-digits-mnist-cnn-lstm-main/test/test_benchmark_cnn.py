import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
import pickle

# mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize to [0,1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])


if not os.path.exists('history'):
    os.makedirs('history')

if not os.path.exists('history/histfile_cnn_e5.pkl'):   # Not Found: Record and save to file
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    with open('history/histfile_cnn_e5.pkl', 'wb') as f:
        pickle.dump(history.history, f)

with open('history/histfile_cnn_e5.pkl', 'rb') as f:
    history = pickle.load(f)



plt.figure("CNN")
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')

plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')

plt.legend()
plt.show()
