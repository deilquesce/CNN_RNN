# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:09:57 2023


train_daniel, test_daniel = fashion_mnist.load_data()
train_daniel['images'] = X_trs

file:///C:/Users/Daniel%20Matthew/Downloads/Building%20an%20Image%20Denoiser%20with%20a%20Keras%20autoencoder%20neural%20network.pdf

@author: Daniel Matthew
"""

# Get the data:
from tensorflow.keras.datasets import fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# reshape dataset to have a single channel
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Initial exploration
print(X_train.size)
print(Y_train.size)
print(X_train.shape)
print(Y_train.shape)

import numpy as np
print(np.amax(X_train))
print(np.amax(Y_train))

# Preprocess data
X_train = X_train / 255
X_test = X_test / 255
import tensorflow as tf
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=10)
print(Y_train.shape)
print(Y_test.shape)

# Visualization
import matplotlib.pyplot as plt
num_row = 4
num_col = 3
num = num_row*num_col
images = X_train[:num]
labels = Y_train[:num]
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_row*num_col):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()
"""
#additional visualization 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
"""
#Data is already split into train test. 

# Build, Train, and Validate CNN Model
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from tensorflow.keras import layers

model = Sequential()
# 1st Layer = Convolution with 32 filter kernels with window size 3x3 and a 'relu' activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 2nd Layer = Max Pooling with window size 2x2
model.add(MaxPooling2D((2, 2)))
# 3rd Layer = Convolution with 32 filter kernels with window size 3x3 and a 'relu' activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 4th Layer = Max Pooling with window size 2x2
model.add(MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# 5th Layer = Full connected layer with 100 neurons (Note: Input to fully connected layer should be flatten first)
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
# Output =   Set output size using info identified in Step c.3 and a softmax activation function
model.add(Dense(10, activation='softmax'))

# compile model\
from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Print summary
print(model.summary())

#train and validate model 
cnn_history_daniel = model.fit(X_train, Y_train, epochs=8, batch_size=256, validation_data=(X_test, Y_test))

#plot the training vs validation
plt.plot(cnn_history_daniel.history['accuracy'], label='accuracy')
plt.plot(cnn_history_daniel.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate the cnn model 
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_acc)

# Create predictions
cnn_predictions_daniel = model.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
y_pred=np.argmax(cnn_predictions_daniel, axis=1)
y_test=np.argmax(Y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#RNN
row, col, pixel = X_train.shape[1:]

# 4D input.
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

x = Input(shape=(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(128))(x)

# Encodes columns of encoded rows.
encoded_rows = TimeDistributed(LSTM(128))(x)
encoded_columns = LSTM(128)(encoded_rows)

# Final predictions and model.
prediction = Dense(10, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Training.
model.fit(X_train, Y_train,
          batch_size=256,
          epochs=8,
          validation_data=(X_test, Y_test))

# Evaluation.
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])