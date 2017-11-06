import os

import keras
import numpy as np
from keras import applications
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
#######
from PIL import Image
from sklearn.model_selection import train_test_split

import util_00 as util

batch_size = 10
num_classes = 4 # number of categories/classes
epochs = 10

x_train, x_test, y_train, y_test = util.get_data()

print(np.shape(x_train))
print(np.shape(x_test))
#x_train = x_train.reshape(-1, 16384,3)
#x_test = x_test.reshape(-1, 16384, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##########################################
# (conv + max-pool) * 2 + dense + output #
##########################################

img_rows, img_cols = 128, 128
#x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 3) 
#x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 3)

model = Sequential() #Sequential model is a linear stack of layers
#1 layer
# input: img_rows x img_cols images with 3 channels -> (?, ?, 3) tensors.
# this applies 32 convolution filters of size 5x5 each.
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#2 layer
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten()) #Convert 3D feature maps to 1D feature vectors
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#Use categorical_crossentropy loss model to train the model 
#SGD - Stochastic gradient descent optimizer.
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

#epochs - how many times you go through the training set
run = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save('model_k_seq.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
print('accuracy_score:', accuracy_score(y_test, y_pred))

mat = confusion_matrix(y_test, y_pred)
sns_plot = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('cnn_confusion.png')
plt.show()


