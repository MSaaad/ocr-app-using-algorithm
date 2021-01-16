import shutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

#import openCV
import cv2
from PIL import Image
import matplotlib.image as mpimg
# importing tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Activation, Dropout, Flatten, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import applications, optimizers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# import zipfile
# with zipfile.ZipFile('content/denoising-dirty-documents.zip', 'r') as zip_ref:
#     zip_ref.extractall()
# with zipfile.ZipFile('content/test.zip', 'r') as test:
#     test.extractall()

# with zipfile.ZipFile('content/train.zip', 'r') as train:
#     train.extractall()

# with zipfile.ZipFile('content/train_cleaned.zip', 'r') as train_cleaned:
#     train_cleaned.extractall()


img = mpimg.imread('content/train/101.png')
imgplot = plt.imshow(img)
plt.title('Before Denoising Image Display')
plt.show()

img = mpimg.imread('content/train_cleaned/101.png')
imgplot = plt.imshow(img)
plt.title('After Denoising Image Display')
plt.show()

x_train = []

path = 'content/train/'
for i in os.listdir(path=path):
    x_train.append((path+str(i)))
print('Total images in train dataset: ', len(x_train))

y_train = []

path = 'content/train_cleaned/'
for i in os.listdir(path=path):
    y_train.append((path+str(i)))

print('Total images in train_cleaned dataset: ', len(y_train))

test = []

path = 'content/test/'
for i in os.listdir(path=path):
    test.append((path+str(i)))

print('Total images in test dataset: ', len(test))
x_train[:5]

for i in x_train[:5]:
    img = cv2.imread(i)
    height, width, channels = img.shape
    print(img.shape)
ht = []
wd = []
for i in x_train:
    img = cv2.imread(i)
    height, width, channels = img.shape
    ht.append(img.shape[0])
    wd.append(img.shape[1])

print('Max Height of image in x_train', max(ht))
print('Min Height of image in x_train', min(ht))
print('Max Width of image in x_train', max(wd))
print('Min Width of image in x_train', max(wd))

ht = []
wd = []
for i in y_train:
    img = cv2.imread(i)
    height, width, channels = img.shape
    ht.append(img.shape[0])
    wd.append(img.shape[1])

print('Max Height of image in y_train', max(ht))
print('Min Height of image in y_train', min(ht))
print('Max Width of image in y_train', max(wd))
print('Min Width of image in y_train', max(wd))


new_shape = (258, 540, 1)


def load_images(path):
    image_list = []
    for i in path:
        img = cv2.imread(i, 0)  # read grayscale image
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        img = img / 255.
        img = np.expand_dims(img, axis=-1)  # we get channel as 1 in output.
        image_list.append(img)
    return image_list


new_x_train = load_images(x_train)
new_y_train = load_images(y_train)
new_test = load_images(test)

new_x_train[0].shape

new_x_train = np.array(new_x_train)
new_y_train = np.array(new_y_train)
new_test = np.array(new_test)

print('Shape of Single image:', new_x_train[0].shape)
print('Shape of All images:', new_x_train.shape)

x_tr, x_val, y_tr, y_val = train_test_split(
    new_x_train, new_y_train, test_size=0.3, random_state=42)
print('Train data:', x_tr.shape)
print('Validation data:', x_val.shape)

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                 padding='same', name='Conv1', input_shape=new_shape))
model.add(BatchNormalization(name='BN1'))
model.add(MaxPool2D((2, 2), padding='same', name='pool1'))

# Decoder
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same', name='Conv2'))
model.add(UpSampling2D((2, 2), name='upsample1'))
model.add(Conv2D(filters=1, kernel_size=(3, 3),
                 activation='sigmoid', padding='same', name='Conv3'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['RootMeanSquaredError'])
history = model.fit(x=x_tr, y=y_tr, batch_size=10, epochs=1,
                    validation_data=(x_val, y_val), )

predictions = model.predict(x=new_test)
# predictions.shape
f, ax = plt.subplots(2, 2, figsize=(16, 10))

# # we need to reshape the image removing the channel as we are using 'plt.imshow'.
ax[0, 0].imshow(new_test[0].reshape(258, 540), cmap='gray', )
ax[0, 1].imshow(predictions[0].reshape(258, 540), cmap='gray')
ax[1, 0].imshow(new_test[1].reshape(258, 540), cmap='gray')
ax[1, 1].imshow(predictions[1].reshape(258, 540), cmap='gray')
plt.show()
