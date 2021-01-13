
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random
import pickle
# RGB is 3x times the size of grayscale data

DATADIR = 'C:/users/saad9/Desktop/ocr-app-using-algorithm/PetImages'  # dataset location

CATEGORIES = ['Dog', 'Cat']

# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)

#     for img in os.listdir(path):
#         img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)

#         plt.imshow(img_array, cmap='gray')
#         plt.show()
#         break
#     break

training_data = []

IMG_SIZE = 50


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(
                    path, img), cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()


print(len(training_data))

random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])

x_train = []
y_test = []

for features, label in training_data:
    x_train.append(features)
    y_test.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# dumping data out
pickle_out = open('x_test.pickle', 'wb')
pickle.dunp(y_test, pickle_out)
pickle_out.close()

pickle_in = open('x_train', 'rb')

x_train = pickle.load(pickle_in)
x_train[1]

# RGB is 3x times the size of grayscale data

DATADIR = 'C:/users/saad9/Desktop/ocr-app-using-algorithm/PetImages'  # dataset location

CATEGORIES = ['Dog', 'Cat']

# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)

#     for img in os.listdir(path):
#         img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)

#         plt.imshow(img_array, cmap='gray')
#         plt.show()
#         break
#     break

training_data = []

IMG_SIZE = 50


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(
                    path, img), cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()


print(len(training_data))

random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])

x_train = []
y_test = []

for features, label in training_data:
    x_train.append(features)
    y_test.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# dumping data out
pickle_out = open('x_test.pickle', 'wb')
pickle.dunp(y_test, pickle_out)
pickle_out.close()

pickle_in = open('x_train', 'rb')

x_train = pickle.load(pickle_in)
x_train[1]
