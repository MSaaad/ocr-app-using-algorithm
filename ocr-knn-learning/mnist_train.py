# neural networks greate at feeding, just dont overfeed

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  # 28x28 images of hadnwritten digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

# to change to 2D images in dataset to a flattenend numbered image
model.add(tf.keras.layers.Flatten())


# 128 units or neurons => relu is default activation neural function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# softmax for probability function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# complex part of model making
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# loss and accuracy calculation
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# saving model
model.save('mnist.model')

new_model = tf.keras.models.load_model('mnist.model')

predictions = new_model.predict([x_test])

# probability predictions
# print(predictions)

number = 3

# change this asset
print(np.argmax(predictions[number]))

# plot showing
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
# print(x_train[0])

# for checking the prediction
plt.imshow(x_test[number], cmap=plt.cm.binary)
plt.show()
