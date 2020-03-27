import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import random



fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(test_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=2)

test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

ran_test_img = random.sample(list(range(len(test_images))), 10)
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[ran_test_img[i]], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[ran_test_img[i]])
    true_label = test_labels[ran_test_img[i]]
    plt.xlabel("Test : {}  Label : {}".format(class_names[predicted_label], 
                                  class_names[true_label]))

plt.show()
