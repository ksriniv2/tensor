# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:39:16 2020

@author: KARTH
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['zero','One', 'Two', 'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine', 'Ten']


train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

plt.figure()
plt.imshow(train_images[4])
plt.colorbar()
plt.grid(False)
plt.show()


train_images = train_images / 255.0

test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE
    callbacks=myCallback()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE
    
    import matplotlib.pyplot as plt
    plt.imshow(x_train[0])
    print(x_train[0])
    print(y_train[0])
    
    x_train=x_train/255.0
    x_test=x_test/255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(x_train,y_train,epochs=5,callbacks=[callbacks]
    )
    
    model.evaluate(x_test,y_test)
    # model fitting
    
    
    return history.epoch, history.history['acc'][-1]

class myCallback(tf.keras.callbacks.Callback):
    def onEpochEnd(self,epoch, logs={}):
        if(logs.get('loss' <0.1)):
            print("Reached 99% accuracy so cancelling training!")
            self.model.stop_training=True