import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_datasets as tfds
mnist = tfds.load(name='mnist')
(train_images, train_labels), (test_images, test_labels) = mnist

