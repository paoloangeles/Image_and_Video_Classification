# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:47:21 2020

@author: Paolo
"""

## import machine learning and neural network libraries
import tensorflow as tf
from tensorflow import keras

## import helper libraries
import numpy as np
import matplotlib.pyplot as plt


## import fashion MNIST dataset

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() ## load data into appropriate variables

## Define class label/names in a list
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Plot example image of data
plt.figure()
plt.imshow(train_images[1]) ## Plots data as an image where input can be RGB data
plt.colorbar() ## Adds colorbar to plot
plt.grid(False) ## Removes gridlines
plt.show()

## Normalise pixel values between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

## Set up layers of model

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), ## flattens 2d array/image into 1d array of 784 pixels
    keras.layers.Dense(128, activation_layer = 'relu'), ## first dense array outputs 128 nodes
    keras.layers.Dense(10) ## second dense array outputs to 10 nodes for the 10 classes of clothing
    ])

## Set up how the model learns

model.compile(optimizer = 'adam', ## improved extension of the gradient descent algorithm
              loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits = True), ## the loss parameter used for multi-label classification, we need to reduce this variable. It changes
                                                                                          ## with the predicted probability of the observation labels - i.e. higher when predicted probability is low for correct label
                                                                                          ## cross entropy is the measure of difference between two probability distributions for a given random variable or set of events
              metrics = ['accuracy'] ## final variable to measure performance of model (i.e. good/bad). Higher accuracies would mean lower losses and hence less needed optimisation.
              )