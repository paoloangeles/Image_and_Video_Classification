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
    keras.layers.Dense(128, activation = 'relu'), ## first dense array outputs 128 nodes
    keras.layers.Dense(128, activation = 'relu'), ## added another later of 128 nodes
    keras.layers.Dense(10) ## second dense array outputs to 10 nodes for the 10 classes of clothing
    ])

## Set up how the model learns
model.compile(optimizer = 'adam', ## improved extension of the gradient descent algorithm
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), ## the loss parameter used for multi-label classification, we need to reduce this variable. It changes
                                                                                          ## with the predicted probability of the observation labels - i.e. higher when predicted probability is low for correct label
                                                                                          ## cross entropy is the measure of difference between two probability distributions for a given random variable or set of events
              metrics = [keras.metrics.SparseCategoricalAccuracy()] ## final variable to measure performance of model (i.e. good/bad). Higher accuracies would mean lower losses and hence less needed optimisation.
              )

## Train and fit model to training data and output accuracies every epoch
model.fit(train_images, train_labels, batch_size = 40, epochs = 10)  


## Test model on remaning 10,000 images and calculate accuracy/loss
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
print('\nTest accuracy: ', test_acc)
print('\nTest loss: ', test_loss)

## Attach a softmax layer to convert logits to probability (easier to interpret), essentially normalises (sum of all to 1) results according to labels
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) 
## Get a probability predictions array for each image - highest probability = image classifier guess
predictions = probability_model.predict(test_images)


## function to show image and prediction
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 100
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()