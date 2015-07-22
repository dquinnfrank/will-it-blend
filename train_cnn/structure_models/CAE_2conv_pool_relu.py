import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Activation
from keras.datasets import mnist
from keras.layers.additional import UnPooling2D

import os
import sys

def get_model(load_name = None, conv_features = 6):

	# This marks index of the layer that has the reduced data, needed for building a new model with the encoder as the first part
	encoded_index = 4

	# The network configuration

	# Create the model
	reconstruction_model = Sequential()

	# Do the convolutional step
	# Input shape (n_images, 1, height, width)
	# Output shape (n_images, conv_features, height, width)
	reconstruction_model.add(Convolution2D(conv_features, 1, 3, 3, border_mode='full'))

	# Non-linear activation
	reconstruction_model.add(Activation("relu"))

	# Do another convolutional step
	# Input shape (n_images, conv_features, height, width)
	# Output shape (n_images, 2 * conv_features, height, width)
	reconstruction_model.add(Convolution2D(2 * conv_features, conv_features, 3, 3, border_mode='full'))

	# Non-linear activation
	reconstruction_model.add(Activation("relu"))

	# Do the max pooling, to reduce the data to (n_images, conv_features, height / 2, width / 2)
	reconstruction_model.add(MaxPooling2D(poolsize=(2,2)))

	# Do a convolutional step to reduce the dimensionality
	# Input shape (n_images, 2 * conv_features, height / 2, width / 2)
	# Output shape (n_images, 1, height / 2, width / 2)
	reconstruction_model.add(Convolution2D(1, 2 *conv_features, 3, 3, border_mode='valid'))

	# Auto encoder ends here

	# Do the unpooling
	# Output shape (n_images, 1, height, width)
	reconstruction_model.add(UnPooling2D(stretch_size=(2,2)))

	# Do a convolutional step, to fix the border
	# Output shape (n_images, 1, height, width)
	reconstruction_model.add(Convolution2D(1, 1, 3, 3, border_mode='valid'))

	# Do a final convolutional step
	# Output shape (n_images, 1, height, width)
	reconstruction_model.add(Convolution2D(1, 1, 3, 3, border_mode='full'))

	# Flatten the whole thing
	reconstruction_model.add(Flatten())

	# The optimizer
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	# Put the model together
	reconstruction_model.compile(loss='mse', optimizer=sgd)

	# Load weights, if optional parameter is set
	if load_name :
		reconstruction_model.load_weights(load_name)

	return reconstruction_model, encoded_index
