# Runs a neural net to do per pixel classification

import numpy as np

import cPickle as pickle

import sys
import os

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# TODO: make this automatic
nb_classes = 13

class Neural_net:

	# Class configuration variables:
	#
	# classifier: the neural net

	# Creates a new model from the sent arguments
	#
	# input_vector_size is the size of the feature vector
	#
	# ouput_vector_size is the size of the label vector (the max class label)
	def __init__(self, input_vector_size, output_vector_size, hidden_nodes=500):

		# Save the batch_size
		self.batch_size = batch_size

		# Create the model container
		self.classifier = Sequential()

		# This is the model configuration

		# First layer takes all of the input features and narrows them down to the number of hidden nodes
		self.classifier.add(Dense(input_vector_size, hidden_nodes))
		self.classifier.add(Activation('relu'))

		# This is the hidden layer
		self.classifier.add(Dense(hidden_nodes, hidden_nodes))
		self.classifier.add(Activation('relu'))

		# This takes the hidden nodes and outputs into the label vector size
		self.classifier.add(Dense(hidden_nodes, output_vector_size))
		self.classifier.add(Activation('softmax'))

		# Using SGD to optimize
		sgd = SGD()

		# Compile the model
		# Using categorical cross entropy as the objective
		self.classifier.compile(loss='categorical_crossentropy', optimizer=sgd)

	# Loads the model configuration from a file
	#
	# file_name is the configuration of the network
	#
	# weight_name is the name of the trained weights, if None the network will be untrained
	#@classmethod
	#def from_file(cls, file_name, weight_name = None):



	# Flattens data so that it is of shape (x, features)
	# Takes the array to be rectified, of shape (x, y, z)
	# Can have any number of x, y dimensions, last dimension is considered to be the features
	# Make sure to save the old array shape, if it is needed
	#
	# Returns the new array
	def rectify_array(self, original_array):

		# Get the original shape
		original_shape = original_array.shape

		# Get the product of all dimensions, except the last
		product = 1
		for index in range(len(original_shape) - 1):

			product *= original_shape[index]

		# Reshape and return
		return original_array.reshape((product, original_shape[-1]))

	# Trains on one set of data
	#
	# train_data is the feature data to train on
	#
	# train_label is the classifications of each example
	#
	# batch_size is the amount to train at once
	#
	# nb_epochs is the number of epochs to run on each batch
	def train_batch(self, train_data, train_labels, batch_size=128, nb_epochs=10, verbose = False):

		# Make the input data into shape (n_examples, features)
		train_data = self.recify_array(train_data)

		# Make the labels into categorical vector
		train_labels = np_utils.to_categorical(train_labels.flatten(), nb_classes)

		# Set the verbose arguments based on sent flag
		if verbose:

			show_accuracy_arg = True

			verbose_arg = 1

		else:

			show_arruracy_arg = False

			verbose_arg = 0

		# Train the network
		self.classifier.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=show_accuracy_arg, verbose=verbose_arg)

		# Verbose finish line
		if verbose:

			print ""

	# Trains on a bunch of pickles
	#
	# source_dir is the directory that contains data and label sub directories
	#
	# save_name is the name to save the model as, no saving if set to None
	# saves temporaries on a regular basis
	def train(self, source_dir, save_name=None, verbose=False):

		# Extra line for verbose printing
		if verbose:

			print ""

		# Set the data and label directories
		data_dir = os.path.join(source_dir, "data")
		label_dir = os.path.join(source_dir, "label")

		# Get the names of the data
		pickle_names = pp.get_names(data_dir)

		# Go through each item
		for item_index, item_name in enumerate(pickle_names):

			if verbose:

				print "\rTraining progress: " + str(item_index + 1) + " / " + str(len(pickle_names)),
				sys.stdout.flush()

			# Get the data
			data_batch = pickle.load(open(os.path.join(data_dir, item_name), 'rb'))

			# Get the labels
			label_batch = pickle.load(open(os.path.join(label_dir, item_name), 'rb'))

			# Train the forest
			self.train_batch(data_batch, label_batch)

			# Save after every 5th batch, if save_name is set
			self.save_model(save_name.split('.')[0] + "_temp." + save_name.split('.')[1])

		# Save the model once done
		self.save_model(save_name)

	def save_model(self):

		pass

	def predict(self):

		pass

if __name__ == "__main__":

	pass
