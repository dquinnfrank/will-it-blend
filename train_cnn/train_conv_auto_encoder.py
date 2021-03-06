import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.datasets import mnist
from keras.layers.additional import UnPooling2D

import os
import sys
import cPickle as pickle

import importlib

# Set the recursion limit high, for pickling the entire network
sys.setrecursionlimit(10000)

# Loads the data from pickles and returns each item in order
# Depth data will be normalized
# Data will be reshaped to conform to keras requirements
# This is a generator function
# source_dir is the directory containing the images, it should have sub directories: data, labels
# data will be of shape (n_images, stack, height, width), Ex: (5000, 1, 48, 64)
# labels will be of shape (n_images, height * width)
# data will be float32, for GPU
# label will be categorical
# TODO: Make this a more general function, the data loading needs to be done in multiple places
def get_data(source_dir):

	# Amount of noise to add to the images
	noise_amount = .2

	# Sub directories
	data_dir = os.path.join(source_dir, "data")
	#label_dir = os.path.join(source_dir, "label")

	# Get the names of all of the data items
	all_names = sorted([ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f)) ])

	# Iterate through all names
	for name in all_names:

		# Corrupt data will cause exceptions
		try:

			# Load the data batch
			print "\nLoading item: ", os.path.join(data_dir, name)
			original_item = pickle.load(open(os.path.join(data_dir, name), 'rb'))

		# The data is corrupt
		except EOFError:

			print "Item corrupt"

			# Ignore this item and load the next one
			 #continue

		# Item is valid
		else:

			# Load the label batch
			#label_item = pickle.load(open(os.path.join(label_dir, name), 'rb'))

			# Get the shape of the input
			(n_images, height, width) = original_item.shape

			# Normalize the depth data
			input_max = np.max(original_item)
			input_min = np.min(original_item)
			input_range = input_max - input_min
			original_item = (original_item - input_min) / input_range

			# Add the stack dimension, needed for correct processing in the convolutional layers
			original_item = np.expand_dims(original_item, axis=0)

			# Reorder axis to (n_images, stack, height, width)
			original_item = np.rollaxis(original_item, 0, 2)

			# Make into GPU friendly float32
			original_item = original_item.astype("float32")

			# The input image is a noisy version of the true image
			noise_item = original_item + noise_amount*original_item.std()*np.random.random(original_item.shape)

			# Generate the next batch
			yield original_item, original_item.reshape(original_item.shape[0], original_item.shape[1] * original_item.shape[2] * original_item.shape[3])

reconstruction_model = (importlib.import_module("structure_models.CAE_2conv_pool_relu")).get_model()

# Train the model on MNIST
# the data, shuffled and split between tran and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Make GPU friendly
#X_train = X_train.astype("float32")

# Normalize
#X_train /= 255

# Add the stack dimension, needed for correct processing in the convolutional layers
#X_train = np.expand_dims(X_train, axis=0)

# Reorder axis to (n_images, stack, height, width)
#X_train = np.rollaxis(X_train, 0, 2)

# Make the input and output
#X_input = X_train + .2*X_train.std()*np.random.random(X_train.shape)
#X_output = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])

# Show shapes for debugging
#print X_input.shape
#print X_output.shape

# Get a new noisy image for each training set
for epoch in range(25):

	print "Running epoch: ", epoch

	# Get each training set
	for X_train, X_target in get_data("../generated_data/set_002_25_tr"):

		# Make the input and output
		#X_input = X_train + .2*X_train.std()*np.random.random(X_train.shape)
		#X_output = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])

		# Train the model
		reconstruction_model.fit(X_train, X_target, batch_size=32, nb_epoch=1)

	# Save the model after each training set
	reconstruction_model.save_weights("../trained_models/person_multi_conv_relu.ke", overwrite=True)
