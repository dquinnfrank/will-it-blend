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
	all_names = [ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f)) ]

	# Iterate through all names
	for name in all_names:

		# Load the data batch
		original_item = pickle.load(open(os.path.join(data_dir, name), 'rb'))

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
		#noise_item = original_item + noise_amount*original_item.std()*np.random.random(original_item.shape)

		# Generate the next batch
		yield original_item, original_item.reshape(original_item.shape[0], original_item.shape[1] * original_item.shape[2] * original_item.shape[3])

# The network configuration

# The number of convolution feature maps to create
conv_features = 6

# The number of hidden units to use in the lower dimaensional projection
# Based of off the existing height and width of the images
# TODO: Make this automatic
height = 480 * .25
width = 640 * .25
hidden_features = (height * width) * .5

# Create the model
reconstruction_model = Sequential()

# Do the convolutional step
# Input shape (n_images, 1, height, width)
# Output shape (n_images, conv_features, height, width)
reconstruction_model.add(Convolution2D(conv_features, 1, 3, 3, border_mode='full'))

# Do the max pooling, to reduce the data to (n_images, conv_features, height / 2, width / 2)
reconstruction_model.add(MaxPooling2D(poolsize=(2,2)))

# Flatten the images, to prepare for the dense layer
# Output shape (n_images, conv_features * (height / 2) * (width / 2))
#reconstruction_model.add(Flatten())

# This is the lower dimensional projection
# Output shape (n_images, hidden_features)
#reconstruction_model.add(Dense(conv_features * (height / 2) * (width / 2)), hidden_features)

# Reshape the output to (n_images, )

# Do the unpooling
# Output shape (n_images, conv_features, height, width)
reconstruction_model.add(UnPooling2D(stretch_size=(2,2)))

# Do the opposite of the first convolution step
# Output shape (n_images, 1, height, width)
reconstruction_model.add(Convolution2D(1, conv_features, 3, 3, border_mode='valid'))

# Flatten the whole thing
reconstruction_model.add(Flatten())

# The optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Put the model together
reconstruction_model.compile(loss='mse', optimizer=sgd)

# Train the model on MNIST
# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Make GPU friendly
X_train = X_train.astype("float32")

# Normalize
X_train /= 255

# Add the stack dimension, needed for correct processing in the convolutional layers
X_train = np.expand_dims(X_train, axis=0)

# Reorder axis to (n_images, stack, height, width)
X_train = np.rollaxis(X_train, 0, 2)

# Make the input and output
#X_input = X_train + .2*X_train.std()*np.random.random(X_train.shape)
#X_output = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])

# Show shapes for debugging
#print X_input.shape
#print X_output.shape

# Get a new noisy image for each training set
for epoch in range(100):

	# Make the input and output
	X_input = X_train + .2*X_train.std()*np.random.random(X_train.shape)
	X_output = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])

	# Train the model
	reconstruction_model.fit(X_input, X_output, batch_size=128, nb_epoch=10)

# Save the model
#pickle.dump(reconstruction_model, open("../trained_models/mnist_recon_1.p" ,'wb'))
reconstruction_model.save_weights("../trained_models/mnist_recon_2.ke")

"""
# Loop for the specified number of epochs
# This outer loop is needed because the images cannot all fit into main memory at once
for current_epoch in range(10):

	print "\nRunning epoch number: ", current_epoch
	print "\n"

	# Process all of the data in source_dir
	for train_data, train_label in get_data("../generated_data/set_002_25_tr"):

		print "data shape", train_data.shape
		print "label shape", train_label.shape

		# Train the model
		reconstruction_model.fit(train_data, train_label, batch_size=128, nb_epoch=1)
		#print "out shape", reconstruction_model.predict(train_data, batch_size=128).shape
"""
