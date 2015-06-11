# Trains a neural network to find human poses
# This is a simple version that has llimited functionality

import os
import sys
import cPickle as pickle
import numpy as np

# Set the recursion limit high, for pickling the entire network
sys.setrecursionlimit(10000)

#from guppy import hpy

# Start the memory monitor
#hp = hpy()

# Need to import the post_processing module from data_generation
#sys.path.insert(0, os.path.join('..', 'data_generation'))
#import post_process as pp

# Keras is the framework for theano based neural nets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

# Number of classes
# TODO: Make this automatic
nb_classes = 13

"""
# Sets up the network
# Returns the constructed network
#
# load_from is the name of the pickle to load the network from
# Leaving load_from as None will create a new network using the sent configuration
#
# These parameters are only needed if setting up a new network
# conv_inter is the number of feature maps to use in the the convolution layers
#
# dense_nodes is the number of nodes to use in the two layer classifier at the end
# Setting this too high may cause memory errors
#
# image_height is the height of the images
#
# image_width is the width of the images
def get_network(load_from=None, conv_inter=32, dense_nodes=512, image_height=48, image_width=64):

	# Check for existing network to load
	if load_from:
	# There is a network to load

		# Load the network
		model = pickle.load(open(load_from,'rb'))

		# Return the network
		return model

	# Need to construct the model
	else:

		# The network
		model = Sequential()

		# Two convolutions followed by max pool
		# Dropout of .25 at end
		# Feature maps is set by the variable conv_inter
		# Input shape: (1, height, width)
		# Output shape: (conv_inter, height / 2, width / 2)
		model.add(Convolution2D(conv_inter, 1, 3, 3, border_mode='full')) 
		model.add(Activation('relu'))
		model.add(Convolution2D(conv_inter, conv_inter, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		model.add(Dropout(0.25))

		# Another two convolutions followed by max pool
		# Dropout of .25 at end
		# Input shape: (conv_inter, height / 2, width / 2)
		# Output shape: (conv_inter * 2, height / 4, width / 4)
		model.add(Convolution2D(conv_inter*2, conv_inter, 3, 3, border_mode='full')) 
		model.add(Activation('relu'))
		model.add(Convolution2D(conv_inter*2, conv_inter*2, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		model.add(Dropout(0.25))

		# Flattens 2D data into 1D
		# Input shape: (2 * conv_inter, height / 4, width / 4)
		# Output shape: ((2 * conv_inter * height * width)/16)
		model.add(Flatten((2 * conv_inter * image_height * image_width) / 16))

		# Two layer dense network
		# Number of dense nodes is set by dense_nodes
		# Dropout of .5 in the middle
		# Input shape: ((2 * conv_inter * height * width)/16)
		# Output shape: (height * width)
		model.add(Dense((2 * conv_inter * image_height * image_width)/16, dense_nodes, init='normal'))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(dense_nodes, image_height * image_width, init='normal'))

		# let's train the model using SGD + momentum (how original).
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mse', optimizer=sgd)

		# Return the model
		return model
"""

# Sets up the network
# Returns the constructed network
#
# load_from is the name of the pickle to load the network from
# Leaving load_from as None will create a new network using the sent configuration
#
# These parameters are only needed if setting up a new network
# conv_inter is the number of feature maps to use in the the convolution layers
#
# dense_nodes is the number of nodes to use in the two layer classifier at the end
# Setting this too high may cause memory errors
#
# image_height is the height of the images
#
# image_width is the width of the images
def get_network(load_from=None, conv_inter=32, dense_nodes=512, image_height=48, image_width=64):

	# Check for existing network to load
	if load_from:
	# There is a network to load

		# Load the network
		model = pickle.load(open(load_from,'rb'))

		# Return the network
		return model

	# Need to construct the model
	else:

		# The network
		model = Sequential()

		# Two convolutions followed by max pool
		# Dropout of .25 at end
		# Feature maps is set by the variable conv_inter
		# Input shape: (1, height, width)
		# Output shape: (conv_inter, height / 2, width / 2)
		model.add(Convolution2D(conv_inter, 1, 3, 3, border_mode='full')) 
		model.add(Activation('relu'))
		model.add(Convolution2D(conv_inter, conv_inter, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		model.add(Dropout(0.25))

		# Another two convolutions followed by max pool
		# Dropout of .25 at end
		# Input shape: (conv_inter, height / 2, width / 2)
		# Output shape: (conv_inter * 2, height / 4, width / 4)
		model.add(Convolution2D(conv_inter*2, conv_inter, 3, 3, border_mode='full')) 
		model.add(Activation('relu'))
		model.add(Convolution2D(conv_inter*2, conv_inter*2, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		model.add(Dropout(0.25))

		# Another two convolutions followed by max pool
		# Dropout of .25 at end
		# Input shape: (conv_inter * 2, height / 4, width / 4)
		# Output shape: (conv_inter * 4, height / 8, width / 8)
		model.add(Convolution2D(conv_inter*4, conv_inter*2, 3, 3, border_mode='full')) 
		model.add(Activation('relu'))
		model.add(Convolution2D(conv_inter*4, conv_inter*4, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		model.add(Dropout(0.25))

		# Flattens 2D data into 1D
		# Input shape: (4 * conv_inter, height / 8, width / 8)
		# Output shape: ((4 * conv_inter * height * width)/64)
		model.add(Flatten((4 * conv_inter * image_height * image_width) / 64))

		# Two layer dense network
		# Number of dense nodes is set by dense_nodes
		# Dropout of .5 in the middle
		# Input shape: ((4 * conv_inter * height * width)/64)
		# Output shape: (height * width * nb_classes)
		model.add(Dense((4 * conv_inter * image_height * image_width)/64, dense_nodes, init='normal'))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(dense_nodes, image_height * image_width * nb_classes, init='normal'))
		model.add(Activation('hard_sigmoid'))

		# let's train the model using SGD + momentum (how original).
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mse', optimizer=sgd)

		# Return the model
		return model

# Takes data of the form: (n_images, height * width), ints 0 - nb_classes
# Outputs data: (n_images, height * width * nb_classes), each vector has bit i set to 1 iff pixel is class i
def make_images_categorical(data_batch):

	global nb_classes

	# Get the number of classes
	nb_classes = np.max(data_batch) + 1

	# Initialize the new data batch
	cate_data = np.zeros((data_batch.shape[0], data_batch.shape[1] * nb_classes))

	# For each pixel, set bit i to 1 iff pixel is of class i
	# Loop through each image
	for image_index in range(data_batch.shape[0]):

		# Loop through each pixel
		for pix_index in range(data_batch.shape[1]):

			# Set only the correct pixel
			cate_data[image_index][pix_index * nb_classes + data_batch[image_index][pix_index]] = 1

	return cate_data

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

	# Sub directories
	data_dir = os.path.join(source_dir, "data")
	label_dir = os.path.join(source_dir, "label")

	# Get the names of all of the data items
	all_names = [ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f)) ]

	# Iterate through all names
	for name in all_names:

		# Load the data batch
		data_item = pickle.load(open(os.path.join(data_dir, name), 'rb'))

		# Load the label batch
		label_item = pickle.load(open(os.path.join(label_dir, name), 'rb'))

		# Get the shape of the input
		(n_images, height, width) = data_item.shape

		# Normalize the depth data
		input_max = np.max(data_item)
		input_min = np.min(data_item)
		input_range = input_max - input_min
		data_item = (data_item - input_min) / input_range

		# Add the stack dimension, needed for correct processing in the convolutional layers
		data_item = np.expand_dims(data_item, axis=0)

		# Reorder axis to to (n_images, stack, height, width)
		data_item = np.rollaxis(data_item, 0, 2)

		# Reshape to (n_images, height * width)
		label_item = label_item.reshape(n_images, height * width)

		# Make each pixel categorical
		label_item = make_images_categorical(label_item)

		# Make into GPU friendly float32
		data_item = data_item.astype("float32")

		# Generate the next batch
		yield data_item, label_item

# Trains the network using all of the data in source_dir
#
# model is the network to train
#
# source_dir is the location of the data to train on
#
# If save_to is set, the network will be saved after each data batch is trained on
#
# n_epochs sets the number of epochs to train for each batch. This is kept low to prevent over fitting on a single batch
#
# n_batch is the number of images to process at once. Set based on the available GPU memory
def train_network(model, source_dir, save_to=None, n_epochs=10, n_batch=32):

	# Loop for the specified number of epochs
	# This outer loop is needed because the images cannot all fit into main memory at once
	for current_epoch in range(n_epochs):

		print "\nRunning epoch number: ", current_epoch
		print "\n"

		# Process all of the data in source_dir
		for train_data, train_label in get_data(source_dir):

			# Train the model
			model.fit(train_data, train_label, batch_size=n_batch, nb_epoch=1)

			# Save the model, if save_to is set
			if save_to:
				pickle.dump(model, open(save_to ,'wb'))

# Tests the model using all of the data in source_dir
#
# model is the model to test
#
# source_dir is the location of the data to train on
#
# n_batch is the number of images to process at once. Set based on the available GPU memory
def test_network(model, source_dir, n_batch=32):

	for test_data, test_label in get_data(source_dir):

		score = model.evaluate(test_data, test_label, batch_size=32)

		print('Test score:', score)

# Loads data, trains network and tests
#
# train_source_dir : The folder to load training data from
#
# test_source_dir : The folder to load testing data from
#
# load_name : The name of a network to load, new network will be created if this is None
#
# save_name : The name to save the network as, not saved if None
def run_network(train_source_dir, test_source_dir, load_name=None, save_name=None):

	# Get the size of the images

	# Open the first image to get the shape
	check_dir = os.path.join(train_source_dir, "data")
	check_name = [ f for f in os.listdir(check_dir) if os.path.isfile(os.path.join(check_dir,f)) ][0]
	check_item = pickle.load(open(os.path.join(check_dir, check_name),'rb'))

	# Set the shape
	(ignore, im_h, im_w) = check_item.shape

	# Get the network
	model = get_network(load_from=load_name, image_height=im_h, image_width=im_w)

	# Train the network
	train_network(model, train_source_dir, save_to=save_name)

	# Test the network
	test_network(model, test_source_dir)

# If this is the main, train the network
# Need the source_dir for the training data and another for the test data
# Optionally, name to load / save the model
if __name__ == "__main__":

	# If no arguments are sent, show the usage
	if (len(sys.argv) < 2):
		# Not enough arguments, print usage
		print "Usage: train_pickle.py train_source_dir test_source_dir [-l load_name -s save_name]"
		print ""
		print "train_source_dir : the location of the training data"
		print "test_source_dir : the location of the testing data"
		print "load_name : the name to load the model from"
		print "save_name : the name to save the model to"
		print ""
		sys.exit(1)

	# Get the source_dir
	train_source_dir = sys.argv[1]
	test_source_dir = sys.argv[2]

	# Get optional argument, if present
	load_name = None
	save_name = None
	if(len(sys.argv) > 4):
		if sys.argv[3] == "-l":
			load_name = sys.argv[4]
		elif sys.argv[3] == "-s":
			save_name = sys.argv[4]

	if(len(sys.argv) > 6):
		if sys.argv[5] == "-l":
			load_name = sys.argv[6]
		elif sys.argv[5] == "-s":
			save_name = sys.argv[6]

	# Run and test the network
	run_network(train_source_dir, test_source_dir, load_name, save_name)

	"""
	# Get the network
	model = get_network(load_from=load_name)

	# Train the network
	train_network(model, train_source_dir, save_to=save_name)

	# Test the network
	test_network(model, test_source_dir)
	"""
