# Trains a model to produce a mask image of body parts

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
from random import shuffle

import importlib

# Takes a batch of images where each pixel corresponds to a class and returns a plane representation
# Each image will have a number of planes equal to the number of classes, there will be a 1 in plane i, index (x, y), where index (x, y) was of class i in the original image
def make_planes(class_batch, total_classes):

	# Make a new array of shape (batch_size, x_max, y_max, total_classes)
	plane_batch = np.zeros((class_batch.shape[0], class_batch.shape[1], class_batch.shape[2], total_classes), dtype=class_batch.dtype)

	print plane_batch.shape

	# Go through each image
	for batch_index in range(class_batch.shape[0]):

		# Go through each pixel
		for x_index in range(class_batch.shape[1]):
			for y_index in range(class_batch.shape[2]):

				# Get the class of the pixel
				pixel_class = class_batch[batch_index][x_index][y_index]

				# Set the pixel (x, y) in plane i, where i is the class of the pixel
				plane_batch[batch_index][x_index][y_index][pixel_class] = 1

	# Return the plane_batch
	return plane_batch

# Loads the data from pickles and returns them in a shuffled order
#
# Depth data will be normalized
# Data will be reshaped to conform to keras requirements
#
# This is a generator function
#
# source_dir is the directory containing the images, it should have sub directories: data, labels
# data will be of shape (n_images, stack, height, width), Ex: (5000, 1, 48, 64)
# labels will be of shape (n_images, height * width)
# data will be float32, for GPU
# label will be float32
#
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
		label_item = make_planes(label_item)

		# Make into GPU friendly float32
		data_item = data_item.astype("float32")

		# Generate the next batch
		yield data_item, label_item

class Mask:

	# Loads the model structure
	#
	# encoder_layer_structure is the name of the structure model to use for the first layers of the network
	#
	# Optionally loads pretrained weights for the encoder
	#
	# Optionally loads a trained version of the same model, this will overwrite pretrained_layer if it is sent
	# trained_model is the name of the weights to load
	def __init__(self, structure_name, encoder_layer_structure = "CAE_2conv_pool_relu", encoder_layer_weight_name = None, trained_model=None):

		# Get the model
		self.model = (importlib.import_module("structure_models." + structure_name)).get_model(encoder_layer_structure = encoder_layer_structure, pretrained_layer_name = encoder_layer_weight_name, load_name = trained_model)


	# Trains the model
	#
	# train_data_dir is the name of the directory that holds both the data and the labels
	#
	# save_name is the name, including path, to save the model as once training is complete
	# Temporary files will regurally be saved as "[save_name]_temp.ke"
	#
	# epochs is the number of times to process the data
	# The display will always show that training is on epoch 0, because each data batch is processed separately
	#
	# batch_size specifies the number of images to process at once
	def train_model(self, train_data_dir, save_name=None, epochs=25, batch_size=32):

		# Get a new noisy image for each training set
		for epoch in range(epochs):

			print "Running epoch: ", epoch

			# Get each training set
			item_count = 0
			for X_train, X_target in get_data(train_data_dir):

				# Train the model
				self.reconstruction_model.fit(X_train, X_target, batch_size=batch_size, nb_epoch=1)

				# Save every 5th item
				if item_count % 5 == 0 and save_name:

					print "Saving temporary"

					# Save the entire network
					self.reconstruction_model.save_weights(save_name[:-3] + "_temp.ke", overwrite=True)

				item_count += 1

			# Save the model after each training set, if save_name is set
			if save_name:

				print "Saving after epoch: ", epoch

				# Save the entire network
				self.reconstruction_model.save_weights(save_name, overwrite=True)

# If this has been run from the commandline, train the network using the given options
if __name__ == "__main__":

	# If there are no arguments, show usage
	if len(sys.argv) < 3:

		print "Usage: train_mask.py structure_name train_data_dir [-p pretained_layer_name -s save_name -e epochs -b batch_size]"

		sys.exit(1)

	# Get the structure name
	structure_name = sys.argv[1]

	# Get the training data directory
	train_data_dir = sys.argv[2]

	# Get the optional arguments
	pretrained_layer_name = None
	save_name = None
	epochs = 25
	batch_size = 32
	arg_index = 3
	while arg_index < len(sys.argv):

		# Flag for save name
		if sys.argv[arg_index] == "-s":

			# Set name
			save_name = sys.argv[arg_index + 1]

			# Skip to the next flag
			arg_index += 2

		# Flag for max epochs
		elif sys.argv[arg_index] == "-e":

			# Set epochs
			epochs = int(sys.argv[arg_index + 1])

			# Skip to the next flag
			arg_index += 2

		# Flag for batch size
		elif sys.argv[arg_index] == "-b":

			# Set the batch size
			batch_size = int(sys.argv[arg_index + 1])

			# Skip to the next flag
			arg_index += 2

		# Flag for pretrained layer name
		elif sys.argv[arg_index] == "-p":

			# Set the pretrained layer name
			pretrained_layer_name = sys.argv[arg_index + 1]

			# Skip to the next flag
			arg_index += 2

		# Flag not known
		else:
			print "Flag not known: ", sys.argv[arg_index]

			# Skip to the next flag
			arg_index += 2

	# Show the network configuration
	print "\nConfiguration"
	print "Model structure: ", structure_name
	print "Training data: ", train_data_dir
	print "Pretrained name: ", pretrained_layer_name
	print "Save name: ", save_name
	print "Epochs: ", epochs
	print "Batch size: ", batch_size
	print ""

	# Create the network
	mask_net_manage = Mask(structure_name, encoder_layer_name = pretrained_layer_name)

	# Train the network
	# Also saves, if save_name is set
	mask_net_manage.train_model(train_data_dir, save_name=save_name, epochs=epochs, batch_size=batch_size)
