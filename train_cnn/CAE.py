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

# This class manages a convolutional auto-encoder
# Main use is training and saving a CAE for use in a different network
class CAE:

	# Loads the model structure
	# The model must be saved in the folder structure_models
	def __init__(self, structure_name):

		self.reconstruction_model = (importlib.import_module("structure_models." + structure_name)).get_model()

	def train_model(self, train_data_dir, save_name=None, epochs=25, batch_s=32):

		# Get a new noisy image for each training set
		for epoch in range(epochs):

			print "Running epoch: ", epoch

			# Get each training set
			for X_train, X_target in get_data(train_data_dir):

				# Train the model
				self.reconstruction_model.fit(X_train, X_target, batch_size=batch_s, nb_epoch=1)

			# Save the model after each training set, if save_name is set
			if save_name:
				self.reconstruction_model.save_weights(save_name, overwrite=True)

# If this is the main, use command line arguments to run the network
if __name__ == "__main__":

	# If there are no arguments, show usage
	if len(sys.argv) < 3:

		print "Usage: CAE structure_name train_data_dir [-s save_name -e epochs -b batch_size]"

		sys.exit(1)

	# Get the structure name
	structure_name = sys.argv[1]

	# Get the training data directory
	train_data_dir = sys.argv[2]

	# Get the optional arguments
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

	# Show the network configuration
	print "\nConfiguration"
	print "Model structure: ", structure_name
	print "Training data: ", train_data_dir
	print "Save name: ", save_name
	print "Epochs: ", epochs
	print "Batch size: ", batch_size
	print ""

	# Create the network
	CAE_manage = CAE(structure_name)

	# Train the network
	# Also saves, if save_name is set
	CAE.train_model(train_data_dir, save_name=save_name, epochs=epochs, batch_size=batch_size)
