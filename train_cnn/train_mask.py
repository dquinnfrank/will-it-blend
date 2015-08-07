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
def get_data(source_dir, noise_amount = .2):

	# Sub directories
	data_dir = os.path.join(source_dir, "data")
	#label_dir = os.path.join(source_dir, "label")

	# Get the names of all of the data items
	all_names = sorted([ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f)) ])

	# Shuffle the data
	shuffle(all_names)

	# Iterate through all names
	for name in all_names:

		# Corrupt data will cause exceptions
		try:

			# Load the data batch
			print "\nLoading item: ", os.path.join(data_dir, name)
			original_item = pickle.load(open(os.path.join(data_dir, name), 'rb'))

		# The data is corrupt
		# Occurs when the file itself has a problem
		except EOFError:

			print "File corrupt"

			# Ignore this item and load the next one
			 #continue

		# Another form of data corruption
		# Occurs when the pickle is not complete, usually an incomplete save
		except ValueError:

			print "Pickle corrupt"

			# Ignore it and go to the next one

		# Item is valid
		else:
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
			yield noise_item, original_item.reshape(original_item.shape[0], original_item.shape[1] * original_item.shape[2] * original_item.shape[3])

class Mask:

	# Loads the model structure
	#
	# Optionally loads a pretrained layer with the parameter pretrained_layer
	# Requires information to be sent as (pretrained_structure_name, pretrained_model_name)
	# pretrained_structure_name is the model to load without any weights initialized
	# pretrained_model_name is the weights that have been saved from a trained model
	# Layers before "encoder_slice" must be compatable in the Mask and CAE models
	#
	# Optionally loads a trained version of the same model, this will overwrite pretrained_layer if it is sent
	# trained_model is the name of the weights to load
	def __init__(self, pretrained_layer=None, trained_model=None):

		pass

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

		pass

# If this has been run from the commandline, train the network using the given options
if __name__ == "__main__":

	pass
