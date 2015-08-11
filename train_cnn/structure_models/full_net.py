import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Activation
from keras.layers.additional import UnPooling2D

# This will return a full convolutional nerual net sutiable for creating mask images from depth images
#
# encoder_layer_structure specifies the first layers of the model that bring the data down in dimensionality
# The parameter is the name of a structure model in the folder structure_models to use
#
# pretrained_layer_name is the name of the trained weights to load into the encoder, if sent
# If not sent, the layers will remain randomly initialized
#
# load_name is the name of an model that has already been trained, that model must have an identical structure to this one
# This will override pretrained_layer_name
#
# The rest of the parameters specify tunable parameters in the net
def get_model(encoder_layer_structure = "CAE_2conv_pool_relu", pretrained_layer_name = None, load_name = None, conv_features = 6, lr=.01, decay=1e-6, momentum=0.9, nesterov=True, loss='mse'):

	# Create the model
	model = Sequential()

	# Load the model structure
	reconstruction_model, encoder_slice = (importlib.import_module("structure_models." + encoder_layer_structure)).get_model()

	# Load the existing weights, if sent
	if pretrained_layer_name:
		reconstruction_model.load_weights(pretrained_layer_name)

	# Add the layers, up to the encoder slice
	for layer_index in range(encoder_slice):

		model.add(reconstruction_model.layers[layer_index])

	# Deallocate the reconstruction_model, to make sure it isn't taking up space on the GPU
	reconstruction_model = None

	# The rest of the network configuration
	# height and width reduced depending on the encoder

	# Convolutional step
	# Input shape (n_images, 1, height, width)
	# Output shape (n_images, conv_features, height, width)
	model.add(Convolution2D(conv_features, 1, 3, 3, border_mode='valid'))

	# Non-linear activation
	model.add(Activation("sigmoid"))

	# Do an Unpooling, to get the data back to the images back to the original size
	# TODO: this is based on knowing the structure of the CAE being loaded
	

	# Do another convolutional step
	# Input shape (n_images, conv_features, height, width)
	# Output shape (n_images, 2 * conv_features, height, width)
	model.add(Convolution2D(2 * conv_features, conv_features, 3, 3, border_mode='valid'))

	# Non-linear activation
	model.add(Activation("sigmoid"))
