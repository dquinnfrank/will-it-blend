# Trains a neural network to find human poses

import os
import sys
import cPickle as pickle
import numpy as np

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Keras is the framework for theano based neural nets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

# Loads the data from pickles and returns each item in order
# This is a generator function
# source_dir is the directory containing the images, it should have sub directories: data, labels
def get_data(source_dir):

	# Sub - directories
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

		# Generate the next batch
		yield data_item, label_item

# Run the network training, if this is the main
if __name__ == "__main__":

	# Network configuration

	# The number of images to process at once. This is decided by the size of the pickles
	batch_size = 3

	# The number of images in the data set
	# MAKE THIS AUTOMATIC
	n_batches = 666

	# The threshold to begin testing at
	# MAKE THIS AUTOMATIC
	n_test_threshold = 606

	# The number of epochs to run
	nb_epochs = 10

	# The number of hidden nodes to have in the classification layer
	n_hidden = 512

	# Stack size
	stack_size = 1

	# The network configuration
	model = Sequential()

	# Convolution layer
	#model.add(Convolution2D(480, 3, 3, 3, border_mode='full')) 
	#model.add(Activation('relu'))
	#model.add(Convolution2D(480, 480, 3, 3))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(poolsize=(2, 2)))
	#model.add(Dropout(0.25))

	# Convolution layer
	#model.add(Convolution2D(batch_size * 2, batch_size, 3, 3, border_mode='full')) 
	#model.add(Activation('relu'))
	#model.add(Convolution2D(batch_size * 2, batch_size * 2, 3, 3)) 
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(poolsize=(2, 2)))
	#model.add(Dropout(0.25))

	# Classification layer
	#model.add(Flatten())
	#model.add(Dense(480*640, n_hidden, init='normal'))
	model.add(Dense(480*640, 480*640, init='normal'))
	model.add(Activation('linear'))

	# Optimizer
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	index = 0
	for d_item, l_item in get_data("/media/master/DAVID_DRIVE/occulsion/set_001_pickled"):
		index += 1

		# Convert to float32 for the GPU's sake
		d_item = d_item.astype("float32")

		# Add an axis
		print (d_item.shape)
		d_item = np.expand_dims(d_item, axis=0)
		print (d_item.shape)
		d_item = np.rollaxis(d_item, 0, 2)
		#d_item = d_item[0]
		print (d_item.shape)
		d_item = d_item[0]
		print (d_item.shape)
		d_item = d_item.flatten()
		l_item = l_item[0].flatten()
		print (d_item.shape)
		print (l_item.shape)

		# Training set
		if index < n_test_threshold:

			print("Training index: ", index)

			# Train the model
			model.fit(d_item, l_item, nb_epoch=nb_epochs)

		# Test set
		else:

			# Evaluation
			score = model.evaluate(d_item, l_item, batch_size=batch_size)
			print('Test score:', score)
