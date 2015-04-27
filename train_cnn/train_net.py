# Trains a neural network to find human poses

import os
import sys
import cPickle as pickle

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Keras is the framework for theano based neural nets
import keras

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

	for d_item, l_item in get_data("/media/master/DAVID_DRIVE/occulsion/set_001_pickled"):

		print "\n"
		print d_item.shape
		print l_item.shape
