# Runs a random forest classifier on depth difference data to train a per pixel system

import numpy as np

import cPickle as pickle

import sys
import os

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Random_forest:

	# Creates the random forest
	#
	# load_name is a name of a pickle containing a random forest classifier, if it is None then a new random_forest will be created with the other parameters
	#
	# Takes the parameters for the random forest, see scikit learn documentation for details, only used if load_name is None
	# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
	# Defaults to parameters similar to the one in the Microsoft paper
	def __init__(self, load_name=None, n_estimators=3, criterion='entropy', max_depth=20):

		# If a load_name has been set, load the classifier from the pickle
		if load_name:

			# Load the classifier
			self.classifier = pickle.load(open(load_name, 'rb'))

		# Create a new forest
		else:

			# Create the forest
			self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

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

	# Trains on one batch
	#
	# train_data is a numpy array where the final dimension is the feature data
	# train_label is a numpy array containing the labels, will be flattened
	def train_batch(self, train_data, train_label):

		# If the data isn't of shape (x, features), flatten it
		train_data = self.rectify_array(train_data)

		# Labels must be 1-D
		train_label = train_label.flatten()

		# Train the forest
		self.classifier.fit(train_data, train_label)

	# Trains on a bunch of pickled data
	#
	# source_dir is the directory that contains data and label sub directories
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

	# Saves the model to a pickle for later use
	#
	# Save name is the path and name to save the model as
	def save_model(self, save_name):

		# Pickle it
		pickle.dump(self.classifier, open(save_name, 'wb'))

	# Predicts a batch of data
	#
	# check_data is an array of feature data
	# It can also be a string that is the name of a pickle with the data
	#
	# returns the predicted labels for the data in the same shape that it was sent
	def predict(self, check_data):

		print check_data

		# If check_data is a string, open it as a pickle
		if isinstance(check_data, str):

			check_data = pickle.load(open(check_data, 'rb'))

		# Get the original shape of the data
		data_shape = check_data.shape

		# Rectify the data
		check_data = self.rectify_array(check_data)

		# Get the predictions
		predictions = self.classifier.predict(check_data)

		# Reshape the predictions to the original shape, considering the loss of the feature dimension
		predictions = predictions.reshape(data_shape[:-1])

		# Return the predictions
		return predictions

	# Tests the classifier and shows the accuracy
	# Will show the accuracy for each pickle in the sub folders
	#
	# Returns a list of accuracies
	#
	# source_dir is a directory that contains data and label sub folders
	def test(self, source_dir, verbose=False):

		# Verbose line
		if verbose:

			print ""

		# Set the data and label directories
		data_dir = os.path.join(source_dir, "data")
		label_dir = os.path.join(source_dir, "label")

		# Get the names of the data
		pickle_names = pp.get_names(data_dir)

		# Go through each item
		acc_list = []
		for item_name in pickle_names:

			# Get the data
			data_batch = pickle.load(open(os.path.join(data_dir, item_name), 'rb'))

			# Get the labels
			label_batch = pickle.load(open(os.path.join(label_dir, item_name), 'rb'))

			# Get the predictions
			predictions = self.predict(data_batch)

			# Get the score
			score = accuracy_score(label_batch, predictions)

			# Add it to the list
			acc_list.append(score)

			# Show the score, if verbose
			if verbose:

				print "\rAccuracy: " + str(score),
				sys.stdout.flush()

# Run the system if this is main
if __name__ == "__main__":

	# Show use if no arguments have been sent
	if len(sys.argv) < 2:

		print "Usage: random_forest.py data_source_dir test_source_dir save_name ex_image_name"

		sys.exit(1)

	# Argument defaults
	test_source_dir = None
	save_name = None
	ex_image_name = None

	# Source directory is required
	data_source_dir = sys.argv[1]

	# Testing is optional
	if len(sys.argv) > 2:

		test_source_dir = sys.argv[2]

	# Saving is optional
	if len(sys.argv) > 3:

		save_name = sys.argv[3]

	# Example image is optional
	if len(sys.argv) > 4:

		ex_image_name = sys.argv[3]

	# Show the configuration
	print "Configuration"
	print "Training data from: ", data_source_dir
	print "Testing data from: ", test_source_dir
	print "Save name: ", save_name
	print "Example image: ", ex_image_name

	# Create the forest trainer
	pixel_classifier = Random_forest()

	# Train the forest
	pixel_classifier.train(data_source_dir, save_name=save_name, verbose=True)

	# Test the forest
	if test_source_dir:

		pixel_classifier.test(test_source_dir, verbose=True)

	# Get an example image
	if ex_image_name:

		# Load the example image
		#ex_image = pickle.load(open(ex_image_name, 'rb'))

		predicted_image = pixel_classifier.predict(ex_image_name)
