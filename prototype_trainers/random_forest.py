# Runs a random forest classifier on depth difference data to train a per pixel system

import numpy as np

import cPickle as pickle
import sys
import os
import h5py
import time

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Random_forest:

	# This tracks if the forest has been trained already
	# Should only be False when it is being trained for the first time
	warm_start = False

	# Creates the random forest
	#
	# load_name is a name of a pickle containing a random forest classifier, if it is None then a new random_forest will be created with the other parameters
	#
	# Takes the parameters for the random forest, see scikit learn documentation for details, only used if load_name is None
	# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
	# Defaults to parameters similar to the one in the Microsoft paper
	#
	# TODO: use @classmethod instead of splitting arguments
	def __init__(self, load_name=None, n_estimators=1, criterion='entropy', max_depth=20, n_jobs=4):

		# If a load_name has been set, load the classifier from the pickle
		if load_name:

			# Load the classifier
			self.classifier = pickle.load(open(load_name, 'rb'))

			# Set the warm start to true
			#warm_start = True

		# Create a new forest
		else:

			# Create the forest
			self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, warm_start=True, n_jobs=n_jobs)

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

	# Trains on one batch, adds and trains trees in the forest
	#
	# train_data is a numpy array where the final dimension is the feature data
	# train_label is a numpy array containing the labels, will be flattened
	#
	# increase_trees : int
	# Adds this many trees to the new forest
	def train_batch(self, train_data, train_label, increase_trees=1):

		# If the data isn't of shape (x, features), flatten it
		#train_data = self.rectify_array(train_data)

		# Labels must be 1-D
		#train_label = train_label.flatten()
		#train_label_view = train_label[:][0]

		# Train the forest
		self.classifier.fit(train_data, train_label)

	# Trains on data stored as a h5 file
	def train(self, file_name, batch_size=2500000, save_name = None, verbose=False):

		# Verbose printing
		if verbose:

			print "Loading data from: ", file_name

			# Get the starting time
			start_time = time.time()


			# Show the start in readable format
			print "Start time: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())

		# Open the data
		h5_file = h5py.File(file_name, 'r')

		# Get the data and label sets
		data_set = h5_file["data"]
		label_set = h5_file["label"]

		# Load the data in chunks
		target_index = batch_size
		while target_index < data_set.shape[0]:

			pass

		# Get the leftovers if there is at least half of the batch size remaining
		if target_index - batch_size > batch_size / 2:

			pass

		# Train the forest
		self.train_batch(data_set, label_set)

		# Verbose
		if verbose:

			# Get the end time and the total time taken
			end_time = time.time()
			total_time = end_time - start_time

			# Show the end time in readable format
			print "Ending time: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())

			# Show the total time
			print "Total time taken in seconds: ", total_time

		# Save the model if save_name is set
		if save_name:

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

		#print check_data

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

		print "Usage: random_forest.py data_source_dir test_source_dir save_name ex_image_dir"

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

	# Example images are optional
	if len(sys.argv) > 4:

		ex_image_dir = sys.argv[4]

	# Show the configuration
	print "Configuration"
	print "Training data from: ", data_source_dir
	print "Testing data from: ", test_source_dir
	print "Save name: ", save_name
	print "Example images from: ", ex_image_name

	# Create the forest trainer
	pixel_classifier = Random_forest()

	# Train the forest
	pixel_classifier.train(data_source_dir, save_name=save_name, verbose=True)

	# Test the forest
	if test_source_dir:

		pixel_classifier.test(test_source_dir, verbose=True)

	# Get an example images
	#if ex_image_dir:

	# Manually load a example image
	first_half = pickle.load(open("/media/CORSAIR/ex_image_pickles/data/2_2_0.p", 'rb'))
	second_half = pickle.load(open("/media/CORSAIR/ex_image_pickles/data/2_2_1.p", 'rb'))

	# Put them together to make the actual image
	full_image = np.append(first_half, second_half, axis=3)

	# Get the predicted image
	#predicted_image = pixel_classifier.predict(ex_image_name)
	predicted_image = pixel_classifier.predict(full_image)

	# Turn the predictions into an image
	view_image = im_p.get_pix_vals(np.squeeze(predicted_image))

	# Reorder the axis to (channels, height, width)
	view_image = np.rollaxis(view_image, 2)
	#view_image = np.rollaxis(view_image, 2, 1)

	# Save the image
	im_p.save_image(view_image, "check_random_forest_ex.jpg")
