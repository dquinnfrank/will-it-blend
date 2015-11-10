# Runs a random forest classifier on depth difference data to train a per pixel system

import numpy as np

import cPickle as pickle
import sys
import os
import h5py
import time
import gc
import argparse

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
	def __init__(self, load_name=None, n_estimators_inc=4, criterion='entropy', max_depth=20, n_jobs=4, verbose = 0):

		# Save the number to increase estimators by
		self.n_estimators_inc = n_estimators_inc

		# If a load_name has been set, load the classifier from the pickle
		if load_name:

			# Load the classifier
			self.classifier = pickle.load(open(load_name, 'rb'))

		# Create a new forest
		else:

			# Create the forest
			self.classifier = RandomForestClassifier(n_estimators=0, criterion=criterion, max_depth=max_depth, warm_start=True, n_jobs=n_jobs, verbose = verbose)

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
	def train_batch(self, train_data, train_label):

		# Increase the number of estimators
		self.classifier.n_estimators += self.n_estimators_inc

		# Train the forest
		self.classifier.fit(train_data, train_label)

	# Trains on data stored as a h5 file
	def train(self, file_name, batch_size=2500000, save_name = None, start_index = None, end_index = None, verbose=False):

		# Verbose printing
		if verbose:

			print "Loading data from: ", file_name

			# Get the starting time
			start_time = time.time()


			# Show the start in readable format
			print "Start time: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())

		# Open the data
		with h5py.File(file_name, 'r') as h5_file:

			# Get the data and label sets
			data_set = h5_file["data"]
			label_set = h5_file["label"]

			# Set the start and end if they were sent as None
			if start_index is None:

				start_index = 0

			if end_index is None:

				end_index = data_set.shape[0]

			if verbose:

				print "Total items to train on: ", end_index - start_index

			# Load the data in chunks
			iteration = 1
			target_index = start_index + batch_size
			while target_index < end_index:

				# Get a batch of data and labels
				data_batch = data_set[start_index : target_index]
				label_batch = label_set[start_index : target_index]

				if verbose:

					print "\rTraining on batch: ", start_index, " to ", target_index, " "*10,
					sys.stdout.flush()

				# Train the forest
				self.train_batch(data_batch, label_batch)

				# Increment the start and target
				start_index += batch_size
				target_index += batch_size

				# Delete this chuck of data to avoid exessive swaping
				del data_batch
				del label_batch
				gc.collect()

				# Save on regular intervals
				if iteration % 5 == 0:

					# The temp save name is based off of the save name and includes the index of the last item trained on
					name, extension = os.path.splitext(save_name)

					temp_save_name = name + "_temp_" + str(target_index) + extension

					self.save_model(temp_save_name)

				iteration += 1

			# If there are enough leftovers, do one last training batch
			if target_index - data_set.shape[0] > batch_size / 2:

				# Set the target index to the end index
				target_index = end_index

				if verbose:

					print "\rTraining on leftovers: ", start_index, " to ", end_index, " "*10,
					sys.stdout.flush()

				# Get the remaining items
				data_batch = data_set[start_index : target_index]
				label_batch = label_set[start_index : target_index]

				# Train
				self.train_batch(data_batch, label_batch)

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
	#
	# Returns a list of accuracies
	#
	# source_dir is a directory that contains data and label sub folders
	def test(self, file_name, start_index, end_index, verbose=False):

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

	# Create parser object
	parser = argparse.ArgumentParser(description="Trains and tests a random decision forest to create body segmentation images")

	# Data source is mandatory
	parser.add_argument("data_source", help="The name, including path and extension, of the data (must be an h5 file)")

	# Test split is optional, specifies the amount of data to be used for training
	parser.add_argument("-t", "--test-split", type=float, dest="test_split", help="The percent of data to be used for training. Float between 0 - 1. Ex: .7 means 70%% will be used for training and the remainder for testing")

	# Name to save the forest as. Optional
	parser.add_argument("-s", "--save-name", dest="save_name", help="The name, including path and extension, to save the forest as. Temporary saves will be made with [save_name]_temp")

	# Directory to load example images from. Optional
	parser.add_argument("-e", "--ex-dir", nargs=2, dest="ex_image_dir", help="First, the name of a directory containing exr images. Second, the name of a directory to save the images to after processing")

	# Get the arguments and unpack them
	args = parser.parse_args()

	data_h5 = os.path.abspath(args.data_source)
	test_split = args.test_split
	save_name = os.path.abspath(args.save_name)
	if args.ex_image_dir:
		ex_image_source = os.path.abspath(args.ex_image_dir[0])
		ex_image_destination = os.path.abspath(args.ex_image_dir[1])
	else:
		ex_image_source = None
		ex_image_destination = None
		

	# Check for valid data file
	try:

		# Get the size of the data set
		with h5py.File(data_h5, 'r') as h5_file:

			# Get the data to check the shape
			data_set_size = h5_file["data"].shape[0]

			# Make sure that there is also a label set
			h5_file["label"]

			# If test_split is set, leave those samples out of training
			if test_split is not None:

				end_index = int(data_set_size * test_split)

			# Otherwise, use all images for training
			else:

				end_index = data_set_size

	# File did not open
	except IOError:

		print "File could not be opened: ", data_h5

		sys.exit(1)

	# File did not have correct format
	except KeyError:

		print "File was malformed: ", data_h5

		sys.exit(1)

	# Unknown error
	except Exception as e:

		print "Unknown error:"
		print e

		sys.exit(1)

	# Show the configuration
	print "\nConfiguration"
	print "Training data from: ", data_h5
	print "Testing split: ", test_split, " at: ", end_index
	print "Save name: ", save_name
	print "Example images from: ", ex_image_source
	print "Saving example images to:", ex_image_destination
	print ""

	# Create the forest trainer
	pixel_classifier = Random_forest(verbose = 1)

	# Train the forest
	pixel_classifier.train(data_h5, save_name=save_name, end_index = end_index, verbose=True)

	# Test the forest
	#if test_source_dir:

		#pixel_classifier.test(test_source_dir, verbose=True)

	# Get an example images
	#if ex_image_dir:

	# Manually load a example image
	#first_half = pickle.load(open("/media/CORSAIR/ex_image_pickles/data/2_2_0.p", 'rb'))
	#second_half = pickle.load(open("/media/CORSAIR/ex_image_pickles/data/2_2_1.p", 'rb'))

	# Put them together to make the actual image
	#full_image = np.append(first_half, second_half, axis=3)

	# Get the predicted image
	#predicted_image = pixel_classifier.predict(ex_image_name)
	#predicted_image = pixel_classifier.predict(full_image)

	# Turn the predictions into an image
	#view_image = im_p.get_pix_vals(np.squeeze(predicted_image))

	# Reorder the axis to (channels, height, width)
	#view_image = np.rollaxis(view_image, 2)
	#view_image = np.rollaxis(view_image, 2, 1)

	# Save the image
	#im_p.save_image(view_image, "check_random_forest_ex.jpg")
