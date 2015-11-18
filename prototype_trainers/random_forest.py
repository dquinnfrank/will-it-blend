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
from sklearn.metrics import accuracy_score, confusion_matrix

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

		if verbose:

			print "Current number of estimators: ", self.classifier.n_estimators
			print "Incrementing estimators by: ", self.n_estimators_inc

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

					print "Training on batch: ", start_index, " to ", target_index, " "*10,
					#sys.stdout.flush()

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

					print "Training on leftovers: ", start_index, " to ", end_index, " "*10,

				# Get the remaining items
				data_batch = data_set[start_index : target_index]
				label_batch = label_set[start_index : target_index]

				# Train
				self.train_batch(data_batch, label_batch)

				# Delete and call garbage collector
				del data_batch
				del label_batch
				gc.collect()

			if verbose:

				# Get the end time and the total time taken
				end_time = time.time()
				total_time = end_time - start_time

				# Show the end time in readable format
				print "Ending time: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())

				# Show the total time
				print "Total time taken in seconds: ", total_time

				# Show model parameters
				print "Total number of trees: ", self.classifier.n_estimators

			# Save the model if save_name is set
			if save_name:

				if verbose:

					print "Saving model as: ", save_name

				self.save_model(save_name)

			if verbose:

				print "Training complete"

	# Saves the model to a pickle for later use
	#
	# Save name is the path and name to save the model as
	def save_model(self, save_name):

		# Pickle it
		pickle.dump(self.classifier, open(save_name, 'wb'))

	# Predicts a batch of data
	#
	# check_data is an array of feature data
	#
	# returns the predicted labels for the data in the same shape that it was sent
	def predict(self, check_data):

		#print check_data

		# If check_data is a string, open it as a pickle
		#if isinstance(check_data, str):

			#check_data = pickle.load(open(check_data, 'rb'))

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

	# Tests the classifier and get the accuracy and confusion matrix
	#
	# file_name : string
	# the name of an h5 file
	#
	# start_index : int
	# Where to start loading the data from
	#
	# end_index : int
	# Where to stop loading data
	def test(self, file_name, start_index, end_index, verbose=False):

		if verbose:

			print "Number of testing samples: ", end_index - start_index

			sys.stdout.flush()

		# Load the data
		with h5py.File(file_name, 'r') as h5_file:

			# Get the data and label sets
			data_set = h5_file["data"]
			label_set = h5_file["label"]

			# Load the specified data and labels from the h5 file
			data_batch = data_set[start_index : end_index]
			label_batch = label_set[start_index : end_index]

			# Get predictions from the forest
			predictions = self.predict(data_batch)

			# Get the score
			score = accuracy_score(label_batch, predictions)

			# Get the confusion matrix
			confusion = confusion_matrix(label_batch, predictions)

		# Show the results
		if verbose:

			# Show score and confusion
			print "Accuracy score: ", score
			print "Confusion matrix:"
			print confusion

		return score, confusion

	# Predicts the pixel labelings for a batch of images
	#
	# source_dir : string
	# Where to load data from
	#
	# destination_dir : string
	# Where to save predictions to
	def batch_image_predict(self, source_dir, destination_dir, verbose=False):

		# TEMP set features to be loaded
		feature_list = pickle.load(open("/media/6a2ce75c-12d0-4cf5-beaa-875c0cd8e5d8/feature_set_01/_feature_list.p"))

		# Make the destination directory
		pp.enforce_path(destination_dir)

		# Load each image individually
		for file_name in sorted([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f)) ]):

			# Get image
			ex_im = np.squeeze(im_p.get_channels(os.path.join(source_dir, file_name), "Z"))

			if verbose:

				print "\rWorking on image: ", file_name,
				sys.stdout.flush()

			# Get the image features
			ex_features = im_p.get_image_features(ex_im, feature_list)

			# Predict the per pixel labels
			predicted_image = self.predict(ex_features)

			# Turn the predictions into an image
			view_image = im_p.get_pix_vals(np.squeeze(predicted_image))

			# Reorder the axis to (channels, height, width)
			view_image = np.rollaxis(view_image, 2)

			# Save the image
			im_p.save_image(view_image, os.path.join(destination_dir, os.path.splitext(file_name)[0] + ".png"))

# Run the system if this is main
if __name__ == "__main__":

	# Create parser object
	parser = argparse.ArgumentParser(description="Trains and tests a random decision forest to create body segmentation images")

	# Loading old model is optional
	parser.add_argument("-l", "--load-name", dest="load_name", help="The name, including path and extension, of an existing forest saved as a pickle to use")

	# Data source is optional
	parser.add_argument("-d", "--data-source", dest="data_source", help="The name, including path and extension, of the data (must be an h5 file). Will be used for training and testing")

	# Train flag is optional, if it is not sent, no training will be done
	parser.add_argument("-r", "--train", dest="train_flag", help="Sets if training will occur. Defaults to false")

	# Test split is optional, specifies the amount of data to be used for training
	parser.add_argument("-t", "--test-split", type=float, dest="test_split", help="The percent of data to be used for training. Float between 0 - 1. Ex: .7 means 70%% will be used for training and the remainder for testing")

	# Name to save the forest as. Optional
	parser.add_argument("-s", "--save-name", dest="save_name", help="The name, including path and extension, to save the forest as. Temporary saves will be made with [save_name]_temp")

	# Directory to load example images from. Optional
	parser.add_argument("-e", "--ex-dir", nargs=2, dest="ex_image_dir", help="First, the name of a directory containing exr images. Second, the name of a directory to save the images to after processing")

	# Get the arguments and unpack them
	args = parser.parse_args()

	# Data source
	if args.data_source:
		data_h5 = os.path.abspath(args.data_source)
	else:
		data_h5 = None

	# Test split
	test_split = args.test_split

	# Save name
	if args.save_name:
		save_name = os.path.abspath(args.save_name)
	else:
		save_name = None

	# Load name
	if args.load_name:
		load_name = os.path.abspath(args.load_name)
	else:
		load_name = None

	# Example images to predict
	if args.ex_image_dir:
		ex_image_source = os.path.abspath(args.ex_image_dir[0])
		ex_image_destination = os.path.abspath(args.ex_image_dir[1])
	else:
		ex_image_source = None
		ex_image_destination = None
		

	# Check for valid data file, if data has been sent
	if data_h5:
		try:

			# Get the size of the data set
			with h5py.File(data_h5, 'r') as h5_file:

				# Get the data to check the shape
				data_set_size = h5_file["data"].shape[0]

				# Make sure that there is also a label set
				h5_file["label"]

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

	# Only need to set end index if there is any training data
	if data_h5:

		# If test_split is set, leave those samples out of training
		if test_split is not None:

			end_index = int(data_set_size * test_split)

		# Otherwise, use all images for training
		else:

			end_index = data_set_size

	# Show the configuration
	print "\nConfiguration"
	if load_name:
		print "Loading from: ", load_name
	if data_h5:
		print "Training data from: ", data_h5
	if test_split is not None:
		print "Testing split: ", test_split, " at: ", end_index
	if save_name:
		print "Save name: ", save_name
	if ex_image_source:
		print "Example images from: ", ex_image_source
		print "Saving example images to:", ex_image_destination
	print ""

	# Create the forest trainer
	pixel_classifier = Random_forest(load_name=load_name, verbose = 1)

	# Train the forest if data source is sent and there is data to train on
	if data_h5 is not None and end_index > 0:

		pixel_classifier.train(data_h5, save_name=save_name, end_index = end_index, verbose=True)

	# Test the forest if the test spilt is sent
	if test_split is not None and data_h5 is not None:

		pixel_classifier.test(data_h5, start_index=end_index, end_index=data_set_size, verbose=True)

	# Process example images if ex_image_dir is sent
	if ex_image_source is not None and ex_image_destination is not None:

		pixel_classifier.batch_image_predict(ex_image_source, ex_image_destination, verbose=True)

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
