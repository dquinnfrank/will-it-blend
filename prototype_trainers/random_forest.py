# Runs a random forest classifier on depth difference data to train a per pixel system

import numpy as np

import cPickle as pickle

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import sys
import os

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
	# Make sure to save the old array, if it is needed
	#
	# Returns the new array
	def rectify_array(original_array):

		# Get the original shape
		original_shape = original_array.shape

		# Get the product of all dimensions, except the last
		product = 1
		for index in range(len(original_shape - 1)):

			product *= original_shape[index]

		# Reshape and return
		return original_array.reshape((product, original_shape[-1]))

	# Trains on one batch
	#
	# train_data is a numpy array where the final dimension is the feature data
	# train_label is a numpy array containing the labels, will be flattened
	def train_batch(self, train_data, train_label):

		# If the data isn't of shape (x, features), flatten it
		train_data = self.rectify_data(train_data)

		# Labels must be 1-D
		train_label = train_label.flatten()

		# Train the forest
		self.classifier.fit(train_data, train_labels)

	# Trains on a bunch of pickled data
	#
	# source_dir is the directory that contains data and label sub directories
	def train(self, source_dir):

		# Set the data and label directories
		data_dir = os.path.join(source_dir, "data")
		label_dir = os.path.join(source_dir, "label")

		# Get the names of the data
		pickle_names = pp.get_names(data_dir)

		# Go through each item
		for item_name in pickle_names:

			# Get the data
			data_batch = pickle.load(open(os.path.join(data_dir, item_name), 'rb'))

			# Get the labels
			label_batch = pickle.load(open(os.path.join(label_dir, item_name), 'rb'))

			# Train the forest
			self.train_batch(data_batch, label_batch)

	# Predicts a batch of data
	#
	# check_data is an array of feature data
	# It can also be a string that is the name of a pickle with the data
	#
	# returns the predicted labels for the data in the same shape that it was sent
	def predict(self, check_data):

		# If check_data is a string, open it as a pickle
		if isinstance(check_data, str):

			check_data = pickle.load(open(check_data, 'rb'))

		# Get the original shape of the data
		data_shape = check_data.shape

		# Rectify the data
		check_data = self.rectify_data(check_data)

		# Get the predictions
		predictions = self.classifier.predict(check_data)

		# Reshape the predictions to the original shape, considering the loss of the feature dimension
		predictions = predictions.reshape(data_shape[:-1])

		# Return the predictions
		return predictions

	# Tests the classifier and shows the accuracy
	# Will show the accuracy for each pickle in the sub folders
	#
	# source_dir is a directory that contains data and label sub folders
	def test(self, source_dir):

		# Set the data and label directories
		data_dir = os.path.join(source_dir, "data")
		label_dir = os.path.join(source_dir, "label")

		# Get the names of the data
		pickle_names = pp.get_names(data_dir)

		# Go through each item
		for item_name in pickle_names:

			# Get the data
			data_batch = pickle.load(open(os.path.join(data_dir, item_name), 'rb'))

			# Get the labels
			label_batch = pickle.load(open(os.path.join(label_dir, item_name), 'rb'))

			# Get the predictions
			predictions = self.predict(data_batch)

			# Get the score
			score = accuracy_score(label_batch, predictions)

			# Show the score
			print "Accuracy: ", score

# Run the system if this is main
#if __name__ == "__main__":

	# Show use if 

"""
# The percentage to be used for training, the remaining data will be used for testing
#train_split = .9

# Get the data
data = pickle.load(open("/media/CORSAIR/depth_features/data/set_001.p", 'rb'))

# Get the labels
labels = pickle.load(open("/media/CORSAIR/depth_features/label/set_001.p", 'rb'))

# Split into training and testing
train_data = data[int(train_split * data.shape[0]):]
train_labels = labels[int(train_split * labels.shape[0]):]

test_data = data[:int(train_split * data.shape[0])]
test_labels = labels[:int(train_split * labels.shape[0])]

# Create the random forest with parameters similar to the microsoft paper
clf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)

# Train the forest
clf.fit(train_data, train_labels)

# Get the score
clf_predict = clf.predict(test_data)
score = accuracy_score(test_labels, clf_predict)

# Show score and other info
print "Random forest"
print "Number of training points: ", data.shape[0]
print "Accuracy: ", score

# Load an example image
#ex_image = pickle.load(open("/media/CORSAIR/ex_images/ex1.py", 'rb'))

# Predict all of the pixels
#ex_prediction = clf.predict(test_data)

# Reorder the axis for saving
#ex_prediction = np.expand_dims(ex_prediction, axis=0)
#ex_prediction = np.rollaxis(ex_prediction, 2, 1)

# Save the image
#im_p.save_image(ex_prediction, "/media/CORSAIR/ex_images/ex1_prediction.jpg")
"""
