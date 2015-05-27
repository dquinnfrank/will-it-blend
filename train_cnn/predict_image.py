# This will predict the labels for a given set of test data

import os
import sys
import cPickle as pickle
import numpy as np

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Keras is the framework for theano based neural nets
# Not needed?
#from keras.models import Sequential

# Used to predict the labeled pixels for a given image
class label_pix:


	# Needs the model being used for predictions
	# model can be the unpacked model, or a string of a file name of a pickled model
	# Does not do any error checking. Make sure that model is valid
	def __init__(self, model):

		# Unpickle the model if argument is a string
		if(isinstance(model, basestring)):
			self.model = pickle.load(open(model, 'rb'))

		# Assume that this is a valid model otherwise
		else:
			self.model = model

	# Get the pixel wise predictions of the image batch
	#
	# Returns the labelings for each pixel, shape (n_images, height, width)
	#
	# images_to_label must be a numpy array of shape (n_images, height, width)
	# images_to_label should be the normalized depth data
	def get_pix_labels(self, images_to_label):

		pix_labels = self.model.predict_proba(images_to_label)

		return pix_labels

# If this is the main, load a pickle of images to predict
# Predict the labels and save the image to the given folder
if __name__ == "__main__":

	# Show usage when run with no arguments
	if len(sys.argv) < 2 :

		print "Usage: predict_image.py load_model_name save_image_dir image_pickle"
		print ""
		print "load_model_name : the name, including path, of the model to use for predictions"
		print "save_image_dir : the name of the directory to save image predictions to, will be created if it doesn't exist"
		print "image_pickle : the name, including path, of the pickle of images to use"
		print ""

		sys.exit(1)

	# Get the arguments
	model_name = sys.argv[1]
	save_dir = sys.argv[2]
	image_name = sys.argv[3]

	# Load the images
	image_batch = pickle.load(open(image_name ,'rb'))

	# Get the shape of the input
	(n_images, height, width) = image_batch.shape

	# Normalize the depth data
	input_max = np.max(image_batch)
	input_min = np.min(image_batch)
	input_range = input_max - input_min
	image_batch = (image_batch - input_min) / input_range

	# Add the stack dimension, needed for correct processing in the convolutional layers
	image_batch = np.expand_dims(image_batch, axis=0)

	# Reorder axis to to (n_images, stack, height, width)
	image_batch = np.rollaxis(image_batch, 0, 2)

	# Enforce the save dir
	pp.enforce_path(save_dir)

	# Create the model class
	predictor = label_pix(model_name)

	# Predict the images
	im_predictions = predictor.get_pix_labels(image_batch)
	#print im_predictions.shape

	# Reshape, from 1D to 2D
	im_predictions = im_predictions.reshape((image_batch.shape[0], image_batch.shape[2], image_batch.shape[3]))
	#print im_predictions.shape

	# Get the pixel values for each labeled image
	new_im = np.empty((im_predictions.shape[0], im_predictions.shape[1], im_predictions.shape[2], 3), dtype=np.uint8)
	for index in range(image_batch.shape[0]):
		new_im[index] = pp.get_pix_vals(im_predictions[index])

	# Reorder the axis from n_images * height * width * channels to n_images * channels * width * height
	new_im = np.rollaxis(new_im, 3, 1)
	new_im = np.rollaxis(new_im, 3, 2)

	#new_im = pp.get_pix_vals(im_predictions)
	#print new_im.shape
	#print new_im.dtype

	# Save all of the images
	# new_im used as intermediate for debugging
	for index in range(new_im.shape[0]):
		pp.save_image(new_im[index], os.path.join(save_dir, "out_" + str(index) + ".png"))
