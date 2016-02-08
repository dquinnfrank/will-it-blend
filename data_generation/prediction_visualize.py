# Loads an hdf5 file of label predictions and makes example images 

import os
import sys

import h5py
import numpy as np
import Image

import post_process as pp; im_p = pp.Image_processing()

# Takes a batch of images and saves them
def batch_save(image_batch, save_prefix, torch_correct = True):

	# Go through each image, get the pixel value predictions, save them
	for index, single_image in enumerate(image_batch):

		print "\rWorking on image " + str(index + 1) + " of " + str(len(image_batch)),
		sys.stdout.flush()

		# Get the prediction array
		label_image = np.array(single_image)

		# If this has come from torch, all of the labels will be shifted up by 1
		if torch_correct:

			label_image -= 1

		# Get pixels
		pixel_image = im_p.get_pix_vals(label_image)

		# Shift the dims to channels, width, height
		pixel_image = np.transpose(pixel_image, (2, 0, 1))

		# Save the image
		im_p.save_image(pixel_image, save_prefix + "_" + str(index) + ".jpg")

# Takes an hdf5 file of predictions and creates visualization images
def make_ex_images(visualize_set, save_path):

	pp.enforce_path(save_path)

	# Open the hdf5 file if a string was sent
	if isinstance(visualize_set, basestring):

		predictions = h5py.File(visualize_set, 'r')

	# Get the batch of images and the true labels, if possible
	prediction_batch = predictions["predictions"]

	# Save them
	print "Saving prediction images\n"
	batch_save(prediction_batch, os.path.join(save_path, "prediction"))

	# Image may not have ground truth
	try:

		true_batch = predictions["true"]

	# Do nothing if there is no true
	except KeyError:

		pass

	# Save true if present
	else:

		print "Saving true images\n"
		batch_save(true_batch, os.path.join(save_path, "true"))

# If this is being run from the commandline, take a source and a destination and visualize the images
if __name__ == "__main__":

	# Get the source and the destination
	source = sys.argv[1]
	destination = sys.argv[2]

	# Visualize
	make_ex_images(source, destination)
