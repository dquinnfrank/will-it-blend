# This module applies post processing to the data
# Loads OpenEXR images, applies functions as needed by the user and saves the results
# Currently, assigns all pixels to the closest class to avoid all confusion when training
# Converts images into jpg for the RGB values
# Saves depth data as a binary file of 64 bit floats

# Copyright notice for OpenEXR
"""
Copyright (c) 2002-2011, Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd. All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:


Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of Industrial Light & Magic nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import pyximport; pyximport.install()
import cython_feature_extraction

import OpenEXR
import Imath
import Image
import sys
from array import array
import struct
import numpy as np
from PIL import Image as PIL_Image

import sys
import os
import errno
import cPickle as pickle
from random import shuffle
import subprocess

# Enforces file path
def enforce_path(path):
    try:
	os.makedirs(path)
    except OSError as exc: # Python >2.5
	if exc.errno == errno.EEXIST and os.path.isdir(path):
	    pass
	else: raise

# Gets a list of files based on the start and end indices
def get_names(source_dir, start_index=None, end_index=None, randomize=False):

	# Get the names of every file
	to_process = sorted([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f)) ])

	# Only include items after the start threshold
	if start_index is not None:
		to_process = sorted([v for v in to_process if int(v.strip(".exr")) >= start_index ])

	# Only include items before the end threshold
	if end_index is not None:
		to_process = sorted([v for v in to_process if int(v.strip(".exr")) <= end_index ])

	# Randomize names, if selected
	if randomize:
		shuffle(to_process)

	return to_process

class Image_processing:

	# The dictionary of pixel values to labels
	# "R G B" : label
	pix_to_label = {
	"0 0 0" : 0, # Not person
	"255 0 0" : 1, # Head L
	"50 0 0" : 2, # Head R
	"0 0 255" : 3, # Torso L
	"0 0 50" : 4, # Torso R
	"255 255 0" : 5, # Upper arm L
	"50 50 0" : 6, # Upper arm R
	"0 255 255" : 7, # Lower arm L
	"0 50 50" : 8, # Lower arm R
	"0 255 0" : 9, # Upper leg L
	"0 50 0" : 10, # Upper leg R
	"255 0 255" : 11, # Lower leg L
	"50 0 50" : 12 # Lower leg R
	}

	# The inverse of the label dictionary
	label_to_pix = {v: k for k, v in pix_to_label.items()}

	# The scale to process all images to
	# TODO: Find a better way to set this
	# The problem is that this is only needed in get_channels, which is at the bottom of most function calls.
	# Thus it requires lots of pass-though arguments if scale is not global
	#scale_factor = 1

	# Start the class
	# scale_factor defaults to 1, images are not scaled
	def __init__(self, sent_scale_factor=1):

		self.scale_factor = sent_scale_factor
		
	# Returns bounds for image processing that avoid issues with unaligned batch sizes
	# Bounds set as None will be automatically set
	def set_bounds(self, source_dir, start_index=None, end_index=None, batch_size=128):
		# If end_index is not sent, set it to the largest file in the set
		if end_index is None:
			end_index = max([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))])
			end_index = int(end_index.strip(".exr"))

		# Make sure that the end index is aligned to the batch size
		end_index = (end_index // batch_size) * batch_size

		# If start_index is not sent, set it to the lowest file in the set
		if start_index is None:

			start_index = int(min([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))]).strip(".exr"))
			
		return start_index, end_index, batch_size

	# Gets the specified channel information from an exr file and places it into a numpy array of floats
	# Scale will change the image size by the sent number
	# For RGB, channels = "RGB"
	# For Depth, channels = "Z"
	# shape : (channels, scale * height, scale * width)
	def get_channels(self, exrfile, channels):

		# Check for file name or open file
		if(isinstance(exrfile, basestring)):
			# This is a string

			# Open the file
			exrfile = OpenEXR.InputFile(exrfile)

		# Set pixel info
		pt = Imath.PixelType(Imath.PixelType.FLOAT)
		dw = exrfile.header()['dataWindow']
		size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

		# Get the pixels, they will be loaded as floats
		pix_float = [Image.fromstring("F", size, exrfile.channel(c, pt)) for c in channels]

		# Initialize the numpy array
		# Images and numpy arrays use different major, thus size must be switched
		np_pix = np.empty((len(pix_float), int(self.scale_factor * size[1]), int(self.scale_factor * size[0])))

		# Place the pixel data into the numpy array
		for index in range(len(pix_float)):
			np_pix[index] = np.array(pix_float[index].resize((int(self.scale_factor * size[0]), int(self.scale_factor * size[1]))))

		# Return the numpy array
		return np_pix

	# Normalizes each channel to be about 0 - 255
	# Changes type to int
	# to_normalize is the image to be normalized, it must be a numpy array of shape channels * height * width
	# OpenEXR images generally have 1 as the highest value, but sometimes the value can be higher
	# This function assumes that the lowest value will always be 0.0
	# The highest value is assumed to be at least 1.0, this is because if an image doesn't show the whole person, the high could be incorrectly very low
	def normalize_image(self, to_normalize):

		# Get the highest and lowest values in the whole image
		# Assume that we always need to normalize each channel by the same amount
		highest_val = np.max(np.max((to_normalize), 1.0))

		# Set the scale
		scale = 255.0 / (highest_val)

		# Normalize the image
		normalized = to_normalize * scale

		return normalized.astype(np.uint8)

	# Assigns a label to each pixel based on the RGB values
	def get_labels(self, rgb_image):

		# Create a new image
		new_image = np.empty((rgb_image.shape[1], rgb_image.shape[2]), dtype=np.uint8)

		# For each pixel, use the label dictionary to assign a label
		for h in range(rgb_image.shape[1]):
			for w in range(rgb_image.shape[2]):

				# Try to assign a label
				try:
					new_image[h][w] = self.pix_to_label[" ".join([str(rgb_image[0][h][w]), str(rgb_image[1][h][w]), str(rgb_image[2][h][w])])]

				# No known label, assign no class
				except KeyError:
					new_image[h][w] = 0

		return new_image

	# Gets the labels for an entire batch of images
	def batch_get_labels(self, rgb_batch):

		# Set the labels for the data
		labels = np.empty((rgb_batch.shape[0], rgb_batch.shape[2], rgb_batch.shape[3]), dtype=np.uint8)
		for index in range(len(labels)):
			labels[index] = self.get_labels(rgb_batch[index])

		return labels

	# Assigns a pixel value based on the label
	def get_pix_vals(self, sent_label_array):

		# Make sure that label array is ints
		label_array = sent_label_array.astype(np.uint8)

		# Create a new image
		new_image = np.empty((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

		# For each pixel, use the label to pix dictionary to assign a pixel value
		for h in range(label_array.shape[0]):
			for w in range(label_array.shape[1]):

				# Try to assign a label
				try:
					pix_vals = (self.label_to_pix[label_array[h][w]]).split()

					new_image[h][w][0] = pix_vals[0]
					new_image[h][w][1] = pix_vals[1]
					new_image[h][w][2] = pix_vals[2]

				# No known label, assign no class
				except KeyError:
					print "no label"
					new_image[h][w][0] = 0
					new_image[h][w][1] = 0
					new_image[h][w][2] = 0

		return new_image

	# Assigns the closest class to each pixel
	# Takes an array and returns a numpy array with the same shape and dtype
	def assign_closest(self, data, possible_classes=[0, 50, 255]):

		# Get all the data as a single array
		# Use float64 to avoid over/under flow issues
		flat_data = (data.flatten()).astype(np.float64)

		# Initialize the new array
		fixed = np.empty(shape=flat_data.shape)

		# Get the differences from each class
		distances = []
		for possible in possible_classes:

			distances.append(np.absolute(flat_data - possible))

		# Assign the lowest distance as the class
		for index in range(flat_data.shape[0]):

			# Get the values of each distance
			vals = []
			for i in range(len(possible_classes)):
				vals.append(distances[i][index])

			# Set the value at the index to be the closest class
			fixed[index] = possible_classes[vals.index(min(vals))]

		# Send the fixed data back reshaped, with the same dtype
		return np.array(fixed.reshape(data.shape), dtype=data.dtype)

	# Converts an exr image to a numpy array
	# Will assign each pixel to the closest class
	# Wrapper for get_channels, normalize_image, assign_closest
	# exrfile is an exrfile, opened or name
	def convert_to_rgb(self, exrfile):

		# Get the image from the exr file
		rgb_image = self.get_channels(exrfile, "RGB")

		# Normalize the image
		rgb_image = self.normalize_image(rgb_image)

		# Fix any noisy pixels
		rgb_image = self.assign_closest(rgb_image)

		# Return the image
		return rgb_image

	# Converts ands saves an exr image to a unit8 image
	# Will assign each pixel to the closest class
	# Wrapper for get_channels, normalize_image, assign_closest, save_image
	# exrfile is an exrfile, opened or name
	# rgb_name is the name to save the image as. Type will be inferred from extension
	def save_to_rgb(self, exrfile, rgb_name):

		# Get the RGB image
		rgb_image = self.convert_to_rgb(exrfile)

		# Save the image
		self.save_image(rgb_image, rgb_name)

	# Checks to see if the given file is valid
	def is_valid(self, exr_name):

		# Attempt to open the file
		try:
			# Open the file
			exrfile = OpenEXR.InputFile(exr_name)

		# File is not valid
		except IOError:

			return False

		# File is valid
		else:

			return True

	# Removes all invalid files from the directory
	# Does not rename files, gaps will be present in the file indexes
	# WARNING: This function deletes things, use wisely
	# will require sudo permission if files are write protected
	def remove_invalid(self, source_dir, start_index=None, end_index=None, verbose=False, dry_run=False):

		if verbose:

			print "Checking directory: ", source_dir
			print "Start index: ", start_index
			print "End index: ", end_index

		# Get the names of all of the files to check
		to_process = get_names(source_dir, start_index, end_index)

		if verbose:

			print "Total items to check: ", len(to_process)

		# Check each image, remove if invalid
		for file_name in to_process:

			# Rejoin the file with the path
			file_name = os.path.join(source_dir, file_name)

			# Sanity check to make sure that this is an exr file
			if  os.path.splitext(file_name)[1] != ".exr":

				if verbose:

					print "Ignoring file (not an exr file): ", file_name

				# Go to the next item
				continue

			# Remove the file if it is not valid
			if not self.is_valid(os.path.join(source_dir, file_name)):

				if verbose:

					print "Removing (file not valid): ", file_name

				# Do not delete if this is a dry_run
				if not dry_run:
					subprocess.call("rm " + file_name, shell=True)

		if verbose:

			print "Remaining items: ", len(get_names(source_dir, start_index, end_index))

	# Saves the given data as a binary array
	# Data will be flattened before saving
	# Whatever program loads the data must know the data type and size
	# to_save is a numpy array to be saved
	# bin_name is the name to save the binary file as. Include file path and extension '.bin'
	# data_type is the python array type to save the data as
	# default is double: 'd'
	def save_binary(self, to_save, bin_name, data_type='d'):

		# Flatten and convert to a Python array
		copy = array(data_type, to_save.flatten())

		# Open the target file in binary write mode
		with open(bin_name, 'wb') as out_file:

			# Save the file
			copy.tofile(out_file)

	# Takes the depth channel from the exr image and saves it as a binary file
	# This is a wrapper for get_channel and save_binary
	# exrfile is an opened exr file or a file name
	# save_name is the name to save the output as
	def save_depth_binary(self, exrfile, save_name):

		# Get the depth channel
		z_channel = self.get_channels(exrfile, "Z")

		# Save the depth channel
		self.save_binary(z_channel, save_name)

	# Takes a numpy array and saves it as an image
	# to_save is the image to be saved, must have shape: channels * width * height
	# TODO: Redo the axis ordering
	def save_image(self, to_save, save_name):

		# Get the channels in the image
		channels = to_save.shape[0]
		width = to_save.shape[1]
		height = to_save.shape[2]

		# Shift the axis to be in the correct order
		im = np.rollaxis(to_save, 1)
		im = np.rollaxis(im, 2, 1)

		# Save the image
		save_im = PIL_Image.fromarray(im)
		save_im.save(save_name)
		
	# Assigns the depth at the target pixel, or a large positive value if the index is off the image
	#
	# returns the depth probe as defined in microsoft paper
	#
	# image is a numpy array of shape (height, width)
	#
	# target_pixel is a tuple containing the coordinates of the pixel to be accessed
	#
	# large_positive is the value to return if the target_pixel is off of the image
	def depth_probe(self, image, target_pixel, large_positive=1000000.0):

		# Get image shape info
		(height, width) = image.shape
		
		# Get target_pixel indexes
		(h_index, w_index) = target_pixel

		# Check that the target pixel is within bounds
		if 0 <= h_index < height and 0 <= w_index < width:
		# Within bounds

			# Assign the target value
			return image[target_pixel]
			
		# Outside of the image
		else:

			# Assign a large positive value
			return large_positive

	# Creates depth difference feature indices randomly
	#
	# returns a list containing tuples of every feature
	# shape (number_features, )
	#
	# number_features is the total number of features to create
	#
	# window is the max offset allowable, total range will be  -1 * window to window 
	def random_feature_list(self, number_features=2000, window=(400, 400)):

		# The list of features to create
		feature_list = []

		# Make the specified number of features
		for feature_index in range(number_features):

			# Create random indices, making sure that they are not already in the list
			# Indices will be within window range
			random_indices = ((np.random.randint(low=-window[0], high=window[0]), np.random.randint(low=-window[1], high=window[1])), (np.random.randint(low=-window[0], high=window[0]), np.random.randint(low=-window[1], high=window[1])))
			while random_indices in feature_list:

				random_indices = ((np.random.randint(low=-window[0], high=window[0]), np.random.randint(low=-window[1], high=window[1])), (np.random.randint(low=-window[0], high=window[0]), np.random.randint(low=-window[1], high=window[1])))

			# Add the feature to the list
			feature_list.append(random_indices)

		return feature_list

	# Gets the features from a specified pixel, given the image
	# Computes the features as set in Equation 1 from Real-Time Human Pose Recognition in Parts from Single Depth Images
	#
	# returns a numpy array of shape (number_features)
	# Each location is one of the feature outputs
	#
	# image is the image to access, must be a numpy array
	#
	# target_pixel is the pixel being classified, must be sent as coordinate pair
	#
	# feature_list is the feature offsets to be computed
	def get_features(self, image, target_pixel, feature_list):

		# Get a copy of the target_pixel as a numpy array, for computations
		target_pixel_np = np.array(target_pixel, dtype=int)

		# The array that will hold the computed features
		feature_array = np.empty((len(feature_list),))

		# Get the depth at the target pixel
		target_pixel_depth = image[target_pixel]

		# Go through each feature in the list
		for index, feature_offsets in enumerate(feature_list):

			# Get the offsets as np arrays and nomalize by the target pixel depth
			feature_first = np.array(feature_offsets[0]) / target_pixel_depth
			feature_second = np.array(feature_offsets[1]) / target_pixel_depth

			# Cast as ints to make sure indexing works correctly
			feature_first = feature_first.astype(int)
			feature_second = feature_second.astype(int)

			# Add the location of the target pixel
			feature_first += target_pixel
			feature_second += target_pixel

			# Transform the features into tuples, for the sake of indexing the image
			feature_first = (feature_first[0], feature_first[1])
			feature_second = (feature_second[0], feature_second[1])

			# DEBUG PRINT
			#print "\nAccessing locations:"
			#print feature_first, "    ", feature_second
			#print "Values at:"
			#print image[feature_first], "    ", image[feature_second]

			# Compute the feature as given by the equation
			feature_array[index] = self.depth_probe(image, feature_first) - self.depth_probe(image, feature_second)

		return feature_array

	# Saves the feature list for later loading
	# TODO: make this more portable, only saves pickles right now
	#
	# feature_list is a list of depth difference features
	#
	# save_name is a string that contains the full path and extension
	# Do not include extension
	def save_features(self, feature_list, save_name):

		# Pickle isn't portable, but it is easy
		pickle.dump(feature_list, open(save_name, 'wb'))

	# Loads a list of features suitable for use in depth difference computations
	# TODO: make this load a more portable format, going to be defined in save_features
	#
	# Returns a list of depth difference features
	#
	# load_name is the name of the features to load. Include path and extension 
	def load_features(self, load_name):

		# Only loads pickles right now
		return pickle.load(open(load_name, 'rb'))

	# Takes a batch of images and generates a training set by randomly sampling pixels
	#
	# returns feature_data, feature_labels. 
	# feature_data is a numpy array with calculated features with shape: (n_images * n_points_per_image, n_features)
	# feature_labels correspond with the data labels and have shape: (n_images * n_points_per_image,)
	#
	# image_batch is a numpy array of shape (n_images, height, width)
	# Each pixel has depth info
	#
	# label_batch are the corresponding labels with shape (n_images, height, width)
	# Each pixel has the labeled body part
	#
	# feature_list is the features to be computed, typically from random_feature_list
	#
	# n_points_per_image is the number of points that will be selected in each image
	#
	# remove_non_person is the chance that a pixel that is not from a person will be ignored
	# Since non-person pixels may be the majority in the set, this prevents them from being overwhelming
	def depth_difference_set(self, image_batch, label_batch, feature_list = None, n_points_per_image = 2000, ignore_non_person = .5, verbose = False):

		# Blank line for verbose printing
		if verbose:

			print ""

		# Get the basic shape info
		(n_images, height, width) = image_batch.shape

		# Create the processed data batch, as float32 to save space and make it run on the GPU
		feature_data = np.empty((n_images * n_points_per_image, len(feature_list)), dtype=np.float32)

		# Create the label batch
		feature_labels = np.empty((n_images * n_points_per_image), dtype=np.uint8)

		# Go through each image
		for image_index, image in enumerate(image_batch):

			# Verbose print the current image being worked on
			if verbose:

				print "\rWorking on batch index: " + str(image_index),
				sys.stdout.flush()

			# Generate the specified number of points per image
			for point_index in range(n_points_per_image):

				# Get a random point in the image
				target_pixel = (np.random.randint(height), np.random.randint(width))

				# If the point is not a person and the random number is greater than the ignore threshold, get a new point
				while label_batch[image_index][target_pixel] == 0 and np.random.rand() < ignore_non_person:

					# Get new random point
					target_pixel = (np.random.randint(height), np.random.randint(width))

				# Get the features from the point
				feature_data[image_index * n_points_per_image + point_index] = self.get_features(image, target_pixel, feature_list)

				# Set the label in the label batch
				feature_labels[image_index * n_points_per_image + point_index] = label_batch[image_index][target_pixel]

		# Put an endline, because previous print has none
		if verbose:

			print ""

		# Return the data and labels
		return feature_data, feature_labels

	# Takes a batch of images and returns the depth difference features for every pixel
	#
	# returns calculated features with the shape: (n_images, height, width, n_features)
	#
	# image_batch is a numpy array of shape (n_images, height, width)
	# Each pixel has depth info
	#
	# feature_list is the features to be computed, typically from random_feature_list
	#
	# Process 1 image by sending it with shape (1, height, width)
	def depth_difference_batch(self, image_batch, feature_list, verbose=False):

		# Verbose info
		if verbose:

			print ""

		# Get the basic shape info
		(n_images, height, width) = image_batch.shape

		# Create the new batch
		processed_images = np.empty(image_batch.shape + (len(feature_list),), dtype=np.float32)

		# Go through each image
		for image_index, image in enumerate(image_batch):

			# Go through each pixel in each image
			for height_index in range(height):
				for width_index in range(width):

					# Verbose progress update
					if verbose:

						print "\rImage index: ", image_index, " Height index: ", height_index, " Width index: ", width_index,
						sys.stdout.flush()

					# Set the target pixel value
					target_pixel = (height_index, width_index)

					# Get the features for the pixel and place them into the processed images
					processed_images[image_index][target_pixel] = self.get_features(image, target_pixel, feature_list)

		# Verbose finish line
		if verbose:

			print ""

		# Return the processed images
		return processed_images
		
	# Processes images from the source directory into depth difference features and saves them to the destination directory as pickles
	#
	# Returns feature_list, for future use
	#
	# feature_list is the set of offsets to use when computing features in get_features
	# Setting it to None will create a random list of features
	#
	# The depth features will be shape: (batch_size, height, width, feature_size)
	# The labels will be of shape: (batch_size, height, width)
	# TODO: fix the pickling issues
	# WARNING: Only call with batch_size = 1
	def process_depth_diff_pickles(self, source_dir, target_dir, start_index=None, end_index=None, batch_size=1, feature_list=None, verbose=False):
	
		# Make sure that the bounds are acceptable
		start_index, end_index, batch_size = self.set_bounds(source_dir, start_index, end_index, batch_size)
		
		# If no feature_list was sent, use the default random list
		if not feature_list:
		
			feature_list = self.random_feature_list()

		# Enforce path to the target directory
		enforce_path(target_dir)

		# Enforce path to the sub directories
		enforce_path(os.path.join(target_dir, "data"))
		enforce_path(os.path.join(target_dir, "label"))

		# Save the feature list
		self.save_features(feature_list, os.path.join(target_dir, "feature_list.p"))

		if verbose:
			print "Items in source directory: ", len(get_names(source_dir))

			print "Start index: ", start_index
			print "End index: ", end_index
			print "Batch size: ", batch_size
			print "Scale: ", self.scale_factor

		# Get the shape of the images
		# Need to get the header from any uncorrupted image
		size = None
		for name in get_names(source_dir, start_index=start_index, end_index=end_index):
			try:
				header = OpenEXR.InputFile(os.path.join(source_dir, name)).header()

			except:
				pass

			else:
				dw = header['dataWindow']
				size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

				break

		# Create a numpy array to hold each batch of raw depth data
		# shape (batch_size, height, width)
		data_batch = np.empty((batch_size, int(self.scale_factor * size[1]), int(self.scale_factor * size[0])))
		label_batch = np.empty((batch_size, int(self.scale_factor * size[1]), int(self.scale_factor * size[0])), dtype='uint8')

		# Create a numpy array to hold the depth difference features
		feature_batch = np.empty((batch_size, int(self.scale_factor * size[1]), int(self.scale_factor * size[0]), len(feature_list)))

		# Loop control
		done = False

		# Loop through all data, getting batches
		index = 0
		while not done and index < end_index:

			# Get the target end index
			target_index = index + batch_size - 1

			# Get the data and labels for the batch
			try:
				if verbose:
					print "Getting batch: ", str(index) + "_" + str(target_index)

				data_batch, label_batch = self.process_to_np(source_dir, index, target_index)

			# Index went past the end index
			except IndexError:
				done = True

			#except Exception, e:
			#	print e

			# There is a problem with one of the files, just ignore it
			except IOError:
				if verbose:
					print "File corrupt: ", str(index) + "_" + str(target_index)

			# The file load was successful
			else:

				# If the data has only 2 dims, it will need to be expanded
				while len(data_batch.shape) <= 2:
					data_batch = np.expand_dims(data_batch, axis=0)
			
				# Get the depth_features
				feature_batch = self.depth_difference_batch(data_batch, feature_list, verbose=verbose)

				# Fix the rgb data
				n_label_batch = self.batch_get_labels(label_batch)

				if verbose:
					print "Saving batch: " + str(index) + "_" + str(target_index)

				# Save the data_batch in 2 parts due to pickling limits
				divided = np.split(feature_batch, 2, axis=3)
				first_half = divided[0]
				second_half = divided[1]
				pickle.dump(first_half, open(os.path.join(target_dir, "data", str(index) + "_" + str(target_index) + "_0.p"), "wb"), protocol=2)
				pickle.dump(second_half, open(os.path.join(target_dir, "data", str(index) + "_" + str(target_index) + "_1.p"), "wb"), protocol=2)
				#pickle.dump(feature_batch, open(os.path.join(target_dir, "data", str(index) + "_" + str(target_index) + ".p"), "wb"), protocol=2)

				# Save the label batch
				pickle.dump(n_label_batch, open(os.path.join(target_dir, "label", str(index) + "_" + str(target_index) + ".p"), "wb"), protocol=2)

			finally:
				# Go to next batch
				index += batch_size

		return feature_list

	# Makes example images
	#
	# source_dir is the directory that includes the exr images
	#
	# target_dir is the directory for the new images, will be enforced
	#
	# feature_list is the features to be extracted from the image
	#
	# verbose controls output
	def make_ex_images(self, source_dir, target_dir, feature_list, start_index = None, end_index= None, verbose = False):

		# Verbose newline
		if verbose:

			print ""

		# Call with batch size of 1, to make each image stand alone
		self.process_depth_diff_pickles(source_dir, target_dir, start_index = start_index, end_index = end_index, batch_size = 1, feature_list = feature_list, verbose = verbose)

	# Processes the images from the source directory and places them into the target dir
	# Takes exr images and creates a png for the rgb data and a binary file for the depth
	# Within the target folder, creates sub folders for rgb and depth binary images
	# If start_index is specified, process will only process files that are higher than the sent value
	# If end_index is specified, process will only process files that are lower than the sent value
	def process_to_file(self, source_dir, target_dir, start_index=None, end_index=None):

		# Names for the subdirs in target dir
		rgb_dir = os.path.join(target_dir, "RGB")
		depth_dir = os.path.join(target_dir, "Depth")

		# Enforce path to target dir
		enforce_path(target_dir)

		# Enforce path to the subdirs
		enforce_path(rgb_dir)
		enforce_path(depth_dir)

		# Get the list of names to process
		to_process = get_names(source_dir, start_index, end_index)

		# Get a list of all files in the source dir and iterate through them
		for exr_name in to_process:

			# Get the name without the extension
			name = exr_name.strip(".exr")

			# Open the exr file
			exrfile = OpenEXR.InputFile(os.path.join(source_dir, exr_name))

			# Create the rgb copy
			self.save_to_rgb(exrfile, os.path.join(rgb_dir, name + ".png"))

			# Create the depth binary
			self.save_depth_binary(exrfile, os.path.join(depth_dir, name + ".bin"))

	# Processes the images from the source directory and returns two numpy arrays: RGB_data, Depth_data
	# RGB_data (n_images, 3, height, width) : the rgb components of the images
	# This data will have the functions 
	# Depth_data (n_images, height, width) : the depth information of the images
	# If start_index is specified, process will only process files that are higher than the sent value
	# If end_index is specified, process will only process files that are lower than the sent value
	# TODO: Fix the squeezing at the end
	def process_to_np(self, source_dir, start_index=None, end_index=None):

		# Get the list of names to process
		to_process = get_names(source_dir, start_index, end_index)

		# Get the shape of the images
		header = OpenEXR.InputFile(os.path.join(source_dir, to_process[0])).header()
		dw = header['dataWindow']
		size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

		# Create the numpy array for the RGB data
		RGB_data = np.empty((len(to_process), 3, int(self.scale_factor * size[1]), int(self.scale_factor * size[0])), dtype=np.uint8)

		# Create the numpy array for the Depth data
		Depth_data = np.empty((len(to_process), 1, int(self.scale_factor * size[1]), int(self.scale_factor * size[0])))

		# Get a list of all files in the source dir and iterate through them
		for index, exr_name in enumerate(to_process):

			# Get the name without the extension
			name = exr_name.strip(".exr")

			# Open the exr file
			exrfile = OpenEXR.InputFile(os.path.join(source_dir, exr_name))

			# Get the rgb data from the file
			RGB_data[index] = self.convert_to_rgb(exrfile)

			# Get the depth data from the file
			Depth_data[index] = self.get_channels(exrfile, "Z")

		# Remove single dimensional rows
		Depth_data = np.squeeze(Depth_data)

		# Return: RGB_data, Depth_data
		return Depth_data, RGB_data

	# Gets the data ready to process for the nn
	# Returns depth images and labels
	# Depth_image : (n_images, height, width)
	# Labels : (n_images, height, width)
	# If start_index is specified, process will only process files that are higher than the sent value
	# If end_index is specified, process will only process files that are lower than the sent value
	def process_to_ready(self, source_dir, start_index=None, end_index=None):

		# Get the data as numpy arrays
		Depth_data, RGB_data = self.process_to_np(source_dir, start_index, end_index)

		# Set the labels for the data
		labels = self.batch_get_labels(RGB_data)

		# Return the data
		return Depth_data, labels

	# Processes images and saves them into pickles
	# Data pickles are  shape (n_items, height, width)
	# Label pickles are shape (n_items, height, width)
	# n_items is equal to or less than the batch size.
	# n_items is typically a few items less than the batch size, due to invalid items that have been removed
	# If start_index is specified, process will only process files that are higher or equal than the sent value
	# If end_index is specified, process will only process files that are lower or equal than the sent value
	def process_to_pickle(self, source_dir, target_dir, start_index=None, end_index=None, batch_size=3, verbose=False):

		# If end_index is not sent, set it to the largest file in the set
		if not end_index:
			end_index = max([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))])
			end_index = int(end_index.strip(".exr"))

		# Make sure that the end index is aligned to the batch size
		end_index = (end_index // batch_size) * batch_size

		# If start_index is not sent, set it to the lowest file in the set
		if start_index:
			index = start_index
		else:
			index = int(min([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))]).strip(".exr"))

		# Enforce path to the target directory
		enforce_path(target_dir)

		# Enforce path to the sub directories
		enforce_path(os.path.join(target_dir, "data"))
		enforce_path(os.path.join(target_dir, "label"))

		if verbose:
			print "Items in source directory: ", len(get_names(source_dir))

			print "Start index: ", start_index
			print "End index: ", end_index
			print "Batch size: ", batch_size
			print "Scale: ", scale_factor

		# Get the shape of the images
		# Need to get the header from any uncorrupted image
		size = None
		for name in get_names(source_dir):
			try:
				header = OpenEXR.InputFile(os.path.join(source_dir, name)).header()

			except:
				pass

			else:
				dw = header['dataWindow']
				size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

				break

		# Create a numpy array to hold each batch
		# shape (batch_size, height, width)
		data_batch = np.empty((batch_size, int(scale_factor * size[1]), int(scale_factor * size[0])))
		label_batch = np.empty((batch_size, int(scale_factor * size[1]), int(scale_factor * size[0])), dtype='uint8')

		# Loop control
		done = False

		# Loop through all data, getting batches
		while not done and index < end_index:

			# Get the target end index
			target_index = index + batch_size - 1

			# Get the data and labels for the batch
			try:
				if verbose:
					print "Getting batch: ", str(index) + "_" + str(target_index)

				data_batch, label_batch = self.process_to_np(source_dir, index, target_index)

			# Index went past the end index
			except IndexError:
				done = True

			#except Exception, e:
			#	print e

			# There is a problem with one of the files, just ignore it
			except IOError:
				if verbose:
					print "File corrupt: ", str(index) + "_" + str(target_index)

			# The file load was successful
			else:

				# Fix the rgb data
				n_label_batch = self.batch_get_labels(label_batch)

				if verbose:
					print "Saving batch: " + str(index) + "_" + str(target_index)

				# Save the data_batch
				pickle.dump(data_batch, open(os.path.join(target_dir, "data", str(index) + "_" + str(target_index) + ".p"), "wb"))

				# Save the label batch
				pickle.dump(n_label_batch, open(os.path.join(target_dir, "label", str(index) + "_" + str(target_index) + ".p"), "wb"))

			finally:
				# Go to next batch
				index += batch_size

# Process all images from the source directory and place them into the target directory
# Enforces target directory
# Needs source and target directory names as commandline arguments
if __name__ == "__main__":

	# Check correct number of arguments
	if (len(sys.argv) < 2):
		# Not enough arguments, print usage
		print ("Usage: post_process.py source_dir target_dir [-s start_index -e end_index -c scale_factor -b batch_size]")
		print("Indices are inclusive")
		sys.exit(1)

	# Get the command line arguments
	source_dir = sys.argv[1]
	target_dir = sys.argv[2]

	# Get optional arguments if present
	arg_index = 3
	start_index = None
	end_index = None
	scale_factor = 1
	while arg_index < len(sys.argv):

		# Get the flag
		get_flag = sys.argv[arg_index]

		# Start index
		if get_flag == "-s":
			start_index = int(sys.argv[arg_index + 1])

			arg_index += 2

		# End index
		elif get_flag == "-e":
			end_index = int(sys.argv[arg_index + 1])

			arg_index += 2

		# Scale factor
		elif get_flag == "-c":
			scale_factor = int(sys.argv[arg_index + 1])

			arg_index += 2

		# Batch size
		elif get_flag == "-b":
			batch_size = int(sys.argv[arg_index + 1])

			arg_index += 2

		# Unknown
		else:

			print "Flag: ", get_flag, ", not known"

	# Process the images to pickles
	im_proc = Image_processing(scale_factor)
	im_proc.process_to_pickle(source_dir, target_dir, start_index=start_index, end_index=end_index, batch_size=batch_size, verbose=True)
