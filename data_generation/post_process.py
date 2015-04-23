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

# Enforces file path
def enforce_path(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# Normalizes each channel to be about 0 - 255
# Changes type to int
# to_normalize is the image to be normalized, it must be a numpy array of shape channels * height * width
# WARNING: image is not ready to save due to the reasons below
# OpenEXR images generally have 1 as the highest value, but sometimes the value can be higher
# This function assumes that the lowest value will always be 0.0
# The highest value is assumed to be at least 1.0, this is because if an image doesn't show the whole person, the high could be incorrectly very low
def normalize_image(to_normalize):

	# Get the highest and lowest values in the whole image
	# Assume that we always need to normalize each channel by the same amount
	highest_val = np.max(np.max((to_normalize), 1.0))

	# Set the scale
	scale = 255.0 / (highest_val)

	# Normalize the image
	normalized = to_normalize * scale

	return normalized.astype(np.uint8)

# Assigns a label to each pixel based on the RGB values
def get_labels(rgb_image):

	# Create a new image
	new_image = np.empty((rgb_image.shape[1], rgb_image.shape[2]), dtype=np.uint8)

	# For each pixel, use the label dictionary to assign a label
	for h in range(rgb_image.shape[1]):
		for w in range(rgb_image.shape[2]):

			# Try to assign a label
			try:
				new_image[h][w] = pix_to_label[" ".join([str(rgb_image[0][h][w]), str(rgb_image[1][h][w]), str(rgb_image[2][h][w])])]

			# No known label, assign no class
			except KeyError:
				new_image[h][w] = 0

	return new_image

# Assigns the closest class to each pixel
# Takes an array and returns a numpy array with the same shape and dtype
def assign_closest(data, possible_classes=[0, 50, 255]):

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
def convert_to_rgb(exrfile):

	# Get the image from the exr file
	rgb_image = get_channels(exrfile, "RGB")

	# Normalize the image
	rgb_image = normalize_image(rgb_image)

	# Fix any noisy pixels
	rgb_image = assign_closest(rgb_image)

	# Return the image
	return rgb_image

# Converts ands saves an exr image to a unit8 image
# Will assign each pixel to the closest class
# Wrapper for get_channels, normalize_image, assign_closest, save_image
# exrfile is an exrfile, opened or name
# rgb_name is the name to save the image as. Type will be inferred from extension
def save_to_rgb(exrfile, rgb_name):

	# Get the RGB image
	rgb_image = convert_to_rgb(exrfile)

	# Save the image
	save_image(rgb_image, rgb_name)

# Gets the specified channel information from an exr file and places it into a numpy array of floats
# For RGB, channels = "RGB"
# For Depth, channels = "Z"
# shape : (channels, height, width)
def get_channels(exrfile, channels):

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
	np_pix = np.empty((len(pix_float), size[1], size[0]))

	# Place the pixel data into the numpy array
	for index in range(len(pix_float)):
		np_pix[index] = np.array(pix_float[index])

	# Return the numpy array
	return np_pix

# Saves the given data as a binary array
# Data will be flattened before saving
# Whatever program loads the data must know the data type and size
# to_save is a numpy array to be saved
# bin_name is the name to save the binary file as. Include file path and extension '.bin'
# data_type is the python array type to save the data as
# default is double: 'd'
def save_binary(to_save, bin_name, data_type='d'):

	# Flatten and convert to a Python array
	copy = array(data_type, to_save.flatten())

	# Open the target file in binary write mode
	with open(bin_name, 'wb') as out_file:

		# Save the file
		(to_save.flatten()).tofile(out_file)

# Takes the depth channel from the exr image and saves it as a binary file
# This is a wrapper for get_channel and save_binary
# exrfile is an opened exr file or a file name
# save_name is the name to save the output as
def save_depth_binary(exrfile, save_name):

	# Get the depth channel
	z_channel = get_channels(exrfile, "Z")

	# Save the depth channel
	save_binary(z_channel, save_name)

# Takes a numpy array and saves it as an image
# to_save is the image to be saved, must have shape: channels * width * height
def save_image(to_save, save_name):

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

# Gets a list of files based on the start and end indices
def get_names(source_dir, start_index=None, end_index=None, randomize=False):

	# Get the names of every file
	to_process = sorted([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f)) ])

	# Only include items after the start threshold
	if start_index:
		to_process = sorted([v for v in to_process if int(v.strip(".exr")) >= start_index ])

	# Only include items before the end threshold
	if end_index:
		to_process = sorted([v for v in to_process if int(v.strip(".exr")) <= end_index ])

	# Randomize names, if selected
	if randomize:
		shuffle(to_process)

	return to_process

# Processes the images from the source directory and places them into the target dir
# Takes exr images and creates a png for the rgb data and a binary file for the depth
# Within the target folder, creates sub folders for rgb and depth binary images
# If start_index is specified, process will only process files that are higher than the sent value
# If end_index is specified, process will only process files that are lower than the sent value
def process_to_file(source_dir, target_dir, start_index=None, end_index=None):

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
		save_to_rgb(exrfile, os.path.join(rgb_dir, name + ".png"))

		# Create the depth binary
		save_depth_binary(exrfile, os.path.join(depth_dir, name + ".bin"))

# Processes the images from the source directory and returns two numpy arrays: RGB_data, Depth_data
# RGB_data (n_images, 3, height, width) : the rgb components of the images
# This data will have the functions 
# Depth_data (n_images, height, width) : the depth information of the images
# If start_index is specified, process will only process files that are higher than the sent value
# If end_index is specified, process will only process files that are lower than the sent value
def process_to_np(source_dir, start_index=None, end_index=None):

	# Get the list of names to process
	to_process = get_names(source_dir, start_index, end_index)

	# Get the shape of the images
	header = OpenEXR.InputFile(os.path.join(source_dir, to_process[0])).header()
	dw = header['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

	# Create the numpy array for the RGB data
	RGB_data = np.empty((len(to_process), 3, size[1], size[0]), dtype=np.uint8)

	# Create the numpy array for the Depth data
	Depth_data = np.empty((len(to_process), 1, size[1], size[0]))

	# Get a list of all files in the source dir and iterate through them
	for index, exr_name in enumerate(to_process):

		# Get the name without the extension
		name = exr_name.strip(".exr")

		# Open the exr file
		exrfile = OpenEXR.InputFile(os.path.join(source_dir, exr_name))

		# Get the rgb data from the file
		RGB_data[index] = convert_to_rgb(exrfile)

		# Get the depth data from the file
		Depth_data[index] = get_channels(exrfile, "Z")

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
def process_to_ready(source_dir, start_index=None, end_index=None):

	# Get the data as numpy arrays
	Depth_data, RGB_data = process_to_np(source_dir, start_index, end_index)

	# Set the labels for the data
	labels = np.empty((RGB_data.shape[0], RGB_data.shape[2], RGB_data.shape[3]), dtype=np.uint8)
	for index in range(len(labels)):
		labels[index] = get_labels(RGB_data[index])

	# Return the data
	return Depth_data, labels

# Processes images and saves them into pickles
# Data pickles are  shape (batch_size, height, width)
# Label pickles are shape (batch_size, height, width)
# If start_index is specified, process will only process files that are higher or equal than the sent value
# If end_index is specified, process will only process files that are lower or equal than the sent value
def process_to_pickle(source_dir, target_dir, start_index=None, end_index=None, batch_size=3):

	# If end_index is not sent, set it to the largest file in the set
	if not end_index:
		end_index = max([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))])
		end_index = int(end_index.strip(".exr"))

	# Make sure that the end index is aligned to the batch size
	end_index = (end_index // batch_size) * batch_size

	# Loop control
	done = False
	if start_index:
		index = start_index
	else:
		index = int(min([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))]).strip(".exr"))

	# Enforce path to the target directory
	enforce_path(target_dir)

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
	data_batch = np.empty((batch_size, size[1], size[0]))
	label_batch = np.empty((batch_size, size[1], size[0]), dtype='uint8')

	# Loop through all data, getting batches
	while not done and index < end_index:

		# Get the target end index
		target_index = index + batch_size - 1

		# Get the data and labels for the batch
		try:
			data_batch, label_batch = process_to_np(source_dir, index, target_index)

		# Index went past the end index
		except IndexError:
			done = True

		# There is a problem with one of the files, just ignore it
		except IOError:
			pass

		# The file load was successful
		else:
			# Save the data_batch
			pickle.dump(data_batch, open(os.path.join(target_dir, str(index) + "_" + str(target_index) + "_data.p"), "wb"))

			# Save the label batch
			pickle.dump(data_batch, open(os.path.join(target_dir, str(index) + "_" + str(target_index) + "_label.p"), "wb"))

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
		print ("Usage: post_process.py source_dir target_dir [-s start_index -e end_index]")
		print("Indices are inclusive")
		sys.exit(1)

	# Get the command line arguments
	source_dir = sys.argv[1]
	target_dir = sys.argv[2]

	start_index = None
	end_index = None
	# Get optional argument, if present
	if(len(sys.argv) > 4):
		if sys.argv[3] == "-s":
			start_index = int(sys.argv[4])
		elif sys.argv[3] == "-e":
			end_index = int(sys.argv[4])

	if(len(sys.argv) > 6):
		if sys.argv[5] == "-s":
			start_index = int(sys.argv[6])
		elif sys.argv[5] == "-e":
			end_index = int(sys.argv[6])

	# Process the images
	process_to_file(source_dir, target_dir, start_index=start_index, end_index=end_index)
