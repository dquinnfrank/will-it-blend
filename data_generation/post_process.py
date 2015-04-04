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

# Converts an exr image to a png
# Will assign each pixel to the closest class
# Wrapper for get_channels, normalize_image, assign_closest, save_image
# exrfile is an exrfile, opened or name
# rgb_name is the name to save the image as. Type will be inferred from extension
def convert_to_rgb(exrfile, rgb_name):

	# Get the image from the exr file
	rgb_image = get_channels(exrfile, "RGB")
s
def save_depth_binary(exrfile, save_name):
	# Normalize the image
	rgb_image = normalize_image(rgb_image)

	# Fix any noisy pixels
	rgb_image = assign_closest(rgb_image)

	# Save the image
	save_image(rgb_image, rgb_name)

# Gets the specified channel information from an exr file and places it into a numpy array of floats
# For RGB, channels = "RGB"
# For Depth, channels = "Z"
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

# Processes the images from the source directory and places them into the target dir
# Takes exr images and creates a jpg for the rgb data and a binary file for the depth
# Within the target folder, creates sub folders for rgb and depth binary images
# If start_index is specified, process will only process files that are higher than the sent value, assuming that 
def process(source_dir, target_dir, start_index=None):

	# Names for the subdirs in target dir
	rgb_dir = os.path.join(target_dir, "RGB")
	depth_dir = os.path.join(target_dir, "Depth")

	# Enforce path to target dir
	enforce_path(target_dir)

	# Enforce path to the subdirs
	enforce_path(rgb_dir)
	enforce_path(depth_dir)

	# If start_index is None, process all items
	if start_index == None:
		to_process = sorted([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f)) ])

	# Otherwise, only process the items after, not including, the specified start_index
	else:
		all_names = sorted([ f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f)) ])

		# Remove items before the threshold
		to_process = sorted([v for v in all_names if int(v.strip(".exr")) > start_index ])

	# Get a list of all files in the source dir and iterate through them
	for exr_name in to_process:

		# Get the name without the extension
		name = exr_name.strip(".exr")

		# Open the exr file
		exrfile = OpenEXR.InputFile(os.path.join(source_dir, exr_name))

		# Create the rgb copy
		convert_to_rgb(exrfile, os.path.join(rgb_dir, name + ".png"))

		# Create the depth binary
		save_depth_binary(exrfile, os.path.join(depth_dir, name + ".bin"))

# Process all images from the source directory and place them into the target directory
# Enforces target directory
# Needs source and target directory names as commandline arguments
if __name__ == "__main__":

	# Check correct number of arguments
	if (len(sys.argv) < 2):
		# Not enough arguments, print usage
		print "Usage: post_process.py source_dir target_dir [start index]"
		sys.exit(1)

	# Get the command line arguments
	source_dir = sys.argv[1]
	target_dir = sys.argv[2]

	start_index = None
	# Get optional argument, if present
	if(len(sys.argv) > 3):
		start_index = int(sys.argv[3])

	# Process the images
	process(source_dir, target_dir, start_index=start_index)
