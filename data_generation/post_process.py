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
import numpy

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

# Converts the exr image to a jpg
# From: http://excamera.com/articles/26/doc/intro.html#conversions
# exrfile is a file name to open or an opened exr file
# jpg_name is the name to save the jpg as. Include file path and extension
def convert_to_jpg(exrfile, jpg_name):

	# Check for file name or open file
	if(isinstance(exrfile, basestring)):
		# This is a string

		# Open the file
		exrfile = OpenEXR.InputFile(exrfile)

	# Set pixel info
	pt = Imath.PixelType(Imath.PixelType.FLOAT)
	dw = exrfile.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

	# Get the pixels
	rgbf = [Image.fromstring("F", size, exrfile.channel(c, pt)) for c in "RGB"]

	# Get the extremas of each channel
	extrema = [im.getextrema() for im in rgbf]
	darkest = min([lo for (lo,hi) in extrema])
	lighest = max([hi for (lo,hi) in extrema])

	# Normalize each channel to 0 - 255
	scale = 255 / (lighest - darkest)
	def normalize_0_255(v):
		return (v * scale) + darkest
	rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]

	# Save the RGB image as a jpg
	Image.merge("RGB", rgb8).save(jpg_name)

# Saves the depth channel as a binary array of floats
# Does not save dimension information, this must be known by whatever program loads the data
# exrfile is a file name to open or an opened exr file
# bin_name is the name to save the binary file as. Include file path and extension '.bin'
def save_depth_binary(exrfile, bin_name):

	# Check for file name or open file
	if(isinstance(exrfile, basestring)):
		# This is a string

		# Open the file
		exrfile = OpenEXR.InputFile(exrfile)

	# Set pixel info
	pt = Imath.PixelType(Imath.PixelType.FLOAT)
	dw = exrfile.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

	# Get the pixels
	zf = [Image.fromstring("F", size, exrfile.channel(c, pt)) for c in "Z"]

	# Convert to a numpy array
	# Roundabout method, but it works
	z_pix = numpy.array(zf[0])

	# Flatten and convert to a Python array
	z_copy = array('d', z_pix.flatten())

	# Open the target file in binary write mode
	with open(bin_name, 'wb') as out_file:

		# Save the file
		z_copy.tofile(out_file)

# Processes the images from the source directory and places them into the target dir
# Takes exr images and creates a jpg for the rgb data and a binary file for the depth
# Within the target folder, creates sub folders for rgb and depth binary images
# If start_index is specified, process will only process files that are higher than the sent value, assuming that 
def process(source_dir, target_dir, start_index=None):

	# Names for the subdirs in target dir
	rgb_dir = target_dir + "/RGB"
	depth_dir = target_dir + "/Depth"

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
		exrfile = OpenEXR.InputFile(source_dir + "/" + exr_name)

		# Create the rgb copy
		#convert_to_jpg(source_dir + "/" + exr_name, rgb_dir + "/" + name + ".jpg")
		convert_to_jpg(exrfile, rgb_dir + "/" + name + ".jpg")

		# Create the depth binary
		#save_depth_binary(source_dir + "/" + exr_name, depth_dir + "/" + name + ".bin")
		save_depth_binary(exrfile, depth_dir + "/" + name + ".bin")

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
