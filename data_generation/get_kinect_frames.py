import freenect
import numpy as np
import Image
import h5py
import time
import sys

# Take kinect depth data and make it suitable for use in the network
# Converts from uint16 in millimeters to float32 in meters
def convert_to_base(image, threshold = 10):

	image = image.astype(np.float32) * .001

	image[image > threshold] = threshold

	print(np.max(image))
	print(np.min(image))

	return image

# Saves an image in a human viewable form, may not be suitable for later mathematical use
# Takes single or multi channel
# Single channel can be sent (height, width)
# Multi channel should be sent as (height, width, channels)
#
# save_name : string
# Include path and extension
def save_image_viewable(image, save_name, threshold=10):

	# Copy the image to avoid changing underlying data
	image = np.copy(image)

	# If this is a single channel image, normalize it
	if len(image.shape) == 2:

		# Threshold the image first, threshold is given in meters
		image[image > threshold * 1000] = threshold * 1000

		image = (image - np.min(image))*(255.0/np.max(image))

	# Get an Image
	image = Image.fromarray(image.astype(np.uint8))

	# Save
	image.save(save_name)

# Saves an image for future mathematical use
# Saves as an hdf5 file
# Depth will be converted to np.float32 at meter scale
# RGB will be left alone
#
# save_name : string
# Include path and extension
def save_image_sci(depth_image, rgb_image, timestamp, save_name):

	# Copy data
	depth_image = np.copy(depth_image)
	rgb_image = np.copy(depth_image)

	# Get the depth data to the correct scale
	depth_image = convert_to_base(depth_image)

	# Open an hdf5 file for saving
	with h5py.File(save_name, 'w') as h5_file:

		# Make the datasets for depth and rgb
		h5_file.create_dataset("depth", data=depth_image)
		h5_file.create_dataset("rgb", data=rgb_image)

		# Save the timestamp
		h5_file.create_dataset("timestamp", data=timestamp)

# Saves both components of the scene
def save_frame(depth_frame, rgb_frame, timestamp, save_handle):

	# Save depth and rgb to hdf5 for use with the neural net
	save_image_sci(depth_frame, rgb_frame, timestamp, save_handle + "_sci.hdf5")

	# Save for making pretty pictures
	save_image_viewable(depth_frame, save_handle + "_depth_view.jpg")
	save_image_viewable(rgb_frame, save_handle + "_rgb_view.jpg")

# Gets the rgb and depth data from the scene
def get_frame():

	# Get the depth
	depth_frame, depth_timestamp = freenect.sync_get_depth()

	# Get the color image
	rgb_frame, rgb_timestamp = freenect.sync_get_video()

	return depth_frame, rgb_frame, depth_timestamp 

# Gets frames from the kinect
#
# save_handle
# Prefix of every image
# Path must slready exist
#
# delay
# Seconds to wait before taking the first image
#
# number_to_take
# Number of images to take in a series
#
# interval
# Seconds to between taking images
def run(save_handle, delay = 0, number_to_take = 1, interval = 1):

	print "Delay:    " + str(delay)
	print "Number:   " + str(number_to_take)
	print "Interval: " + str(interval)

	# Wait for the initial delay
	time.sleep(delay)

	# Take the specified number of images
	for index_to_take in range(number_to_take):

		print "Taking image number: " + str(index_to_take)

		# Take the depth and the rgb
		depth_image, rgb_image, timestamp = get_frame()

		# Save both
		save_frame(depth_image, rgb_image, timestamp, save_handle + "_" + str(index_to_take).zfill(3))

		# Between images, wait for the specified interval, not needed on last image
		if index_to_take < number_to_take - 1:

			time.sleep(interval)

	print ""

# If this is run from the commandline, use user commands to take pictures
if __name__ == "__main__":

	# Get the save handle
	save_base = sys.argv[1]

	# Default parameters
	delay = 2
	number_to_take = 3
	interval = .5

	print "Default configuration"
	print "Save base: " + save_base
	print "Delay:     " + str(delay)
	print "Number:    " + str(number_to_take)
	print "Interval:  " + str(interval)

	# Continue until user selects quit
	quit_flag = False
	while not quit_flag:

		print "q to quit"
		print "Or enter a number to take that many images"

		command = raw_input("Enter command: ")

		# Quitting
		if command == 'q':

			quit_flag = True

			print "Quitting"

		# Getting images
		else:

			# Incase this isn't a number
			try:

				num = int(command)

			# Not an int
			except ValueError:

				print "That's not an int"


			# Good user, take pictures
			else:

				run(save_base, delay, num, interval)
