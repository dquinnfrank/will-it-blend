# Import the human class
from scene_manager import Human
from scene_manager import save_image
from scene_manager import random_occulsion

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

# Command line arguments

# Type 0 is an easy data set that has the person facing the camera and no occulsion
# Type 1 has random orientation and occulsions

# Set all of the flags
save_path = None		# REQUIRED
number_of_images = None		# REQUIRED
debug_flag = False		# OPTIONAL
offset = 0			# OPTIONAL
data_set_type = 0		# OPTIONAL

# Parse all of the arguments
for index, flag in enumerate(sys.argv):

	# Save path, following argument is the save path
	if flag == "SAVE_PATH":
		save_path = sys.argv[index+1]

	# Number to generate, following argument is the number of images to generate
	if flag == "GENERATE_NUMBER":
		number_of_images = int(sys.argv[index+1])

	# Debug outputs
	if flag == "DEBUG_OUTPUT":
		if sys.argv[index+1] == "TRUE":
			debug_flag = True

	# Offset, number following this flag is where numbering should start
	if flag == "OFFSET":
		offset = int(sys.argv[index+1])

	# Type of data set to create, following number gives the data set code
	if flag == "TYPE":
		data_set_type = int(sys.argv[index+1])

# Check for required arguments, if any are missing, quit
if(save_path is None or number_of_images is None):
	print("Arguments missing")
	sys.exit(1)

# Enforce the save path
enforce_path(save_path)

# Enforce vert save path
enforce_path(save_path.rstrip("/") + "_verts/")

# Set the path where data should be loaded from
path = os.getcwd() + "/blend_data/"

# Create a human
print("Creating a person")
person = Human(path, "neutral_person", debug_flag)
print("Person created")

# Generate the specified number of images
for image_index in range(offset, offset + number_of_images):

	# Create a random pose
	person.random_pose()

	# For the harder data set, add rotation to the person and a simple occulsion
	if data_set_type == 1:

		# Create a random rotation
		person.random_rotation()

		# Add occulsion
		random_occulsion(debug_flag)

	# For the next hardest data set, use clutter objects
	if data_set_type == 2:

		pass

	# Save the key vertices
	person.save_key_verts(save_path.rstrip("/") + "_verts/" + str(image_index).zfill(12))

	# Save the image
	save_image(save_path + str(image_index).zfill(12), debug_flag)
