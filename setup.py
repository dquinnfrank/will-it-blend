# Handles files that must be created or copied for the project to function
# Creates path.txt
# Copies all scripts in data_generation/blender_scripts to the blender modules folder

from shutil import copy
import sys
from os import listdir
from os.path import isfile, join

# No arguments means that usage must be shown
if len(sys.argv) < 2:

	print ("This program performs setup operations")
	print ("Blender will require additional one time setup, see README")
	print ("\nArguments:")
	print ("\n-p path_to_blender : set the path to blender")
	print ("This must be the full path to the blender executable")
	print ("\n-c : copy scripts to blender")
	print ("This copies any files in blender_scripts to the modules folder in blender")

# Get all of the arguments
else:

	# Start at the first argument
	index = 1

	# Go until all arguments have been processed
	while index < len(sys.argv):

		# Get the flag
		flag = sys.argv[index]

		# Set path flag
		if flag == "-p":

			# Create a file called data_generation/path.txt that contains the path to blender
			with open("data_generation/path.txt", "w") as write_path:

				write_path.write(sys.argv[index + 1])

			print ("Path file created")

			# Move index to next flag
			index += 2

		# Copy all of the scripts
		elif flag == "-c":

			# Get the path to blender
			try:
				get_path = open("data_generation/path.txt", 'r')
				path = (get_path.readline()).strip("blender")
				get_path.close()

			# Path is not set right now
			except:
				print ("Path to blender must be set before this operation can be performed")

			# Copy all files in data_generation/blender_scripts to [path_to_blender folder]/2.72/scripts/modules/
			else:

				copy_from = "data_generation/blender_scripts"

				for item in [ f for f in listdir(copy_from) if isfile(join(copy_from,f)) ]:
					copy(copy_from + '/' + item, path + "2.72/scripts/modules/")

				print ("Scripts copied")

			# Move index
			index += 1

		# Not known
		else:

			print ("Argument: " + flag + " not known")
