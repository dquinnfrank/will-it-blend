# Handles files that must be created or copied for the project to function
# Creates path.txt
# Copies all scripts in data_generation/blender_scripts to the blender modules folder
# Runs the OpenEXR setup

from shutil import copy
import sys
from os import listdir
from os.path import isfile, join
import subprocess

# No arguments means that usage must be shown
if len(sys.argv) < 2:

	print ("This program performs setup operations")
	print ("Blender will require additional one time setup, see README")
	print ("\nArguments:")
	print ("\n-p path_to_blender : set the path to blender")
	print ("This must be the full path to the blender executable")
	print ("\n-c : copy scripts to blender")
	print ("\n-e : Runs the OpenEXR setup script")
	print ("This copies any files in blender_scripts to the modules folder in blender")

# Get all of the arguments
else:

	# Set the path to the path config file
	path_filename = join("data_generation", "path.txt")

	# Start at the first argument
	index = 1

	# Go until all arguments have been processed
	while index < len(sys.argv):

		# Get the flag
		flag = sys.argv[index]

		# Set path flag
		if flag == "-p":

			# Create a file called data_generation/path.txt that contains the path to blender
			with open(path_filename, "w") as write_path:

				write_path.write(sys.argv[index + 1])

			print ("Path file created")

			# Move index to next flag
			index += 2

		# Copy all of the scripts
		elif flag == "-c":

			# Get the path to blender
			try:
				get_path = open(path_filename, 'r')
				path = (get_path.readline()).strip("blender")
				get_path.close()

			# Path is not set right now
			except:
				print ("Path to blender must be set before this operation can be performed")

			# Copy all files in data_generation/blender_scripts to [path_to_blender folder]/2.72/scripts/modules/
			else:

				copy_from = join("data_generation", "blender_scripts")

				for item in [ f for f in listdir(copy_from) if isfile(join(copy_from,f)) ]:
					copy(join(copy_from, item), join(path, "2.72", "scripts", "modules"))

				print ("Scripts copied")

			# Move index
			index += 1

		# Runs the OpenEXR script
		elif flag == "-e":

			# Try to run the OpenEXR script
			try:

				subprocess.call("python setup.py install", cwd=join("data_generation", "OpenEXR-1.2.0"), shell=True)

			# Something is wrong with the OpenEXR tools
			except:

				print("OpenEXR setup could not be run.")
				print("Make sure that OpenEXR-1.2 is in the data_generation folder.")

			# Setup worked
			else:

				print("OpenEXR setup run. Check output to make sure install was successful.")

			# Move index
			index += 1
				
		# Not known
		else:

			print ("Argument: " + flag + " not known")
