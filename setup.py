# Handles files that must be created or copied for the project to function
# Creates path.txt
# Copies all scripts in data_generation/blender_scripts to the blender modules folder
# Runs the OpenEXR setup

from shutil import copy
from shutil import rmtree
import sys
from os import listdir
from os.path import isfile, join
import subprocess
import platform

# Set the path to the path config file
path_filename = join("data_generation", "path.txt")

# Writes the path file
# Takes the path to blender
def write_path(path_to):

	# Create a file called data_generation/path.txt that contains the path to blender
	with open(path_filename, "w") as write_path:

		write_path.write(path_to)

	print ("\nPath file created")

# Copy all scripts to the blender modules folder
def copy_scripts():

	# Get the path to blender
	try:
		get_path = open(path_filename, 'r')
		path = (get_path.readline()).strip("blender")
		get_path.close()

	# Path is not set right now
	except:
		print ("\nPath to blender must be set before this operation can be performed")

	# Copy all files in data_generation/blender_scripts to [path_to_blender folder]/2.72/scripts/modules/
	else:

		copy_from = join("data_generation", "blender_scripts")

		for item in [ f for f in listdir(copy_from) if isfile(join(copy_from,f)) ]:
			copy(join(copy_from, item), join(path, "2.72", "scripts", "modules"))

		print ("\nScripts copied")

# Prints instructions on how to install the OpenEXR bindings
# The installation has been moved ouside of the install script since it is dependent on the system configuration
def install_OpenEXR():

	# Show link to OpenEXR bindings
	print("\nInformation about OpenEXR bindings for python can be found at:")
	print("http://www.excamera.com/sphinx/articles-openexr.html")
	print("\nTo install prerequisites on Debian-based Linux run (you will need write permissions):")
	print("apt-get install libopenexr-dev")
	print("\nTo install the python bindings run (you will need write permissions):")
	print("pip install openexr")

	"""
	# Try to run the OpenEXR script
	try:
		print("")
		status = subprocess.call("python setup.py install", cwd=join("data_generation", "OpenEXR-1.2.0"), shell=True)

	# Something is wrong with the OpenEXR tools
	except:

		print("\nOpenEXR setup could not be run.")
		print("Make sure that OpenEXR-1.2.0 is in the data_generation folder.")

	# Setup worked
	else:

		# Delete the temporary build folder, if present
		try:
			rmtree('build')

		except:
			pass

		# Get the status, 0 means successful
		if status == 0:
			print("\nOpenEXR setup run. Check output to make sure install was successful.")

		else:
			print("\nOpenEXR setup did not complete.") 
			print("You may need to install the OpenEXR C++ library. On Linux run: ")
			print("sudo apt-get install libopenexr-dev")
	"""

# No arguments means that usage must be shown
if len(sys.argv) < 2:

	print ("This program performs setup operations")
	print ("Blender will require additional one time setup, see README")
	print ("\nArguments:")
	print ("\n-a path_to_blender : runs all setup actions")
	print ("path_to_blender must be the full path to the blender executable")
	print ("\n-p path_to_blender : set the path to blender")
	print ("path_to_blender must be the full path to the blender executable")
	print ("\n-c : copy scripts to blender")
	print ("This copies any files in blender_scripts to the modules folder in blender")
	#print ("\n-e : Runs the OpenEXR setup script")
	print ("\n-e : Shows information to install the OpenEXR bindings for python")
	print ("")

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

			# Get the sent path
			path_to = sys.argv[index + 1]

			# Make the path file
			write_path(path_to)

			# Move index to next flag
			index += 2

		# Copy all of the scripts
		elif flag == "-c":

			# Copy the scripts
			copy_scripts()

			# Move index
			index += 1

		# Runs the OpenEXR script
		elif flag == "-e":

			# Install OpenEXR
			install_OpenEXR()

			# Move index
			index += 1

		# Runs all setup actions
		elif flag == "-a":

			# Get the sent path
			path_to = sys.argv[index + 1]

			# Make the path file
			write_path(path_to)

			# Copy the scripts
			copy_scripts()

			# Install OpenEXR
			install_OpenEXR()

			# Leave the loop
			break
		
		# Runs the setup actions for ubuntu, installs all prerequistes
		elif flag == "-u":
			
			# Install all apt-get things
			subprocess.call("sudo apt-get install build-essential cmake git libhdf5-serial-dev hdf5-tools python-pip python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose freenect python-freenect")
			
			# Install all pip things
			subprocess.call("sudo pip install future h5py")
			
			# Install torch
			subprocess.call("git clone https://github.com/torch/distro.git ~/torch --recursive; cd ~/torch; bash install-deps; ./install.sh")
			
			# Update torch
			subprocess.call("luarocks install nn cunn cutorch")
			
			# Get hdf5 for torch
			subprocess.call("git clone git@github.com:deepmind/torch-hdf5.git ~/torch-hdf5.git; cd ~/torch-hdf5; luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR='/usr/lib/x86_64-linux-gnu/'")
			
			# Get lutorpy
			subprocess.call("git clone https://github.com/imodpasteur/lutorpy.git; cd lutorpy; sudo python setup.py install")
			
			# Set the path to this directory
			subprocess.call("echo 'export PD_ROOT=`pwd`' >> ~/.bashrc; source ~/.bashrc")
			
		# Not known
		else:

			print ("\nArgument: " + flag + " not known")
