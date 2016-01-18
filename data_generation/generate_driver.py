# This is the driver that loads arguments for the image generator
# Runs the image generator

import time
import subprocess
import sys
import os

# All options, with defaults
# TODO: Fix the list weirdness
parameters = {"SAVE_PATH" : None, "GENERATE_NUMBER" : None, "DEBUG_OUTPUT" : ["FALSE"], "OFFSET" : ['0'], "TYPE" : ['1']}

# Required parameters
required = ["GENERATE_NUMBER", "SAVE_PATH"]

# Set the file path for program options
config_path = "configs/"

# Set the default configuration file name
config_file = "default.txt"

# Load the path to blender
with open("path.txt") as load_path:
	blender_path = (load_path.readline()).strip(" \n")
	
# Load the current directory
current_dir = os.getcwd()

# Check for alternate configuration file name sent as argument
if len(sys.argv) > 1:

	# Use the sent file instead
	config_file = sys.argv[1]

# Get the info from the config file
with open(config_path + config_file) as load_file:

	# Iterate through every line
	for line in load_file:
	
		# Ignore comments
		if(line[0] != '#'):
		
			# Get the command and the value
			contents = line.split()
			command = contents[0]
			contents = contents[1:]
			
			# Set the command
			try:
				parameters[command] = contents
				
			# Command was not valid
			except KeyError:
				print("Command: " + command + ", is not valid")
				
			# Unanticpated error
			except:
				print("Unanticpated error: ", sys.exc_info()[0])
				
# Check for required parameters
for check in required:

	# Check for unset parameter
	if parameters[check] == None:
	
		# Parameter was not set
		
		# Exit
		print("Required parameter: " + check + ", was not set")
		sys.exit(1)
		
# Display the settings
print("Running generator with the following parameters:")
for item in parameters:
	print(item + " ", parameters[item])

# Create the command
command_line_entry = blender_path + " -b " + "\"" + current_dir + "/blend_data/base_scene.blend\"" + " -P generate_image.py"
for item in parameters:
	#command_line_entry += [item] + parameters[item]
	command_line_entry += " " + item + " " + ' '.join(parameters[item])

# Check debug parameter
if parameters["DEBUG_OUTPUT"][0] == "FALSE":
	# Suppress the output of the image generator
	out = open(os.devnull, 'w')
	
else:
	# Do not redirect the output
	out = None

# Time the execution
start_time = time.time()

# Run the command
subprocess.call(command_line_entry, shell=True, stdout=out)

# Get total time
total_time = time.time() - start_time

# Close file if needed
if out != None:
	out.close()

# Show total time
print("\nProgram terminated\n")
print("Total time: ", total_time)
