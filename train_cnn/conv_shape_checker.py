# Shows you what the shape of output images will be as convolutions are done
# Easily catch errors before putting everything into keras model
# Needs file that describes the net

import sys

# Format for describing the net

# Specify the original height, width, and stacks
# H W S

# For convolutions:
# C input_stacks output_stacks border_mode height_window width_window
# border_mode is:
# F for full
# V for valid

# For max pooling:
# M height_pool width_pool

# For unpooling:
# U height_stretch width_stretch

# Convolution transformations
# All arguments except border_mode are ints
# Returns new height, width, stacks
def convolutional(input_height, intput_width, input_stacks, border_mode, height_window, width_window):

# Get the file name
file_name = sys.argv[1]

# Open the file
with open(file_name) as net_file:

	# This list holds the layers of the net
	layers = []

	# Go through every line in the file
	first = True
	for line in net_file:

		# If this is the first line, get the configuration
		if first:

			first = False

			# Get the height, width, and stacks of the original images
			(original_height, original_width, original_stacks) = line.split()

		# This is a layer in the net
		else:

			# Save the line in the layers list
			layers.append(line.strip("\n"))

print "\n"
print "Original height: ", original_height
print "Original width: ", original_width
print "Original stacks: ", original_stacks

print "\n"
print "Layers:"
for item in layers:
	print item

# Trace the stacks, height and width through the net
# Shows when the input stacks of a layer does not match the output of the last layer
# Shows the changes in height and width due to poolings, stretchings, and border_modes

# These variables track trough the layers, they are initialized to the original shape
height = original_height
width = original_width
stacks = original_stacks

# Go through each layer
for layer_description in layers:

	# Get the type of layer
	layer_type = layer_description[0]

	# Switch based on the layer

	# Convolutional layer
	if layer_type == "C":

		pass

	# Max pooling layer
	elif layer_type == "M":

		pass

	# Unpooling layer
	elif layer_type == "U":

		pass

	# Idiot trap
	# Ignores this layer
	else:

		print "Layer type not known: ", layer_type
