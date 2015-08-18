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

# For max pooling
# M height_pool width_pool ignore_border height_padding width_padding

# For unpooling:
# U height_stretch width_stretch

# Outputs the current info about the network
def display_info(layer_type, height, width, stacks):

	# Get a full description for the layer type
	# TODO: Make a dictionary out of this

	# Convolutional
	if layer_type == "C":

		expanded_layer_type = "Convolutional"

	# Max pooling
	elif layer_type == "M":

		expanded_layer_type = "Max Pooling"

	# Unpooling
	elif layer_type == "U":

		expanded_layer_type = "Unpooling"

	# Ending layer
	elif layer_type == "E":

		expanded_layer_type = "Ending"

	# Start layer
	elif layer_type == "S":

		expanded_layer_type = "Starting"

	# Not known
	else:

		expanded_layer_type = "Not Known"

	print "Layer: " + expanded_layer_type.ljust(15) + "\tHeight: " + str(height).ljust(5) + "\tWidth: " + str(width).ljust(5) + "\tStacks: " + str(stacks).ljust(5)

# Convolution transformations
# Info on shape changes from: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html
# All arguments except border_mode are ints
# Returns new height, width, stacks
def convolution(input_height, input_width, input_stacks, to_output_stacks, border_mode, height_window, width_window):

	# Change the height and width based on the border mode

	# Full mode
	if border_mode == "F":

		# Calculate the height
		output_height = input_height + height_window - 1

		# Calculate the width
		output_width = input_width + width_window - 1

	# Valid mode
	elif border_mode == "V":

		# Calculate the height
		output_height = input_height - height_window + 1

		# Calculate the width
		output_width = input_width - width_window + 1

	# Not know
	# Assume that the size stays the same
	else:

		output_height = input_height

		output_width = input_width

	# TODO: Check for expected input stack
	output_stacks = to_output_stacks

	# Return the new values
	return output_height, output_width, output_stacks

# Max pooling transformations
# Info from: http://deeplearning.net/software/theano/library/tensor/signal/downsample.html
# All arguments except ignore_border are ints
# Returns new height, width, stacks
def max_pool(input_height, input_width, input_stacks, ignore_border, height_stride, width_stride, height_padding, width_padding):

	# Change the height and width, based on the border mode

	# Border is being ignored
	if ignore_border == "T":

		# Use integer division to get the size
		output_height = input_height // height_stride

		output_width = input_width // width_stride

	# Border is not being ignored
	elif ignore_border == "F":

		# Use ceiling division to get the size
		output_height = -(-input_height // height_stride)

		output_width = -(-input_width // width_stride)

	# Wrong input
	# Assume that the shape stays the same, probably the wrong thing to do
	else:

		output_height = input_height

		output_width = input_width

	# Add the padding
	output_height = output_height + 2 * height_padding

	output_width = output_width + 2 * width_padding

	# stacks stay the same
	output_stacks = input_stacks

	# Return the new values
	return output_height, output_width, output_stacks

# Unpooling transformations
# All arguments are ints
# Returns new height, width, stacks
def unpool(input_height, input_width, input_stacks, height_stretch, width_stretch):

	# Multiply both height and width by their respective stretches
	output_height = height_stretch * input_height

	output_width = width_stretch * input_width

	# Stacks stay the same
	output_stacks = input_stacks

	# Return the new values
	return output_height, output_width, output_stacks

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

			# Change into ints
			original_height = int(original_height)
			original_width = int(original_width)
			original_stacks = int(original_stacks)

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

print "\n"
print "Transformation information:"
print "\n"

# Trace the stacks, height and width through the net
# Shows when the input stacks of a layer does not match the output of the last layer
# Shows the changes in height and width due to poolings, stretchings, and border_modes

# These variables track trough the layers, they are initialized to the original shape
height = original_height
width = original_width
stacks = original_stacks

# Show the start layer
display_info("S", original_height, original_width, original_stacks)

# Go through each layer
for layer_description in layers:

	# Get the type of layer
	layer_type = layer_description[0]

	# Switch based on the layer

	# Convolutional layer
	if layer_type == "C":

		# Get the data about the layer
		# C input_stacks output_stacks border_mode height_window width_window
		layer_info = layer_description.split()

		to_input_stacks = int(layer_info[1])
		to_output_stacks = int(layer_info[2])
		border_mode = layer_info[3]
		height_window = int(layer_info[4])
		width_window = int(layer_info[5])

		# Get the new values
		height, width, stacks = convolution(height, width, to_input_stacks, to_output_stacks, border_mode, height_window, width_window)

	# Max pooling layer
	elif layer_type == "M":

		# Get the data about the layer
		# M height_pool width_pool ignore_border height_padding width_padding
		layer_info = layer_description.split()

		height_pool = int(layer_info[1])
		width_pool = int(layer_info[2])
		ignore_border = layer_info[3]
		height_padding = int(layer_info[4])
		width_padding = int(layer_info[5])

		# Get the new values
		height, width, stacks = max_pool(height, width, stacks, ignore_border, height_pool, width_pool, height_padding, width_padding)

	# Unpooling layer
	elif layer_type == "U":

		# Get the data about the layer
		# U height_stretch width_stretch
		layer_info = layer_description.split()

		height_stretch = int(layer_info[1])
		width_stretch = int(layer_info[2])

		# Get the new values
		height, width, stacks = unpool(height, width, stacks, height_stretch, width_stretch)

	# Error input
	# Ignores this layer
	else:

		print "Layer type not known: ", layer_type

	# Output the updated info
	display_info(layer_type, height, width, stacks)

# Display some final info
print "\n"
print "Starting and ending dimensions: "
display_info("S", original_height, original_width, original_stacks)
display_info("E", height, width, stacks)
print "\n"
print "The flattened dimensions of the start and the end are:"
print "Flatten starting: ", original_height * original_width * original_stacks
print "Flatten ending:   ", height * width * stacks
print "Ratio:            ", height * width * stacks / float(original_height * original_width * original_stacks)
