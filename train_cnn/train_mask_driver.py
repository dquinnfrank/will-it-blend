# Loads configuration info from a file, then runs train_mask.py
# File should have this structure, order doesn't matter:
#
# send_email [string]
# early_stop [int]
# structure_name [string]
# train_data_dir [string]
# pretrained_structure_name [string]
# pretrained_weight_name [string]
# save_name [string]
# epochs [int]
# batch_size [int]
# threshold [int]
#
# These are just for the driver and not the network and are optional:
# send_email: the address to send updates to
# early_stop: the maximum number of stale cycles to allow
#
# These are required:
#
# structure_name: the name within the structure models folder for the full net
# train_data_dir: the name of the directory that has the data to train on, must be processed into pickles by post_process.py
#
# These are optional:
# pretrained_structure_name: the name of the encoder layer in the structure_models folder
# pretrained_weight_name: the name of the weights to load into the pretrained layer
# save_name: the name to save the network weights as
# epochs: the number of epochs to run
# batch_size: the batch size to use
# threshold: the value to threshold the data to
# early_stop: if the network is not doing better after this many cycles, stop early. Keep the number of items in each data pickle in mind

import sys
import os
import smtplib
from email.mime.text import MIMEText
import traceback

# Import the full mask neural net
import train_mask

# Sends an email to about the program execution
# email always comes from auto.terminal.message@gmail.com
def send_status(send_to, subject, message, network_name=None):

	# The email to send from
	send_from = "auto.terminal.message@gmail.com"

	# The password of the sending email address
	send_password = "Unsecure"

	# Open connection
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login(send_from, send_password)

	# Create the message, starting with the network name
	send_message = "Computer name: " + os.uname()[1] + "\n"

	# Add the network name, if sent
	if network_name:

		send_message += "Network name: " + network_name + "\n"

	# Add the message
	send_message += "\nInfo:\n" + message

	# Set the body of the message
	msg = MIMEText(send_message)

	# Set the message meta
	msg['Subject'] = subject
	msg['From'] = send_from
	msg['To'] = send_to

	print "\nSending message to: ", send_to
	print "The message:\n"
	print msg.as_string()

	# Send the message
	server.sendmail(send_from, send_to, msg.as_string())

	# End the session
	server.quit()

# Default file to use, if no configurations have been sent
default_file = "configurations/train_mask.default.txt"

# Get the configuration file name
if len(sys.argv) > 1:

	load_file = sys.argv[1]

# Use default if no file has been sent
else:

	load_file = default_file

# These are mandatory arguments, if they are still None after file load, the file is misconfigured
structure_name = None
train_data_dir = None

# These are the default settings for optional arguments
pretrained_structure_name = None
pretrained_weight_name = None
save_name = None
epochs = 25
batch_size = 32
early_stop = None
send_email = None
early_stop = None
threshold = None

# Load the file and get the arguments
with open(load_file, "r") as config_file:

	# Go through each line in the file
	for line in config_file:

		# Get the info from the line
		(command, value) = line.split()

		# command tells which variable to set

		if command == "structure_name":

			structure_name = value

		elif command == "train_data_dir":

			train_data_dir = value

		elif command == "pretrained_structure_name":

			pretrained_structure_name = value

		elif command == "pretrained_weight_name":

			pretrained_weight_name = value

		elif command == "save_name":

			save_name = value

		elif command == "epochs":

			epochs = int(value)

		elif command == "batch_size":

			batch_size = int(value)

		elif command == "early_stop":

			early_stop = int(value)

		elif command == "send_email":

			send_email = value

		elif command == "early_stop":

			early_stop = int(value)

		elif command == "threshold":

			threshold = int(value)

		# Command was not known
		else:

			print "Command not known: ", command

# Check that the default settings have been set, exits if required settings are not set
if not structure_name or not train_data_dir:

	# Required settings are missing, exit
	print "Required settings missing:"
	if not structure_name:
		print "structure_name"
	if not train_data_dir:
		print "train_data_dir"

	sys.exit(1)

# Show the configuration
print "\nConfiguration"
print "Send email: ", send_email
print "Model structure: ", structure_name
print "Training data: ", train_data_dir
print "Pretrained structure name: ", pretrained_structure_name
print "Pretrained weight name: ", pretrained_weight_name
print "Save name: ", save_name
print "Epochs: ", epochs
print "Batch size: ", batch_size
print "Early stop: ", early_stop
print "\n"

# Create the network with the settings
mask_net_manage = train_mask.Mask(structure_name, encoder_layer_structure = pretrained_structure_name, encoder_layer_weight_name = pretrained_weight_name)

# Train will raise exceptions when problems occur
try:

	# Train the network
	loss = mask_net_manage.train_model(train_data_dir, save_name=save_name, epochs=epochs, batch_size=batch_size, data_threshold=threshold, stale_max=early_stop)

# This is the early stop exception catch
except UserWarning as warning_message:

	print "Early stop triggered:\n"
	print warning_message

	# Send the message as an email
	if send_email:

		send_status(send_email, "Early stop", "".join(traceback.format_exception(sys.exc_type, sys.exc_value, sys.exc_traceback)), network_name=save_name)

# This is for unexpected errors
except:

	print "Unexpected error occured:\n"

	traceback.print_exc()

	# Send the message as an email
	if send_email:

		send_status(send_email, "Unexpected error", "".join(traceback.format_exception(sys.exc_type, sys.exc_value, sys.exc_traceback)), network_name=save_name)

# Training completed without issue
else:

	# Send the all clear message, with the loss
	send_status(send_email, "Complete", "Loss: " + str(loss), network_name=save_name)
