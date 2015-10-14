# Loads the image pickles and makes them into a hdf5 object

import sys
import os
import cPickle as pickle

import h5py
import numpy as np

# Get the name of the source dir
source_dir = sys.argv[1]

# Get the name of the file to be created
destination_name = sys.argv[2]

# For progress printing
print ""

# Create the file
with h5py.File(destination_name, 'w') as h5_file:

	# Create the dataset
	data_set = h5_file.create_dataset("data", (0, 2000), maxshape=(None, 2000))

	# Create the label set
	label_set = h5_file.create_dataset("label", (0,), maxshape=(None,))

	# Go through each item in the source directory
	check_dir = os.path.join(source_dir, "data")
	for file_name in sorted([ f for f in os.listdir(check_dir) if os.path.isfile(os.path.join(check_dir,f)) ]):

		# Get the full path and name
		load_data = os.path.join(source_dir, "data", file_name)
		load_label = os.path.join(source_dir, "label", file_name)

		# Load the data pickle
		data_batch = pickle.load(open(load_data, 'rb'))

		# Load the label pickle
		label_batch = pickle.load(open(load_label, 'rb'))

		# Go through each image in the batch
		for (data_plane, label_plane) in zip(data_batch, label_batch):

			# Increase the size of the sets
			data_set.resize(len(data_set) + 1, axis = 0)
			label_set.resize(len(label_set) + 1, axis = 0)

			# Add the data to the data_set
			data_set[-1] = np.copy(data_plane)

			# Add the label to the label_set
			label_set[-1] = np.copy(label_plane)

			print "\rPickle: ", file_name, " Total set index: ", data_set.shape[0],
			sys.stdout.flush()
