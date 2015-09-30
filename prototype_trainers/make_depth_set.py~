# Takes processed image pickles and makes random depth features out of them

import sys

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp

import numpy as np
import cPickle as pickle

import os

# Get the name of this set
set_name = sys.argv[1]

# Get the name of the source directory
# This directory should contain sub directories: data, label
source_root_dir = sys.argv[2]

# Get the max size of the data set
# Processing will end after this many items have been added
# Set really high to process all images
#max_items = sys.argv[3]

# Set the data and label directories
data_dir = os.path.join(source_root_dir, "data")
label_dir = os.path.join(source_root_dir, "label")

# Enforce the destination directories
pp.enforce_path(set_name)
pp.enforce_path(os.path.join(set_name, "data"))
pp.enforce_path(os.path.join(set_name, "label"))

# Create the post_process object
im_p = pp.Image_processing()

# Get a list of items in the data directory
image_pickle_names = pp.get_names(data_dir)

# Get a random list of features
rand_features = im_p.random_feature_list()

# Save the features for future use
im_p.save_features(rand_features, set_name + "feature_list.p")

# Get the batch size of the images
image_range = image_pickle_names[0].strip(".p").split("_")
source_batch_size = int(image_range[1]) - int(image_range[0]) + 1

# Get the total number of examples that will be created
# 2000 pixels will be choosen from each image
#total_examples = len(image_pickle_names) * source_batch_size * 2000
#print len(image_pickle_names) * source_batch_size
#print total_examples
#print (total_examples * 2000 * 32) / (8 * 10**9)

# Check to see if this is past the max items
#if total_examples > max_items:

# Create a numpy array to hold the specified amount of images and labels
#data_set = np.empty((total_examples, len(rand_features)), dtype=np.float32)
#label_set = np.empty((total_examples,), dtype=np.uint8)

# Load pickles and process them
running_index = 0
for item_name in image_pickle_names:

	print "Loading pickle: ", item_name

	# Load the data and the labels
	data_batch = pickle.load(open(os.path.join(data_dir, item_name), 'rb'))
	label_batch = pickle.load(open(os.path.join(label_dir, item_name), 'rb'))

	# Get the processed batches
	data_processed, label_processed = im_p.depth_difference_set(data_batch, label_batch, rand_features, verbose = True)

	# Save the set
	pickle.dump(data_processed, open(os.path.join(set_name, "data", str(running_index).zfill(5) + "_data.p"), 'wb'), protocol=2)
	pickle.dump(label_processed, open(os.path.join(set_name, "label", str(running_index).zfill(5) + "_label.p"), 'wb'), protocol=2)

	# Increment running index
	running_index += 1
