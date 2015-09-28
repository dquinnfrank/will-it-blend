# Takes processed image pickles and makes random depth features out of them

import post_process as pp

import numpy as np
import cPickle as pickle

import sys
import os

# Get the name of this set
set_name = sys.argv[1]

# Get the name of the source directory
# This directory should contain sub directories: data, label
source_root_dir = sys.argv[2]

# Get the max size of the data set
# Processing will end after this many items have been added
max_items = sys.argv[3]

# Set the data and label directories
data_dir = os.path.join(source_root_dir, "data")
label_dir = os.path.join(source_root_dir, "label")

# Create the post_process object
im_p = pp.Image_processing()

# Get a list of items in the data directory
image_pickles = get_names(data_dir)

# Get a random list of features
rand_features = im_p.random_feature_list()

# Save the features

# Create a numpy array to hold the specified 
