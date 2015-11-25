import sys
import os
import errno
from PIL import Image
import numpy as np
import lmdb

# Need to add the path to caffe before importing
caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, os.path.join(caffe_root, "python"))

# Turn off GUI in matplotlib, to run headless
import matplotlib
matplotlib.use('Agg')

# Now caffe can be imported
import caffe

# Making an lmdb:
# https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045 

# Requirements for FCN-32s:
# https://gist.github.com/shelhamer/80667189b218ad570e82

# Enforces file path
def enforce_path(path):
    try:
	os.makedirs(path)
    except OSError as exc: # Python >2.5
	if exc.errno == errno.EEXIST and os.path.isdir(path):
	    pass
	else: raise

# Split for training and testing
split = .9

# Labeled images used by FCN
label_source_dir = "./59_context_labels"

# The original RGB images, not all are used in the label set
image_source_dir = "./VOCdevkit/VOC2010/JPEGImages"

# Training data lmdb
train_data_db = lmdb.open('pascal-context-train-lmdb',  map_size=int(1e12))

# Training label lmdb
train_label_db = lmdb.open('pascal-context-train-gt59-lmdb', map_size=int(1e12))

# Testing data lmdb
test_data_db = lmdb.open('pascal-context-val-lmdb', map_size=int(1e12))

# Testing label lmdb
test_label_db = lmdb.open('pascal-context-val-gt59-lmdb', map_size=int(1e12))

# Train data
train_data_txn = train_data_db.begin(write=True)

# Train label
train_label_txn = train_label_db.begin(write=True)

# Test data
test_data_txn = test_data_db.begin(write=True)

# Test label
test_label_txn = test_label_db.begin(write=True)

# Iterate through the names of the images in the label directory
file_names = sorted([ f for f in os.listdir(label_source_dir) if os.path.isfile(os.path.join(label_source_dir,f)) ])
for im_index, label_im_name in enumerate(file_names):

	# The base name of the image
	base_name = os.path.splitext(label_im_name)[0]

	# Load the label image
	label_im = np.array(Image.open(os.path.join(label_source_dir, label_im_name)))

	# Get that image in the RGB source
	data_im = np.array(Image.open(os.path.join(image_source_dir, base_name + ".jpg")))

	# Switch from RGB to BGR
	data_im = data_im[:,:,::-1]

	# Change to using floats
	data_im = data_im.astype(np.float32)

	# Make sure this is np.unit8
	label_im = label_im.astype(np.uint8)

	# Subtract the mean from each channel
	data_im -= np.array([104.00698793, 116.66876762, 122.67891434])

	# Switch dimensions from (height, width, channels) to (channels, height, width)
	data_im = np.rollaxis(data_im, 2)

	# Make a new single dimension
	label_im = np.expand_dims(label_im, axis=0)

	# Get image into a datum
	data_im_datum = caffe.io.array_to_datum(data_im.astype(float))
	label_im_datum = caffe.io.array_to_datum(label_im)

	# If this index is below the split, place data and label into the training lmdb
	if im_index < split * len(file_names):

		# Place the data image
		train_data_txn.put('{:0>10d}'.format(im_index), data_im_datum.SerializeToString())

		# Place the label datum item
		train_label_txn.put('{:0>10d}'.format(im_index), label_im_datum.SerializeToString())

		# Show progress
		print "\rCompleted item: ", im_index + 1, " of ", len(file_names), ". Saved to training set",
		sys.stdout.flush()

	# Goes into the testing set
	else:

		# Place the data image
		test_data_txn.put('{:0>10d}'.format(im_index), data_im_datum.SerializeToString())

		# Place the label datum item
		test_label_txn.put('{:0>10d}'.format(im_index), label_im_datum.SerializeToString())

		# Show progress
		print "\rCompleted item: ", im_index + 1, " of ", len(file_names), ". Saved to testing set",
		sys.stdout.flush()

train_data_db.close()

train_label_db.close()

test_data_db.close()

test_label_db.close()
