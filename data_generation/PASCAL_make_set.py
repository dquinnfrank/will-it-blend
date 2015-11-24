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

# Labeled images used by FCN
label_source_dir = "./59_context_labels"

# The original RGB images, not all are used in the label set
image_source_dir = "./VOCdevkit/VOC2010/JPEGImages"

# Create the lmdb
in_db = lmdb.open('image-lmdb',  map_size=int(1e12))

# Open the lmdb
with in_db.begin(write=True) as in_txn:

	# Iterate through the names of the images in the label directory
	file_names = sorted([ f for f in os.listdir(label_source_dir) if os.path.isfile(os.path.join(label_source_dir,f)) ])
	for im_index, label_im_name in enumerate(file_names):

		# Show progress
		print "\rWorking on item: ", im_index, " of ", len(file_names),
		sys.stdout.flush()

		# The base name of the image
		base_name = os.path.splitext(label_im_name)[0]

		# Load the label image
		label_im_RGB = np.array(Image.open(os.path.join(label_source_dir, label_im_name)))

		# Get that image in the RGB source
		data_im_RGB = np.array(Image.open(os.path.join(image_source_dir, base_name + ".jpg")))

		# Switch from RGB to BGR
		data_im_BGR = data_im_RGB[:,:,::-1]

		# Change to using floats
		data_im_BGR = data_im_BGR.astype(np.float32)

		# Subtract the mean from each channel
		data_im_BGR -= np.array([104.00698793, 116.66876762, 122.67891434])

		# Switch dimensions from (height, width, channels) to (channels, height, width)
		data_im_BGR = np.rollaxis(data_im_BGR, 2)

		# Get image into a datum, need to change image type to float
		data_im_datum = caffe.io.array_to_datum(data_im_BGR.astype(float))

		# Put the image into the lmdb
		in_txn.put('{:0>10d}'.format(im_index), data_im_datum.SerializeToString())

in_db.close()
