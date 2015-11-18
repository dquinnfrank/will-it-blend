import sys
import os
import errno
from PIL import Image
import numpy as np
import caffe
import lmdb

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
label_source_dir = "/media/6a2ce75c-12d0-4cf5-beaa-875c0cd8e5d8/59_context_labels"

# The original RGB images, not all are used in the label set
image_source_dir = "/media/6a2ce75c-12d0-4cf5-beaa-875c0cd8e5d8/VOCdevkit/VOC2010/JPEGImages"

# Where to save the output
output_dir = "/media/6a2ce75c-12d0-4cf5-beaa-875c0cd8e5d8/check_aligment"

# Make the root directory for the output
enforce_path(output_dir)

# Subdirectories for separating images and labels
image_out_dir = os.path.join(output_dir, "data")
label_out_dir = os.path.join(output_dir, "label")

# Make the subdirectories for the images and labels
enforce_path(image_out_dir)
enforce_path(label_out_dir)

# Create the lmdb
in_db = lmdb.open('image-lmdb',  map_size=int(1e12))

# Open the lmdb
with in_db.begin(write=True) as in_txn:

	# Iterate through the names of the images in the label directory
	for im_index, label_im_name in enumerate(sorted([ f for f in os.listdir(label_source_dir) if os.path.isfile(os.path.join(label_source_dir,f)) ])):

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

		# Switch from RGB to BGR and subtract the mean from each channel
		#data_im_BGR = np.empty(data_im_RGB.shape, dtype=np.float32)
		#data_im_BGR[:,:,0] = data_im_RGB[:,:,2] - 104.00698793
		#data_im_BGR[:,:,1] = data_im_RGB[:,:,1] - 116.66876762
		#data_im_BGR[:,:,2] = data_im_RGB[:,:,0] - 122.67891434

		# Switch dimensions from (height, width, channels) to (channels, height, width)
		data_im_BGR = np.rollaxis(data_im_BGR, 2)

		# Get image into a datum
		data_im_datum = caffe.io.array_to_datum(data_im_BGR)

		# Put the image into the lmdb
		in_txn.put('{:0>10d}'.format(im_index), data_im_datum.SerializeToString())

		# Save them to the output dir
		#rgb_im.save(os.path.join(image_out_dir, base_name + ".png"))
		#label_im.save(os.path.join(label_out_dir, base_name + ".png"))
in_db.close()
