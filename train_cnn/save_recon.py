# Shows the reconstructed mnist data

import os
import errno
import sys
import cPickle as pickle
import numpy as np

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.datasets import mnist
from keras.layers.additional import UnPooling2D

import importlib

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Enforces file path
def enforce_path(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# Get the name of the model to use
model_name = sys.argv[1]
print "Using model: ", model_name

# Get the name of the save file directory
save_dir = sys.argv[2]
print "Saving to: ", save_dir

# Enforce the save path
enforce_path(save_dir)

model = (importlib.import_module("structure_models.CAE_2conv_pool_relu")).get_model(load_name = model_name)

# Load the data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The amount of images to use
to_use = 20

# Get a slice of the images
images_to_use = X_train[:to_use]

# Make into floats
images_to_use = images_to_use.astype("float32")

# Add noise
images_to_use = images_to_use + .2*images_to_use.std()*np.random.uniform(-1, 1, images_to_use.shape)
images_to_use = np.maximum(0, np.minimum(255, images_to_use))

# Normalize the images
images_to_use /= 255

# Add the stack dimension, needed for correct processing in the convolutional layers
images_to_use = np.expand_dims(images_to_use, axis=0)

# Reorder axis to (n_images, stack, height, width)
images_to_use = np.rollaxis(images_to_use, 0, 2)

# Set to GPU friendly float32
images_to_use = images_to_use.astype("float32")

# Add noise
#images_to_use = images_to_use + .4*images_to_use.std()*np.random.uniform(-1, 1, images_to_use.shape)

# Keep the numbers within bounds
#images_to_use = np.maximum(0, np.minimum(1, images_to_use))

# Get the image reconstructions
predicted_images = model.predict(images_to_use)

# Get the test score
score = model.evaluate(images_to_use, X_train[:to_use].reshape(to_use, 28 * 28), batch_size=128)

print "Test score: ", score

# Denormalize all of the images
images_to_use *= 255
predicted_images *= 255

# Make sure that all values can be represented by a uint8
predicted_images = np.maximum(0, np.minimum(255, predicted_images))

# Make them all into ints
images_to_use = images_to_use.astype(np.uint8)
predicted_images = predicted_images.astype(np.uint8)

# Reshape the images
images_to_use = images_to_use.reshape((to_use, 28, 28))
predicted_images = predicted_images.reshape((to_use, 28, 28))

# Create a matrix to show all of the images side by side
compare_images = np.empty((to_use, 28 * 2, 28), dtype=np.uint8)

#print compare_images.shape
#print images_to_use.shape
#print predicted_images.shape

# Place every image so that the original is on the left and the reconstructed is on the right
for image_index in range(to_use):

	for h_index in range(28):

		for w_index in range(28):

			# Original
			compare_images[image_index][w_index][h_index] = images_to_use[image_index][h_index][w_index]
			
			# Reconstructed
			compare_images[image_index][w_index + 28][h_index] = predicted_images[image_index][h_index][w_index]

# Get the pixel values for each labeled image
new_im = np.empty((compare_images.shape[0], compare_images.shape[1], compare_images.shape[2], 3), dtype=np.uint8)
for im_index in range(compare_images.shape[0]):
	for w_index in range(compare_images.shape[1]):
		for h_index in range(compare_images.shape[2]):
			new_im[im_index][w_index][h_index][0] = compare_images[im_index][w_index][h_index]
			new_im[im_index][w_index][h_index][1] = compare_images[im_index][w_index][h_index]
			new_im[im_index][w_index][h_index][2] = compare_images[im_index][w_index][h_index]

# Add the channel dim
#compare_images = np.expand_dims(compare_images, axis=0)
new_im = np.rollaxis(new_im, 3, 1)
#print new_im.shape
#print new_im.dtype
#print np.max(new_im)
#print type(new_im)

# Save all of the images
for save_index in range(to_use):

	pp.save_image(new_im[save_index], save_dir + "/" + str(save_index).zfill(4) + ".png")
