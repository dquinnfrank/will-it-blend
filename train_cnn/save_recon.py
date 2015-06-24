# Shows the reconstructed mnist data

import os
import sys
import cPickle as pickle
import numpy as np

from keras.datasets import mnist

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Get the name of the model to use
model_name = sys.argv[1]
print "Using model: ", model_name

# Get the name of the save file directory
save_dir = sys.argv[2]
print "Saving to: ", save_dir

# Load the model
model = pickle.load(open(model_name, 'rb'))

# Load the data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The amount of images to use
to_use = 20

# Get a slice of the images
images_to_use = X_train[:to_use]

# Normalize the images
images_to_use /= 225

# Add the stack dimension, needed for correct processing in the convolutional layers
images_to_use = np.expand_dims(images_to_use, axis=0)

# Reorder axis to (n_images, stack, height, width)
images_to_use = np.rollaxis(images_to_use, 0, 2)

# Set to GPU friendly float32
images_to_use = images_to_use.astype("float32")

# Get the image reconstructions
predicted_images = model.predict(images_to_use)

# Denormalize all of the images
images_to_use *= 255
predicted_images *= 255

# Make them all into ints
images_to_use = images_to_use.astype(np.uint8)
predicted_images = predicted_images.astype(np.uint8)

# Reshape the images
images_to_use = images_to_use.reshape((to_use, 28, 28))
predicted_images = predicted_images.reshape((to_use, 28, 28))

# Create a matrix to show all of the images side by side
compare_images = np.empty((to_use, 28 * 2, 28), dtype=np.uint8)

print compare_images.shape
print images_to_use.shape
print predicted_images.shape

# Place every image so that the original is on the left and the reconstructed is on the right
for image_index in range(to_use):

	for h_index in range(28):

		for w_index in range(28):

			# Original
			compare_images[image_index][w_index][h_index] = images_to_use[image_index][h_index][w_index]
			
			# Reconstructed
			compare_images[image_index][w_index + 28][h_index] = predicted_images[image_index][h_index][w_index]

# Add the channel dim
compare_images = np.expand_dims(compare_images, axis=0)
compare_images = np.rollaxis(compare_images, 0, 2)
print compare_images.shape
print compare_images.dtype
print np.max(compare_images)

# Save all of the images
for save_index in range(to_use):

	pp.save_image(compare_images[save_index], save_dir + "/" + str(save_index).zfill(4))
