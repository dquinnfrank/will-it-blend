import numpy as np
cimport numpy as np

# Takes a batch of images where each pixel corresponds to a class and returns a plane representation
# Each image will have a number of planes equal to the number of classes, there will be a 1 in plane i, index (x, y), where index (x, y) was of class i in the original image
# Pass vectorize batch as an array of zeros with shape class_batch.shape + (total_classes,)
def vectorize(np.ndarray[np.uint8_t, ndim=3] class_batch, np.ndarray[np.uint8_t, ndim=4] vectorized_batch):

	cdef Py_ssize_t batch_index, x_index, y_index

	cdef np.uint8_t pixel_class

	# Go through each image
	for batch_index in range(class_batch.shape[0]):

		# Go through each pixel
		for x_index in range(class_batch.shape[1]):
			for y_index in range(class_batch.shape[2]):

				# Get the class of the pixel
				pixel_class = class_batch[batch_index, x_index, y_index]

				# Set the pixel (x, y) in plane i, where i is the class of the pixel
				vectorized_batch[batch_index, x_index, y_index, pixel_class] = 1
