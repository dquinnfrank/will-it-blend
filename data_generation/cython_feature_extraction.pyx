import numpy as np
cimport numpy as np

def get_features(np.ndarray[np.float32_t, ndim=2] image, np.ndarray[np.int32_t, ndim=2] feature_list, np.ndarray[np.float32_t] results, int target_x, int target_y):

	# This is the value to give to any pixels that are off of the image
	cdef np.float32_t large_positive = 1000000

	cdef Py_ssize_t feature_index

	# The depth at the target pixel
	cdef np.float32_t depth_at = image[target_x, target_y]

	# If depth_at is 0, make it an epsilon value
	if (depth_at == 0):

		depth_at = np.finfo(np.float32).eps

	# The offsets of the features
	cdef int first_offset_x, first_offset_y, second_offset_x, second_offset_y

	# The values at the offsets
	cdef np.float32_t first_value, second_value

	# Go though each feature in the list
	for feature_index in range(feature_list.shape[0]):

		# Calculate the offsets based on the depth at the target pixel
		first_offset_x = int(feature_list[feature_index, 0] / depth_at)
		first_offset_y = int(feature_list[feature_index, 1] / depth_at)
		second_offset_x = int(feature_list[feature_index, 2] / depth_at)
		second_offset_y = int(feature_list[feature_index, 3] / depth_at)

		# Add the value of the target pixel to each
		first_offset_x += target_x
		first_offset_y += target_y
		second_offset_x += target_x
		second_offset_y += target_y

		# Check to make sure that the targets are on the image
		# If not, give them large positive values
		if (0 > first_offset_x or first_offset_x >= image.shape[0] or 0 > first_offset_y or first_offset_y >= image.shape[1]):

			# Off the image
			first_value = large_positive

		# use the value at the targeted pixel
		else:

			first_value = image[first_offset_x, first_offset_y]

		# Check to make sure that the targets are on the image
		# If not, give them large positive values
		if (0 > second_offset_x or second_offset_x >= image.shape[0] or 0 > second_offset_y or second_offset_y >= image.shape[1]):

			# Off the image
			second_value = large_positive

		# use the value at the targeted pixel
		else:

			second_value = image[second_offset_x, second_offset_y]

		# Set the feature in the result array as the difference between the first and second offset values
		results[feature_index] = first_value - second_value

def get_depth_features_image(np.ndarray[np.float32_t, ndim=2] image, np.ndarray[np.int32_t, ndim=2] feature_list, np.ndarray[np.float32_t, ndim=3] result_image):

	cdef Py_ssize_t height_index, width_index

	# Go through each pixel in the image and get the features for the pixel
	for height_index in range(image.shape[0]):

		for width_index in range(image.shape[1]):

			get_features(image, feature_list, result_image[height_index, width_index], height_index, width_index)
