import sys

import numpy as np

from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing()

# The cuda function to run the feature extraction
feature_extractor = """
__global__ void DepthDifference(float* source_image, float* computed_features, int* first_offset, int* second_offset)
{
	// TEMP MANUAL SET
	int bound_x = 480;
	int bound_y = 640;

	// The value that gets assigned to pixels off of the image
	float large_positive = 1000000;

	// The target offsets
	int first_target_x;
	int first_target_y;
	int second_target_x;
	int second_target_y;

	// The depth at the pixel being worked on
	float depth_at;

	// The depth at the target pixels
	float depth_first;
	float depth_second;

	// Get thread location
	int row_location = blockIdx.y * blockDim.y + threadIdx.y;
	int col_location = blockIdx.x * blockDim.x + threadIdx.x;

	// Exit if this is out of bounds
	if(row_location > bound_y || col_location > bound_x) return;

	// Get the depth at the pixel
	depth_at = source_image[row_location * bound_y + col_location];

	// Get the offsets, normalized to the depth of the center pixel
	first_target_x = first_offset[0] / depth_at;	
	first_target_y = first_offset[1] / depth_at;

	second_target_x = second_offset[0] / depth_at;
	second_target_y = second_offset[1] / depth_at;

	// Depth probe the target locations

	// First feature offset
	if(first_target_x < 0 || first_target_x > bound_x || first_target_y < 0 || first_target_y > bound_y)
	// Targets that are off of the image get a large positive value
	{
		depth_first = large_positive;
	}

	else
	{
		depth_first = source_image[first_target_x * bound_y + first_target_y];
	}

	// Second feature offset
	if(second_target_x < 0 || second_target_x > bound_x || second_target_y < 0 || second_target_y > bound_y)
	// Targets that are off of the image get a large positive value
	{
		depth_second = large_positive;
	}

	else
	{
		depth_second = source_image[second_target_x * bound_y + second_target_y];
	}

	// Get the difference between the locations and save it into the computed features array, at the location of this thread
	computed_features[row_location * bound_y + col_location] = depth_first - depth_second;
}
"""

# The name of the image to test
exr_file_name = "/home/master/ex_images/000000000002.exr"

# Manually set a feature to test
manual_feature_first = gpuarray.to_gpu(np.array([0, 400], dtype=np.uint8))
manual_feature_second = gpuarray.to_gpu(np.array([-300, 200], dtype=np.uint8))

# Load a test image
test_image = np.squeeze(im_p.get_channels(exr_file_name, "Z").astype(np.float32))

# Convert it into a gpu array
test_image_gpu = gpuarray.to_gpu(test_image)

# Create an empty array for the result
result = gpuarray.empty(test_image.shape, np.float32)

# Get the shape of the image
(height, width) = test_image.shape

# Set the number of tiles
num_tiles = 50

# Set the size of each tile
# Act as if the image is square and make enough tiles to cover it
tile_size = int(np.ceil(max(height, width) / float(num_tiles)))

# Compile the cuda code
compiled = compiler.SourceModule(feature_extractor)

# Get the function
get_features_gpu = compiled.get_function("DepthDifference")

# Compute the features
print num_tiles
print tile_size
get_features_gpu(test_image_gpu, result, manual_feature_first, manual_feature_second, grid=(10, 10), block=(10, 10, 1),)
