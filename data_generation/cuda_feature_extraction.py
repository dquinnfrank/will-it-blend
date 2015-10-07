"""
__global__ void DepthDifference(float* source_image, float* computed_features, int* first_offset, int* second_offset, int bound_x, int bound_y)
{
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
	int col_location = blockIdx.x * blockDim.x + threadIdx.x

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
	if(first_target_x < 0 || first_target_x > bound_x || first_target_y < 0 || first_target_y > bound_y)
	// Targets that are off of the image get a large positive value
	{
		depth_first = large_positive;
	}

	else
	{
		depth_first = source_image[first_target_x * bound_y + first_target_y]
	}
}
"""
