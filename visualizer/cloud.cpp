// Processes the point clouds generated by the cnn by removing outliers
// Finds cloud centers to get joint locations
// Shows clouds

#include "cloud.h"

// Gives the RGB for the given label
// Changes label if fix_lua is set
void label_to_pix(int& label, int& r, int& g, int& b, bool fix_lua)
{
	// LUA indexes from 1, fix by subtracting 1
	if (fix_lua)
	{
		label--;
	}

	// Non person
	if (label == 0)
	{
		r = 0;
		g = 0;
		b = 0;
	}

	// Head L
	else if (label == 1)
	{
		r = 255;
		g = 0;
		b = 0;
	}

	// Head R
	else if (label == 2)
	{
		r = 50;
		g = 0;
		b = 0;
	}

	// Torso L
	else if (label == 3)
	{
		r = 0;
		g = 0;
		b = 255;
	}

	// Torso R
	else if (label == 4)
	{
		r = 0;
		g = 0;
		b = 50;
	}

	// Upper Arm L
	else if (label == 5)
	{
		r = 255;
		g = 255;
		b = 0;
	}

	// Upper Arm R
	else if (label == 6)
	{
		r = 50;
		g = 50;
		b = 0;
	}

	// Lower Arm L
	else if (label == 7)
	{
		r = 0;
		g = 255;
		b = 255;
	}

	// Lower Arm R
	else if (label == 8)
	{
		r = 0;
		g = 50;
		b = 50;
	}

	// Upper Leg L
	else if (label == 9)
	{
		r = 0;
		g = 255;
		b = 0;
	}

	// Upper Leg R
	else if (label == 10)
	{
		r = 0;
		g = 50;
		b = 0;
	}

	// Lower Leg L
	else if (label == 11)
	{
		r = 255;
		g = 0;
		b = 255;
	}

	// Lower Leg R
	else if (label == 12)
	{
		r = 50;
		g = 0;
		b = 50;
	}
}

// Gives the name of the label
// index should already be fixed from lua
void label_to_name(int& label, string& name)
{
	// Non person
	if (label == 0)
	{
		name = "Non person";
	}

	// Head L
	else if (label == 1)
	{
		name = "Head L";
	}

	// Head R
	else if (label == 2)
	{
		name = "Head R";
	}

	// Torso L
	else if (label == 3)
	{
		name = "Torso L";
	}

	// Torso R
	else if (label == 4)
	{
		name = "Torso R";
	}

	// Upper Arm L
	else if (label == 5)
	{
		name = "Upper Arm L";
	}

	// Upper Arm R
	else if (label == 6)
	{
		name = "Upper Arm R";
	}

	// Lower Arm L
	else if (label == 7)
	{
		name = "Lower Arm L";
	}

	// Lower Arm R
	else if (label == 8)
	{
		name = "Lower Arm R";
	}

	// Upper Leg L
	else if (label == 9)
	{
		name = "Upper Leg L";
	}

	// Upper Leg R
	else if (label == 10)
	{
		name = "Upper Leg R";
	}

	// Lower Leg L
	else if (label == 11)
	{
		name = "Lower Leg L";
	}

	// Lower Leg R
	else if (label == 12)
	{
		name = "Lower Leg R";
	}
}

// Gives the label given the pix
void pix_to_label(int r, int g, int b, int& label, bool break_for_lua)
{
	// non person
	if (r == 0 && g == 0 && b == 0)
	{
		label = 0;
	}

	// Head L
	else if (r == 255 && g == 0 && b == 0)
	{
		label = 1;
	}

	// Head R
	else if (r == 50 && g == 0 && b == 0)
	{
		label = 2;
	}

	// Torso L
	else if (r == 0 && g == 0 && b == 255)
	{
		label = 3;
	}

	// Torso R
	else if (r == 0 && g == 0 && b == 50)
	{
		label = 4;
	}

	// Upper arm L
	else if (r == 255 && g == 255 && b == 0)
	{
		label = 5;
	}

	// Upper arm R
	else if (r == 50 && g == 50 && b == 0)
	{
		label = 6;
	}

	// Lower arm L
	else if (r == 0 && g == 255 && b == 255)
	{
		label = 7;
	}

	// Lower arm R
	else if (r == 0 && g == 50 && b == 50)
	{
		label = 8;
	}

	// Upper leg L
	else if (r == 0 && g == 255 && b == 0)
	{
		label = 9;
	}

	// Upper leg R
	else if (r == 0 && g == 50 && b == 0)
	{
		label = 10;
	}

	// Lower leg L
	else if (r == 255 && g == 0 && b == 255)
	{
		label = 11;
	}

	// Lower leg R
	else if (r == 50 && g == 0 && b == 50)
	{
		label = 12;
	}

	// Finally, add 1 for lua sake
	if (break_for_lua)
	{
		label++;
	}
}

// Gives the XYZ position in world of the point using assumed intrinsics
void get_world_XYZ(double depth, int h_index, int w_index, double& X, double& Y)
{
	// These are the intrinsics
	// TODO: make these configurable
	double focal_x = 580;
	double focal_y = 580;
	double center_x = 314;
	double center_y = 252;

	// X position
	X = ((h_index - center_y) * depth) / focal_y;

	// Y position
	Y = ((w_index - center_x) * depth) / focal_x;

	// Depth does not change
}

// Constructor
person_cloud::person_cloud(int class_max, int im_height, int im_width)
{

	// Basic size constraints
	num_classes = class_max;
	height = im_height;
	width = im_width;

	// Initialize the clouds
	for (int i = 0; i < num_classes; i++)
	{
		// Add a new blank cloud to the map
		part_clouds[i] =  boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();
	}

}

// Makes a cloud out of the given info
// Uses an hdf5 file
void person_cloud::make_cloud(string file_name, int load_index)
{
	// Loading python hdf5 files into c++
	// http://stackoverflow.com/questions/25568446/loading-data-from-hdf5-to-vector-in-c

	// Open the H5 file
	H5::H5File file(file_name, H5F_ACC_RDONLY);

	// Name of the sub fields in the hdf5 file
	string data_name = "data";
	string true_name = "true";
	string pred_name = "predictions";

	// Open the data sets
	H5::DataSet depth = file.openDataSet(data_name);
	H5::DataSet truth = file.openDataSet(true_name);
	H5::DataSet preds = file.openDataSet(pred_name);

	// Get the shape of the data sets (all have the same shape)
	hid_t dspace = H5Dget_space(depth.getId());
	hsize_t shape[3];
	H5Sget_simple_extent_dims(dspace, shape, NULL);

	// Get the height and width of the images
	int height = shape[1];
	int width = shape[2];

	// Data space for specifing the image to be loaded
	H5::DataSpace depth_space = depth.getSpace();
	H5::DataSpace preds_space = preds.getSpace();

	// The shape of the data to get, getting one image
	hsize_t get_shape[3];
	get_shape[0] = 1;
	get_shape[1] = height;
	get_shape[2] = width;

	// The offset, start at the selected image
	hsize_t image_at[3];
	image_at[0] = load_index;
	image_at[1] = 0;
	image_at[2] = 0;

	// Load the depth image
	float depth_image[1][height][width];
	depth_space.selectHyperslab(H5S_SELECT_SET, get_shape, image_at);
	depth.read(depth_image, H5::PredType::IEEE_F32LE, H5::DataSpace::ALL, depth_space);

	// Load the predictions
	float label_image[1][height][width];
	preds_space.selectHyperslab(H5S_SELECT_SET, get_shape, image_at);
	preds.read(label_image, H5::PredType::IEEE_F32LE, H5::DataSpace::ALL, preds_space);

	// Flatten the images
	// TODO: Can this be done directly in hdf5 loading?
	float depth_flat[height*width];
	int label_flat[height*width];

	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			depth_flat[width*h + w] = depth_image[0][h][w];
			label_flat[width*h + w] = label_image[0][h][w];
		}
	}

	// Call the other function
	make_cloud(depth_flat, height, width, label_flat, height, width);
}

// Makes a person cloud given arrays with the depth and predictions
void person_cloud::make_cloud(float* depth_data, int depth_h, int depth_w, int* prediction_data, int prediction_h, int prediction_w)
{
	// Temporary point to hold values
	pcl::PointXYZRGB temp_point;

	// Holds the XYZ
	double temp_x, temp_y, temp_z;

	// Holds the RGB
	int temp_r, temp_g, temp_b;

	// Holds the label for the point
	int label;

	// Go through each point in the file at the specified index
	for (int h_index = 0; h_index < depth_h; h_index++)
	{
		for (int w_index = 0; w_index < depth_w; w_index++)
		{
			// Get the label at this point
			label = prediction_data[prediction_w*h_index + w_index];

			// Get the RGB of this point and fix the label
			label_to_pix(label, temp_r, temp_g, temp_b, true);

			//cout << "\rAdding on point: " << h_index << " " << w_index << " Label at: " << label;

			// Ignore non person points
			if (label != 0)
			{
				// Get the depth at this point
				temp_z = depth_data[depth_w*h_index + w_index];

				// Get the XYZ of the point
				get_world_XYZ(temp_z, h_index, w_index, temp_x, temp_y);

				// Set the values in the temp point
				temp_point.x = temp_x;
				temp_point.y = temp_y;
				temp_point.z = temp_z;
				temp_point.r = temp_r;
				temp_point.g = temp_g;
				temp_point.b = temp_b;

				// Add the temp point to the cloud that corresponds to this label
				part_clouds[label]->points.push_back(temp_point);

			}
		}
	}
	//cout << endl;
}

// Removes bad points from the cloud that are considered wrong
// Pixels at the threshold, outliers
// http://www.pcl-users.org/How-to-use-Crop-Box-td3888183.html
void person_cloud::trim_cloud(double threshold)
{

	// Go through each point and remove threshold points
	for(int cloud_index = 0; cloud_index < num_classes; cloud_index++)
	{
		// Holds the output of the filtering
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudOut( new pcl::PointCloud<pcl::PointXYZRGB> );
		pcl::PointCloud<pcl::PointXYZRGB> cloudOut;

		// The box to allow points within is defined by a cube with sides of length threshold
		pcl::CropBox<pcl::PointXYZRGB> cropFilter;

		// Set the min point
		Eigen::Vector4f min_point(-.5 * threshold, -.5 * threshold, 0.0, 0.0);

		// Set the max point
		Eigen::Vector4f max_point(.5 * threshold, .5 * threshold, threshold, 0.0);

		// Set the parameters for the crop box filter
		cropFilter.setInputCloud( part_clouds[cloud_index] );
                cropFilter.setMin( min_point );
                cropFilter.setMax( max_point );

		// Run the filter
		cropFilter.filter(cloudOut);
		
		// Set the output as the new part cloud
		part_clouds[cloud_index] = cloudOut.makeShared();
	}

}

// Gets the centers of the point clouds
// TODO: make this throw out and mark uncertain centers
void person_cloud::get_centers()
{
	// Go through each body part
	for (int part_index = 1; part_index < num_classes; part_index++)
	{
		// Check for valid cloud
		// TODO: make this

		// Get the center of the part
		compute3DCentroid(*part_clouds[part_index], part_centers[part_index]);
	}
}

// Gets vectors for each cloud
// uses PCA
void person_cloud::get_vectors()
{
	// Holds all of the eigen vectors temporarily
	Eigen::Matrix3f temp_eigen_vectors;

	// Runs PCA
	pcl::PCA<pcl::PointXYZRGB> pca;

	// Go through each body part
	for (int part_index = 1; part_index < num_classes; part_index++)
	{
		// Set the cloud input
		pca.setInputCloud(part_clouds[part_index]);

		// Get the eigen vectors
		temp_eigen_vectors = pca.getEigenVectors();

		// Save the largest
		part_vectors[part_index] = temp_eigen_vectors.col(0);
	}
}

// Shows the cloud for visualization
void person_cloud::show_cloud()
{
	//pcl::visualization::CloudViewer viewer("Cloud View");
	pcl::visualization::PCLVisualizer viewer("Pose View");

	// Construct a combined cloud from all of the parts
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	*cloud = *part_clouds[0];
	for (int i = 1; i < num_classes; i++)
	{
		*cloud += *part_clouds[i];
	}

	// Add markers for each part center
	Eigen::Vector4f temp_location;
	pcl::PointXYZ temp_point;
	pcl::PointXYZ temp_target;
	Eigen::Vector3f temp_offset;
	int r, g, b;
	string name;
	for (int part_index = 1; part_index < num_classes; part_index++)
	{
		// Get the position of the point
		temp_location = part_centers[part_index];

		// Set the XYZ point
		temp_point.x = temp_location[0];
		temp_point.y = temp_location[1];
		temp_point.z = temp_location[2];

		// Get the RGB for this point
		label_to_pix(part_index, r, g, b);

		// Get the name of this part
		label_to_name(part_index, name);

		// Make a sphere to mark this point
		viewer.addSphere(temp_point, .01, r, g, b, name + "_center");

		// Set the target for the other end of the vector
		temp_offset = part_vectors[part_index];
		temp_target.x = temp_point.x + temp_offset[0];
		temp_target.y = temp_point.y + temp_offset[1];
		temp_target.z = temp_point.z + temp_offset[2];

		// Make a line to show the vector of this point
		viewer.addLine(temp_point, temp_target, r, g, b, name + "_vector");
	}

	//cout << "Number of points in the cloud: " << cloud->points.size() << endl;

	//viewer.showCloud(cloud);
	viewer.addPointCloud(cloud);

	// Set the camera to be where it is in blender
	// sort out terrible documentation to figure out how, good luck with that
	// http://docs.pointclouds.org/trunk/classpcl_1_1visualization_1_1_p_c_l_visualizer.html#ab2039927ec8f5a9771202f269987ec72
	// http://www.pcl-users.org/PCLVisualizer-viewer-camera-setting-td2968333.html
	//viewer.setCameraPosition(0,0,0,0,1,0);

	// Spin lock until window exit
	//while (!viewer.wasStopped())
	//{

	//}
	viewer.spin();
}

int main(int argc, char** argv)
{
	// Print usage if not enough args
	if (argc < 3)
	{
		cout << "Needs arguments: file_name index" << endl;

		return 0;
	}

	// Name of the visualization file
	string set_file_name = argv[1];

	// Index to visualize
	int to_visualize_index = atoi(argv[2]);

	cout << "Using file: " << set_file_name << endl;
	cout << "At index: " << to_visualize_index << endl;

	// Make the cloud handler
	person_cloud the_cloud;

	cout << "Class initialized" << endl;

	// Make the cloud
	the_cloud.make_cloud(set_file_name, to_visualize_index);

	cout << "Cloud constructed" << endl;

	// Remove bad points
	the_cloud.trim_cloud();

	cout << "Cloud trimmed" << endl;

	// Get the centers of the parts
	the_cloud.get_centers();

	cout << "Got part centers" << endl;

	// Get the vectors of the parts
	the_cloud.get_vectors();

	cout << "Got part vectors" << endl;

	// Show the cloud
	the_cloud.show_cloud();

	return 0;
}
