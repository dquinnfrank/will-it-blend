// Processes the point clouds generated by the cnn by removing outliers
// Finds cloud centers to get joint locations
// Shows clouds

#include <iostream>
#include <string>
#include <cstdlib>
#include <map>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/pca.h>

#include "H5Cpp.h"

using namespace std;
using namespace H5;
using namespace Eigen;

// Gives the RGB for the given label
// Changes label if fix_lua is set
void label_to_pix(int& label, int& r, int& g, int& b, bool fix_lua = false)
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
void pix_to_label(int r, int g, int b, int& label, bool break_for_lua = false)
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

// Handles point clouds
class person_cloud
{
	private:

	// The max number of classes
	// TODO: make this automatic / configurable
	int num_classes;

	public:

	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

	// Holds point clouds for every separate body part
	// Keys are the labels
	map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> part_clouds;

	// Holds the centers for each body part
	map<int, Vector4f> part_centers;

	// Holds the vectors for each body part
	map<int, Vector3f> part_vectors;

	// Marks which centers are valid
	map<int, bool> part_valid;

	// The name of the file to load from
	string set_file_name;

	// Constructor
	// Needs a file to load the data from
	person_cloud(string file_name)
	{

		// Set the max number of classes
		num_classes = 13;

		// H5 file name
		set_file_name = file_name;

		// Initialize the clouds
		for (int i = 0; i < num_classes; i++)
		{
			// Add a new blank cloud to the map
			part_clouds[i] =  boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();
		}

	}

	// Destructor
	~person_cloud()
	{
		// Not needed?
		// Delete all of the clouds
		//for (int i = 0; i < num_classes; i++)
		//{
		//	delete part_clouds[i];
		//	part_clouds[i] = NULL;
		//}
	}

	// Makes a cloud out of the given info
	// Need to call after running the creating the class
	void make_cloud(int image_index = 0)
	{
		// Loading python hdf5 files into c++
		// http://stackoverflow.com/questions/25568446/loading-data-from-hdf5-to-vector-in-c

		// Open the H5 file
		H5File file(set_file_name, H5F_ACC_RDONLY);

		// Name of the sub fields in the hdf5 file
		string data_name = "data";
		string true_name = "true";
		string pred_name = "predictions";

		// Open the data sets
		DataSet depth = file.openDataSet(data_name);
		DataSet truth = file.openDataSet(true_name);
		DataSet preds = file.openDataSet(pred_name);

		// Get the shape of the data sets (all have the same shape)
		hid_t dspace = H5Dget_space(depth.getId());
		hsize_t shape[3];
		H5Sget_simple_extent_dims(dspace, shape, NULL);

		// Get the height and width of the images
		int height = shape[1];
		int width = shape[2];

		// Data space for specifing the image to be loaded
		DataSpace depth_space = depth.getSpace();
		DataSpace preds_space = preds.getSpace();

		// The shape of the data to get, getting one image
		hsize_t get_shape[3];
		get_shape[0] = 1;
		get_shape[1] = height;
		get_shape[2] = width;

		// The offset, start at the selected image
		hsize_t image_at[3];
		image_at[0] = image_index;
		image_at[1] = 0;
		image_at[2] = 0;
cout << "1" << endl;
cout << depth_space[0] << endl;
		// Load the depth image
		float depth_image[1][height][width];
		depth_space.selectHyperslab(H5S_SELECT_SET, get_shape, image_at);
		depth.read(depth_image, PredType::NATIVE_FLOAT, DataSpace::ALL, depth_space);
cout << "2" << endl;
		// Load the predictions
		float label_image[1][height][width];
		preds_space.selectHyperslab(H5S_SELECT_SET, get_shape, image_at);
		preds.read(label_image, PredType::NATIVE_FLOAT, DataSpace::ALL, preds_space);

		// Temporary point to hold values
		pcl::PointXYZRGB temp_point;

		// Holds the XYZ
		double temp_x, temp_y, temp_z;

		// Holds the RGB
		int temp_r, temp_g, temp_b;

		// Holds the label for the point
		int label;

		// Go through each point in the file at the specified index
		for (int h_index = 0; h_index < height; h_index++)
		{
			for (int w_index = 0; w_index < width; w_index++)
			{
				// Get the label at this point
				label = label_image[0][h_index][w_index];

				// Get the RGB of this point and fix the label
				label_to_pix(label, temp_r, temp_g, temp_b, true);

				//cout << "\rAdding on point: " << h_index << " " << w_index << " Label at: " << label;

				// Ignore non person points
				if (label != 0)
				{
					// Get the depth at this point
					temp_z = depth_image[0][h_index][w_index];

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
	void trim_cloud(double threshold=9.5)
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
			Vector4f min_point(-.5 * threshold, -.5 * threshold, 0.0, 0.0);

			// Set the max point
			Vector4f max_point(.5 * threshold, .5 * threshold, threshold, 0.0);

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
	void get_centers()
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
	void get_vectors()
	{
		// Holds all of the eigen vectors temporarily
		Matrix3f temp_eigen_vectors;

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
	void show_cloud()
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
		Vector4f temp_location;
		pcl::PointXYZ temp_point;
		pcl::PointXYZ temp_target;
		Vector3f temp_offset;
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

};

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
	person_cloud the_cloud(set_file_name);

	cout << "Class initialized" << endl;

	// Make the cloud
	the_cloud.make_cloud(to_visualize_index);

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
