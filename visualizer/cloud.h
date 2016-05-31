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

//using namespace H5;
//using namespace Eigen;

// Gives the RGB for the given label
// Changes label if fix_lua is set
void label_to_pix(int& label, int& r, int& g, int& b, bool fix_lua = false);

// Gives the name of the label
// index should already be fixed from lua
void label_to_name(int& label, string& name);

// Gives the label given the pix
void pix_to_label(int r, int g, int b, int& label, bool break_for_lua = false);

// Gives the XYZ position in world of the point using assumed intrinsics
void get_world_XYZ(double depth, int h_index, int w_index, double& X, double& Y);

// Takes depth images and predictions, outputs clouds
class person_cloud
{

	public:

	// Constructor needs to know the basic dimensions of images and class predictions
	person_cloud(int class_max = 13, int im_height = 480, int im_width = 640);

	// Creates the cloud from float arrays of depth and int arrays of predictions
	// Prediction indices are redundant, but important to include for wrapping
	void make_cloud(float* depth_data, int depth_h, int depth_w, int* prediction_data, int prediction_h, int prediction_w);

	// Creates the cloud from an hdf5 file
	// Needs the index of the image to load
	// MAJOR BUG: CAN ONLY LOAD IMAGE 0
	void make_cloud(string file_name, int load_index);

	// Removes bad points from the cloud that are considered wrong
	// Pixels at the threshold, outliers
	// http://www.pcl-users.org/How-to-use-Crop-Box-td3888183.html
	void trim_cloud(double threshold=9.5);

	// Gets the centers of the point clouds
	// TODO: make this throw out and mark uncertain centers
	void get_centers();

	// Gets vectors for each cloud
	// uses PCA
	void get_vectors();

	// Shows the cloud for visualization
	void show_cloud();

	private:

	// The total number of classes in the prediction images
	int num_classes;

	// The dimensions of the images
	int height;
	int width;

	// Holds point clouds for every separate body part
	// Keys are the labels
	map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> part_clouds;

	// Holds the centers for each body part
	map<int, Eigen::Vector4f> part_centers;

	// Holds the vectors for each body part
	map<int, Eigen::Vector3f> part_vectors;

	// Marks which centers are valid
	map<int, bool> part_valid;

	// Tracks the frame number
	int frame_number = 0;
};
