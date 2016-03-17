// Processes the point clouds generated by the cnn by removing outliers
// Finds cloud centers to get joint locations
// Shows clouds

#include <iostream>
#include <string>
#include <cstdlib>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>

#include "H5Cpp.h"

using namespace std;
using namespace H5;

// Handles point clouds
class person_cloud
{

	public:

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

	// Constructor
	// Needs a file to load the data from
	person_cloud(string file_name)
	{

		// Initialize the cloud
		cloud = new pcl::PointCloud<pcl::PointXYZRGB>;

	}

	// Makes a cloud out of the given info
	// Need to call after running the creating the class
	void make_cloud()
	{

		// Temporary point to hold values
		pcl::PointXYZRGB temp;

	}

	// Removes bad points from the cloud that are considered wrong
	// Pixels at the threshold, outliers
	void trim_cloud(double threshold=10.0)
	{

		// Go through each point and remove threshold points
		
		
	}

	// Shows the cloud for visualization
	void show_cloud()
	{
		pcl::visualization::CloudViewer viewer("Cloud View");

		viewer.showCloud(cloud);

		// Spin lock until window exit
		while (!viewer.wasStopped())
		{

		}
	}

}

int main(int argc, char** argv)
{
	// Print usage if not enough args
	if (argc < 3)
	{
		cout << "Needs arguments: file_name index" << endl;

		return 0;
	}

	// Name of the sub fields in the hdf5 file
	string data_name = "data";
	string true_name = "true";
	string pred_name = "predictions";

	// Name of the visualization file
	string set_file_name = argv[1];

	// Index to visualize
	int to_visualize_index = atoi(argv[2]);

	cout << "Using file: " << set_file_name << endl;
	cout << "At index: " << to_visualize_index << endl;

	// Open the file
	H5File file(set_file_name, H5F_ACC_RDONLY);

	cout << "File opened" << endl;

	// Open the data sets
	DataSet depth = file.openDataSet(data_name);
	DataSet truth = file.openDataSet(true_name);
	DataSet preds = file.openDataSet(pred_name);

	return 0;
}
