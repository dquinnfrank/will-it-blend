#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "H5Cpp.h"

using namespace std;
using namespace H5;

int main(int argc, char** argv)
{
	cout << "Hello clouds and hdf5" << endl;

	return 0;
}
