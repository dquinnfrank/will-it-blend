# Shows the person as a point cloud
# http://stackoverflow.com/questions/7591204/how-to-display-point-cloud-in-vtk-in-different-colors
# http://www.vtk.org/pipermail/vtkusers/2011-February/065697.html

import sys

import numpy
import vkt
import hdf5

import post_process as pp; im_p = pp.Image_processing()

class Cloud_vis:

	def __init__(self, vis_file_name):

		# Get the data and image predictions
		self.depth = h5py.File(vis_file_name, 'r')["data"]
		self.truth = h5py.File(vis_file_name, 'r')["true"]
		self.predictions = h5py.File(vis_file_name, 'r')["predictions"]

		# Create the cloud
		self.vtkPolyData = vtk.vtkPolyData()

		# Set all of the cloud variables 
		self.clear_points()

		#mapper = vtk.vtkPolyDataMapper()
		#mapper.SetInput(self.vtkPolyData)
		#mapper.SetColorModeToDefault()
		#mapper.SetScalarRange(zMin, zMax)
		#mapper.SetScalarVisibility(1)

		# To render
		self.vtkActor = vtk.vtkActor()
		#self.vtkActor.SetMapper(mapper)

		# The intrinsics of the camera
		# http://cs.gmu.edu/~xzhou10/doc/kinect-study.pdf
		# http://stackoverflow.com/questions/31265245/extracting-3d-coordinates-given-2d-image-points-depth-map-and-camera-calibratio
		self.focal_x = 580.0
		self.focal_y = 580.0
		self.center_x = 314.0
		self.center_y = 252.0

	def create_cloud(self, image_index = 0):

		# Go through each pixel in the image at the image index
		for h_index in range(self.depth.shape[1]):
			for w_index in range(self.depth.shape[2]):

				# Get the label at this location and correct for Lua bad indexing
				this_label = self.predictions[image_index][h_index][w_index] - 1

				# Ignore non person pixels
				if(this_label != 0):

					# Get the depth at this point
					this_depth = self.depth[image_index][h_index][w_index]

					coordinates = [0,0,0]

					# Get the x, y coordinates of this point
					coordinates[0] = ((h_index - self.center_y) * this_depth) / self.focal_y
					coordinates[1] = ((w_index - self.center_x) * this_depth) / self.focal_x
					coordinates[2] = this_depth

					# Set the location of this point
					self.

					# Get the RGB values for this point
					rgb = im_p.label_to_pix[this_label].split()
					rgb = [int(i) for i in rgb]

					# Add the point
					self.add_point(coordinates, rgb)

					# Set the colors for the point
					#self.Colors.InsertNextTuple3(int(rgb[0]),int(rgb[1]),int(rgb[2]))

	# Adds a single point to the cloud that will have the given color
	def add_point(self, point_location, point_color):

		# Make a new point
		pointId = self.vtkPoints.InsertNextPoint(point_location[:])

		# Add a cell for display
		self.vtkCells.InsertNextCell(1)
		self.vtkCells.InsertCellPoint(pointId)

		# Set the color of this point
		self.vtkColor.InsertNextTuple3(point_color[:])

		# Update
		self.vtkCells.Modified()
		self.vtkPoints.Modified()
		self.vtkColor.Modified()

	# Creates / resets the point cloud
	def clear_points(self):

		# Create the points
		self.vtkPoints = vtk.vtkPoints()
		self.vtkCells = vtk.vtkCellArray()

		# Create the colors
		self.vtkColor = vtk.vtkUnsignedCharArray()
		self.vtkColor.SetName('ColorArray')
		self.vtkColor.SetNumberOfComponents(3)

		# Add to the cloud
		self.vtkPolyData.SetPoints(self.vtkPoints)
		self.vtkPolyData.SetVerts(self.vtkCells)

		self.vtkPolyData.GetPointData().SetScalars(self.vtkColor)
		self.vtkPolyData.GetPointData().SetActiveScalars('ColorArray')
