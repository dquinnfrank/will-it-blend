# Shows the person as a point cloud
# http://stackoverflow.com/questions/7591204/how-to-display-point-cloud-in-vtk-in-different-colors
# http://www.vtk.org/pipermail/vtkusers/2011-February/065697.html

import sys

import numpy
import vtk
import h5py

import post_process as pp; im_p = pp.Image_processing()

class Cloud_vis:

	def __init__(self, vis_file_name):

		# Get the data and image predictions
		self.h5_file = h5py.File(vis_file_name, 'r')
		#self.depth = h5py.File(vis_file_name, 'r')["data"]
		#self.truth = h5py.File(vis_file_name, 'r')["true"]
		#self.predictions = h5py.File(vis_file_name, 'r')["predictions"]
		self.depth = self.h5_file["data"]
		self.truth = self.h5_file["true"]
		self.predictions = self.h5_file["predictions"]

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

		#first_flag = True

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

					#if first_flag:

						#print coordinates

						#first_flag = False

					# Get the RGB values for this point
					rgb = im_p.label_to_pix[this_label].split()
					rgb = [int(i) for i in rgb]

					# Add the point
					self.add_point(coordinates, rgb)

					# Set the colors for the point
					#self.Colors.InsertNextTuple3(int(rgb[0]),int(rgb[1]),int(rgb[2]))

	# Shows the constructed cloud
	def show(self):

		# Add to the cloud
		self.vtkPolyData.SetPoints(self.vtkPoints)
		self.vtkPolyData.SetVerts(self.vtkCells)

		self.vtkPolyData.GetPointData().SetVectors(self.vtkColor)

		self.vtkPolyData.Modified()
		self.vtkPolyData.Update()

		# Renderer
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInput(self.vtkPolyData)
		renderer = vtk.vtkRenderer()
		self.vtkActor.SetMapper(mapper)
		renderer.AddActor(self.vtkActor)
		renderer.SetBackground(.2, .3, .4)
		renderer.ResetCamera()
		#renderer.SetPosition([-3, 0, 11])

		# Render Window
		renderWindow = vtk.vtkRenderWindow()
		renderWindow.AddRenderer(renderer)

		# Interactor
		renderWindowInteractor = vtk.vtkRenderWindowInteractor()
		renderWindowInteractor.SetRenderWindow(renderWindow)

		# Begin Interaction
		renderWindow.Render()
		renderWindowInteractor.Start()

	# Adds a single point to the cloud that will have the given color
	def add_point(self, point_location, point_color):

		# Make a new point
		pointId = self.vtkPoints.InsertNextPoint(point_location[:])

		# Add a cell for display
		self.vtkCells.InsertNextCell(1)
		self.vtkCells.InsertCellPoint(pointId)

		# Set the color of this point
		self.vtkColor.InsertNextTuple3(point_color[0],point_color[1],point_color[2])

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
		#self.vtkPolyData.SetPoints(self.vtkPoints)
		#self.vtkPolyData.SetVerts(self.vtkCells)

		self.vtkPolyData.GetPointData().SetScalars(self.vtkColor)
		self.vtkPolyData.GetPointData().SetActiveScalars('ColorArray')

# If this is run as main, visualize the cloud from the given file
if __name__ == "__main__":

	# Get the name of the file
	file_name = sys.argv[1]

	# Create the visualizer
	visualizer = Cloud_vis(file_name)

	# Visualize
	visualizer.create_cloud()

	# Show
	visualizer.show()
