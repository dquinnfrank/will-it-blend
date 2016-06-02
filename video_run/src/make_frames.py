import numpy as np

# Gets frames from a depth sensor, uses a trained FCN to predict the body segmentations
# Returns the depth data and the predictions
class segment_frame:

	# Gets a frame from the sensor
	# Returns as a numpy array
	# get_frame()
	#
	# Segments a depth image and produces a segmented image
	# get_prediction()

	# Needs to know:
	# TODO: make the threshold part of a post processing stage?
	def __init__(self, model_name, sensor_type = "kinect", predictor_type = "torch", threshold = 10):

		# Set the sensor
		self.set_sensor(sensor_type, threshold = threshold)

		# Set the predictor
		self.set_predictor(predictor_type, model_name)

		# Set the depth storage
		self.current_depth = None;

		# Save the threshold
		self.threshold = threshold

	# Set the type of sensor to use
	# kinect : a microsoft 360 kinect
	def set_sensor(sensor_type, threshold = None):

		# None threshold, use max value
		if threshold is None:

			threshold = np.iinfo(np.float32).max

		# Uses microsoft kinect for 360
		if sensor_type == "kinect":

			import freenect

			# Take kinect depth data and make it suitable for use in the network
			# Converts from uint16 in millimeters to float32 in meters
			# Thresholds the data
			def convert_to_base(depth_frame, threshold):

				depth_frame = depth_frame.astype(np.float32) * .001

				depth_frame[depth_frame > threshold] = threshold

			# Get the frame from the kinect and convert it into the base
			self.get_frame = lambda threshold: convert_to_base(freenect.sync_get_depth()[0], threshold = threshold)

		# Sensor type not known
		else:

			raise ValueError("Sensor type not known: " + sensor_type)

	# Sets the type of predictor
	# torch : torch based model
	def set_predictor(predictor_type, model_name):

		# Uses torch
		if predictor_type == "torch":

			import lutorpy

			# Load the model
			self.model = torch.load(model_name)

			# The function to predict given a depth image
			self.get_prediction = lambda depth_image : self.model._forward(torch.fromNumpyArray(depth_image))

		# Model library not known
		else:

			raise ValueError("Model type not known: " + predictor_type)

	# Updates the depth frame, needs to be called before the get methods
	def update_depth():

		self.current_depth = self.get_frame(self.threshold)

	def get_depth():

		return self.current_depth.flatten()

	# Predicts the segmentation of the current depth frame
	def get_segmentation():

		return self.get_prediction(self.current_depth).astype(np.uint8).flatten()
	
