# Imports for all classes
import bpy
import random
import math
import numpy as np
import itertools

# Save the scene object for easy of use later
scene = bpy.data.scenes['Scene']

# Constants for all classes

# Define list access: X, Y, Z = 0, 1, 2
X = 0
Y = 1
Z = 2

# Define list access: lower, upper = 0, 1
LOWER = 0
UPPER = 1

# Deselects all objects
# Prevents unwanted deletions
def deselect_all():

	# Go through each object in the default scene
	for obj in scene.objects:

		# Make sure the object is not selected
		obj.select = False

# Removes an object
def remove_object(object_name):

	# Make sure that nothing we don't want is selected
	#deselect_all()

	# Select the target object
	try:

		#bpy.data.objects[object_name].select = True

		to_delete = scene.objects[object_name]

	# Object doesn't exist, so it doesn't need deleting
	except KeyError:

		pass

	# Remove the object
	else:

		#bpy.ops.object.delete()

		scene.objects.unlink(to_delete)

		bpy.data.objects.remove(to_delete)

# Handles a human mesh
# Mesh must be exported from makehuman
# Mesh must use rigify
class Human:

	# Joint constraints for all humans
	# "bone" : [[x_upper, x_lower],[y_upper, y_lower],[z_upper, z_lower]]
	bone_constraints = {
			"head" : [[-20, 50],[-50, 50],[-30, 30]],
			"neck" : [[-10, 10],[],[-10, 10]],
			"shoulder.L" : [[-10, 30],[-30, 75],[-20, 40]],
			"shoulder.R" : [[-10, 30],[-30, 75],[-20, 40]],
			"chest" : [[-20, 20],[-10, 10],[-10, 10]],
			"hips" : [[-60, 20],[-50, 50],[-30, 30]],
			"thigh.fk.L" : [[-110, 30],[-30, 10],[-20, 20]],
			"thigh.fk.R" : [[-110, 30],[-30, 10],[-20, 20]],
			"upper_arm.fk.L" : [[-45, 100],[-50, 50],[-35, 35]],
			"upper_arm.fk.R" : [[-45, 100],[-50, 50],[-35, 35]],
			"forearm.fk.L" : [[-20, 90],[],[]],
			"forearm.fk.R" : [[-20, 90],[],[]],
			"shin.fk.L" : [[0, 120],[],[]],
			"shin.fk.R" : [[0, 120],[],[]],
			}

	# Initializes the class
	# Needs the name of the human mesh, will load the mesh into the scene
	def __init__(self, data_directory, load_mesh, debug_flag=False, limb_collision_radius=.15, body_collision_radius=.2, skin_name="left_right_colored.png"):

		# Set the data directory
		self.data_directory = data_directory

		# Set the name of the mesh
		self.mesh_name = (load_mesh.split('/')[-1]).strip(".mhx")

		# Load the mesh into the scene
		load_name = data_directory + load_mesh + ".mhx"
		bpy.ops.import_scene.makehuman_mhx(filepath=load_name)

		# Set the mesh object
		self.acting_mesh = bpy.data.objects[self.mesh_name]

		# Texture the mesh with the color coded skin
		img = bpy.data.images.load(data_directory + skin_name)
		cTex = bpy.data.textures.new('ColorTex', type = 'IMAGE')
		cTex.image = img

		# Select skin material
		mat = bpy.data.materials[self.mesh_name + ":Skin"]

		# Add texture slot for color texture
		mtex = mat.texture_slots.add()
		mtex.texture = cTex

		# Make the skin shadeless
		mat.use_shadeless = True

		# Remove the eye textures
		bpy.data.textures[self.mesh_name + ":HighPolyEyes:diffuse"].type = 'NONE'

		# Set the debug flag
		self.debug_flag = debug_flag

		# Initialize the joint position matrix
		self.joint_positions = {
				# Head
				"881" : None,
				# Neck
				"809" : None,
				# Left Shoulder
				"8276" : None,
				# Right Shoulder
				"1604" : None,
				# Left elbow
				"10042" : None,
				# Right elbow
				"3391" : None,
				# Chest
				"1891" : None,
				# Left Hand
				"8799" : None,
				# Right Hand
				"2131" : None,
				# Hips
				"4317" : None,
				# Left Thigh
				"10938" : None,
				# Right Thigh
				"4308" : None,
				# Left Knee
				"11249" : None,
				# Right Knee
				"4631" : None,
				# Left Foot
				"12888" : None,
				# Right Foot
				"6291" : None
				}

		# Set collision radii for each part
		self.collision_radii = {
				# Head
				"881" : limb_collision_radius,
				# Neck
				"809" : limb_collision_radius,
				# Left Shoulder
				"8276" : limb_collision_radius,
				# Right Shoulder
				"1604" : limb_collision_radius,
				# Left elbow
				"10042" : limb_collision_radius,
				# Right elbow
				"3391" : limb_collision_radius,
				# Chest
				"1891" : body_collision_radius,
				# Left Hand
				"8799" : limb_collision_radius,
				# Right Hand
				"2131" : limb_collision_radius,
				# Hips
				"4317" : body_collision_radius,
				# Left Thigh
				"10938" : limb_collision_radius,
				# Right Thigh
				"4308" : limb_collision_radius,
				# Left Knee
				"11249" : limb_collision_radius,
				# Right Knee
				"4631" : limb_collision_radius,
				# Left Foot
				"12888" : limb_collision_radius,
				# Right Foot
				"6291" : limb_collision_radius
				}

		# Set the original mesh
		original = bpy.data.objects[self.mesh_name + ":Body"]

		# Copy the mesh with modifiers
		copy = original.to_mesh(scene=bpy.data.scenes['Scene'], apply_modifiers=True, settings='PREVIEW')

		# Apply the world transformation
		#copy.transform(original.matrix_world)

		# Get the positions of every joint
		for joint_name in self.joint_positions.keys():

			# Get the position of the joint
			loc = copy.vertices[int(joint_name)].co

			# Set the position
			self.joint_positions[joint_name] = [loc[0], loc[1], loc[2]]

		# Remove copy
		bpy.data.meshes.remove(copy)

		# Add the occulsion square
		bpy.ops.mesh.primitive_plane_add(radius=.5, location=(0, -2, 0))

		# DEBUG PRINT
		if(debug_flag):
			print("\nDEBUG: Class: human, Name: " + self.mesh_name)
			print("Running with debug outputs")
			print("END DEBUG MESSAGE\n")

	# Outputs debug statements, if the flag is set
	def debug_print(self, message):

		# Only print if the debug flag is set
		if(self.debug_flag):

			# Add class identifier before showing message
			print("\nDEBUG: Class: human, Name: " + self.mesh_name)
			print(message)
			print("END DEBUG MESSAGE\n")

	# Moves a bone to a certain rotation
	# Takes the name of the bone and the target rotation in degrees [X, Y, Z]
	def rotate_bone(self, bone_name, target_rotation):

		# DEBUG PRINT
		self.debug_print("Rotating bone: " + bone_name + ", to: " + " ".join([str(i) for i in target_rotation]))

		# Select the bone based on the given name
		acting_bone = self.acting_mesh.pose.bones[bone_name]

		# Set the rotation mode to XYZ euler
		acting_bone.rotation_mode = 'XYZ'

		# Set the rotation to the sent targets
		# Must be set in radians
		# Some degrees of freedom may be locked, item will be None if this is the case
		if(target_rotation[X] is not None):
			acting_bone.rotation_euler[X] = math.radians(target_rotation[X])
		if(target_rotation[Y] is not None):
			acting_bone.rotation_euler[Y] = math.radians(target_rotation[Y])
		if(target_rotation[Z] is not None):
			acting_bone.rotation_euler[Z] = math.radians(target_rotation[Z])

	# Gets the current pose of the person
	# This will update the values in joint_positions
	def update_position_info(self):

		# Update the scene to have the most recent information
		scene.update()

		# Set the original mesh
		original = bpy.data.objects[self.mesh_name + ":Body"]

		# Copy the mesh with modifiers
		copy = original.to_mesh(scene=scene, apply_modifiers=True, settings='PREVIEW')

		# Apply the world transformation
		#copy.transform(original.matrix_world)

		# Get the positions of every joint
		for joint_name in self.joint_positions.keys():

			# Get the position of the joint
			loc = copy.vertices[int(joint_name)].co

			# Set the position
			self.joint_positions[joint_name] = [loc[0], loc[1], loc[2]]

			# Show the position
			# DEBUG PRINT
			self.debug_print("Vertex: " + joint_name + ", is at: " + str(loc[0]) + ", " + str(loc[1]) + " " + str(loc[2]))

		# Remove copy
		bpy.data.meshes.remove(copy)

	# Check the pose of the person to ensure that there is no clipping
	# Returns bool for valid pose
	# Currently, checks to see if the hands, elbows, feet, or knees are too close to any of the other points
	# If one point is too close to another, the pose is invalid
	# Radius can be tuned in the class initializer
	# This is not a robust method, it could use improvement
	def check_pose(self):

		# Remember to that order matters for the sub list items

		# Update the pose info
		self.update_position_info()

		# Set the set of line segments
		# Each number is an ID of a vertex
		line_segments = [
			["8799", "10042"],
			["2131", "3391"],
			["11249", "12888"],
			["4631", "6291"],
			["8276", "10042"],
			["1604", "3391"],
			["10938", "11249"],
			["4308", "4631"],
			["881", "809"],
			["1891", "4317"]
			]

		# Set the line segments to be considered when checking
		to_check = [["8799", "10042"],["2131", "3391"],["11249", "12888"],["4631", "6291"]]

		# Create the lines
		lines = {}
		for pair in line_segments:

			# Set the ends of the lines
			lines[pair[0] + " " + pair[1]] = [np.array(self.joint_positions[pair[0]]), np.array(self.joint_positions[pair[1]])]

			# DEBUG PRINT
			#self.debug_print("Line: " + pair[0] + " " + pair[1] + " is at: " + str(lines[pair[0] + " " + pair[1]][0]) + " " + str(lines[pair[0] + " " + pair[1]][1]))

		# Check to see if any of the key lines are colliding with another part
		for key_line in to_check:

			# Check each other line
			for other_line in line_segments:

				# Do not check if the lines contain a similar point
				if(not(key_line[0] in other_line or
					key_line[1] in other_line)):

					# DEBUG PRINT
					#self.debug_print("Checking key line: " + str(key_line) + " with other line " + str(other_line))

					# Get the two lines
					first_line = [np.array(self.joint_positions[key_line[0]]), np.array(self.joint_positions[key_line[1]])]
					second_line = [np.array(self.joint_positions[other_line[0]]), np.array(self.joint_positions[other_line[1]])]

					# Get the distance between the two lines
					distance = closestDistanceBetweenLines(first_line[0],first_line[1],second_line[0],second_line[1])

					# DEBUG PRINT
					#self.debug_print("Distance between lines: " + str(key_line) + " and " + str(other_line) + " is " + str(distance))

					# If the distance is less than the collision radius, then the parts are colliding
					check_radius = (self.collision_radii[other_line[0]] + self.collision_radii[other_line[1]])/2
					if distance < check_radius:

						# One collision makes the pose invalid
						return False

		# If no check above returned false, then the pose is valid
		return True

	# Creates a random pose
	# Will throw away invalid poses and start over until a valid one is found. Possible, but unlikely, for this to never terminate
	def random_pose(self):

		# DEBUG PRINT
		self.debug_print("Creating a random pose")

		# Reset the root rotation
		self.rotate_bone("root", [0, 0, 0])

		# Wait for a valid pose
		valid = False
		while(not valid):

			# Loop for each bone in the person to be moved
			random_pose = {}
			for bone_name in self.bone_constraints.keys():

				# Set the current bone being used
				acting_bone = self.bone_constraints[bone_name]

				# Generate a random number in the range of the constraints
				# Ignore locked degrees of freedom
				# X rotation
				x_rotation = None
				if(acting_bone[X] != []):
					x_rotation = random.randrange(acting_bone[X][LOWER], acting_bone[X][UPPER])
				# Y rotation
				y_rotation = None
				if(acting_bone[Y] != []):
					y_rotation = random.randrange(acting_bone[Y][LOWER], acting_bone[Y][UPPER])
				# Z rotation
				z_rotation = None
				if(acting_bone[Z] != []):
					z_rotation = random.randrange(acting_bone[Z][LOWER], acting_bone[Z][UPPER])

				# Add this bone rotation to the pose dictionary
				random_pose[bone_name] = [x_rotation, y_rotation, z_rotation]

			# Move all of the bones to the target rotations
			for bone_name in random_pose.keys():

				# Move the bone
				self.rotate_bone(bone_name, random_pose[bone_name])

			# Check for collisions
			# Pose is invalid if there are collisions
			valid = self.check_pose()

			# DEBUG PRINT
			if(not valid):
				self.debug_print("Pose not valid")
			else:
				self.debug_print("Pose valid")

	# Adds rotation to the entire person
	# the ranges will use the restrictions sent as (min, max)
	# lock_plane will make sure that the person is always upright facing the camera, overrides other options
	def random_rotation(self, lock_plane=False, x_range=None, y_range=None, z_range=None):

		if lock_plane:

			x_range = (0,0)
			z_range = (0,0)

		# Any ranges left as None should have no restriction
		#for r in [x_range, y_range, z_range]:

			#if r is None:

				#r = (0,360)

		if x_range is None:
			x_range = (0,360)
		if y_range is None:
			y_range = (0,360)
		if z_range is None:
			z_range = (0,360)

		# Generate a random rotation
		if x_range == (0,0):
			rand_x = 0
		else:
			rand_x = random.randrange(*x_range)
		if y_range == (0,0):
			rand_y = 0
		else:
			rand_y = random.randrange(*y_range)
		if z_range == (0,0):
			rand_z = 0
		else:
			rand_z = random.randrange(*z_range)

		random_rot = [rand_x, rand_y, rand_z]

		# Apply it to the entire person
		self.rotate_bone("root", random_rot)

	# Saves the positions of key vertices with the following scheme
	# Head, neck, ect are not labeled in the file, the appear in the order below
	# Each item contains xyz coordinates for 12 items per line
	# Head: front back right left
	# Neck: front back right left
	# Chest: front back right left
	# Abdomen: front back right left
	# Upper Arm Right: front back outer inner
	# Upper Arm Left: front back outer inner
	# Lower Arm Right: front back outer inner
	# Lower Arm Left: front back outer inner
	# Upper Leg Right: front back outer inner
	# Upper Leg Left: front back outer inner
	# Lower Leg Right: front back outer inner
	# Lower Leg Left: front back outer inner
	def save_key_verts(self, save_name):

		# Set the vertices to save the locations of
		key_verts = [
			# Head
			133, # front
			955, # back
			5757, # right
			12352, # left
			# Neck
			797,
			855,
			754,
			7530,
			# Chest
			1893,
			3974,
			3966,
			10528,
			# Abdomen
			4110,
			4188,
			4166,
			10807,
			# Upper Arm Right
			8385,
			1723,
			1721,
			1716,
			# Upper Arm Left
			1713,
			8395,
			8393,
			8388,
			# Lower Arm Right
			3874,
			3863,
			3455,
			3452,
			# Lower Arm Left
			10539,
			10528,
			10125,
			10120,
			# Upper Leg Right
			6755,
			6744,
			6749,
			6739,
			# Upper Leg Left
			13351,
			13340,
			13345,
			13335,
			# Lower Leg Right
			4764,
			4759,
			4763,
			6413,
			# Lower Leg Left
			11382,
			11377,
			11381,
			13010
			]

		# Update the scene to have the most recent information
		scene.update()

		# Set the original mesh
		original = bpy.data.objects[self.mesh_name + ":Body"]

		# Copy the mesh with modifiers
		copy = original.to_mesh(scene=scene, apply_modifiers=True, settings='PREVIEW')

		# Save the vertex locations as x,y,z four to a row
		with open(save_name + ".txt", "w") as save_file:

			# Save each vertex
			for index, vert in enumerate(key_verts):

				# Newline every 4 vertices
				if index % 4 == 0 and index != 0:
					save_file.write("\n")

				# Get the location of the vertex
				loc = copy.vertices[vert].co

				# Save the location
				save_file.write(" ".join([str(i) for i in loc]) + " ")


		# Remove copy
		bpy.data.meshes.remove(copy)

# End of Human

# Handles non person objects in the scene
class Clutter:

	# The area where objects can be active
	# Uses the XYZ indices defined above
	"""
	active_area = [
			# X
			(-1.25, 1.25),
			# Y
			(-1.5, 1.5),
			# Z
			(-1.0, 1.0)
			]
	"""
	active_area = [
			# X
			(-.75, .75),
			# Y
			(-.75, .75),
			# Z
			(-1.0, 1.0)
			]

	# A list of created object names
	# Used to keep track of what this class is responsible for in the scene
	current_objects = []

	# Constructor
	#
	# object_range : [min, max]
	# A number of random objects in this range will be choosen for each image
	def __init__(self, object_range=(10,30), debug_flag=False):

		# Save the object range
		self.object_range = object_range

		# Save the debug_flag
		self.debug_flag = debug_flag

	# Removes all objects this class is responsible for
	def remove_all_objects(self):

		# Go through the list of all clutter objects and remove each one
		for clutter_object in self.current_objects:

			remove_object(clutter_object)

		# Reset object list
		self.current_objects = []

	# Gives a random location within the active area
	def random_location(self):

		return (random.uniform(self.active_area[X][LOWER], self.active_area[X][UPPER]), random.uniform(self.active_area[Y][LOWER], self.active_area[Y][UPPER]), random.uniform(self.active_area[Z][LOWER], self.active_area[Z][UPPER]))

	# Gives a random rotation
	# No restrictions on rotations
	def random_rotation(self):

		return (random.uniform(0.0,math.pi * 2),random.uniform(0.0,math.pi * 2),random.uniform(0.0,math.pi * 2))

	# Clears all current objects
	def clear_objects(self):

		# Make sure that no objects are selected
		deselect_all()

		# Get rid of all current clutter objects
		self.remove_all_objects()

	# Adds clutter to the scene, removes previous clutter objects, if reset is set
	# Will generate a random number of random objects from the total possible objects
	def add_clutter(self, reset = True):

		# The number of different types of objects available
		# TODO: use a list of possible functions?
		total_object_types = 4

		if reset:

			self.clear_objects()

		# Set the number of objects to create
		num_to_create = random.randint(self.object_range[LOWER], self.object_range[UPPER])

		# Create that many objects
		for object_index in range(num_to_create):

			# Choose an type of object at random
			rand_type = random.randint(0, total_object_types)

			# Make the specified object

			# Sphere
			if rand_type == 0:

				self.add_sphere()

			# Cube
			elif rand_type == 1:

				self.add_cube()

			# Plane
			elif rand_type == 2:

				self.add_plane()

			# Cone
			elif rand_type == 3:

				self.add_cone()

	# Adds a sphere to the scene, parameters set to None will be randomly set
	def add_sphere(self, size=None, location=None):

		# Deselect all other objects
		deselect_all()

		if size is None:

			# The range of the sphere radius
			radius_range = (.1, .25)

			# Get a random size
			size = random.uniform(radius_range[LOWER], radius_range[UPPER])

		if location is None:

			# Get a random location within the active area
			location = self.random_location()

		# Create a sphere within the active area
		bpy.ops.mesh.primitive_uv_sphere_add(size = size, location = location)

		# Get the name of the new object
		name = scene.objects.active.name

		# Add it to the responsibility list
		self.current_objects.append(name)

	# Adds a cube to the scene, parameters set to None will be randomly set
	def add_cube(self, size=None, location=None, rotation=None):

		# Deselect all other objects
		deselect_all()

		if size is None:

			# The range of the cube radius
			radius_range = (.1, .25)

			# Get a random size
			size = random.uniform(radius_range[LOWER], radius_range[UPPER])

		if location is None:

			# Get a random location within the active area
			location = self.random_location()

		if rotation is None:

			# Get a random rotation
			rotation = self.random_rotation()

		# Create a sphere within the active area
		bpy.ops.mesh.primitive_cube_add(radius = size, location = location, rotation = rotation)

		# Get the name of the new object
		name = scene.objects.active.name

		# Add it to the responsibility list
		self.current_objects.append(name)

	# Adds a plane (geometric) to the scene, parameters set to None will be randomly set
	def add_plane(self, size=None, location=None, rotation=None):

		# Deselect all other objects
		deselect_all()

		if size is None:

			# The range of the plane radius
			radius_range = (.1, .25)

			# Get a random size
			size = random.uniform(radius_range[LOWER], radius_range[UPPER])

		if location is None:

			# Get a random location within the active area
			location = self.random_location()

		if rotation is None:

			# Get a random rotation
			rotation = self.random_rotation()

		# Create a sphere within the active area
		bpy.ops.mesh.primitive_plane_add(radius = size, location = location, rotation = rotation)

		# Get the name of the new object
		name = scene.objects.active.name

		# Add it to the responsibility list
		self.current_objects.append(name)

	# Adds a cone to the scene, parameters set to None will be randomly set
	def add_cone(self, radius1=None, radius2=None, depth=None, location=None, rotation=None):

		# Deselect all other objects
		deselect_all()

		# The potential radius of each end of the cone
		radius_range = (0.01, .1)

		# Get random dimensions for the radii set to None
		if radius1 is None:
			radius1 = random.uniform(radius_range[LOWER], radius_range[UPPER])
		if radius2 is None:
			radius2 = random.uniform(radius_range[LOWER], radius_range[UPPER])

		# The potential length of the cone
		if depth is None:
			length_range = (.2, 1)

			depth = random.uniform(length_range[LOWER], length_range[UPPER])

		if location is None:

			# Get a random location within the active area
			location = self.random_location()

		if rotation is None:

			# Get a random rotation
			rotation = self.random_rotation()

		# Create the cone
		bpy.ops.mesh.primitive_cone_add(radius1=radius1, radius2=radius2, depth=depth, location=location, rotation=rotation)

		# Get the name of the new object
		name = scene.objects.active.name

		# Add it to the responsibility list
		self.current_objects.append(name)

# TODO: move into the clutter class
# Moves the occulsion square to a random amount of occulsion
# Creates the object if it is not already present
def random_occulsion(square_radius=.5, z_min=-.75, z_max=-.40, debug_flag=False):

	# The name of the large occulder is "Billboard"

	# Deselect every thing else
	deselect_all()

	# Get random location
	# y location is fixed at -2 so that it will always be blocking the person
	# x doesn't change
	# z moves and determines the amount of occulsion
	target_location = (0, -2, random.uniform(z_min, z_max))

	# Set rotation
	# x axis is fixed so that the plane is perpendicular to the camera's view
	target_rotation = (math.pi/2, 0, 0)

	# Check for object already present in the scene
	try:

		occulsion_square = bpy.data.objects["Billboard"]

	# Object was not in the scene
	# Create it
	except:

		# Create the plane object
		bpy.ops.mesh.primitive_plane_add(radius=square_radius, location=target_location, rotation=target_rotation)

		# Name it
		scene.objects.active.name = "Billboard"

	# Object is already present
	# Move it to the target rotation
	else:

		# Move object
		occulsion_square.location = target_location

		# Rotate object
		occulsion_square.rotation_euler = target_rotation

	# Show the location of the object
	finally:

		# DEBUG PRINT
		if(debug_flag):
			print("\nDEBUG: Function: random_occulsion")
			print("Occulsion z location: " + str(target_location[2]))
			print("END DEBUG MESSAGE\n")

# Saves the current scene with the given name
# Uses openexr to save the depth of the image
# Pass path, but do not pass extension
# Default resolution is 640x480
def save_image(save_name, resolution=(640,480) , debug_flag=False):

	# DEBUG PRINT
	if(debug_flag):
		print("\nDEBUG: Function: save_image")
		print("Saving image as: " + save_name)
		print("END DEBUG MESSAGE\n")

	# Set the resolution of the image
	scene.render.resolution_x = resolution[X]
	scene.render.resolution_y = resolution[Y]
	scene.render.resolution_percentage = 100

	# Set the image name
	scene.render.filepath = save_name

	# Render and save the image
	bpy.ops.render.render(write_still=True)

# From: http://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
def closestDistanceBetweenLines(a0,a1,b0,b1,clamp=True):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return distance, and the two closest points

        Use the clamp option to limit results to line segments
    '''
    A = a1 - a0
    B = b1 - b0

    _A = A / np.linalg.norm(A)
    _B = B / np.linalg.norm(B)
    cross = np.cross(_A, _B);


    # If denominator is 0, lines are parallel
    denom = np.linalg.norm(cross)**2

    if (denom == 0):
        return None

    # Calculate the dereminent and return points
    t = (b0 - a0);
    det0 = np.linalg.det([t, _B, cross])
    det1 = np.linalg.det([t, _A, cross])

    t0 = det0/denom;
    t1 = det1/denom;

    pA = a0 + (_A * t0);
    pB = b0 + (_B * t1);


    # Clamp results to line segments if requested
    if clamp:
        if t0 < 0:
            pA = a0
        elif t0 > np.linalg.norm(A):
            pA = a1

        if t1 < 0:
            pB = b0
        elif t1 > np.linalg.norm(B):
            pB = b1


    d = np.linalg.norm(pA-pB)

    return d
