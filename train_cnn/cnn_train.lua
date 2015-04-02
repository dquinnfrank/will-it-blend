-- Trains and tests a cnn for partially occulded human detection
-- Requires the path to the data set and the name of the file to save the trained model as

-- Needed to unpack binary strings
-- From:
-- http://www.inf.puc-rio.br/~roberto/struct/
require "struct"
--[[
Copyright notice for struct
/******************************************************************************
* Copyright (C) 2010-2012 Lua.org, PUC-Rio.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/
--]]

-- For the nerual net
require 'nn'
require 'optim'
require 'cunn'

-- For the loading
require 'image'
require 'paths'

-- Loads a binary file of floats into a Tensor, returns the tensor
-- Full file path, name, and extension must be specified
-- Default size is 640*480
-- Default file type is "d", check struct documentation for more details
function bin_to_tensor(file_name, sent_size, sent_data)

	-- Default to size 307200 if size is not specified
	local size = sent_size
	if not sent_size then

		size = 307200

	end

	-- Default file type is "d" if data_type is not specified
	local data_type = sent_data
	if not sent_data then

		data_type = "d"

	end

	-- Load the file
	input = assert(io.open(file_name, "rb"))

	-- Get all of the info from the file in a string
	data = input:read("*all")

	-- Put all of the data into a Tensor
	local temp_tensor = torch.Tensor(size)
	local tensor_index = 1
	local data_index = 1
	while data_index < size * struct.size(data_type) do

		temp_tensor[tensor_index], data_index = struct.unpack(data_type, data, data_index)

		tensor_index = tensor_index + 1

	end

	-- Send the tensor back
	return temp_tensor

end

-- Gets the files with the lowest and highest index
-- Needs the path to the directory
-- returns low, high
function get_extreme_files(directory)

	local low = 0
	local high = 0
	local first = true

	local popen = io.popen
	for filename in popen('ls -a "'..directory..'"'):lines() do

		local index = tonumber(paths.basename(filename, ".jpg"))

		if index then

			if first then
				high = index
				low = index
				first = false

			else
				if index > high then
					high = index

				elseif index < low then
					low = index

				end

			end
		end
	end

	return low, high

end

-- Generates a random order 
function get_random_order(min_index, max_index)

	math.randomseed( os.time() )

	local rnd,trem,getn,ins = math.random,table.remove,table.getn,table.insert;

	-- Make a table of all values in the range
	local all_range = {}
	for i = min_index, max_index do
		ins(all_range, i)
	end

	local rand_order = {};
	while getn(all_range) > 0 do
		ins(rand_order, trem(all_range, rnd(getn(all_range))));
	end

	return rand_order
end

-- Gets a batch of training data
-- Requires a list of file names to open
-- Returns a Tensor of size [numImages] x height x width x channels
--function get_batch(list_to_open)

-- Check for correct number of arguments
if #arg < 2 then
	-- Not enough arguments, show usage
	print ("Usage: cnn_train input_dir output_name")
	print ("\nOptional arguments:")
	print ("\n-b batch_size : Sets the batch size")
	print ("Default: batch_size is 100")
	print ("\n-l load_file_name : Loads a model from the given file to continue training on")
	print ("Default: create a new model")
	print ("Will overwrite the load file if output_name has the same name")

	-- Quit program
	os.exit()
end

-- Get the source folder of the data
input_dir = arg[1]

-- Show file being loaded
print ("\nData directory: ")
print (input_dir)

-- Get the location to save the cnn to
output_name = arg[2]

-- Show output name
print ("\nSaving to: ")
print (output_name)

-- Default batch size is 100
batch_size = 100

-- Default load option is to create a new model
load_model_name = nil

if #arg > 2 then

	-- Get all remaining optional arguments
	local index = 3
	while index < #arg do

		-- Get the flag
		local flag = arg[index]

		-- Switch based on the option

		-- Batch size
		if (flag == "-b") then

			-- Get batch_size
			batch_size = tonumber(arg[index+1])

			-- Set the index to the next flag
			index = index + 2

		-- Load name
		elseif (flag == "-l") then

			-- Get the load_model_name
			load_model_name = arg[index+1]

			-- Set the index to the next flag
			index = index + 2

		-- Flag not known
		else

			-- Show error message
			print("Flag: "..flag.." is not known")

			-- Set the index to the next flag
			index = index + 1

		end

	end
end

-- Show batch_size
print ("\nBatch size: ")
print (batch_size)

-- Load the existing model, if applicable
if load_model_name then

	print ("\nLoad model: ")
	print (load_model_name)

	-- Load the model from a file
	model_auto = torch.load(load_model_name)

else

	print ("\nCreating new model")

	-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
	model:add(nn.SpatialSubtractiveNormalization(1, image.gaussian1D(15)))
	model:add(nn.SpatialConvolution(1, 16, 5, 5))
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
	model:add(nn.SpatialSubtractiveNormalization(16, image.gaussian1D(15)))
	model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 128, 4), 5, 5))
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- stage 3 : standard 2-layer neural network
	model:add(nn.Reshape(128*5*5))
	model:add(nn.Linear(128*5*5, 3 * 480 * 640))
	--model:add(nn.Tanh())
	--model:add(nn.Linear(200,#classes))
--[[
	ninput = 480 * 640
	nhidden = 300
	noutput = 3 * 480 * 640

	reshaper = nn.Reshape(ninput)

	model = nn.Sequential()
	model:add(nn.Reshape(ninput))
	model:add(nn.Linear(ninput,nhidden))
	model:add(nn.ReLU())
	model:add(nn.Linear(nhidden, noutput))
	model:add(nn.LogSoftMax())
]]--
	model:cuda()

	-- we put the mlp in a new container:
	model_auto = nn.Sequential()
	model_auto:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
	model_auto:add(model)
	model_auto:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
end

-- Get the lowest, highest indexed file in the input dir
min_index, max_index = get_extreme_files(paths.concat(input_dir, "RGB"))

-- Get a random ordering of the file indices in the given range
rand_order = get_random_order(min_index, max_index)

-- Process data and train the model

