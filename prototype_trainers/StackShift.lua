-- This layer will take input from a convolutional net and make it suitable for use with the Multi Margin criterion
-- For forward propagations, takes shape (batch_size, feature_planes, height, width) and turns it into (batch_size, height * width, feature_planes)
-- Backwards will do the reverse of this

-- WARNING: must call forward at least once before calling backward
-- TODO: figure out if this is a problem
-- This is due to the fact that the flattened dimensions need to have the orignal shapes restored

local StackShift, Parent = torch.class('nn.StackShift', 'nn.Module')

function StackShift:__init()

	Parent.__init(self)

end

-- Will save input shape in self.orignal_shape : (batch_size, feature_planes, height, width)
function StackShift:updateOutput(input)

	-- Save the dimensions
	self.original_shape = input:size()

	--print ("shape")
	--print (self.original_shape)

	-- Create a copy of the input
	self.output:resizeAs(input):copy(input)

	-- Permute the dimensions
	--self.output = self.output:permute(1,3,4,2)
	self.output = self.output:permute(2,3,1)

	-- Flatten the height and width dimensions and squeeze the batch (should be 1)
	--self.output = self.output:reshape(self.original_shape[1], self.original_shape[3] * self.original_shape[4], self.original_shape[2])
	--self.output = self.output:reshape(self.original_shape[3] * self.original_shape[4], self.original_shape[2])
	self.output = self.output:reshape(self.original_shape[2] * self.original_shape[3], self.original_shape[1])

	return self.output

end

function StackShift:updateGradInput(input, gradOutput)

	-- Create a copy of the grad input
	self.gradInput:resizeAs(gradOutput):copy(gradOutput)

	-- Reverse the permutation
	--self.gradInput = self.gradInput:permute(1,3,2)
	self.gradInput = self.gradInput:permute(2,1)

	-- Undo the partial flattening
	self.gradInput = self.gradInput:reshape(self.original_shape)

	return self.gradInput

end

--function StackShift:__tostring__()

--	return torch.type(self)

--end
