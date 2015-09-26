require 'torch'
require 'nn'
require 'optim'
require 'cunn'

-- Function to provide random weights sized to tensor size weight. 
-- All weights set to zero except those weightsPerRow value of weights
-- chosen uniformly based on the number of elements in each row, these
-- weights are set to a randn value.
function newWeight(weight)
	matsize = weight:size()
	colsize = matsize[2]
	p = 1/colsize
	-- matrix to provide uniform probabilty to multinomial function
	probmat = torch.Tensor(matsize):fill(p)
	-- Decide which weightsPerRow indices should be replaced with randn values
	indicesToWeight = torch.multinomial(probmat, opt.weightsSelect, true)
	newWeightMat = torch.zeros(matsize)
	for i = 1, indicesToWeight:size()[1] do
		for j = 1, indicesToWeight:size()[2] do
			colToAddWeight = indicesToWeight[{i, j}]
			newWeightMat[{i, colToAddWeight}] = torch.randn(1):squeeze()
        end
    end
	return newWeightMat
end

-- returns autoencoder

function getModel()
	criterion = nn.MSECriterion()
	criterion.sizeAverage = false

	-- set up of model layers
	layerxy = nn.Linear(opt.xSize, opt.ySize)
	layeryz = nn.Linear(opt.ySize, opt.zSize)
	layerzy = nn.Linear(opt.zSize, opt.ySize)
	layeryx = nn.Linear(opt.ySize, opt.xSize)

	-- initialize weights
	layerxy.weight = newWeight(layerxy.weight)
	layeryz.weight = newWeight(layeryz.weight)
	layerzy.weight = newWeight(layerzy.weight)
	layeryx.weight = newWeight(layeryx.weight)

	-- set up model
	model = nn.Sequential()
	model:add(layerxy)
	model:add(nn.ReLU())
	model:add(layeryz)
	model:add(nn.ReLU())
	model:add(layerzy)
	model:add(nn.ReLU())
	model:add(layeryx) 

	return model, criterion
end
