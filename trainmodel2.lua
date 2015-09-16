require 'nn'
require 'optim'
require 'getbatchnew'
require 'cunn'
require 'torch'
require 'trainrelu2'

criterion = nn.MSECriterion()
criterion.sizeAverage = false

pretrainedModel = torch.load('model2.t7')
pretrainedModel = pretrainedModel:double()
layer1weight = (pretrainedModel:get(1).weight):clone()
layer2weight = (pretrainedModel:get(4).weight):clone()

model = nn.Sequential()
model:add(nn.Linear(514, 250))
model:add(nn.ReLU())
model:add(nn.Linear(250, 50))
model:add(nn.ReLU())
model:add(nn.Linear(50,250))
model:add(nn.ReLU())
model:add(nn.Linear(250, 514))

-- get add pretrained weights
model:get(1).weight = layer1weight
model:get(3).weight = layer2weight

-- tie parameters
model:get(5).weight = model:get(3).weight:t()
model:get(5).gradWeight = model:get(3).gradWeight:t()
model:get(7).weight = model:get(1).weight:t()
model:get(7).gradWeight = model:get(1).gradWeight:t()

--model = model:cuda()
--criterion = criterion:cuda()

train(model:cuda(), criterion:cuda())

--torch.save("model2_trained.t7", model)