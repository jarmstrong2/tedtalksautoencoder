require 'nn'
require 'getbatchnew'
require 'pretrain'
require 'cunn'

criterion = nn.MSECriterion()
criterion.sizeAverage = false

-- 514 -> 250 -> 514
ae1 = nn.Sequential()
ae1:add(nn.Dropout())
ae1:add(nn.Linear(514, 250))
ae1:add(nn.ReLU())
ae1:add(nn.Linear(250,514))

train(ae1:cuda(), criterion:cuda(), -1)

-- 514 -> 250 -> 50 -> 250
ae2 = nn.Sequential()
ae2:add(nn.Linear(514, 250))
ae2:add(nn.ReLU())
ae2:add(nn.Dropout())
ae2:add(nn.Linear(250, 50))
ae2:add(nn.ReLU())
ae2:add(nn.Linear(50,250))

ae2:get(1).weight = (ae1:get(2).weight):clone()

train(ae2:cuda(), criterion:cuda(), 2)

torch.save("model2.t7", ae2)
