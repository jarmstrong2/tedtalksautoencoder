require 'model'
require 'train'
require 'torch'

weightsPerRow = 15

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-weightsSelect' , 15, 'number of row elems in a weights matrix to set to randn value')
cmd:option('-xSize' , 514, 'The basic setup of the autoencoder x -> y -> z -> y -> x')
cmd:option('-ySize' , 1000, 'The basic setup of the autoencoder x -> y -> z -> y -> x')
cmd:option('-zSize' , 50, 'The basic setup of the autoencoder x -> y -> z -> y -> x')
cmd:option('-batchSize' , 128, 'Batch size for training')
cmd:option('-validationSize' , 1024, 'Number of element to use from validation set for validation')
cmd:option('-lr' , 1e-2, 'Learning rate')
cmd:option('-numIters' , 500000, 'Number of iterations for learning')
cmd:option('-numItersToSave' , 5000, 'The number of iterations after which we  save the model, loss, and validation')
cmd:option('-folderPath' , "/home/jarmstrong/sda/", 'Path to folder with training, validation and test data')
cmd:option('-modLossFile' , "losses.t7", 'model loss file name')
cmd:option('-valLossFile' , "valLosses.t7", 'validation loss file name')
cmd:option('-modFile' , "model.t7", 'model file name')
cmd:text()
opt = cmd:parse(arg)

model, criterion = getModel()
train(model:cuda(), criterion:cuda())