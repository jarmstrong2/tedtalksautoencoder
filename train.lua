require 'nn'
require 'optim'
require 'getbatch'
require 'torch'

function train(model, criterion)
        function gettensor(set)
                isstarted = false
                for i = 1, opt.validationSize do
                        if isstarted then
                                valsettensor = torch.cat(valsettensor, set[i], 2)
                        else
                                valsettensor = set[i]:clone()
                                isstarted = true
                        end
                end
                valsettensor = valsettensor:t()
                return valsettensor
        end

        function validation()
                output = model:forward(valset:cuda())

                target = valset
                
                loss = criterion:forward(output, target:cuda())

                return loss
        end

        valset = torch.load(opt.folderPath .. 'val.t7')
        valset = gettensor(valset)
        minval = 1/0

	parameters, gradParameters = model:getParameters()

	local feval = function(x)
		-- get new parameters
                if x ~= parameters then
                	parameters:copy(x)
                end

                -- reset gradients
                gradParameters:zero()

                input = getbatch(opt.batchSize)
                input = input:cuda()
                --print(input)

                -- forward --
                output = model:forward(input)

                target = input
                
                loss = criterion:forward(output, target)

                -- backward --
                grad_criterion = criterion:backward(output, target)
                
                model:backward(input, grad_criterion)

                return loss, gradParameters
	end

	local optim_state = {learningRate = opt.lr}
	local iterations = numberOfIterations
        local lossTable = {}
        local valLossTable = {}

	for i = 1, opt.numIters do

		local _, loss = optim.adam(feval, parameters, optim_state)
		collectgarbage()


		if i % opt.numItersToSave == 0 then
			print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParameters:norm()))

                        newval = validation()

                        table.insert(lossTable, loss[1])
                        table.insert(valLossTable, newval/opt.validationSize)

                        torch.save(opt.modLossFile, lossTable)
                        torch.save(opt.valLossFile, valLossTable)

                        print(string.format("validation loss = %6.8f", newval))
                        if newval < minval then
                                minval = newval
                                print("model saved")
                                torch.save(opt.modFile, model)
                        end
		end
	end
end
