require 'nn'
require 'optim'
require 'getbatchnew'

function train(model, criterion)
        function gettensor(set)
                isstarted = false
                for i = 1, 1024 do
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

        valset = torch.load('val.t7')
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

                input = getbatch(128)
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

	local optim_state = {learningRate = 1e-2}
	local iterations = 500000

	for i = 1, iterations do

		local _, loss = optim.adam(feval, parameters, optim_state)
		collectgarbage()
		--print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParameters:norm()))

		if i % 1000 == 0 then
			print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParameters:norm()))

                        newval = validation()
                        print(string.format("validation loss = %6.8f", newval))
                        if newval < minval then
                                minval = newval
                                print("model saved")
                                torch.save("model1_2_trained.t7", model)
                        end
		end
	end
end
