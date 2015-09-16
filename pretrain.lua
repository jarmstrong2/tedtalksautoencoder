require 'nn'
require 'optim'

function train(model, criterion, nonCorruptedIndex)
	parameters, gradParameters = model:getParameters()

	local feval = function(x)
		-- get new parameters
        if x ~= parameters then
        	parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        input = getbatch(128)
        --print(input)
        input = input:cuda()

        -- forward --
        output = model:forward(input)

        if nonCorruptedIndex == -1 then
        	-- dealing with initial mel log spectra
        	target = input
        else
        	target = model:get(nonCorruptedIndex).output
        end

        loss = criterion:forward(output, target)

        -- backward --
        grad_criterion = criterion:backward(output, target)
        
        model:backward(input, grad_criterion)

        return loss, gradParameters
	end

	local optim_state = {learningRate = 1e-3, learningRateDecay = 1e-7}
	local iterations = 50000

	for i = 1, iterations do

		local _, loss = optim.adam(feval, parameters, optim_state)
		collectgarbage()
		--print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParameters:norm()))


		if i % 1000 == 0 then
			print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParameters:norm()))
		end
	end
end
