require 'math'
require 'torch'

function hasinf(tensor)
	for i = 1,514 do
		if tensor[{1, i}] == (1/0) or tensor[{1, i}] == -(1/0) then
			return true
		end
	end
	return false
end

function getLineData(oldcount, curcount)
	started = false

	while (not started) or (foundinf) do
		started = true

		if curcount > 582000 then
			curcount = 1
		end

		datacount = math.floor((curcount - 1)/97000) + 1

		if math.floor((oldcount - 1)/97000) ~= math.floor((curcount - 1)/97000) then
			data = getNewData(datacount)
		end

		tensorToReturn = data[curcount - ((datacount - 1) * 97000)]

		if hasinf(tensorToReturn) then
			curcount = curcount + 1
			foundinf = true
		else
			foundinf = false
		end
	end

	return tensorToReturn, curcount
end

function getNewData(filenumber)
	dataload = torch.load('newtrain'..tostring(filenumber)..'.t7')
	return dataload
end

function getbatch(batchsize)
	batchtensor = nil
	for batchindex = 1, batchsize do
		dataelem, curcount = getLineData(oldcount, count)
		oldcount = curcount
		count = curcount + 1
		if batchtensor == nil then
			batchtensor = dataelem
		else
			batchtensor = torch.cat(batchtensor, dataelem, 1)
		end
	end
	return batchtensor
end

data = nil
count = 1
oldcount = 0