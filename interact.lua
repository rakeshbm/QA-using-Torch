require("hdf5")
require("nn")
require("rnn")
require("nngraph")
require("io")
require("string")
require("csv")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-modelfile', '', 'model file')
cmd:option('-datafile', 'new_qa20.hdf5', 'data file')
cmd:option('-cuda',true,'whether to use cuda')
cmd:option('-max_history',50,'max history')
cmd:option('-max_sent_len',50,'max sent len')

function main()
  -- Parse input params
  local D0 = 20 --no. of nodes in output LUT
  opt = cmd:parse(arg)
  if opt.modelfile == '' then
      print("Please load a valid trained model!")
  end
  if opt.cuda then
      require("cunn")
      --print("Using Cuda")
  else
      --print("Using CPU")
  end
  load()
  te_mask = torch.ones(opt.max_history, D0)
  model = torch.load(opt.modelfile) --loading the saved trained model
  --print(string.format("Using the model: %s", opt.modelfile))
  user_queries() --call interactive session
end

function user_queries()  
  
  colors = require 'ansicolors'
  print(colors("\n%{bluebg,bright,underline}Welcome to ITC Policies Search Engine"))
  testX = test_stories
  testQ = test_questions
  testA = test_answers
  
  if opt.cuda then
    testX = testX:cuda()
    testQ = testQ:cuda()
  end

  local Y_hat = torch.zeros(testA:size(1)) --initializes "answer" probabilities with zeros

  --Interactive session begins
  local answer, query
  csv = require("csv")
  filePath = '/home/naveen/Documents/MemN2N/word_to_idx.csv'
  repeat
    q_id = 0
    bool = true
    exit = false
    --input a query
    print(colors("%{yellow,bright}\nHow may I help you?\n"))
    io.flush()
    query = io.read()
    query = string.lower(query)

    if not (string.find(string.sub(query, -1, -1), '[a-z]') or string.find(string.sub(query, -1, -1), '[0-9]')) then 
	query = string.sub(query, 1, -2) --removes "?"
    end
    
    if query == "exit" or query == "thank you" or query == "no thanks" or query == "thanks" or query == "no" then 
	print(colors("%{yellow}\nThank you for using the portal!\n")) exit = true
	goto next_iter
    end
 
    question_idx = {} --encoded question tensor
    question_idx[1] = 3
    for i in string.gmatch(query, "%S+") do
	f = csv.open(filePath,{separator=","})
	for fields in f:lines(".*") do
        	if fields[1] == i then
			table.insert(question_idx, tonumber(fields[2]))
                end
        end
	f:close()
    end
    
    question_idx[#question_idx+1] = 4 --#question_idx size of the encoded query, add end token
    for i=1,pad_qlen do
    	if question_idx[i] == nil then question_idx[i] = 2 end --padding
    end
   
    for i=1,testQ:size(1) do
    	local c=0
        for j=1,testQ:size(2) do
        	if testQ[i][j] == question_idx[j] then
           		c = c+1
           	end
        end
        if c == pad_qlen then
    	bool = true
                q_id = i
                break
        end
    end
    if not bool then
        print(colors("%{red,bright}Sorry! Processing failed. Please retry."))
    	goto next_iter
    end
    
    if opt.cuda then question_idx = torch.CudaTensor(question_idx) end
    
    x = getStory(testX, opt.max_history, opt.max_sent_len)
    preds = model.model:forward({x, question_idx, te_mask})

    m, res = torch.max(preds:float(),1)
    
    f = csv.open('/home/naveen/Documents/MemN2N/word_to_idx.csv',{separator=","})
    for fields in f:lines(".*") do
    	local a = tonumber(fields[2])
        if a == res[1] then pred_facts = fields[1] end
    end

    --Printing answer
    for i=1,#pred_facts do
	fact_num = tonumber(string.sub(pred_facts,i,i)) 
    	for j=1,opt.max_sent_len do
                if tonumber(x[fact_num][j]) ~= 3 and tonumber(x[fact_num][j]) ~= 4 and tonumber(x[fact_num][j]) ~= 2 then
			f = csv.open(filePath,{separator=","})
			for fields in f:lines(".*") do
	    			local a = tonumber(fields[2])
	        		if a == tonumber(x[fact_num][j]) then io.write(colors('%{green,bright}'..fields[1])) io.write(" ") end
    			end
			f:close()
		end
    	end
        io.write("\n")
    end
    io.write("\n")
    ::next_iter::
  until exit
  --Interactive session ends
end

function getStory(X,q_id,max_history,max_sent_len)
  local story = X[ {q_id, {num_history - max_history+1, max_history}, {1, max_sent_len} }]
  
  -- detect empty memories and clear out theta potentials
  local num_empty_sentences = torch.sum(story[{{},1}]:eq(idx_pad))

  if te_mask then
    te_mask:fill(1)
    if num_empty_sentences > 0 then
      te_mask[{{1,num_empty_sentences}}]:fill(0)
    end
  end
  return story
end

function load()
  -- get the data out of the datafile
  local fd = hdf5.open('new_qa20.hdf5', 'r')
  local data = fd:all()
  idx_start = data.idx_start[1]
  idx_end   = data.idx_end[1]
  idx_pad   = data.idx_pad[1]
  idx_rare  = data.idx_rare[1]
  pad_qlen  = data.max_pad_qlen[1]
  
  test_stories   = data.test_stories:long()
  test_questions = data.test_questions:long()
  test_answers   = data.test_answers:long()

  num_history = test_stories:size(2)
  len_sentence = test_stories:size(3)

  opt.max_history = math.min(opt.max_history, num_history)
  opt.max_sent_len = math.min(opt.max_sent_len, len_sentence)
end

main()
