

from asyncio import wait_for


class ParamFinder:
  
  
  def _get_params_list(self, lineno, line_nums, words):
    start_idx = line_nums.index(lineno)
    
    end_line = 0
    params = []
    for i in range(start_idx, len(words)):
      end_line = line_nums[i]
      word = words[i]
      params.append(word)
      if (params[-2:] == [")", ":"]) or (params[-2:] ==  ["]", ":"]):
        break
    return params, end_line
  
  def _tups_to_lists(self, tups):
    line_nums = []
    words = []
    for lineno, word in tups:
      line_nums.append(lineno)
      words.append(word)      
    return line_nums, words

  def _identify_possible_params(self, params_list):
    new_list = []
    for params in params_list:
      temp = []
      passed_def = False
      done = False
      first_param_idx = 0
      for i, param in enumerate(params):
        if param == "def":
          passed_def = True
          first_param_idx = i + 3          
                  
        if passed_def and not done:
          if i >= first_param_idx:
            try:
              if (param == ")" or param == "]") and param[i+1] == ":":
                done = False       
            except IndexError:
              done = False
            if param != "cls" and param != "self":
              temp.append(param)
      new_list.append(temp)
    return new_list
              
  def _remove_non_params(self, params_list):
    keep_list = []
    for params in params_list:
      keep = []
      for i, param in enumerate(params):
        if i == 0:
          keep.append(param)
        else:
          if params[i-1] == "," or params[i-1] == "(":
            try:
              if params[i+1] != "]":
                keep.append(param)
            except:
              pass
      keep_list.append(keep)
    return keep_list
  
  def _filter_non_alpha(self, params_list):
    keep_list = []
    for params in params_list:
      keep = []
      for param in params:
        if param[0].isalpha():
          keep.append(param)
      keep_list.append(keep)
    return keep_list
  
  
  def __call__(self, start_lines, tups):
    line_nums, words = self._tups_to_lists(tups)

    params_list = []
    end_lines = []
       
    for lineno in start_lines:
      params, end_line = self._get_params_list(lineno=lineno, line_nums=line_nums, words=words)
      params_list.append(params)
      end_lines.append(end_line)


    params_list = self._identify_possible_params(params_list=params_list)
    params_list = self._remove_non_params(params_list=params_list)
    params_list = self._filter_non_alpha(params_list=params_list)
    return params_list, end_lines


