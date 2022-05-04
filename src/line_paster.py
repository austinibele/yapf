
from sys import prefix


class LinePaster:
  def __init__(self, tab_spaces=4):
    self.tab_spaces = tab_spaces

  def _get_file_lines(self, filename):
    with open(filename) as file:
        file_lines = file.readlines()
        file_lines = [line.rstrip() for line in file_lines]
    return file_lines  
  
  def _create_docstring(self, params, column):
    space = ' '
    prefix = space*(column+self.tab_spaces)
    
    new_lines = []
    new_lines.append(prefix + "'''")
    new_lines.append(prefix + "desc*")
    new_lines.append(prefix)
    new_lines.append(prefix + "Args:")
    for i, param in enumerate(params):
      line = prefix + space*4 + param + space*4 + "(type*): "
      new_lines.append(line)
            
    new_lines.append(prefix)
    new_lines.append(prefix + "'''")
    return new_lines

      
  def _paste_docstring(self, docstring, file_lines, end_line):
    idx = end_line + self.lines_added
    self.lines_added += len(docstring)
    for i, line in enumerate(docstring):
      file_lines.insert(idx+i, line)
    return file_lines

  
  def _write_file(self, file_lines, filename):
    with open('test.py', 'w') as f:
        f.write('\n'.join(file_lines))
  
  def __call__(self, filename, params_list, end_lines, columns):
    file_lines = self._get_file_lines(filename) 
    
    self.lines_added = 0
    for i, params in enumerate(params_list):
      docstring = self._create_docstring(params, columns[i])
      file_lines = self._paste_docstring(docstring, file_lines, end_lines[i])
    
    self.lines_added = 0
    self._write_file(file_lines=file_lines, filename=filename)