from fileinput import filename
from re import I
import unittest
from src.line_paster import LinePaster

file_name = "tests/fixtures/src/dataset/dataset_splitter.py"

params_list = [['dataset', 'val_size', 'train_transform', 'val_transform'], [], ['key', 'val']]
end_lines = [10,11,14]  
columns = [4, 8, 8]

class TestLinePaster:
  def setUp(self):
    self.line_paster = LinePaster()
    self.file_name = file_name
    self.params_list = params_list
    self.end_lines = end_lines
    self.columns = columns
    
  def test_get_file_lines(self):
    self.line_paster._get_file_lines(self.file_name)

  def test_create_docstring(self):
    docstring = self.line_paster._create_docstring(params=self.params_list[0], column=self.columns[0])
    assert docstring == ["        '''", '        desc*', '        ', '        Args:', '            dataset    (type*): ', '            val_size    (type*): ', '            train_transform    (type*): ', '            val_transform    (type*): ', '        ', "        '''"]
    
  def test_paste_docstring(self):
    docstring = self.line_paster._create_docstring(params=self.params_list[0], column=self.columns[0])
    file_lines = self.line_paster._get_file_lines(self.file_name)
    self.line_paster.lines_added = 0
    new_lines = self.line_paster._paste_docstring(docstring=docstring, file_lines = file_lines, end_line=self.end_lines[0])
    assert new_lines[self.end_lines[0]:self.end_lines[0]+len(docstring)] == docstring
   
  def test_write_file(self):
    file_lines = self.line_paster._get_file_lines(self.file_name)
    self.line_paster._write_file(file_lines=file_lines, filename=None)
    for i in range(len(file_lines)-1):      
      self.line_paster._get_file_lines(self.file_name)[i]
      self.line_paster._get_file_lines("tests/fixtures/outputs/test_write_file.py")[i]
   
  def test_call(self):
      self.line_paster(filename=self.file_name, params_list=self.params_list, end_lines=self.end_lines, columns=self.columns)
    
if __name__ == '__main__':
  test = TestLinePaster()
  test.setUp()
  test.test_get_file_lines()
  test.test_create_docstring()
  test.test_paste_docstring()
  test.test_write_file()
  test.test_call()