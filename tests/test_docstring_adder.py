import unittest
from src.docstring_adder import DocstringAdder

filename = "tests/fixtures/src/dataset/dataset_splitter.py"

class TestDocstringAdder(unittest.TestCase):
  def setUp(self):
    self.adder = DocstringAdder()
    
  def test_AddDocstrings(self):
    params_list, end_lines, columns = self.adder(filename=filename)
    assert params_list == [['dataset', 'val_size', 'train_transform', 'val_transform'], [], ['key', 'val']]
    assert end_lines == [10,11,14]  
    assert columns == [4, 8, 8]
    
if __name__ == '__main__':
  test = TestDocstringAdder()
  test.setUp()
  test.test_AddDocstrings()