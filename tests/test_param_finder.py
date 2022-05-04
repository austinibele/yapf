import pytest
from src.param_finder import ParamFinder

# @pytest.fixture
# def start_lines():
start_lines = [8, 11, 14]

# @pytest.fixture
# def tups():
tups = [(1, 'from'), (1, 'copy'), (1, 'import'), (1, 'deepcopy'), (1, ''), (2, 'from'), (2, 'typing'), (2, 'import'), (2, 'Tuple'), (2, ''), (3, 'from'), (3, 'torch'), (3, '.'), (3, 'utils'), (3, '.'), (3, 'data'), (3, 'import'), (3, 'Dataset'), (3, ','), (3, 'Subset'), (3, ''), (5, 'class'), (5, 'DatasetSplitter'), (5, ':'), (5, ''), (7, ''), (7, '@'), (7, 'classmethod'), (7, ''), (8, 'def'), (8, 'ordered_split'), (8, '('), (8, 'cls'), (8, ','), (9, 'dataset'), (9, ','), (9, 'val_size'), (9, ','), (9, 'train_transform'), (9, ','), (9, 'val_transform'), (10, ')'), (10, '->'), (10, 'Tuple'), (10, '['), (10, 'Dataset'), (10, ','), (10, 'Dataset'), (10, ']'), (10, ':'), (10, ''), (11, ''), (11, 'def'), (11, 'test'), (11, '('), (11, ')'), (11, ':'), (11, ''), (12, ''), (12, 'pass'), (12, ''), (14, ''), (14, 'def'), (14, 'test2'), (14, '('), (14, 'key'), (14, ':'), (14, 'str'), (14, '='), (14, 'None'), (14, ','), (14, 'val'), (14, ':'), (14, 'int'), (14, '='), (14, '5'), (14, ')'), (14, ':'), (14, ''), (15, ''), (15, 'pass'), (15, ''), (17, ''), (17, 'dataset_length'), (17, '='), (17, 'int'), (17, '('), (17, 'dataset'), (17, '.'), (17, '__len__'), (17, '('), (17, ')'), (17, ')'), (17, ''), (18, 'split_idx'), (18, '='), (18, 'int'), (18, '('), (18, 'dataset_length'), (18, '*'), (18, '('), (18, '1'), (18, '-'), (18, 'val_size'), (18, ')'), (18, ')'), (18, ''), (19, 'train_indices'), (19, '='), (19, 'range'), (19, '('), (19, '0'), (19, ','), (19, 'split_idx'), (19, ')'), (19, ''), (20, 'test_indices'), (20, '='), (20, 'range'), (20, '('), (20, 'split_idx'), (20, ','), (20, 'dataset_length'), (20, ')'), (20, ''), (21, 'train'), (21, '='), (21, 'Subset'), (21, '('), (21, 'dataset'), (21, ','), (21, 'train_indices'), (21, ')'), (21, ''), (22, 'train'), (22, '.'), (22, 'dataset'), (22, '.'), (22, 'transform'), (22, '='), (22, 'train_transform'), (22, ''), (23, 'val'), (23, '='), (23, 'Subset'), (23, '('), (23, 'deepcopy'), (23, '('), (23, 'dataset'), (23, ')'), (23, ','), (23, 'test_indices'), (23, ')'), (23, ''), (24, 'val'), (24, '.'), (24, 'dataset'), (24, '.'), (24, 'transform'), (24, '='), (24, 'val_transform'), (24, ''), (25, 'return'), (25, 'train'), (25, ','), (25, 'val'), (25, ''), (28, ''), (28, ''), (28, '')]



def test_call(start_lines, tups):
  param_finder = ParamFinder()
  params_list, end_lines = param_finder(start_lines=start_lines, tups=tups)
  assert params_list == [['dataset', 'val_size', 'train_transform', 'val_transform'], [], ['key', 'val']]
  assert end_lines == [10,11,14]
  
if __name__ == '__main__':
  sl = start_lines
  tps = tups
  test_call(start_lines=sl, tups=tps)