# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Comment splicer for lib2to3 trees.

The lib2to3 syntax tree produced by the parser holds comments and whitespace in
prefix attributes of nodes, rather than nodes themselves. This module provides
functionality to splice comments out of prefixes and into nodes of their own,
making them easier to process.

  SpliceComments(): the main function exported by this module.
"""

from lib2to3 import pygram
from lib2to3 import pytree
from lib2to3.pgen2 import token
from yapf.pytree import pytree_utils
from src.param_finder import ParamFinder

class DocstringAdder:
  def __init__(self):
    self.start_lines = []
    self.tups = []
    self.columns = []
    self.param_finder = ParamFinder()
      
  def AddDocstrings(self, tree):
    """Given a pytree, splice comments into nodes of their own right.

    Extract comments from the prefixes where they are housed after parsing.
    The prefixes that previously housed the comments become empty.

    Args:
      tree: a pytree.Node - the tree to work on. The tree is modified by this
          function.
    """
    # The previous leaf node encountered in the traversal.
    # This is a list because Python 2.x doesn't have 'nonlocal' :)
    
    self._find_def_start_lines(tree)
    self._get_tups(tree)

    self.param_finder(start_lines = self.start_lines, tups = self.tups)


  def _get_tups(self, node):
    for i, child in enumerate(node.children[:]):
      if isinstance(child, pytree.Node):
        # Nodes don't have prefixes.
        self._go_deeper_end(child)
      else:
        self.tups.append((int(child.lineno), str(child).strip()))
          
  def _go_deeper_end(self, node):
    self._get_tups(node=node)

  def _go_deeper_start(self, node):
    self._find_def_start_lines(node=node)
      
  def _find_def_start_lines(self, node):
    for child in node.children[:]:
      if isinstance(child, pytree.Node):
        # Nodes don't have prefixes.
        self._go_deeper_start(child)
      else:
        if str(child).lstrip()[0:3] == 'def':
          if int(child.lineno) not in self.start_lines:            
            self.start_lines.append(int(child.lineno))
            self.columns.append(child.column)

    