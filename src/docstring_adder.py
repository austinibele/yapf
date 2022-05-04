from lib2to3 import pytree

from pytest import param
from src.param_finder import ParamFinder
from yapf.pytree import pytree_utils
from yapf.yapflib import file_resources
from yapf.yapflib import py3compat

class DocstringAdder:
  def __init__(self):
    self.start_lines = []
    self.tups = []
    self.columns = []
    self.param_finder = ParamFinder()
      

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

  def __call__(self, filename):
    """Given a pytree, splice comments into nodes of their own right.

    Extract comments from the prefixes where they are housed after parsing.
    The prefixes that previously housed the comments become empty.

    Args:
      tree: a pytree.Node - the tree to work on. The tree is modified by this
          function.
    """
    original_source, newline, encoding = ReadFile(filename, logger=None)
    tree = pytree_utils.ParseCodeToTree(original_source)
    self._find_def_start_lines(tree)
    self._get_tups(tree)
    params_list, end_lines = self.param_finder(start_lines=self.start_lines, tups=self.tups)
    return params_list, end_lines, self.columns
    


def ReadFile(filename, logger=None):
  """Read the contents of the file.

  An optional logger can be specified to emit messages to your favorite logging
  stream. If specified, then no exception is raised. This is external so that it
  can be used by third-party applications.

  Arguments:
    filename: (unicode) The name of the file.
    logger: (function) A function or lambda that takes a string and emits it.

  Returns:
    The contents of filename.

  Raises:
    IOError: raised if there was an error reading the file.
  """
  try:
    encoding = file_resources.FileEncoding(filename)

    # Preserves line endings.
    with py3compat.open_with_encoding(
        filename, mode='r', encoding=encoding, newline='') as fd:
      lines = fd.readlines()

    line_ending = file_resources.LineEnding(lines)
    source = '\n'.join(line.rstrip('\r\n') for line in lines) + '\n'
    return source, line_ending, encoding
  except IOError as e:  # pragma: no cover
    if logger:
      logger(e)
    e.args = (e.args[0], (filename, e.args[1][1], e.args[1][2], e.args[1][3]))
    raise
  except UnicodeDecodeError as e:  # pragma: no cover
    if logger:
      logger('Could not parse %s! Consider excluding this file with --exclude.',
             filename)
      logger(e)
    e.args = (e.args[0], (filename, e.args[1][1], e.args[1][2], e.args[1][3]))
    raise