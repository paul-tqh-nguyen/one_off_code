
'''

This module contains test for the shallow convertion from ASTs to MLIR. No passes are tested here.

'''

# TODO update this doc string.

import pytest

from tibs import parser
from tibs.misc_utilities import *

# TODO make sure all these imports are used

# def test_mlir_from_print():
#     input_string = '''
# x = [[1,2,3], [4,5,6], [7,8,9]]
# print x
# '''
#     module_ast = parser.parseSourceCode(input_string)
#     mlir_string = module_ast.emit_mlir()
#     print(f"mlir_string {repr(mlir_string)}")
#     assert False
