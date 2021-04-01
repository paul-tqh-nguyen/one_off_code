import pytest

from tibs import parser
from tibs.misc_utilities import *

EXPECTED_INPUT_OUTPUT_PAIRS = (
    pytest.param('True', True),
    pytest.param('False', False),
    pytest.param('Fal\
se', False),
)

@pytest.mark.parametrize("input_string,expected_result", EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_atomic_boolean(input_string, expected_result):
    module_node = parser.parseSourceCode('x = '+input_string)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert isinstance(tensor_type_node, TensorTypeASTNode)
    assert tensor_type_node.base_type_name is None
    assert tensor_type_node.shape is None
    value_node = assignment_node.value
    assert isinstance(value_node, BooleanLiteralASTNode)
    result = value_node.value
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
