import pytest

from tibs import parser

VALID_IDENTIFIERS = (
        'var',
        'var_1',
        '_var',
        '_var_',
        '_var_1',
        'var_1_',
        '_var_1_',
        '_12345_',
        'VAR',
        'Var',
        'vAr213feEF',
        'vAr21\
3feEF',
)

@pytest.mark.parametrize('input_string', VALID_IDENTIFIERS)
def test_parser_atomic_identifier(input_string):
    module_node = parser.parseSourceCode(input_string+' = 1')
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    assert assignment_node.value == IntegerLiteralASTNode(value=1)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert tensor_type_node.base_type_name is None
    assert tensor_type_node.shape is None
    assert isinstance(variable_node, VariableASTNode)
    result = variable_node.name
    assert type(result) is str
    assert result == input_string, f'''
input_string: {repr(input_string)}
result: {repr(result)}
'''
