import pytest

from tibs import parser, type_inference
from tibs.ast_node import (
    PrintStatementASTNode,
    ComparisonExpressionASTNode,
    ExpressionASTNode,
    TensorTypeASTNode,
    VectorExpressionASTNode,
    FunctionDefinitionASTNode,
    ForLoopASTNode,
    WhileLoopASTNode,
    ConditionalASTNode,
    ScopedStatementSequenceASTNode,
    ReturnStatementASTNode,
    BooleanLiteralASTNode,
    IntegerLiteralASTNode,
    FloatLiteralASTNode,
    StringLiteralASTNode,
    NothingTypeLiteralASTNode,
    VariableASTNode,
    NegativeExpressionASTNode,
    ExponentExpressionASTNode,
    MultiplicationExpressionASTNode,
    DivisionExpressionASTNode,
    AdditionExpressionASTNode,
    SubtractionExpressionASTNode,
    NotExpressionASTNode,
    AndExpressionASTNode,
    XorExpressionASTNode,
    OrExpressionASTNode,
    GreaterThanExpressionASTNode,
    GreaterThanOrEqualToExpressionASTNode,
    LessThanExpressionASTNode,
    LessThanOrEqualToExpressionASTNode,
    EqualToExpressionASTNode,
    NotEqualToExpressionASTNode,
    StringConcatenationExpressionASTNode,
    FunctionCallExpressionASTNode,
    AssignmentASTNode,
    ModuleASTNode,
) # TODO reorder these in according to their declaration
from tibs.misc_utilities import *

# TODO make sure all these imports are used

EXPECTED_INPUT_OUTPUT_PAIRS = (
    pytest.param('', ''),
    pytest.param('adsf', 'adsf'),
    pytest.param('  adsf  ', '  adsf  '),
    pytest.param(r'  ad\"sf  ', r'  ad"sf  '),
    pytest.param('ad\nsf', 'ad\nsf'),
    pytest.param(r'''a
d
s
f''', 'a\nd\ns\nf'),
)

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_atomic_string(input_string, expected_result):
    module_node = parser.parseSourceCode(f'x = "{input_string}"')
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
    assert isinstance(value_node, StringLiteralASTNode)
    result = value_node.value
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_type_inference_atomic_string(input_string, expected_result):
    module_node = parser.parseSourceCode(f'x = "{input_string}"')
    module_node = type_inference.perform_type_inference(module_node)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert tensor_type_node == TensorTypeASTNode(base_type_name='String', shape=[])
