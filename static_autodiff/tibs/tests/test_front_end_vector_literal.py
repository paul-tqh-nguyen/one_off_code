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

TEST_CASES = (
    (
        '[1,2,3]',
        VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
        TensorTypeASTNode(base_type_name='Integer', shape=[3])
    ),
    (
        '[1,2,\
 3]',
        VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
        TensorTypeASTNode(base_type_name='Integer', shape=[3])
    ),
    (
        '[[1,2,3], [4,5,6], [7,8,9]]',
        VectorExpressionASTNode(values=[
            VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
            VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=4), IntegerLiteralASTNode(value=5), IntegerLiteralASTNode(value=6)]),
            VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=7), IntegerLiteralASTNode(value=8), IntegerLiteralASTNode(value=9)])
        ]),
        TensorTypeASTNode(base_type_name='Integer', shape=[3, 3])
    ),
    (
        '[[[1,2], [3, 4]], [[5,6], [7,8]]]',
        VectorExpressionASTNode(values=[
            VectorExpressionASTNode(values=[
                VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2)]),
                VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=3), IntegerLiteralASTNode(value=4)])]),
            VectorExpressionASTNode(values=[
                VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=5), IntegerLiteralASTNode(value=6)]),
                VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=7), IntegerLiteralASTNode(value=8)])])
        ]),
        TensorTypeASTNode(base_type_name='Integer', shape=[2,2,2])
    ),
    (
        '[1.1, 2.2, 3.3]',
        VectorExpressionASTNode(values=[FloatLiteralASTNode(value=1.1), FloatLiteralASTNode(value=2.2), FloatLiteralASTNode(value=3.3)]),
        TensorTypeASTNode(base_type_name='Float', shape=[3])
    ),
    (
        '[1.1 + 2.2, 3.3 * 4.4]',
        VectorExpressionASTNode(values=[
            AdditionExpressionASTNode(left_arg=FloatLiteralASTNode(value=1.1), right_arg=FloatLiteralASTNode(value=2.2)),
            MultiplicationExpressionASTNode(left_arg=FloatLiteralASTNode(value=3.3), right_arg=FloatLiteralASTNode(value=4.4))
        ]),
        TensorTypeASTNode(base_type_name='Float', shape=[2])
    ),
    (
        '[True, False, True or False]',
        VectorExpressionASTNode(values=[
            BooleanLiteralASTNode(value=True),
            BooleanLiteralASTNode(value=False),
            OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=False))]),
        TensorTypeASTNode(base_type_name='Boolean', shape=[3])
    ),
)

@pytest.mark.parametrize('input_string, parse_result, expected_type', TEST_CASES)
def test_parser_vector_literal(input_string, parse_result, expected_type):
    del expected_type
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
    result = assignment_node.value
    assert result == parse_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
parse_result: {repr(parse_result)}
'''

@pytest.mark.parametrize('input_string, parse_result, expected_type', TEST_CASES)
def test_type_inference_vector_literal(input_string, parse_result, expected_type):
    del parse_result
    module_node = parser.parseSourceCode('x = '+input_string)
    type_inference.perform_type_inference(module_node)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert isinstance(tensor_type_node, TensorTypeASTNode)
    assert tensor_type_node == expected_type
