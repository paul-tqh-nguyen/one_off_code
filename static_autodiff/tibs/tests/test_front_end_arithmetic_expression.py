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

INTEGER_TEST_CASES = (
    ('1 + 2', AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)), 'Integer'),
    ('1 + 2 - 3',
     SubtractionExpressionASTNode(
         left_arg=AdditionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=IntegerLiteralASTNode(value=2)
         ),
         right_arg=IntegerLiteralASTNode(value=3)
     ),
     'Integer'),
    ('1 + 2 * 3',
     AdditionExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=MultiplicationExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('(1 + 2) * 3',
     MultiplicationExpressionASTNode(
         left_arg=AdditionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=IntegerLiteralASTNode(value=2)
         ),
         right_arg=IntegerLiteralASTNode(value=3)
     ),
     'Integer'),
    ('1 / 2 * 3',
     MultiplicationExpressionASTNode(
         left_arg=DivisionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=IntegerLiteralASTNode(value=2)
         ),
         right_arg=IntegerLiteralASTNode(value=3)
     ),
     'Integer'),
    ('1 / 2 + 3',
     AdditionExpressionASTNode(
         left_arg=DivisionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=IntegerLiteralASTNode(value=2)
         ),
         right_arg=IntegerLiteralASTNode(value=3)
     ),
     'Integer'),
    ('1 / (2 + 3)',
     DivisionExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=AdditionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('1 ^ 2',ExponentExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)), 'Integer'),
    ('1 ^ (2 + 3)',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=AdditionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('1 ** 2 ^ 3',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('1 ^ 2 ** 3',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('1 / 2 ** 3',
     DivisionExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('1 ** 2 - 3',
     SubtractionExpressionASTNode(
         left_arg=ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=IntegerLiteralASTNode(value=2)
         ),
         right_arg=IntegerLiteralASTNode(value=3)
     ),
     'Integer'),
    ('1 ^ (2 + 3)',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=AdditionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)
         )
     ),
     'Integer'),
    ('1 ^ (2 - -3)',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=SubtractionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=3))
         )
     ),
     'Integer'),
    ('1 ** (2 --3)',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=SubtractionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=3))
         )
     ),
     'Integer'),
    ('1 ** f(a:=1)',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=FunctionCallExpressionASTNode(
             arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
             function_name='f')
     ),
     'Integer'),
)

@pytest.mark.parametrize('input_string, expected_result, expected_result_type', INTEGER_TEST_CASES)
def test_parser_arithmetic_expression(input_string, expected_result, expected_result_type):
    del expected_result_type
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
    assert isinstance(result, ExpressionASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result, expected_result_type', INTEGER_TEST_CASES)
def test_type_inference_arithmetic_expression(input_string, expected_result, expected_result_type):
    del expected_result
    module_node = parser.parseSourceCode('x = '+input_string)
    type_inference.perform_type_inference(module_node, {
        'f': FunctionDefinitionASTNode(
            function_body=ScopedStatementSequenceASTNode(statements=[ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])]),
            function_name='f',
            function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
            function_signature=[])
    })
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert isinstance(tensor_type_node, TensorTypeASTNode)
    assert tensor_type_node.base_type_name is expected_result_type
    assert tensor_type_node.shape == []
    