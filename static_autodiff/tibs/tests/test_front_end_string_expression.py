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

EXPECTED_INPUT_OUTPUT_PAIRS = tuple(pytest.param(*args, id=f'case_{i}') for i, args in enumerate([
    (' "asd" << "dsa" ', StringConcatenationExpressionASTNode(left_arg=StringLiteralASTNode(value='asd'), right_arg=StringLiteralASTNode(value='dsa'))),
    (' "asd" << variable ', StringConcatenationExpressionASTNode(left_arg=StringLiteralASTNode(value='asd'), right_arg=VariableASTNode(name='variable'))),
    (' variable << "asd" ', StringConcatenationExpressionASTNode(left_arg=VariableASTNode(name='variable'), right_arg=StringLiteralASTNode(value='asd'))),
    (' variable << variable ', StringConcatenationExpressionASTNode(left_arg=VariableASTNode(name='variable'), right_arg=VariableASTNode(name='variable'))),
    (' f(x:=1) << variable ', StringConcatenationExpressionASTNode(
        left_arg=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
        right_arg=VariableASTNode(name='variable'))),
    (' variable << f(x:=1) ', StringConcatenationExpressionASTNode(
        left_arg=VariableASTNode(name='variable'),
        right_arg=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'))),
    (' f(x:=1) << f(x:=1) ', StringConcatenationExpressionASTNode(
        left_arg=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
        right_arg=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'))),
    (' f(x:=1) << "asd" ', StringConcatenationExpressionASTNode(
        left_arg=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
        right_arg=StringLiteralASTNode(value='asd'))),
    (' "asd" << f(x:=1) ', StringConcatenationExpressionASTNode(
        left_arg=StringLiteralASTNode(value='asd'),
        right_arg=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'))),
    (' [["a", "b"], ["c", "d"]] << [["A", "B"], ["C", "D"]] ', StringConcatenationExpressionASTNode(
        left_arg=VectorExpressionASTNode(values=[
            VectorExpressionASTNode(values=[
                StringLiteralASTNode(value='a'),
                StringLiteralASTNode(value='b'),
            ]),
            VectorExpressionASTNode(values=[
                StringLiteralASTNode(value='c'),
                StringLiteralASTNode(value='d'),
            ])
        ]),
        right_arg=VectorExpressionASTNode(values=[
            VectorExpressionASTNode(values=[
                StringLiteralASTNode(value='A'),
                StringLiteralASTNode(value='B'),
            ]),
            VectorExpressionASTNode(values=[
                StringLiteralASTNode(value='C'),
                StringLiteralASTNode(value='D'),
            ])
        ])
    ))
]))

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_string_expression(input_string, expected_result):
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
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_type_inference_string_expression(input_string, expected_result):
    module_node = parser.parseSourceCode('x = '+input_string)
    module_node = type_inference.perform_type_inference(module_node, {
        'f': FunctionDefinitionASTNode(
            function_name='f',
            function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
            function_return_types=[TensorTypeASTNode(base_type_name='String', shape=[])],
            function_body=ReturnStatementASTNode(return_values=[StringLiteralASTNode(value='blah')])
        ),
        'variable': TensorTypeASTNode(base_type_name='String', shape=[])
    })
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert isinstance(tensor_type_node, TensorTypeASTNode)
    assert tensor_type_node.base_type_name is 'String'
    assert tensor_type_node.shape in ([], [2,2])
    assert all(EMPTY_TENSOR_TYPE_AST_NODE != node for node in module_node.traverse())
