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
        'f() # comment',
        FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            )
        }
    ),
    (
        'f(a:=1) # comment',
        FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
            function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[(VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            )
        }
    ),
    (
        'f(a:=1, b:=2)',
        FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
                (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2))
            ],
            function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[
                    (VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                    (VariableASTNode(name='b'), TensorTypeASTNode(base_type_name='Integer', shape=[]))
                ],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            )
        }
    ),
    (
        'f(a:=1e3, b:=y)',
        FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), FloatLiteralASTNode(value=1000.0)),
                (VariableASTNode(name='b'), VariableASTNode(name='y'))
            ],
            function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[
                    (VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Float', shape=[])),
                    (VariableASTNode(name='b'), TensorTypeASTNode(base_type_name='String', shape=[3, 3]))
                ],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            ),
            'y': TensorTypeASTNode(base_type_name='String', shape=[3, 3])
        }
    ),
    (
        'f(a:=1, b:=2, c:= True)',
        FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
                (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2)),
                (VariableASTNode(name='c'), BooleanLiteralASTNode(value=True))
            ],
            function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[
                    (VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                    (VariableASTNode(name='b'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                    (VariableASTNode(name='c'), TensorTypeASTNode(base_type_name='Boolean', shape=[]))
                ],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            )
        }
    ),
    (
        'f(a:=1+2, b:= True or False)',
        FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), AdditionExpressionASTNode(
                    left_arg=IntegerLiteralASTNode(value=1),
                    right_arg=IntegerLiteralASTNode(value=2)
                )),
                (VariableASTNode(name='b'), OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
            ],
            function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[
                    (VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                    (VariableASTNode(name='b'), TensorTypeASTNode(base_type_name='Boolean', shape=[]))
                ],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            )
        }
    ),
    (
        'f(a := 1, b := g(arg:=True), c := Nothing)',
        FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
                (VariableASTNode(name='b'), FunctionCallExpressionASTNode(
                    arg_bindings=[
                        (VariableASTNode(name='arg'), BooleanLiteralASTNode(value=True)),
                    ],
                    function_name='g')),
                (VariableASTNode(name='c'), NothingTypeLiteralASTNode())
            ],
            function_name='f'),
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[
                    (VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                    (VariableASTNode(name='b'), TensorTypeASTNode(base_type_name='Boolean', shape=[])),
                    (VariableASTNode(name='c'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))
                ],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            ),
            'g': FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[
                    (VariableASTNode(name='arg'), TensorTypeASTNode(base_type_name='Booean', shape=[]))
                ],
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=False)])
            )
        }
    ),
)

@pytest.mark.parametrize('input_string, expected_result, type_inference_table', TEST_CASES)
def test_parser_function_call(input_string, expected_result, type_inference_table):
    del type_inference_table
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
    assert isinstance(result, FunctionCallExpressionASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result, type_inference_table', TEST_CASES)
def test_type_inference_function_call(input_string, expected_result, type_inference_table):
    module_node = parser.parseSourceCode('x = '+input_string)
    module_node = type_inference.perform_type_inference(
        module_node,
        type_inference_table
    )
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert isinstance(tensor_type_node, TensorTypeASTNode)
    assert tensor_type_node == TensorTypeASTNode(base_type_name='NothingType', shape=[])
    result = assignment_node.value
    assert isinstance(result, FunctionCallExpressionASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
    assert all(EMPTY_TENSOR_TYPE_AST_NODE != node for node in module_node.traverse())
