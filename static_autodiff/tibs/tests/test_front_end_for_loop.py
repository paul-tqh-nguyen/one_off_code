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
    ('for x:(1,10,y) func(x:=1)', ForLoopASTNode(
        body=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
            function_name='func'
        ),
        iterator_variable_name='x',
        minimum=IntegerLiteralASTNode(value=1),
        supremum=IntegerLiteralASTNode(value=10),
        delta=VariableASTNode(name='y'))),
    ('for x:(1,10) func(x:=1)', ForLoopASTNode(
        body=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
            function_name='func'
        ),
        iterator_variable_name='x',
        minimum=IntegerLiteralASTNode(value=1),
        supremum=IntegerLiteralASTNode(value=10),
        delta=IntegerLiteralASTNode(value=1))),
    ('''
for x:(1,10, 2) 
    func(x:=1)
''',
     ForLoopASTNode(
         body=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='func'),
         iterator_variable_name='x',
         minimum=IntegerLiteralASTNode(value=1),
         supremum=IntegerLiteralASTNode(value=10),
         delta=IntegerLiteralASTNode(value=2))),
    ('''
for x:(1+0,10) {
    Nothing
}
''',
     ForLoopASTNode(
         body=ScopedStatementSequenceASTNode(statements=[NothingTypeLiteralASTNode()]),
         iterator_variable_name='x',
         minimum=AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=0)),
         supremum=IntegerLiteralASTNode(value=10),
         delta=IntegerLiteralASTNode(value=1))
    ),
    ('''
for x:(1, -10, -1) {
    True or True
}
''',
     ForLoopASTNode(
         body=ScopedStatementSequenceASTNode(statements=[
             OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True)),
         ]),
         iterator_variable_name='x',
         minimum=IntegerLiteralASTNode(value=1),
         supremum=NegativeExpressionASTNode(arg=IntegerLiteralASTNode(value=10)),
         delta=NegativeExpressionASTNode(arg=IntegerLiteralASTNode(value=1))
     )),
    ('for x:(1,10, 2) while False 1', ForLoopASTNode(
        body=WhileLoopASTNode(
            condition=BooleanLiteralASTNode(value=False),
            body=IntegerLiteralASTNode(value=1)),
        iterator_variable_name='x',
        minimum=IntegerLiteralASTNode(value=1),
        supremum=IntegerLiteralASTNode(value=10),
        delta=IntegerLiteralASTNode(value=2))),
)

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_for_loop(input_string, expected_result):
    module_node = parser.parseSourceCode(input_string)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, ForLoopASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_type_inference_for_loop(input_string, expected_result):
    '''Type inference should be a no-op.'''
    module_node = parser.parseSourceCode(input_string)
    type_inference.perform_type_inference(
        module_node,
        {
            'y': TensorTypeASTNode(base_type_name='Integer', shape=[]),
            'func': FunctionDefinitionASTNode(
                function_name='func',
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=True)])
            )
        }
    )
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, ForLoopASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
