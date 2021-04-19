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
    ('if False while True f(x:=1)', ConditionalASTNode(
        condition=BooleanLiteralASTNode(value=False),
        then_body=WhileLoopASTNode(
            condition=BooleanLiteralASTNode(value=True),
            body=FunctionCallExpressionASTNode(
                arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
                function_name='f')),
        else_body=None)),
    ('if True xor False while False and True 3 else \n   5', ConditionalASTNode(
        condition=XorExpressionASTNode(
            left_arg=BooleanLiteralASTNode(value=True),
            right_arg=BooleanLiteralASTNode(value=False)
        ),
        then_body=WhileLoopASTNode(
            condition=AndExpressionASTNode(
                left_arg=BooleanLiteralASTNode(value=False),
                right_arg=BooleanLiteralASTNode(value=True)
            ),
            body=IntegerLiteralASTNode(value=3)),
        else_body=IntegerLiteralASTNode(value=5))),
    ('''
if False {
    while not False {
        if True 1 else { f(x:=2) }
    }
} else 5
''',
     ConditionalASTNode(
         condition=BooleanLiteralASTNode(value=False),
         then_body=ScopedStatementSequenceASTNode(statements=[
             WhileLoopASTNode(
                 condition=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=False)),
                 body=ScopedStatementSequenceASTNode(statements=[
                     ConditionalASTNode(
                         condition=BooleanLiteralASTNode(value=True),
                         then_body=IntegerLiteralASTNode(value=1),
                         else_body=ScopedStatementSequenceASTNode(statements=[
                             FunctionCallExpressionASTNode(
                                 arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=2))],
                                 function_name='f')
                         ])
                     )
                 ])
             )]),
         else_body=IntegerLiteralASTNode(value=5))),
)

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_conditional(input_string, expected_result):
    module_node = parser.parseSourceCode(input_string)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, ConditionalASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_type_inference_conditional(input_string, expected_result):
    '''Type inference should be a no-op.'''
    module_node = parser.parseSourceCode(input_string)
    module_node = type_inference.perform_type_inference(module_node, {
        'f': FunctionDefinitionASTNode(
            function_name='f',
            function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
            function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
            function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=True)])
        )
    })
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, ConditionalASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
