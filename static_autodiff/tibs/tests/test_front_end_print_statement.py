import pytest

from tibs import parser, type_inference
from tibs.ast_node import (
    EMPTY_TENSOR_TYPE_AST_NODE,
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
    ('''print "1
2
3" ''', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('1\n2\n3')])),
    ('print ("a" << "b" << "c") ', PrintStatementASTNode(values_to_print=[
        StringConcatenationExpressionASTNode(
            StringConcatenationExpressionASTNode(StringLiteralASTNode('a'), StringLiteralASTNode('b')),
            StringLiteralASTNode('c')
        ),
    ])),
    ('print "if" ', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('if')])),
    ('print "1" 2 "3"', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('1'), IntegerLiteralASTNode(value=2), StringLiteralASTNode('3')])),
    ('print "1" -2 "3"', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('1'), NegativeExpressionASTNode(IntegerLiteralASTNode(value=2)), StringLiteralASTNode('3')])),
    ('print "1" f(a:=1)', PrintStatementASTNode(values_to_print=[
        StringLiteralASTNode('1'),
        FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
            function_name='f')
    ])),
    ('print True False xor True 3', PrintStatementASTNode(values_to_print=[
        BooleanLiteralASTNode(value=True),
        XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True)),
        IntegerLiteralASTNode(value=3)
    ])),
)

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_print_statement(input_string, expected_result):
    module_node = parser.parseSourceCode(input_string)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, PrintStatementASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_type_inference_print_statement(input_string, expected_result):
    '''Verify that type inference is a no-op.'''
    module_node = parser.parseSourceCode(input_string)
    module_node = type_inference.perform_type_inference(
        module_node,
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[(VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            )
        }
    )
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, PrintStatementASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
    assert all(EMPTY_TENSOR_TYPE_AST_NODE != node for node in module_node.traverse())
