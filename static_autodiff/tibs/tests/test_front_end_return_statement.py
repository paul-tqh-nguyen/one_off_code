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
    ('''Nothing  \
\t # comment''', NothingTypeLiteralASTNode()),
    ('False', BooleanLiteralASTNode(value=False)),
    ('123', IntegerLiteralASTNode(value=123)),
    ('1E2', FloatLiteralASTNode(value=100.0)),
    ('some_variable', VariableASTNode(name='some_variable')),
    ('False or True', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True))),
    ('1 ** 2 ^ 3',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)))),
    ('g(a:=1, b:=2, c:=Nothing)',
     FunctionCallExpressionASTNode(
         arg_bindings=[
             (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
             (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2)),
             (VariableASTNode(name='c'), NothingTypeLiteralASTNode()),
         ],
         function_name='g')),
)

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_return_statement(input_string, expected_result):
    module_node = parser.parseSourceCode(f'''
function f() -> NothingType {{
    {input_string}
    return
}}
''')
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    function_definition_node = only_one(module_node.statements)
    assert isinstance(function_definition_node, FunctionDefinitionASTNode)
    assert function_definition_node.function_name == 'f'
    assert function_definition_node.function_signature == []
    assert function_definition_node.function_return_types == [TensorTypeASTNode(base_type_name='NothingType', shape=[])]
    function_body = function_definition_node.function_body
    assert isinstance(function_body, ScopedStatementSequenceASTNode)
    expression_node, return_statement_node = function_body.statements
    assert isinstance(return_statement_node, ReturnStatementASTNode)
    assert only_one(return_statement_node.return_values) == NothingTypeLiteralASTNode()
    assert expression_node == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

@pytest.mark.parametrize('input_string, expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_type_inference_return_statement(input_string, expected_result):
    del expected_result
    input_string = f'''
function f() -> NothingType {{
    {input_string}
    return
}}
'''
    ast = parser.parseSourceCode(input_string)
    ast_with_type_inference = type_inference.perform_type_inference(ast, {
        'some_variable': TensorTypeASTNode(base_type_name='Boolean', shape=[3]),
        'g': FunctionDefinitionASTNode(
            function_name='g',
            function_signature=[
                (VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                (VariableASTNode(name='b'), TensorTypeASTNode(base_type_name='Integer', shape=[])),
                (VariableASTNode(name='c'), TensorTypeASTNode(base_type_name='NothingType', shape=[])),
            ],
            function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
            function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=True)])
        )
    })
    assert ast == ast_with_type_inference
