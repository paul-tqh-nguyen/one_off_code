import pytest

from tibs import parser
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

EXPECTED_INPUT_OUTPUT_PAIRS = (
    ('', NothingTypeLiteralASTNode()),
    ('  \
\t # comment', NothingTypeLiteralASTNode()),
    ('Nothing', NothingTypeLiteralASTNode()),
    ('False', BooleanLiteralASTNode(value=False)),
    ('123', IntegerLiteralASTNode(value=123)),
    ('1E2', FloatLiteralASTNode(value=100.0)),
    ('some_variable', VariableASTNode(name='some_variable')),
    ('[f(x:=1), [2, 3], some_variable, Nothing]',
     VectorExpressionASTNode(values=[
         FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
         VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
         VariableASTNode(name='some_variable'),
         NothingTypeLiteralASTNode()])),
    ('False or True', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True))),
    ('1 ** 2 ^ 3',
     ExponentExpressionASTNode(
         left_arg=IntegerLiteralASTNode(value=1),
         right_arg=ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=2),
             right_arg=IntegerLiteralASTNode(value=3)))),
    ('f(a:=1, b:=2, c:=Nothing)',
     FunctionCallExpressionASTNode(
         arg_bindings=[
             (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
             (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2)),
             (VariableASTNode(name='c'), NothingTypeLiteralASTNode()),
         ],
         function_name='f')),
)

@pytest.mark.parametrize("input_string,expected_result", EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_return_statement(input_string, expected_result):
    module_node = parser.parseSourceCode(f'''
function f() -> NothingType {{
    return {input_string}
}}
''')
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    function_definition_node = only_one(module_node.statements)
    assert isinstance(function_definition_node, FunctionDefinitionASTNode)
    assert function_definition_node.function_name == 'f'
    assert function_definition_node.function_signature == []
    function_body = function_definition_node.function_body
    assert isinstance(function_body, ScopedStatementSequenceASTNode)
    return_statement_node = only_one(function_body.statements)
    assert isinstance(return_statement_node, ReturnStatementASTNode)
    result = only_one(return_statement_node.return_values)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
