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
    ('while True func(x:=1)', WhileLoopASTNode(
        condition=BooleanLiteralASTNode(value=True),
        body=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
            function_name='func'
        ))),
    ('while False xor True return 3', WhileLoopASTNode(
        condition=XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True)),
        body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=3)]))),
    ('''
while not False {
    Nothing
}
''',
     WhileLoopASTNode(
         condition=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=False)),
         body=ScopedStatementSequenceASTNode(statements=[NothingTypeLiteralASTNode()]))),
    ('while False for x:(1,10, 2) 1', WhileLoopASTNode(
        condition=BooleanLiteralASTNode(value=False),
        body=ForLoopASTNode(
            body=IntegerLiteralASTNode(value=1),
            iterator_variable_name='x',
            minimum=IntegerLiteralASTNode(value=1),
            supremum=IntegerLiteralASTNode(value=10),
            delta=IntegerLiteralASTNode(value=2)))),
    ('while False for x:(1,10, 2) if True 1 else 2', WhileLoopASTNode(
        condition=BooleanLiteralASTNode(value=False),
        body=ForLoopASTNode(
            body=ConditionalASTNode(
                condition=BooleanLiteralASTNode(value=True),
                then_body=IntegerLiteralASTNode(value=1),
                else_body=IntegerLiteralASTNode(value=2)
            ),
            iterator_variable_name='x',
            minimum=IntegerLiteralASTNode(value=1),
            supremum=IntegerLiteralASTNode(value=10),
            delta=IntegerLiteralASTNode(value=2)))),
)

@pytest.mark.parametrize("input_string,expected_result", EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_while_loop(input_string, expected_result):
    module_node = parser.parseSourceCode(input_string)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    result = only_one(module_node.statements)
    assert isinstance(result, WhileLoopASTNode)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
