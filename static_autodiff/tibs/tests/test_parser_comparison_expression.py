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
    ('1 < -2', LessThanExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2)))),
    ('1 <= 3-2', LessThanOrEqualToExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=SubtractionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=3), right_arg=IntegerLiteralASTNode(value=2)))),
    ('1/10 > -2', GreaterThanExpressionASTNode(
        left_arg=DivisionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=10)),
        right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2)))),
    ('1*3 >= -2', GreaterThanOrEqualToExpressionASTNode(
        left_arg=MultiplicationExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=3)),
        right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2)))),
    ('1^3 == 3**-2', EqualToExpressionASTNode(
        left_arg=ExponentExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=3)),
        right_arg=ExponentExpressionASTNode(left_arg=IntegerLiteralASTNode(value=3), right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2))))),
    ('01 != -y', NotEqualToExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=NegativeExpressionASTNode(VariableASTNode(name='y')))),
    ('1 < -2 != -y',
     AndExpressionASTNode(
         left_arg=LessThanExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2))),
         right_arg=NotEqualToExpressionASTNode(left_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2)), right_arg=NegativeExpressionASTNode(VariableASTNode(name='y'))))
    ),
    ('1^3 == 3*\
*-2', EqualToExpressionASTNode(
    left_arg=ExponentExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=3)),
    right_arg=ExponentExpressionASTNode(left_arg=IntegerLiteralASTNode(value=3), right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=2))))),
    ('1 < f(a:=1)', LessThanExpressionASTNode(
        left_arg=IntegerLiteralASTNode(value=1),
        right_arg=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
            function_name='f'))),
)

@pytest.mark.parametrize("input_string,expected_result", EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_comparison_expression(input_string, expected_result):
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
    assert isinstance(result, (ComparisonExpressionASTNode, AndExpressionASTNode))
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
