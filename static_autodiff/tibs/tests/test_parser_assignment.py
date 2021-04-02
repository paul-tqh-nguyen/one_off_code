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

# TODO make sure all these imports are used

EXPECTED_INPUT_OUTPUT_PAIRS = (
    ('x = 1', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1))),
    ('x, y = f(a:=12)', AssignmentASTNode(variable_type_pairs=[
        (VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None)),
        (VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))
    ], value=FunctionCallExpressionASTNode(
        arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=12))],
        function_name='f')
    )),
    ('x: NothingType = Nothing', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode())),
    ('x = Nothing', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=NothingTypeLiteralASTNode())),
    ('x: NothingType<> = Nothing', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode())),
    ('x: NothingType<1,?> = Nothing', AssignmentASTNode(
        variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[IntegerLiteralASTNode(value=1), None]))],
        value=NothingTypeLiteralASTNode())),
    ('x: Integer = 1', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))),
    ('x: Integer<> = 1', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))),
    ('x: Integer<2,3,4> = 1', AssignmentASTNode(
        variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(
            base_type_name='Integer',
            shape=[
                IntegerLiteralASTNode(value=2),
                IntegerLiteralASTNode(value=3),
                IntegerLiteralASTNode(value=4),
            ]))],
        value=IntegerLiteralASTNode(value=1)
    )),
    ('x: Float<?> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None]))], value=VariableASTNode(name='value'))),
    ('x: Float<???> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value'))),
    ('x: Float<?, ?, ?> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]))], value=VariableASTNode(name='value'))),
    ('x: Boolean<?, 3, ?> = value', AssignmentASTNode(
        variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[None, IntegerLiteralASTNode(value=3), None]))],
        value=VariableASTNode(name='value')
    )),
    ('x: Float<??\
?> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value'))),
)

@pytest.mark.parametrize('input_string,expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_assignment(input_string, expected_result):
    module_node = parser.parseSourceCode(input_string)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    for variable_node, tensor_type_node in assignment_node.variable_type_pairs:
        assert isinstance(variable_node, VariableASTNode)
        assert isinstance(variable_node.name, str)
        assert isinstance(tensor_type_node, TensorTypeASTNode)
    result = assignment_node
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
