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
    ('f() # comment', FunctionCallExpressionASTNode(arg_bindings=[], function_name='f')),
    ('f(a:=1) # comment',
     FunctionCallExpressionASTNode(
         arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
         function_name='f')
    ),
    ('f(a:=1, b:=2)', FunctionCallExpressionASTNode(
        arg_bindings=[
            (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
            (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2))
        ],
        function_name='f')),
    ('f(a:=1e3, b:=y)', FunctionCallExpressionASTNode(
        arg_bindings=[
            (VariableASTNode(name='a'), FloatLiteralASTNode(value=1000.0)),
            (VariableASTNode(name='b'), VariableASTNode(name='y'))
        ],
        function_name='f')),
    ('f(a:=1, b:=2, c:= True)', FunctionCallExpressionASTNode(
        arg_bindings=[
            (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
            (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2)),
            (VariableASTNode(name='c'), BooleanLiteralASTNode(value=True))
        ],
        function_name='f')),
    ('f(a:=1+2, b:= True or False)', FunctionCallExpressionASTNode(
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
        function_name='f')),
    ('f(a := 1, b := g(arg:=True), c := Nothing)', FunctionCallExpressionASTNode(
        arg_bindings=[
            (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
            (VariableASTNode(name='b'), FunctionCallExpressionASTNode(
                arg_bindings=[
                    (VariableASTNode(name='arg'), BooleanLiteralASTNode(value=True)),
                ],
                function_name='g')),
            (VariableASTNode(name='c'), NothingTypeLiteralASTNode())
        ],
        function_name='f')),
)

@pytest.mark.parametrize('input_string,expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_function_call(input_string, expected_result):
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
