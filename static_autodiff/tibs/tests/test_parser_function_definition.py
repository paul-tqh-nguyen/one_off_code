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

INPUT_STRINGS = (
    'function f() -> NothingType {}',
    '''
function f() -> NothingType {
    True
    x = Nothing
    return 
}
''',
    '''
function f(a: Float<1,2,3>, b: NothingType) -> Float<1,2,3> {
    0000
    _var_ = Nothing
    0123
    return a
}
''',
    '''
function f(a: NothingType, b: Boolean<???>) -> Integer<2,2> { # comment
    x: NothingType = Nothing # comment
    y: Boolean<?, 3, ?> = b
    g(x:=x, y:=y)
    return [[[1,2], [3, 4]], [[5,6], [7,8]]] # comment
} # comment
''',
    '''function f(a: NothingType, b: Boolean<???>) -> Integer<2,2> {}''',
    '''

function f(a: NothingType, b: Boolean<???>) -> Integer<2,2> {

}

''',
    '''

function f(a: NothingType, b: Boolean<???>) -> Integer<2,2> {
    for x:(1,10) return func(x:=1)
}

''',
    '''function f(a: NothingType, b: Boolean<???>) -> Integer<2,2> return 123''',
    '''function f(a: NothingType, b: Boolean<???>) -> Integer<2,2> 123''',
    '''
dummy_var # comment

    # comment

function f(a: NothingType, b: Boolean<?, ?, ?>) -> Integer<2,2> g(b:=1, a:=3, b:=123)
''',
)

@pytest.mark.parametrize('input_string', INPUT_STRINGS)
def test_parser_function_definition(input_string):
    # TODO test that the parses are correct
    parser.parseSourceCode(input_string)

# TODO add type inference tests
