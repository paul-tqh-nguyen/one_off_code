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
    ('', ModuleASTNode(statements=[])),
    ('   ', ModuleASTNode(statements=[])),
    ('\t   \n', ModuleASTNode(statements=[])),
    ('\n\n\n \t   \n', ModuleASTNode(statements=[])),
    ('1 + 2',
     ModuleASTNode(statements=[
         AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
     ])),
    ('x = 1 ; 1 + 2 ; x: Integer = 1 ; Nothing',
     ModuleASTNode(statements=[
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
         NothingTypeLiteralASTNode(),
     ])),
    ('x = 1 ; x: Integer = 1 ; 1 + 2 # y = 123',
     ModuleASTNode(statements=[
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
         AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
     ])
    ),
    ('''

# comment

x
Nothing
x = 1 # comment
x: Integer = 1 # comment
init_something()
init_something(x:=x)
x = 1 ; x: Integer = 1 # y = 123
1 + 2
{ x = 1 ; x: Integer = 1 } # y = 123
{ x: Integer = 1 } ; 123 # y = 123
x = 1 ; { x: Integer = 1 } ; 123 # y = 123
x = 1 ; {
    x: Integer = 1
    x: Integer = 1 # comment
} # y = 123
x = 1 ; { x: Integer = 1
} # y = 123

for x:(1,10, 2) {
    True or True
    return
}

for x\
:\
(1,10, 2) {
    True or True
    return
}

''',
     ModuleASTNode(statements=[
         VariableASTNode(name='x'),
         NothingTypeLiteralASTNode(),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
         FunctionCallExpressionASTNode(arg_bindings=[], function_name='init_something'),
         FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), VariableASTNode(name='x'))], function_name='init_something'),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
         AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
         ScopedStatementSequenceASTNode(statements=[
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
         ]),
         ScopedStatementSequenceASTNode(statements=[
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
         ]),
         IntegerLiteralASTNode(value=123),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         ScopedStatementSequenceASTNode(statements=[
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
         ]),
         IntegerLiteralASTNode(value=123),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         ScopedStatementSequenceASTNode(statements=[
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
         ]),
         AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
         ScopedStatementSequenceASTNode(statements=[
             AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
         ]),
         ForLoopASTNode(
             body=ScopedStatementSequenceASTNode(statements=[
                 OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True)),
                 ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
             ]),
             iterator_variable_name='x',
             minimum=IntegerLiteralASTNode(value=1),
             supremum=IntegerLiteralASTNode(value=10),
             delta=IntegerLiteralASTNode(value=2)
         ),
         ForLoopASTNode(
             body=ScopedStatementSequenceASTNode(statements=[
                 OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True)),
                 ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
             ]),
             iterator_variable_name='x',
             minimum=IntegerLiteralASTNode(value=1),
             supremum=IntegerLiteralASTNode(value=10),
             delta=IntegerLiteralASTNode(value=2)
         ),
     ])),
)

@pytest.mark.parametrize("input_string,expected_result", EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_module(input_string, expected_result):
    result = parser.parseSourceCode(input_string)
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
