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
    ('not False', NotExpressionASTNode(arg=BooleanLiteralASTNode(value=False))),
    ('not True', NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True))),
    ('not (True)', NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True))),
    ('True and True', AndExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True))),
    ('False and False', AndExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=False))),
    ('True and False', AndExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=False))),
    ('False and True', AndExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True))),
    ('True xor True', XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True))),
    ('False xor False', XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=False))),
    ('True xor False', XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=False))),
    ('False xor True', XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True))),
    ('True or True', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True))),
    ('False or False', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=False))),
    ('True or False', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=False))),
    ('False or True', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True))),
    ('True and True and False and True',
     AndExpressionASTNode(
         left_arg=AndExpressionASTNode(
             left_arg=AndExpressionASTNode(
                 left_arg=BooleanLiteralASTNode(value=True),
                 right_arg=BooleanLiteralASTNode(value=True)
             ),
             right_arg=BooleanLiteralASTNode(value=False)
         ),
         right_arg=BooleanLiteralASTNode(value=True)
     )),
    ('not True and True', AndExpressionASTNode(left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)), right_arg=BooleanLiteralASTNode(value=True))),
    ('not True and True xor False',
     XorExpressionASTNode(
         left_arg=AndExpressionASTNode(
             left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
             right_arg=BooleanLiteralASTNode(value=True)
         ),
         right_arg=BooleanLiteralASTNode(value=False)
     )),
    ('False or not True and True xor False',
     OrExpressionASTNode(
         left_arg=BooleanLiteralASTNode(value=False),
         right_arg=XorExpressionASTNode(
             left_arg=AndExpressionASTNode(
                 left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
                 right_arg=BooleanLiteralASTNode(value=True)
             ),
             right_arg=BooleanLiteralASTNode(value=False)
         )
     )),
    ('True xor False or not True and True xor False',
     OrExpressionASTNode(
         left_arg=XorExpressionASTNode(
             left_arg=BooleanLiteralASTNode(value=True),
             right_arg=BooleanLiteralASTNode(value=False)
         ),
         right_arg=XorExpressionASTNode(
             left_arg=AndExpressionASTNode(
                 left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
                 right_arg=BooleanLiteralASTNode(value=True),
             ),
             right_arg=BooleanLiteralASTNode(value=False)
         )
     )),
    ('True xor (False or not True) and True xor False',
     XorExpressionASTNode(
         left_arg=XorExpressionASTNode(
             left_arg=BooleanLiteralASTNode(value=True),
             right_arg=AndExpressionASTNode(
                 left_arg=OrExpressionASTNode(
                     left_arg=BooleanLiteralASTNode(value=False),
                     right_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
                 ),
                 right_arg=BooleanLiteralASTNode(value=True)
             )
         ),
         right_arg=BooleanLiteralASTNode(value=False)
     )),
    ('not (1==2)', NotExpressionASTNode(arg=EqualToExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)))),
    ('not f(a:=1)', NotExpressionASTNode(arg=FunctionCallExpressionASTNode(
        arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
        function_name='f'))),
)

@pytest.mark.parametrize('input_string,expected_result', EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_boolean_expression(input_string, expected_result):
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
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
