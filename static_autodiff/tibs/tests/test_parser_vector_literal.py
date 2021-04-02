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
    ('[1,2,3]', VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)])),
    ('[1,2,\
 3]', VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)])),
    ('[[1,2,3], [4,5,6], [7,8,9]]',
     VectorExpressionASTNode(values=[
         VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
         VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=4), IntegerLiteralASTNode(value=5), IntegerLiteralASTNode(value=6)]),
         VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=7), IntegerLiteralASTNode(value=8), IntegerLiteralASTNode(value=9)])
     ])),
    ('[[[1,2], [3, 4]], [[5,6], [7,8]]]',
     VectorExpressionASTNode(values=[
         VectorExpressionASTNode(values=[
             VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2)]),
             VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=3), IntegerLiteralASTNode(value=4)])]),
         VectorExpressionASTNode(values=[
             VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=5), IntegerLiteralASTNode(value=6)]),
             VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=7), IntegerLiteralASTNode(value=8)])])
     ])),
    ('[1.1, 2.2, 3.3]', VectorExpressionASTNode(values=[FloatLiteralASTNode(value=1.1), FloatLiteralASTNode(value=2.2), FloatLiteralASTNode(value=3.3)])),
    ('[1.1 + 2.2, 3.3 * 4.4]',
     VectorExpressionASTNode(values=[
         AdditionExpressionASTNode(left_arg=FloatLiteralASTNode(value=1.1), right_arg=FloatLiteralASTNode(value=2.2)),
         MultiplicationExpressionASTNode(left_arg=FloatLiteralASTNode(value=3.3), right_arg=FloatLiteralASTNode(value=4.4))
     ])),
    ('[True, False, True or False]',
     VectorExpressionASTNode(values=[
         BooleanLiteralASTNode(value=True),
         BooleanLiteralASTNode(value=False),
         OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=False))])),
    ('[f(x:=1), [2, 3], some_variable, Nothing, "string"]',
     VectorExpressionASTNode(values=[
         FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
         VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
         VariableASTNode(name='some_variable'),
         NothingTypeLiteralASTNode(),
         StringLiteralASTNode('string'),
     ])),
)

@pytest.mark.parametrize("input_string,expected_result", EXPECTED_INPUT_OUTPUT_PAIRS)
def test_parser_vector_literal(input_string, expected_result):
    for input_string, expected_result in expected_input_output_pairs:
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
