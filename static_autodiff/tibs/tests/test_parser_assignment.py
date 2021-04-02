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

TEST_CASES = (
    (
        'x = 1',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
        {}
    ),
    (
        'x, y = f(a:=12)',
        AssignmentASTNode(variable_type_pairs=[
            (VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None)),
            (VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))
        ], value=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=12))],
            function_name='f')
        ),
        AssignmentASTNode(variable_type_pairs=[
            (VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[])),
            (VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Float', shape=[2]))
        ], value=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=12))],
            function_name='f')
        ),
        { # TODO verify that this body is correct
            'f': FunctionDefinitionASTNode(
                function_body=ScopedStatementSequenceASTNode(statements=[ReturnStatementASTNode(return_values=[
                    BooleanLiteralASTNode(value=True),
                    VectorExpressionASTNode(values=[FloatLiteralASTNode(value=2.2), FloatLiteralASTNode(value=3.3)])
                ])]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[]), TensorTypeASTNode(base_type_name='Float', shape=[2])],
                function_signature=[])
        }
    ),
    (
        'x: NothingType = Nothing',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode()),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode()),
        {}
    ),
    (
        'x = Nothing',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=NothingTypeLiteralASTNode()),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode()),
        {}
    ),
    (
        'x: NothingType<> = Nothing',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode()),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode()),
        {}
    ),
    (
        'x: NothingType<1,?> = Nothing',
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[IntegerLiteralASTNode(value=1), None]))],
            value=NothingTypeLiteralASTNode()
        ),
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[IntegerLiteralASTNode(value=1), None]))],
            value=NothingTypeLiteralASTNode()
        ),
        {}
    ),
    (
        'x: Integer = 1',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
        {}
    ),
    (
        'x: Integer<> = 1',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
        {}
    ),
    (
        'x: Integer<2,3,4> = 1',
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(
                base_type_name='Integer',
                shape=[
                    IntegerLiteralASTNode(value=2),
                    IntegerLiteralASTNode(value=3),
                    IntegerLiteralASTNode(value=4),
                ]))],
            value=IntegerLiteralASTNode(value=1)
        ),
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(
                base_type_name='Integer',
                shape=[
                    IntegerLiteralASTNode(value=2),
                    IntegerLiteralASTNode(value=3),
                    IntegerLiteralASTNode(value=4),
                ]))],
            value=IntegerLiteralASTNode(value=1)
        ),
        {}
    ),
    (
        'x: Float<?> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None]))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None]))], value=VariableASTNode(name='value')),
        {}
    ),
    (
        'x: Float<???> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        {}
    ),
    (
        'x: Float<?, ?, ?> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]))], value=VariableASTNode(name='value')),
        {}
    ),
    (
        'x: Boolean<?, 3, ?> = value',
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[None, IntegerLiteralASTNode(value=3), None]))],
            value=VariableASTNode(name='value')
        ),
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[None, IntegerLiteralASTNode(value=3), None]))],
            value=VariableASTNode(name='value')
        ),
        {}
    ),
    (
        'x: Float<??\
?> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        {}
    ),
)

@pytest.mark.parametrize('input_string, unprocessed_results, type_inference_results, type_inference_table', TEST_CASES)
def test_parser_assignment(input_string, unprocessed_results, type_inference_results, type_inference_table):
    del type_inference_results
    del type_inference_table
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
    assert result == unprocessed_results, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(unprocessed_results)}
'''

@pytest.mark.parametrize('input_string, unprocessed_results, type_inference_results, type_inference_table', TEST_CASES)
def test_type_inference_assignment(input_string, unprocessed_results, type_inference_results, type_inference_table):
    del unprocessed_results
    module_node = parser.parseSourceCode(input_string)
    type_inference.perform_type_inference(module_node, type_inference_table)
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    for variable_node, tensor_type_node in assignment_node.variable_type_pairs:
        assert isinstance(variable_node, VariableASTNode)
        assert isinstance(variable_node.name, str)
        assert isinstance(tensor_type_node, TensorTypeASTNode)
    result = assignment_node
    assert result == type_inference_results, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(type_inference_results)}
'''
