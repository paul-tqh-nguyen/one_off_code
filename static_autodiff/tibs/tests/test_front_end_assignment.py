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
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[]), TensorTypeASTNode(base_type_name='Float', shape=[2])],
                function_body=ScopedStatementSequenceASTNode(statements=[ReturnStatementASTNode(return_values=[
                    BooleanLiteralASTNode(value=True),
                    VectorExpressionASTNode(values=[FloatLiteralASTNode(value=2.2), FloatLiteralASTNode(value=3.3)])
                ])])
            )
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
        'x: NothingType<1,?> = [[Nothing, Nothing, Nothing]]',
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[1, None]))],
            value=VectorExpressionASTNode(values=[
                VectorExpressionASTNode(values=[
                    NothingTypeLiteralASTNode(),
                    NothingTypeLiteralASTNode(),
                    NothingTypeLiteralASTNode()
                ]),
            ])
        ),
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[1, None]))],
            value=VectorExpressionASTNode(values=[
                VectorExpressionASTNode(values=[
                    NothingTypeLiteralASTNode(),
                    NothingTypeLiteralASTNode(),
                    NothingTypeLiteralASTNode()
                ]),
            ])
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
        'x: Integer<2,3,1> = [[[1], [2], [3]], [[1], [2], [3]]]',
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(
                base_type_name='Integer',
                shape=[2, 3, 1]
            ))],
            value=VectorExpressionASTNode(values=[
                VectorExpressionASTNode(values=[
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=3)])
                ]),
                VectorExpressionASTNode(values=[
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=3)])
                ])
            ])
        ),
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(
                base_type_name='Integer',
                shape=[2, 3, 1]
            ))],
            value=VectorExpressionASTNode(values=[
                VectorExpressionASTNode(values=[
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=3)])
                ]),
                VectorExpressionASTNode(values=[
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2)]),
                    VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=3)])
                ])
            ])
        ),
        {'bogus_distractor_var': TensorTypeASTNode(base_type_name='String', shape=[2, 3, 1])}
    ),
    (
        'x: Float<?> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None]))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None]))], value=VariableASTNode(name='value')),
        {'value': TensorTypeASTNode(base_type_name='Float', shape=[None])}
    ),
    (
        'x: Float<???> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        {'value': TensorTypeASTNode(base_type_name='Float', shape=[1,2,3,4,5])}
    ),
    (
        'x: Float<?, ?, ?> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]))], value=VariableASTNode(name='value')),
        {'value': TensorTypeASTNode(base_type_name='Float', shape=[None, 1234, None])}
    ),
    (
        'x: Boolean<?, 3, ?> = value',
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[None, 3, None]))],
            value=VariableASTNode(name='value')
        ),
        AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[None, 3, None]))],
            value=VariableASTNode(name='value')
        ),
        {'value': TensorTypeASTNode(base_type_name='Boolean', shape=[9999, 3, None])}
    ),
    (
        'x: Float<??\
?> = value',
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value')),
        {'value': TensorTypeASTNode(base_type_name='Float', shape=[None, None, None, None, None])}
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
