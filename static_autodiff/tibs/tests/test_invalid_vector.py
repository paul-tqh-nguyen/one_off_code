
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

INVALID_EXAMPLES = [pytest.param(*args, id=f'case_{i}') for i, args in enumerate([
    (
        '''
while [True, True]
    func(x:=1)
''',
        type_inference.TypeInferenceConsistencyError,
        'has the following inconsistent types',
        {}
    ),
    (
        '[[2, 3], x]',
        type_inference.TypeInferenceFailure,
        'must strictly contain tensor expressions with the same base type',
        {'x': TensorTypeASTNode(base_type_name='Boolean', shape=[9999, 3, None])}
    ),
    (
        '[[2, 3], Nothing]',
        type_inference.TypeInferenceFailure,
        'must strictly contain tensor expressions with the same base type',
        {}
    ),
    (
        '[[2, 3], "string"]',
        type_inference.TypeInferenceFailure,
        'must strictly contain tensor expressions with the same base type',
        {}
    ),
    (
        '[f(x:=1), [2, 3]]',
        type_inference.TypeInferenceFailure,
        'must strictly contain tensor expressions with the same base type',
        {
            'f': FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=True)])
            )
        }
    ),
])]

@pytest.mark.parametrize('input_string, exception_type, error_match_string, type_inference_table', INVALID_EXAMPLES)
def test_invalid_vector(input_string, exception_type, error_match_string, type_inference_table):
    with pytest.raises(exception_type, match=error_match_string):
        result = parser.parseSourceCode(input_string)
        type_inference.perform_type_inference(result, type_inference_table)
