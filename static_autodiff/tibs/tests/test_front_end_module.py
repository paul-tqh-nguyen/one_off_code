import pytest
import os

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

# TODO make sure all these imports are used

TEST_CASES = tuple(pytest.param(*args, id=f'{os.path.basename(__file__)}_{i}') for i, args in enumerate([
    (
        '',
        ModuleASTNode(statements=[]),
        ModuleASTNode(statements=[])
    ),
    (
        '   ',
        ModuleASTNode(statements=[]),
        ModuleASTNode(statements=[])
    ),
    (
        '\t   \n',
        ModuleASTNode(statements=[]),
        ModuleASTNode(statements=[])
    ),
    (
        '\n\n\n \t   \n',
        ModuleASTNode(statements=[]),
        ModuleASTNode(statements=[])
    ),
    (
        '1 + 2',
        ModuleASTNode(statements=[
            AdditionExpressionASTNode(
                left_arg=IntegerLiteralASTNode(value=1),
                right_arg=IntegerLiteralASTNode(value=2)
            ),
        ]),
        ModuleASTNode(statements=[
            AdditionExpressionASTNode(
                left_arg=IntegerLiteralASTNode(value=1),
                right_arg=IntegerLiteralASTNode(value=2)
            ),
        ])
    ),
    (
        'x = 1 ; 1 + 2 ; x: Integer = 1 ; Nothing',
        ModuleASTNode(statements=[
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
            AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            NothingTypeLiteralASTNode(),
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            NothingTypeLiteralASTNode(),
        ])
    ),
    (
        'x = 1 ; x: Integer = 1 ; 1 + 2 # y = 123',
        ModuleASTNode(statements=[
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
        ])
    ),
    (
        '''

# comment

function init_something(x: Integer) -> NothingType return

x: Integer = 1 # comment
x
Nothing
x = 1 # comment
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

function f(a: NothingType) -> NothingType 
{

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
}

function g() -> NothingType, Boolean {
    while False xor True return Nothing, False
    return Nothing,  True
}
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='init_something',
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            ),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            VariableASTNode(name='x'),
            NothingTypeLiteralASTNode(),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)),
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
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1)
            ),
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
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[(VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    ForLoopASTNode(
                        body=ScopedStatementSequenceASTNode(statements=[
                            OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True)
                            ),
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
                    )
                ])
            ),
            FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    WhileLoopASTNode(
                        condition=XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True)),
                        body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode(), BooleanLiteralASTNode(value=False)])
                    ),
                    ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode(), BooleanLiteralASTNode(value=True)])
                ])
            )
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='init_something',
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
            ),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            VariableASTNode(name='x'),
            NothingTypeLiteralASTNode(),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), VariableASTNode(name='x'))], function_name='init_something'),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
            ScopedStatementSequenceASTNode(statements=[
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
            ]),
            ScopedStatementSequenceASTNode(statements=[
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
            ]),
            IntegerLiteralASTNode(value=123),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)
            ),
            ScopedStatementSequenceASTNode(statements=[
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
            ]),
            IntegerLiteralASTNode(value=123),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            ScopedStatementSequenceASTNode(statements=[
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
            ]),
            AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1)),
            ScopedStatementSequenceASTNode(statements=[
                AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))
            ]),
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[(VariableASTNode(name='a'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    ForLoopASTNode(
                        body=ScopedStatementSequenceASTNode(statements=[
                            OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True)
                            ),
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
                    )
                ])
            ),
            FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    WhileLoopASTNode(
                        condition=XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True)),
                        body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode(), BooleanLiteralASTNode(value=False)])
                    ),
                    ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode(), BooleanLiteralASTNode(value=True)])
                ])
            )
        ])
    ),
    (
        '''
function f() -> Integer, Boolean {
    function g() -> Integer
        return 1234
    return g(), True
}
f()
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    FunctionDefinitionASTNode(
                        function_name='g',
                        function_signature=[],
                        function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                        function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                    ),
                    ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                ])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='f')
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    FunctionDefinitionASTNode(
                        function_name='g',
                        function_signature=[],
                        function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                        function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                    ),
                    ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                ])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='f')
        ])
    ),
    (
        '''
function f() -> Integer, Boolean {
    function g() -> Integer
        return 1234
    return g(), True
}
f()
function g() -> Integer
    return 1234
g()
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    FunctionDefinitionASTNode(
                        function_name='g',
                        function_signature=[],
                        function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                        function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                    ),
                    ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                ])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
            FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='g')
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    FunctionDefinitionASTNode(
                        function_name='g',
                        function_signature=[],
                        function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                        function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                    ),
                    ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                ])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
            FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='g')
        ])
    ),
    (
        '''
{
    function f() -> Integer, Boolean {
        function g() -> Integer
            return 1234
        return g(), True
    }
    f()
    function g() -> Integer
        return 1234
    g()
}
''',
        ModuleASTNode(statements=[
            ScopedStatementSequenceASTNode(statements=[
                FunctionDefinitionASTNode(
                    function_name='f',
                    function_signature=[],
                    function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                    function_body=ScopedStatementSequenceASTNode(statements=[
                        FunctionDefinitionASTNode(
                            function_name='g',
                            function_signature=[],
                            function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                            function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                        ),
                        ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                    ])
                ),
                FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
                FunctionDefinitionASTNode(
                    function_name='g',
                    function_signature=[],
                    function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                    function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                ),
                FunctionCallExpressionASTNode(arg_bindings=[], function_name='g')
            ])
        ]),
        ModuleASTNode(statements=[
            ScopedStatementSequenceASTNode(statements=[
                FunctionDefinitionASTNode(
                    function_name='f',
                    function_signature=[],
                    function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                    function_body=ScopedStatementSequenceASTNode(statements=[
                        FunctionDefinitionASTNode(
                            function_name='g',
                            function_signature=[],
                            function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                            function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                        ),
                        ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                    ])
                ),
                FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
                FunctionDefinitionASTNode(
                    function_name='g',
                    function_signature=[],
                    function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                    function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                ),
                FunctionCallExpressionASTNode(arg_bindings=[], function_name='g')
            ])
        ])
    ),
    (
        '''
function f() -> Integer, Boolean {
    function g() -> Integer
        return 1234
    return g(), True
}
f()
function g() -> Integer
    return 1234
g()
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    FunctionDefinitionASTNode(
                        function_name='g',
                        function_signature=[],
                        function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                        function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                    ),
                    ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                ])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
            FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='g')
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_name='f',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[]), TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_body=ScopedStatementSequenceASTNode(statements=[
                    FunctionDefinitionASTNode(
                        function_name='g',
                        function_signature=[],
                        function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                        function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                    ),
                    ReturnStatementASTNode(return_values=[FunctionCallExpressionASTNode(arg_bindings=[], function_name='g'), BooleanLiteralASTNode(value=True)])
                ])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='f'),
            FunctionDefinitionASTNode(
                function_name='g',
                function_signature=[],
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
            ),
            FunctionCallExpressionASTNode(arg_bindings=[], function_name='g')
        ])
    )
]))

@pytest.mark.parametrize('input_string, parse_result, type_inference_result', TEST_CASES)
def test_parser_module(input_string, parse_result, type_inference_result):
    del type_inference_result
    result = parser.parseSourceCode(input_string)
    assert result == parse_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
parse_result: {repr(parse_result)}
'''

@pytest.mark.parametrize('input_string, parse_result, type_inference_result', TEST_CASES)
def test_type_inference_module(input_string, parse_result, type_inference_result):
    del parse_result
    result = parser.parseSourceCode(input_string)
    type_inference.perform_type_inference(result)
    assert result == type_inference_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
type_inference_result: {repr(type_inference_result)}
'''
