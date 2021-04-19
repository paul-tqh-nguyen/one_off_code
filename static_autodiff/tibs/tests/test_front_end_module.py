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
function f(x: Integer) -> Integer {
    x = 100
    for i:(1,10, 2) {
        x = x + i
    }
    return 1234
}
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_body=ScopedStatementSequenceASTNode(statements=[
                    AssignmentASTNode(
                        variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))],
                        value=IntegerLiteralASTNode(value=100)
                    ),
                    ForLoopASTNode(
                        body=ScopedStatementSequenceASTNode(statements=[
                            AssignmentASTNode(
                                variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))],
                                value=AdditionExpressionASTNode(
                                    left_arg=VariableASTNode(name='x'),
                                    right_arg=VariableASTNode(name='i')
                                )
                            )
                        ]),
                        delta=IntegerLiteralASTNode(value=2),
                        iterator_variable_name='i',
                        minimum=IntegerLiteralASTNode(value=1),
                        supremum=IntegerLiteralASTNode(value=10)),
                    ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                ]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            )
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_body=ScopedStatementSequenceASTNode(statements=[
                    AssignmentASTNode(
                        variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                        value=IntegerLiteralASTNode(value=100)
                    ),
                    ForLoopASTNode(
                        body=ScopedStatementSequenceASTNode(statements=[
                            AssignmentASTNode(
                                variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                                value=AdditionExpressionASTNode(
                                    left_arg=VariableASTNode(name='x'),
                                    right_arg=VariableASTNode(name='i')
                                )
                            )
                        ]),
                        delta=IntegerLiteralASTNode(value=2),
                        iterator_variable_name='i',
                        minimum=IntegerLiteralASTNode(value=1),
                        supremum=IntegerLiteralASTNode(value=10)),
                    ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                ]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            )
        ])
    ),
    (
        '''
function f(x: Integer) -> Integer {
    for i:(1,10, 2) {
        x = x + i
        if x >= 2 return 0
    }
    return 1234
}
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_body=ScopedStatementSequenceASTNode(statements=[
                    ForLoopASTNode(
                        body=ScopedStatementSequenceASTNode(statements=[
                            AssignmentASTNode(
                                variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))],
                                value=AdditionExpressionASTNode(
                                    left_arg=VariableASTNode(name='x'),
                                    right_arg=VariableASTNode(name='i')
                                )
                            ),
                            ConditionalASTNode(
                                condition=GreaterThanOrEqualToExpressionASTNode(
                                    left_arg=VariableASTNode(name='x'),
                                    right_arg=IntegerLiteralASTNode(value=2)
                                ),
                                then_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=0)]),
                                else_body=None
                            )
                        ]),
                        delta=IntegerLiteralASTNode(value=2),
                        iterator_variable_name='i',
                        minimum=IntegerLiteralASTNode(value=1),
                        supremum=IntegerLiteralASTNode(value=10)),
                    ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                ]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            )
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_body=ScopedStatementSequenceASTNode(statements=[
                    ForLoopASTNode(
                        body=ScopedStatementSequenceASTNode(statements=[
                            AssignmentASTNode(
                                variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                                value=AdditionExpressionASTNode(
                                    left_arg=VariableASTNode(name='x'),
                                    right_arg=VariableASTNode(name='i')
                                )
                            ),
                            ConditionalASTNode(
                                condition=GreaterThanOrEqualToExpressionASTNode(
                                    left_arg=VariableASTNode(name='x'),
                                    right_arg=IntegerLiteralASTNode(value=2)
                                ),
                                then_body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=0)]),
                                else_body=None
                            )
                        ]),
                        delta=IntegerLiteralASTNode(value=2),
                        iterator_variable_name='i',
                        minimum=IntegerLiteralASTNode(value=1),
                        supremum=IntegerLiteralASTNode(value=10)),
                    ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=1234)])
                ]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Integer', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            )
        ])
    ),
    (
        '''
x: Float = 10.2
function f(x: Integer) -> NothingType return
''',
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=10.2)
            ),
            FunctionDefinitionASTNode(
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            )
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=10.2)
            ),
            FunctionDefinitionASTNode(
                function_body=ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='NothingType', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            )
        ]),
    ),
    (
        '''
function f(x: Integer) -> Boolean return True
y = 3
i = f(x:=1234)
for i:(1,10, 2)
    y = y + i
False xor i
''',
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=True)]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=IntegerLiteralASTNode(value=3),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1234))], function_name='f'),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=IntegerLiteralASTNode(value=10),
                delta=IntegerLiteralASTNode(value=2)
            ),
            XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=VariableASTNode(name='i')),
        ]),
        ModuleASTNode(statements=[
            FunctionDefinitionASTNode(
                function_body=ReturnStatementASTNode(return_values=[BooleanLiteralASTNode(value=True)]),
                function_name='f',
                function_return_types=[TensorTypeASTNode(base_type_name='Boolean', shape=[])],
                function_signature=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))]
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                value=IntegerLiteralASTNode(value=3),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name='Boolean', shape=[]))],
                value=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1234))], function_name='f'),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=IntegerLiteralASTNode(value=10),
                delta=IntegerLiteralASTNode(value=2)
            ),
            XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=VariableASTNode(name='i')),
        ]),
    ),
    (
        '''
y = 3
i = "string"
for i:(1,10*y, 2)
    y = y + i
"asd" << i << "\n"
''',
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=IntegerLiteralASTNode(value=3),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=StringLiteralASTNode(value='string'),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=MultiplicationExpressionASTNode(
                    left_arg=IntegerLiteralASTNode(value=10),
                    right_arg=VariableASTNode(name='y'),
                ),
                delta=IntegerLiteralASTNode(value=2)
            ),
            StringConcatenationExpressionASTNode(
                left_arg=StringConcatenationExpressionASTNode(left_arg=StringLiteralASTNode(value='asd'), right_arg=VariableASTNode(name='i')),
                right_arg=StringLiteralASTNode(value='\n')
            ),
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                value=IntegerLiteralASTNode(value=3),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name='String', shape=[]))],
                value=StringLiteralASTNode(value='string'),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=MultiplicationExpressionASTNode(
                    left_arg=IntegerLiteralASTNode(value=10),
                    right_arg=VariableASTNode(name='y'),
                ),
                delta=IntegerLiteralASTNode(value=2)
            ),
            StringConcatenationExpressionASTNode(
                left_arg=StringConcatenationExpressionASTNode(left_arg=StringLiteralASTNode(value='asd'), right_arg=VariableASTNode(name='i')),
                right_arg=StringLiteralASTNode(value='\n')
            )
        ]),
    ),
    (
        '''
y = 3
i = 1234.5678
for i:(1,10, y-1)
    y = y + i
i ** 2.0
''',
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=IntegerLiteralASTNode(value=3),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=FloatLiteralASTNode(value=1234.5678),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=IntegerLiteralASTNode(value=10),
                delta=SubtractionExpressionASTNode(
                    left_arg=VariableASTNode(name='y'),
                    right_arg=IntegerLiteralASTNode(value=1),
                )
            ),
            ExponentExpressionASTNode(
                left_arg=VariableASTNode(name='i'),
                right_arg=FloatLiteralASTNode(value=2.0)
            ),
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                value=IntegerLiteralASTNode(value=3),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=1234.5678),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Integer', shape=[]))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=IntegerLiteralASTNode(value=10),
                delta=SubtractionExpressionASTNode(
                    left_arg=VariableASTNode(name='y'),
                    right_arg=IntegerLiteralASTNode(value=1),
                )
            ),
            ExponentExpressionASTNode(
                left_arg=VariableASTNode(name='i'),
                right_arg=FloatLiteralASTNode(value=2.0)
            ),
        ]),
    ),
    (
        '''
y = 3.0
i = 4.0
for i:(i,10.0, 0.50)
    y = y + i
i ** 2.0
''',
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=FloatLiteralASTNode(value=3.0),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=FloatLiteralASTNode(value=4.0),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=VariableASTNode(name='i'),
                supremum=FloatLiteralASTNode(value=10.0),
                delta=FloatLiteralASTNode(value=0.5)
            ),
            ExponentExpressionASTNode(
                left_arg=VariableASTNode(name='i'),
                right_arg=FloatLiteralASTNode(value=2.0)
            ),
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=3.0),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=4.0),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=VariableASTNode(name='i'),
                supremum=FloatLiteralASTNode(value=10.0),
                delta=FloatLiteralASTNode(value=0.5)
            ),
            ExponentExpressionASTNode(
                left_arg=VariableASTNode(name='i'),
                right_arg=FloatLiteralASTNode(value=2.0)
            ),
        ]),
    ),
    (
        '''
y = 3.0
i = 4.0
for i:(i*2.0,10.0**i, i*0.50)
    y = y + i
i ** 2.0
''',
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=FloatLiteralASTNode(value=3.0),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name=None, shape=None))],
                value=FloatLiteralASTNode(value=4.0),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=MultiplicationExpressionASTNode(
                    left_arg=VariableASTNode(name='i'),
                    right_arg=FloatLiteralASTNode(value=2.0),
                ),
                supremum=ExponentExpressionASTNode(
                    left_arg=FloatLiteralASTNode(value=10.0),
                    right_arg=VariableASTNode(name='i'),
                ),
                delta=MultiplicationExpressionASTNode(
                    left_arg=VariableASTNode(name='i'),
                    right_arg=FloatLiteralASTNode(value=0.50),
                )
            ),
            ExponentExpressionASTNode(
                left_arg=VariableASTNode(name='i'),
                right_arg=FloatLiteralASTNode(value=2.0)
            ),
        ]),
        ModuleASTNode(statements=[
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=3.0),
            ),
            AssignmentASTNode(
                variable_type_pairs=[(VariableASTNode(name='i'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                value=FloatLiteralASTNode(value=4.0),
            ),
            ForLoopASTNode(
                body=AssignmentASTNode(
                    variable_type_pairs=[(VariableASTNode(name='y'), TensorTypeASTNode(base_type_name='Float', shape=[]))],
                    value=AdditionExpressionASTNode(
                        left_arg=VariableASTNode(name='y'),
                        right_arg=VariableASTNode(name='i')
                    ),
                ),
                iterator_variable_name='i',
                minimum=MultiplicationExpressionASTNode(
                    left_arg=VariableASTNode(name='i'),
                    right_arg=FloatLiteralASTNode(value=2.0),
                ),
                supremum=ExponentExpressionASTNode(
                    left_arg=FloatLiteralASTNode(value=10.0),
                    right_arg=VariableASTNode(name='i'),
                ),
                delta=MultiplicationExpressionASTNode(
                    left_arg=VariableASTNode(name='i'),
                    right_arg=FloatLiteralASTNode(value=0.50),
                )
            ),
            ExponentExpressionASTNode(
                left_arg=VariableASTNode(name='i'),
                right_arg=FloatLiteralASTNode(value=2.0)
            ),
        ]),
    ),
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
    result = type_inference.perform_type_inference(result)
    assert result == type_inference_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
type_inference_result: {repr(type_inference_result)}
'''
