
import pytest
import itertools

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
from tibs.misc_utilities import *

# TODO verify that the above imports are used (check all other test files as well)
# TODO make sure these imports are ordered in some way

def test_parser_nothing_literal():
    module_node = parser.parseSourceCode('x = Nothing')
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    variable_node, tensor_type_node = only_one(assignment_node.variable_type_pairs)
    assert isinstance(variable_node, VariableASTNode)
    assert variable_node.name is 'x'
    assert isinstance(tensor_type_node, TensorTypeASTNode)
    assert tensor_type_node.base_type_name == 'NothingType'
    assert tensor_type_node.shape == []
    value_node = assignment_node.value
    assert isinstance(value_node, NothingTypeLiteralASTNode)
    result = value_node
    expected_result = NothingTypeLiteralASTNode()
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_assignment():
    expected_input_output_pairs = [
        ('x = 1', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1))),
        ('x, y = f(a:=12)', AssignmentASTNode(variable_type_pairs=[
            (VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None)),
            (VariableASTNode(name='y'), TensorTypeASTNode(base_type_name=None, shape=None))
        ], value=FunctionCallExpressionASTNode(
            arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=12))],
            function_name='f')
        )),
        ('x: NothingType = Nothing', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode())),
        ('x = Nothing', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=NothingTypeLiteralASTNode())),
        ('x: NothingType<> = Nothing', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[]))], value=NothingTypeLiteralASTNode())),
        ('x: NothingType<1,?> = Nothing', AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='NothingType', shape=[IntegerLiteralASTNode(value=1), None]))],
            value=NothingTypeLiteralASTNode())),
        ('x: Integer = 1', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))),
        ('x: Integer<> = 1', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))),
        ('x: Integer<2,3,4> = 1', AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(
                base_type_name='Integer',
                shape=[
                    IntegerLiteralASTNode(value=2),
                    IntegerLiteralASTNode(value=3),
                    IntegerLiteralASTNode(value=4),
                ]))],
            value=IntegerLiteralASTNode(value=1)
        )),
        ('x: Float<?> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None]))], value=VariableASTNode(name='value'))),
        ('x: Float<???> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value'))),
        ('x: Float<?, ?, ?> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]))], value=VariableASTNode(name='value'))),
        ('x: Boolean<?, 3, ?> = value', AssignmentASTNode(
            variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Boolean', shape=[None, IntegerLiteralASTNode(value=3), None]))],
            value=VariableASTNode(name='value')
        )),
        ('x: Float<??\
?> = value', AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Float', shape=None))], value=VariableASTNode(name='value'))),
    ]
    for input_string, expected_result in expected_input_output_pairs:
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
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_comments():
    expected_input_output_pairs = [
        ('x = 1 # comment', ModuleASTNode(statements=[AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name=None, shape=None))], value=IntegerLiteralASTNode(value=1))])),
        ('x: Integer \
= 1 # comment', ModuleASTNode(statements=[AssignmentASTNode(variable_type_pairs=[(VariableASTNode(name='x'), TensorTypeASTNode(base_type_name='Integer', shape=[]))], value=IntegerLiteralASTNode(value=1))])),
        ('Nothing # comment', ModuleASTNode(statements=[NothingTypeLiteralASTNode()])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_function_call():
    expected_input_output_pairs = [
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
    ]
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
        assert isinstance(result, FunctionCallExpressionASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_module():
    expected_input_output_pairs = [
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
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_return_statement():
    expected_input_output_pairs = [
        ('', NothingTypeLiteralASTNode()),
        ('  \
\t # comment', NothingTypeLiteralASTNode()),
        ('Nothing', NothingTypeLiteralASTNode()),
        ('False', BooleanLiteralASTNode(value=False)),
        ('123', IntegerLiteralASTNode(value=123)),
        ('1E2', FloatLiteralASTNode(value=100.0)),
        ('some_variable', VariableASTNode(name='some_variable')),
        ('[f(x:=1), [2, 3], some_variable, Nothing]',
         VectorExpressionASTNode(values=[
             FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
             VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
             VariableASTNode(name='some_variable'),
             NothingTypeLiteralASTNode()])),
        ('False or True', OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True))),
        ('1 ** 2 ^ 3',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)))),
        ('f(a:=1, b:=2, c:=Nothing)',
         FunctionCallExpressionASTNode(
             arg_bindings=[
                 (VariableASTNode(name='a'), IntegerLiteralASTNode(value=1)),
                 (VariableASTNode(name='b'), IntegerLiteralASTNode(value=2)),
                 (VariableASTNode(name='c'), NothingTypeLiteralASTNode()),
             ],
             function_name='f')),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(f'''
function f() -> NothingType {{
    return {input_string}
}}
''')
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        function_definition_node = only_one(module_node.statements)
        assert isinstance(function_definition_node, FunctionDefinitionASTNode)
        assert function_definition_node.function_name == 'f'
        assert function_definition_node.function_signature == []
        function_body = function_definition_node.function_body
        assert isinstance(function_body, ScopedStatementSequenceASTNode)
        return_statement_node = only_one(function_body.statements)
        assert isinstance(return_statement_node, ReturnStatementASTNode)
        result = only_one(return_statement_node.return_values)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_function_definition():
    input_strings = [
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
    ]
    # TODO test that the parses are correct
    for input_string in input_strings:
        parser.parseSourceCode(input_string)

def test_parser_for_loop():
    expected_input_output_pairs = [
        ('for x:(1,10,y) func(x:=1)', ForLoopASTNode(
            body=FunctionCallExpressionASTNode(
                arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
                function_name='func'
            ),
            iterator_variable_name='x',
            minimum=IntegerLiteralASTNode(value=1),
            supremum=IntegerLiteralASTNode(value=10),
            delta=VariableASTNode(name='y'))),
        ('for x:(1,10) func(x:=1)', ForLoopASTNode(
            body=FunctionCallExpressionASTNode(
                arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
                function_name='func'
            ),
            iterator_variable_name='x',
            minimum=IntegerLiteralASTNode(value=1),
            supremum=IntegerLiteralASTNode(value=10),
            delta=IntegerLiteralASTNode(value=1))),
        ('for x:(1,10, 2) func(x:=1)', ForLoopASTNode(
            body=FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='func'),
            iterator_variable_name='x',
            minimum=IntegerLiteralASTNode(value=1),
            supremum=IntegerLiteralASTNode(value=10),
            delta=IntegerLiteralASTNode(value=2))),
        ('''
for x:(1+0,10) {
    Nothing
}
''',
         ForLoopASTNode(
             body=ScopedStatementSequenceASTNode(statements=[NothingTypeLiteralASTNode()]),
             iterator_variable_name='x',
             minimum=AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=0)),
             supremum=IntegerLiteralASTNode(value=10),
             delta=IntegerLiteralASTNode(value=1))
        ),
        ('''
for x:(1, -10, -1) {
    True or True
    return
}
''',
         ForLoopASTNode(
             body=ScopedStatementSequenceASTNode(statements=[
                 OrExpressionASTNode(left_arg=BooleanLiteralASTNode(value=True), right_arg=BooleanLiteralASTNode(value=True)),
                 ReturnStatementASTNode(return_values=[NothingTypeLiteralASTNode()])
             ]),
             iterator_variable_name='x',
             minimum=IntegerLiteralASTNode(value=1),
             supremum=NegativeExpressionASTNode(arg=IntegerLiteralASTNode(value=10)),
             delta=NegativeExpressionASTNode(arg=IntegerLiteralASTNode(value=1))
         )),
        ('for x:(1,10, 2) while False 1', ForLoopASTNode(
            body=WhileLoopASTNode(
                condition=BooleanLiteralASTNode(value=False),
                body=IntegerLiteralASTNode(value=1)),
            iterator_variable_name='x',
            minimum=IntegerLiteralASTNode(value=1),
            supremum=IntegerLiteralASTNode(value=10),
            delta=IntegerLiteralASTNode(value=2))),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        result = only_one(module_node.statements)
        assert isinstance(result, ForLoopASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_while_loop():
    expected_input_output_pairs = [
        ('while True func(x:=1)', WhileLoopASTNode(
            condition=BooleanLiteralASTNode(value=True),
            body=FunctionCallExpressionASTNode(
                arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
                function_name='func'
            ))),
        ('while False xor True return 3', WhileLoopASTNode(
            condition=XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True)),
            body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=3)]))),
        ('''
while not False {
    Nothing
}
''',
         WhileLoopASTNode(
             condition=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=False)),
             body=ScopedStatementSequenceASTNode(statements=[NothingTypeLiteralASTNode()]))),
        ('while False for x:(1,10, 2) 1', WhileLoopASTNode(
            condition=BooleanLiteralASTNode(value=False),
            body=ForLoopASTNode(
                body=IntegerLiteralASTNode(value=1),
                iterator_variable_name='x',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=IntegerLiteralASTNode(value=10),
                delta=IntegerLiteralASTNode(value=2)))),
        ('while False for x:(1,10, 2) if True 1 else 2', WhileLoopASTNode(
            condition=BooleanLiteralASTNode(value=False),
            body=ForLoopASTNode(
                body=ConditionalASTNode(
                    condition=BooleanLiteralASTNode(value=True),
                    then_body=IntegerLiteralASTNode(value=1),
                    else_body=IntegerLiteralASTNode(value=2)
                ),
                iterator_variable_name='x',
                minimum=IntegerLiteralASTNode(value=1),
                supremum=IntegerLiteralASTNode(value=10),
                delta=IntegerLiteralASTNode(value=2)))),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        result = only_one(module_node.statements)
        assert isinstance(result, WhileLoopASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_conditional():
    expected_input_output_pairs = [
        ('if False while True func(x:=1)', ConditionalASTNode(
            condition=BooleanLiteralASTNode(value=False),
            then_body=WhileLoopASTNode(
                condition=BooleanLiteralASTNode(value=True),
                body=FunctionCallExpressionASTNode(
                    arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))],
                    function_name='func')),
            else_body=None)),
        ('if True xor False while False and True return 3 else 5', ConditionalASTNode(
            condition=XorExpressionASTNode(
                 left_arg=BooleanLiteralASTNode(value=True),
                 right_arg=BooleanLiteralASTNode(value=False)
             ),
            then_body=WhileLoopASTNode(
                condition=AndExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=True)
                ),
                body=ReturnStatementASTNode(return_values=[IntegerLiteralASTNode(value=3)])),
            else_body=IntegerLiteralASTNode(value=5))),
        ('''
if False {
    while not False {
        if True 1 else { f(x:=2) }
    }
} else return 3, 5
''',
         ConditionalASTNode(
             condition=BooleanLiteralASTNode(value=False),
             then_body=ScopedStatementSequenceASTNode(statements=[
                 WhileLoopASTNode(
                     condition=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=False)),
                     body=ScopedStatementSequenceASTNode(statements=[
                         ConditionalASTNode(
                             condition=BooleanLiteralASTNode(value=True),
                             then_body=IntegerLiteralASTNode(value=1),
                             else_body=ScopedStatementSequenceASTNode(statements=[
                                 FunctionCallExpressionASTNode(
                                     arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=2))],
                                     function_name='f')
                             ])
                         )
                     ])
                 )]),
             else_body=ReturnStatementASTNode(return_values=[
                 IntegerLiteralASTNode(value=3),
                 IntegerLiteralASTNode(value=5)
             ]))),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        result = only_one(module_node.statements)
        assert isinstance(result, ConditionalASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_print_statement():
    expected_input_output_pairs = [
        ('''print "1
2
3" ''', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('1\n2\n3')])),
        ('print "if" ', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('if')])),
        ('print "1" 2 "3"', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('1'), IntegerLiteralASTNode(value=2), StringLiteralASTNode('3')])),
        ('print "1" -2 "3"', PrintStatementASTNode(values_to_print=[StringLiteralASTNode('1'), NegativeExpressionASTNode(IntegerLiteralASTNode(value=2)), StringLiteralASTNode('3')])),
        ('print "1" f(a:=1)', PrintStatementASTNode(values_to_print=[
            StringLiteralASTNode('1'),
            FunctionCallExpressionASTNode(
                arg_bindings=[(VariableASTNode(name='a'), IntegerLiteralASTNode(value=1))],
                function_name='f')
        ])),
        ('print True False xor True 3', PrintStatementASTNode(values_to_print=[
            BooleanLiteralASTNode(value=True),
            XorExpressionASTNode(left_arg=BooleanLiteralASTNode(value=False), right_arg=BooleanLiteralASTNode(value=True)),
            IntegerLiteralASTNode(value=3)
        ])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        result = only_one(module_node.statements)
        assert isinstance(result, PrintStatementASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
