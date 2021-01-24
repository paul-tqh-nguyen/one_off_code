
import pytest

from leibniz import parser
from leibniz.parser import (
    ExpressionASTNode,
    TypeASTNode,
    BooleanLiteralASTNode,
    RealLiteralASTNode,
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
    FunctionCallExpressionASTNode,
    AssignmentASTNode,
    ModuleASTNode,
)
from leibniz.misc_utilities import *

# TODO verify that the above imports are used

def test_parser_invalid_():
    invalid_input_strings = [
        '''
###
init()

x: Integer

''',
    ]
    for input_string in invalid_input_strings:
        with pytest.raises(parser.ParseError, match='Could not parse the following:'):
            parser.parseSourceCode(input_string)

def test_parser_atomic_boolean():
    expected_input_output_pairs = [
        ('True', True),
        ('False', False),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TypeASTNode)
        assert assignment_node.identifier_type.value is None
        value_node = assignment_node.value
        assert isinstance(value_node, BooleanLiteralASTNode)
        result = value_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_atomic_real():
    expected_input_output_pairs = [
        ('123', 123),
        ('0.123', 0.123),
        ('.123', 0.123),
        # ('1.e2', 100.0),
        # ('1.E2', 100.0),
        # ('1.0e2', 100.0),
        ('1e0', 1.0),
        ('1e2', 100.0),
        ('1E2', 100.0),
        ('1E-2', 0.01),
        # ('.23E2', 23.0),
        # ('1.23e-2', 0.0123),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TypeASTNode)
        assert assignment_node.identifier_type.value is None
        value_node = assignment_node.value
        assert isinstance(value_node, RealLiteralASTNode)
        result = value_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_atomic_identifier():
    valid_identifiers = [
        'var',
        'var_1',
        '_var',
        '_var_',
        '_var_1',
        'var_1_',
        '_var_1_',
        '_12345_',
        'VAR',
        'Var',
        'vAr213feEF',
    ]
    for input_string in valid_identifiers:
        module_node = parser.parseSourceCode(input_string+' = 1')
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert assignment_node.value == RealLiteralASTNode(value=1) # TODO change this "==" to is_equivalent
        assert assignment_node.identifier_type.value is None
        identifier_node = assignment_node.identifier
        assert isinstance(identifier_node, VariableASTNode)
        result = identifier_node.name
        assert type(result) is str
        assert result == input_string, f'''
input_string: {repr(input_string)}
result: {repr(result)}
'''

def test_parser_boolean_expression():
    expected_input_output_pairs = [
        ('not False', NotExpressionASTNode(arg=BooleanLiteralASTNode(value=False))),
        ('not True', NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True))),
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
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TypeASTNode)
        assert assignment_node.identifier_type.value is None
        result = assignment_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_arithmetic_expression():
    expected_input_output_pairs = [
        ('1 + 2', AdditionExpressionASTNode(left_arg=RealLiteralASTNode(value=1), right_arg=RealLiteralASTNode(value=2))),
        ('1 + 2 - 3',
         SubtractionExpressionASTNode(
             left_arg=AdditionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=1),
                 right_arg=RealLiteralASTNode(value=2)
             ),
             right_arg=RealLiteralASTNode(value=3)
         )),
        ('1 + 2 * 3',
         AdditionExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=MultiplicationExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('(1 + 2) * 3',
         MultiplicationExpressionASTNode(
             left_arg=AdditionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=1),
                 right_arg=RealLiteralASTNode(value=2)
             ),
             right_arg=RealLiteralASTNode(value=3)
         )),
        ('1 / 2 * 3',
         MultiplicationExpressionASTNode(
             left_arg=DivisionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=1),
                 right_arg=RealLiteralASTNode(value=2)
             ),
             right_arg=RealLiteralASTNode(value=3)
         )),
        ('1 / 2 + 3',
         AdditionExpressionASTNode(
             left_arg=DivisionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=1),
                 right_arg=RealLiteralASTNode(value=2)
             ),
             right_arg=RealLiteralASTNode(value=3)
         )),
        ('1 / (2 + 3)',
         DivisionExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=AdditionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('1 ^ 2',ExponentExpressionASTNode(left_arg=RealLiteralASTNode(value=1), right_arg=RealLiteralASTNode(value=2))),
        ('1 ^ (2 + 3)',
         ExponentExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=AdditionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('1 ** 2 ^ 3',
         ExponentExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('1 ^ 2 ** 3',
         ExponentExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('1 / 2 ** 3',
         DivisionExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('1 ** 2 - 3',
         SubtractionExpressionASTNode(
             left_arg=ExponentExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=1),
                 right_arg=RealLiteralASTNode(value=2)
             ),
             right_arg=RealLiteralASTNode(value=3)
         )),
        ('1 ^ (2 + 3)',
         ExponentExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=AdditionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=RealLiteralASTNode(value=3)
             )
         )),
        ('1 ^ (2 - -3)',
         ExponentExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=SubtractionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=NegativeExpressionASTNode(RealLiteralASTNode(value=3))
             )
         )),
        ('1 ** (2 --3)',
         ExponentExpressionASTNode(
             left_arg=RealLiteralASTNode(value=1),
             right_arg=SubtractionExpressionASTNode(
                 left_arg=RealLiteralASTNode(value=2),
                 right_arg=NegativeExpressionASTNode(RealLiteralASTNode(value=3))
             )
         )),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TypeASTNode)
        assert assignment_node.identifier_type.value is None
        result = assignment_node.value
        assert isinstance(result, ExpressionASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_assignment():
    expected_input_output_pairs = [
        ('x = 1', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1))),
        ('x: Real = 1', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TypeASTNode)
        result = assignment_node
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_comments():
    expected_input_output_pairs = [
        ('x = 1 # comment', ModuleASTNode(statements=[AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1))])),
        ('x: Real = 1 # comment', ModuleASTNode(statements=[AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))])),
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
        ('f() # comment', FunctionCallExpressionASTNode(arg_bindings=[], function_name=VariableASTNode(name='f'))),
        ('f(a:=1) # comment',
         FunctionCallExpressionASTNode(
             arg_bindings=[(VariableASTNode(name='a'), RealLiteralASTNode(value=1))],
             function_name=VariableASTNode(name='f'))
        ),
        ('f(a:=1, b:=2)', FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), RealLiteralASTNode(value=1)),
                (VariableASTNode(name='b'), RealLiteralASTNode(value=2))
            ],
            function_name=VariableASTNode(name='f'))),
        ('f(a:=1e3, b:=y)', FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), RealLiteralASTNode(value=1000.0)),
                (VariableASTNode(name='b'), VariableASTNode(name='y'))
            ],
            function_name=VariableASTNode(name='f'))),
        ('f(a:=1, b:=2, c:= True)', FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), RealLiteralASTNode(value=1)),
                (VariableASTNode(name='b'), RealLiteralASTNode(value=2)),
                (VariableASTNode(name='c'), BooleanLiteralASTNode(value=True))
            ],
            function_name=VariableASTNode(name='f'))),
        ('f(a:=1+2, b:= True or False)', FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), AdditionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=RealLiteralASTNode(value=2)
                )),
                (VariableASTNode(name='b'), OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
            ],
            function_name=VariableASTNode(name='f'))),
        ('f(a := 1, b := g(arg:=True))', FunctionCallExpressionASTNode(
            arg_bindings=[
                (VariableASTNode(name='a'), RealLiteralASTNode(value=1)),
                (VariableASTNode(name='b'), FunctionCallExpressionASTNode(
                    arg_bindings=[
                        (VariableASTNode(name='arg'), BooleanLiteralASTNode(value=True)),
                    ],
                    function_name=VariableASTNode(name='g')))
            ],
            function_name=VariableASTNode(name='f'))),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TypeASTNode)
        assert assignment_node.identifier_type.value is None
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
             AdditionExpressionASTNode(left_arg=RealLiteralASTNode(value=1), right_arg=RealLiteralASTNode(value=2)),
         ])),
        ('x = 1 ; 1 + 2 ; x: Real = 1',
         ModuleASTNode(statements=[
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
             AdditionExpressionASTNode(left_arg=RealLiteralASTNode(value=1), right_arg=RealLiteralASTNode(value=2)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
         ])),
        ('x = 1 ; x: Real = 1 ; 1 + 2 # y = 123',
         ModuleASTNode(statements=[
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
             AdditionExpressionASTNode(left_arg=RealLiteralASTNode(value=1), right_arg=RealLiteralASTNode(value=2)),
         ])
        ),
        ('''

x
x = 1 # comment
x: Real = 1 # comment
init_something()
init_something(x:=x)
x = 1 ; x: Real = 1 # y = 123
1 + 2

''',
         ModuleASTNode(statements=[
             VariableASTNode(name='x'),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
             FunctionCallExpressionASTNode(arg_bindings=[], function_name=VariableASTNode(name='init_something')),
             FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), VariableASTNode(name='x'))], function_name=VariableASTNode(name='init_something')),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
             AdditionExpressionASTNode(left_arg=RealLiteralASTNode(value=1), right_arg=RealLiteralASTNode(value=2)),
         ])
        ),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
