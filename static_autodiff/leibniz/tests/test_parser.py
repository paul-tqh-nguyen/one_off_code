
import pytest

from leibniz import parser
from leibniz.parser import (
    ExpressionASTNode,
    TensorTypeASTNode,
    VectorExpressionASTNode,
    FunctionDefinitionExpressionASTNode,
    ReturnStatementASTNode,
    BooleanLiteralASTNode,
    IntegerLiteralASTNode,
    FloatLiteralASTNode,
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
    FunctionCallExpressionASTNode,
    AssignmentASTNode,
    ModuleASTNode,
) # TODO reorder these in according to their declaration
from leibniz.misc_utilities import *

# TODO verify that the above imports are used

def test_parser_invalid():
    invalid_input_strings = [
        '''
###
init()

x: Integer

''',
        'x = {}',
        'x = {function}',
        'x = {Nothing}',
        'x = {Integer}',
        'x = {Float}',
        'x = {Boolean}',
        'x: function = 1',
        'x: Nothing = 1',
        'x: True = 1',
        'x: False = 1',
        'x: not = 1',
        'x: and = 1',
        'x: xor = 1',
        'x: or = 1',
        'x: ** = 1',
        'x: ^ = 1',
        'x: * = 1',
        'x: / = 1',
        'x: + = 1',
        'x: - = 1',
        'Float',
        'Boolean',
        'Integer',
        'NothingType',
        'Float = 1',
        'Boolean = 1',
        'Integer = 1',
        'NothingType = 1',
        'Float(x:=1)',
        'Boolean(x:=1)',
        'Integer(x:=1)',
        'NothingType(x:=1)',
        'True = 1',
        'False = 1',
        'not = 1',
        'and = 1',
        'xor = 1',
        'or = 1',
        'return = 1',
        'True(x:=1)',
        'False(x:=1)',
        'not(x:=1)',
        'and(x:=1)',
        'xor(x:=1)',
        'or(x:=1)',
        'return(x:=1)',
        'return',
        'function(x:=1)',
        'function',
        'function = 1',
        'function: Integer = 1',
    ]
    for input_string in invalid_input_strings:
        with pytest.raises(parser.ParseError, match='Could not parse the following:'):
            parser.parseSourceCode(input_string)

def test_parser_nothing_literal():
    module_node = parser.parseSourceCode('x = Nothing')
    assert isinstance(module_node, ModuleASTNode)
    assert isinstance(module_node.statements, list)
    assignment_node = only_one(module_node.statements)
    assert isinstance(assignment_node, AssignmentASTNode)
    assert isinstance(assignment_node.identifier, VariableASTNode)
    assert assignment_node.identifier.name is 'x'
    assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
    assert assignment_node.identifier_type.base_type_name is None
    assert assignment_node.identifier_type.shape is None
    value_node = assignment_node.value
    assert isinstance(value_node, NothingTypeLiteralASTNode)
    result = value_node
    expected_result = NothingTypeLiteralASTNode()
    assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

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
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        value_node = assignment_node.value
        assert isinstance(value_node, BooleanLiteralASTNode)
        result = value_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_atomic_integer():
    expected_input_output_pairs = [
        ('123', 123),
        ('0', 0),
        ('0000', 0),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        value_node = assignment_node.value
        assert isinstance(value_node, IntegerLiteralASTNode)
        result = value_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_atomic_float():
    expected_input_output_pairs = [
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
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        value_node = assignment_node.value
        assert isinstance(value_node, FloatLiteralASTNode)
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
        assert assignment_node.value == IntegerLiteralASTNode(value=1) # TODO change this "==" to is_equivalent
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        identifier_node = assignment_node.identifier
        assert isinstance(identifier_node, VariableASTNode)
        result = identifier_node.name
        assert type(result) is str
        assert result == input_string, f'''
input_string: {repr(input_string)}
result: {repr(result)}
'''

def test_parser_vector_literal():
    expected_input_output_pairs = [
        ('[1,2,3]', VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)])),
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
        ('[f(x:=1), [2, 3], some_variable, Nothing]',
         VectorExpressionASTNode(values=[
             FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), IntegerLiteralASTNode(value=1))], function_name='f'),
             VectorExpressionASTNode(values=[IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)]),
             VariableASTNode(name='some_variable')
         ])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode('x = '+input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        result = assignment_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
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
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        result = assignment_node.value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_arithmetic_expression():
    expected_input_output_pairs = [
        ('1 + 2', AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2))),
        ('1 + 2 - 3',
         SubtractionExpressionASTNode(
             left_arg=AdditionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=1),
                 right_arg=IntegerLiteralASTNode(value=2)
             ),
             right_arg=IntegerLiteralASTNode(value=3)
         )),
        ('1 + 2 * 3',
         AdditionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=MultiplicationExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('(1 + 2) * 3',
         MultiplicationExpressionASTNode(
             left_arg=AdditionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=1),
                 right_arg=IntegerLiteralASTNode(value=2)
             ),
             right_arg=IntegerLiteralASTNode(value=3)
         )),
        ('1 / 2 * 3',
         MultiplicationExpressionASTNode(
             left_arg=DivisionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=1),
                 right_arg=IntegerLiteralASTNode(value=2)
             ),
             right_arg=IntegerLiteralASTNode(value=3)
         )),
        ('1 / 2 + 3',
         AdditionExpressionASTNode(
             left_arg=DivisionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=1),
                 right_arg=IntegerLiteralASTNode(value=2)
             ),
             right_arg=IntegerLiteralASTNode(value=3)
         )),
        ('1 / (2 + 3)',
         DivisionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=AdditionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('1 ^ 2',ExponentExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2))),
        ('1 ^ (2 + 3)',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=AdditionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('1 ** 2 ^ 3',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('1 ^ 2 ** 3',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('1 / 2 ** 3',
         DivisionExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=ExponentExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('1 ** 2 - 3',
         SubtractionExpressionASTNode(
             left_arg=ExponentExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=1),
                 right_arg=IntegerLiteralASTNode(value=2)
             ),
             right_arg=IntegerLiteralASTNode(value=3)
         )),
        ('1 ^ (2 + 3)',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=AdditionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=IntegerLiteralASTNode(value=3)
             )
         )),
        ('1 ^ (2 - -3)',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=SubtractionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=3))
             )
         )),
        ('1 ** (2 --3)',
         ExponentExpressionASTNode(
             left_arg=IntegerLiteralASTNode(value=1),
             right_arg=SubtractionExpressionASTNode(
                 left_arg=IntegerLiteralASTNode(value=2),
                 right_arg=NegativeExpressionASTNode(IntegerLiteralASTNode(value=3))
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
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
        result = assignment_node.value
        assert isinstance(result, ExpressionASTNode)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_assignment():
    expected_input_output_pairs = [
        ('x = 1', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=IntegerLiteralASTNode(value=1))),
        ('x: NothingType = Nothing', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='NothingType', shape=[]), value=NothingTypeLiteralASTNode())),
        ('x = Nothing', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=NothingTypeLiteralASTNode())),
        ('x: NothingType<> = Nothing', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='NothingType', shape=[]), value=NothingTypeLiteralASTNode())),
        ('x: NothingType<1,?> = Nothing', AssignmentASTNode(
            identifier=VariableASTNode(name='x'),
            identifier_type=TensorTypeASTNode(base_type_name='NothingType', shape=[IntegerLiteralASTNode(value=1), None]),
            value=NothingTypeLiteralASTNode())),
        ('x: Integer = 1', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1))),
        ('x: Integer<> = 1', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1))),
        ('x: Integer<2,3,4> = 1', AssignmentASTNode(
            identifier=VariableASTNode(name='x'),
            identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[
                IntegerLiteralASTNode(value=2),
                IntegerLiteralASTNode(value=3),
                IntegerLiteralASTNode(value=4),
            ]),
            value=IntegerLiteralASTNode(value=1)
        )),
        ('x: Float<?> = value', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Float', shape=[None]), value=VariableASTNode(name='value'))),
        ('x: Float<???> = value', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Float', shape=None), value=VariableASTNode(name='value'))),
        ('x: Float<?, ?, ?> = value', AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Float', shape=[None, None, None]), value=VariableASTNode(name='value'))),
        ('x: Boolean<?, 3, ?> = value', AssignmentASTNode(
            identifier=VariableASTNode(name='x'),
            identifier_type=TensorTypeASTNode(base_type_name='Boolean', shape=[None, IntegerLiteralASTNode(value=3), None]),
            value=VariableASTNode(name='value')
        )),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        module_node = parser.parseSourceCode(input_string)
        assert isinstance(module_node, ModuleASTNode)
        assert isinstance(module_node.statements, list)
        assignment_node = only_one(module_node.statements)
        assert isinstance(assignment_node, AssignmentASTNode)
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        result = assignment_node
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_comments():
    expected_input_output_pairs = [
        ('x = 1 # comment', ModuleASTNode(statements=[AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=IntegerLiteralASTNode(value=1))])),
        ('x: Integer = 1 # comment', ModuleASTNode(statements=[AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1))])),
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
        assert isinstance(assignment_node.identifier, VariableASTNode)
        assert assignment_node.identifier.name is 'x'
        assert isinstance(assignment_node.identifier_type, TensorTypeASTNode)
        assert assignment_node.identifier_type.base_type_name is None
        assert assignment_node.identifier_type.shape is None
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
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=IntegerLiteralASTNode(value=1)),
             AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1)),
             NothingTypeLiteralASTNode(),
         ])),
        ('x = 1 ; x: Integer = 1 ; 1 + 2 # y = 123',
         ModuleASTNode(statements=[
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=IntegerLiteralASTNode(value=1)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1)),
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

''',
         ModuleASTNode(statements=[
             VariableASTNode(name='x'),
             NothingTypeLiteralASTNode(),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=IntegerLiteralASTNode(value=1)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1)),
             FunctionCallExpressionASTNode(arg_bindings=[], function_name='init_something'),
             FunctionCallExpressionASTNode(arg_bindings=[(VariableASTNode(name='x'), VariableASTNode(name='x'))], function_name='init_something'),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name=None, shape=None), value=IntegerLiteralASTNode(value=1)),
             AssignmentASTNode(identifier=VariableASTNode(name='x'), identifier_type=TensorTypeASTNode(base_type_name='Integer', shape=[]), value=IntegerLiteralASTNode(value=1)),
             AdditionExpressionASTNode(left_arg=IntegerLiteralASTNode(value=1), right_arg=IntegerLiteralASTNode(value=2)),
         ])
        ),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        print('\n'*8) # TODO remove this
        result = parser.parseSourceCode(input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_return_statement():
    expected_input_output_pairs = [
        ('', NothingTypeLiteralASTNode()),
        ('  \t # comment', NothingTypeLiteralASTNode()),
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
        assert isinstance(function_definition_node, FunctionDefinitionExpressionASTNode)
        assert function_definition_node.function_name == 'f'
        assert function_definition_node.function_signature == []
        return_statement_node = only_one(function_definition_node.function_body_statements)
        assert isinstance(return_statement_node, ReturnStatementASTNode)
        result = only_one(return_statement_node.return_values)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

# def test_parser_function_definition():
#     expected_input_output_pairs = [
#         ('''
# function f() {
#     return {}
# }
# ''', NothingTypeLiteralASTNode()),
#     ]
#     for input_string, expected_result in expected_input_output_pairs:
#         module_node = parser.parseSourceCode(input_string)
#         assert isinstance(module_node, ModuleASTNode)
#         assert isinstance(module_node.statements, list)
#         function_definition_node = only_one(module_node.statements)
#         assert isinstance(function_definition_node, FunctionDefinitionExpressionASTNode)
#         assert function_definition_node.function_name == 'f'
#         assert function_definition_node.function_signature == []
#         return_statement_node = only_one(function_definition_node.function_body_statements)
#         assert isinstance(return_statement_node, ReturnStatementASTNode)
#         result = only_one(return_statement_node.return_values)
#         assert result == expected_result, f'''
# input_string: {repr(input_string)}
# result: {repr(result)}
# expected_result: {repr(expected_result)}
# '''