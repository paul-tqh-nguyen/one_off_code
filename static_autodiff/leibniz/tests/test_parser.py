
import pytest

from leibniz import parser
from leibniz.parser import (
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
    FunctionCallASTNode,
    DeclarationASTNode,
    SubtheoryASTNode,
)
from leibniz.misc_utilities import *

# TODO verify that the above imports are used

def test_parser_atomic_boolean():
    expected_input_output_pairs = [
        ('True', True),
        ('False', False),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        subtheory_node = parser.parseSourceCode('x = '+input_string)
        declaration_node = only_one(subtheory_node.declarations)
        value_node = declaration_node.value
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
        ('1e2', 100.0),
        ('1E2', 100.0),
        ('1E-2', 0.01),
        # ('.23E2', 23.0),
        # ('1.23e-2', 0.0123),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        subtheory_node = parser.parseSourceCode('x = '+input_string)
        declaration_node = only_one(subtheory_node.declarations)
        value_node = declaration_node.value
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
        subtheory_node = parser.parseSourceCode(input_string+' = 1')
        declaration_node = only_one(subtheory_node.declarations)
        identifier_node = declaration_node.identifier
        result = identifier_node.name
        assert type(result) is str
        assert result == input_string, f'''
input_string: {repr(input_string)}
result: {repr(result)}
'''

def test_parser_boolean_expression():
    expected_input_output_pairs = [
        ('not False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=NotExpressionASTNode(
                    arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('not True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=NotExpressionASTNode(
                    arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('True and True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AndExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('False and False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AndExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('True and False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AndExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('False and True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AndExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('True xor True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=XorExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('False xor False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=XorExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('True xor False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=XorExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('False xor True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=XorExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('True or True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('False or False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('True or False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=True),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))
        ])),
        ('False or True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('True and True and False and True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AndExpressionASTNode(
                    left_arg=AndExpressionASTNode(
                        left_arg=AndExpressionASTNode(
                            left_arg=BooleanLiteralASTNode(value=True),
                            right_arg=BooleanLiteralASTNode(value=True)
                        ),
                        right_arg=BooleanLiteralASTNode(value=False)
                    ),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('not True and True', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AndExpressionASTNode(
                    left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
                    right_arg=BooleanLiteralASTNode(value=True)
                ))
        ])),
        ('not True and True xor False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=XorExpressionASTNode(
                    left_arg=AndExpressionASTNode(
                        left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
                        right_arg=BooleanLiteralASTNode(value=True)
                    ),
                    right_arg=BooleanLiteralASTNode(value=False)
                ))])),
        ('False or not True and True xor False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=OrExpressionASTNode(
                    left_arg=BooleanLiteralASTNode(value=False),
                    right_arg=XorExpressionASTNode(
                        left_arg=AndExpressionASTNode(
                            left_arg=NotExpressionASTNode(arg=BooleanLiteralASTNode(value=True)),
                            right_arg=BooleanLiteralASTNode(value=True)
                        ),
                        right_arg=BooleanLiteralASTNode(value=False)
                    )
                ))])),
        ('True xor False or not True and True xor False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=OrExpressionASTNode(
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
                ))])),
        ('True xor (False or not True) and True xor False', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=XorExpressionASTNode(
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
                ))])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_arithmetic_expression():
    expected_input_output_pairs = [
        ('1 + 2', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AdditionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=RealLiteralASTNode(value=2)
                ))
        ])),
        ('1 + 2 - 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=SubtractionExpressionASTNode(
                    left_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 + 2 * 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AdditionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=MultiplicationExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('(1 + 2) * 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=MultiplicationExpressionASTNode(
                    left_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 / 2 * 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=MultiplicationExpressionASTNode(
                    left_arg=DivisionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 / 2 + 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=AdditionExpressionASTNode(
                    left_arg=DivisionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 / (2 + 3)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=DivisionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ^ 2', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=RealLiteralASTNode(value=2)
                ),
            )])),
        ('1 ^ (2 + 3)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ** 2 ^ 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ^ 2 ** 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 / 2 ** 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=DivisionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ** 2 - 3', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=SubtractionExpressionASTNode(
                    left_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 ^ (2 + 3)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ^ (2 - -3)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=SubtractionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=NegativeExpressionASTNode(RealLiteralASTNode(value=3))
                    )
                ),
            )])),
        ('1 ** (2 --3)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=SubtractionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=NegativeExpressionASTNode(RealLiteralASTNode(value=3))
                    )
                ),
            )])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_declaration():
    expected_input_output_pairs = [
        ('x', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=None)])),
        ('x = 1', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1))])),
        ('x: Integer', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Integer'), value=None)])),
        ('x: Real = 1', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))])),
        ('x ; x = 1 ; x: Integer ; x: Real = 1', SubtheoryASTNode(declarations=[
            DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=None),
            DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
            DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Integer'), value=None),
            DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
        ])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_comments():
    expected_input_output_pairs = [
        ('x # comment', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=None)])),
        ('x = 1 # comment', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1))])),
        ('x: Integer # comment # comment ', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Integer'), value=None)])),
        ('x: Real = 1 # comment', SubtheoryASTNode(declarations=[DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))])),
        (
            'x ; x = 1 ; x: Integer ; x: Real = 1 # y = 123',
            SubtheoryASTNode(declarations=[
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=None),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Integer'), value=None),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))
            ])
        ),
        (
            '''

x # comment
x = 1 # comment
x: Integer # comment
x: Real = 1 # comment
x ; x = 1 ; x: Integer ; x: Real = 1 # y = 123

''',
            SubtheoryASTNode(declarations=[
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=None),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Integer'), value=None),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=None),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Integer'), value=None),
                DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))
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

def test_parser_function_call():
    expected_input_output_pairs = [
        ('x = f() # comment', SubtheoryASTNode(declarations=[
            DeclarationASTNode(identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None), value=FunctionCallASTNode(arg_bindings=[], function_name=VariableASTNode(name='f')))
        ])),
        ('x = f(a:=1) # comment', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=FunctionCallASTNode(
                    arg_bindings=[(VariableASTNode(name='a'), RealLiteralASTNode(value=1))],
                    function_name=VariableASTNode(name='f'))
            )])),
        ('x = f(a:=1, b:=2)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=FunctionCallASTNode(
                    arg_bindings=[
                        (VariableASTNode(name='a'), RealLiteralASTNode(value=1)),
                        (VariableASTNode(name='b'), RealLiteralASTNode(value=2))
                    ],
                    function_name=VariableASTNode(name='f'))
            )])),
        ('x = f(a:=1e3, b:=y)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=FunctionCallASTNode(
                    arg_bindings=[
                        (VariableASTNode(name='a'), RealLiteralASTNode(value=1000.0)),
                        (VariableASTNode(name='b'), VariableASTNode(name='y'))
                    ],
                    function_name=VariableASTNode(name='f'))
            )])),
        ('x = f(a:=1, b:=2, c:= True)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=FunctionCallASTNode(
                    arg_bindings=[
                        (VariableASTNode(name='a'), RealLiteralASTNode(value=1)),
                        (VariableASTNode(name='b'), RealLiteralASTNode(value=2)),
                        (VariableASTNode(name='c'), BooleanLiteralASTNode(value=True))
                    ],
                    function_name=VariableASTNode(name='f'))
            )])),
        ('x = f(a:=1+2, b:= True or False)', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=FunctionCallASTNode(
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
                    function_name=VariableASTNode(name='f'))
            )])),
        ('x = f(a := 1, b := g(arg:=True))', SubtheoryASTNode(declarations=[
            DeclarationASTNode(
                identifier=VariableASTNode(name='x'), identifier_type=TypeASTNode(value=None),
                value=FunctionCallASTNode(
                    arg_bindings=[
                        (VariableASTNode(name='a'), RealLiteralASTNode(value=1)),
                        (VariableASTNode(name='b'), FunctionCallASTNode(
                            arg_bindings=[
                                (VariableASTNode(name='arg'), BooleanLiteralASTNode(value=True)),
                            ],
                            function_name=VariableASTNode(name='g')))
                    ],
                    function_name=VariableASTNode(name='f'))
            )])),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string)
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
