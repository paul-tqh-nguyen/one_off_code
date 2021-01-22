
import pytest

from leibniz import parser
from leibniz.parser import (
    TypeASTNode,
    BooleanLiteralASTNode,
    RealLiteralASTNode,
    NegativeExpressionASTNode,
    ExponentExpressionASTNode,
    MultiplicationExpressionASTNode,
    DivisionExpressionASTNode,
    AdditionExpressionASTNode,
    SubtractionExpressionASTNode,
    AtomicDeclarationASTNode,
    SubtheoryASTNode,
)

# TODO verify that the above imports are used

def test_parser_atomic_boolean():
    expected_input_output_pairs = [
        ('True', True),
        ('False', False),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string).asList()[0][3].value
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_atomic_real():
    expected_input_output_pairs = [ # TODO Make all these cases work
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
        result = parser.parseSourceCode('x = '+input_string).asList()[0][3].value
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
        result = parser.parseSourceCode(input_string+' = 1').asList()[0][0]
        assert type(result) is str
        assert result == input_string, f'''
input_string: {repr(input_string)}
result: {repr(result)}
'''

def test_parser_boolean_expression():
    expected_input_output_pairs = [
        ('not False', ['not', False]),
        ('not True', ['not', True]),
        
        ('True and True', [True, 'and', True]),
        ('False and False', [False, 'and', False]),
        ('True and False', [True, 'and', False]),
        ('False and True', [False, 'and', True]),

        ('True xor True', [True, 'xor', True]),
        ('False xor False', [False, 'xor', False]),
        ('True xor False', [True, 'xor', False]),
        ('False xor True', [False, 'xor', True]),

        ('True or True', [True, 'or', True]),
        ('False or False', [False, 'or', False]),
        ('True or False', [True, 'or', False]),
        ('False or True', [False, 'or', True]),

        ('True and True and False and True', [True, 'and', True, 'and', False, 'and', True]),
        ('not True and True', [['not', True], 'and', True]),
        ('not True and True xor False', [[['not', True], 'and', True], 'xor', False]),
        ('False or not True and True xor False', [False, 'or', [[['not', True], 'and', True], 'xor', False]]),
        ('True xor False or not True and True xor False', [[True, 'xor', False], 'or', [[['not', True], 'and', True], 'xor', False]]),
        ('True xor (False or not True) and True xor False', [True, 'xor', [[False, 'or', ['not', True]], 'and', True], 'xor', False]),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string).asList()[0][3]
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_arithmetic_expression():
    expected_input_output_pairs = [
        ('1 + 2', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=AdditionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=RealLiteralASTNode(value=2)
                ))
        ])),
        ('1 + 2 - 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=SubtractionExpressionASTNode(
                    left_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 + 2 * 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=AdditionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=MultiplicationExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('(1 + 2) * 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=MultiplicationExpressionASTNode(
                    left_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 / 2 * 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=MultiplicationExpressionASTNode(
                    left_arg=DivisionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 / 2 + 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=AdditionExpressionASTNode(
                    left_arg=DivisionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 / (2 + 3)', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=DivisionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ^ 2', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=RealLiteralASTNode(value=2)
                ),
            )])),
        ('1 ^ (2 + 3)', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ** 2 ^ 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ^ 2 ** 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 / 2 ** 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=DivisionExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ** 2 - 3', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=SubtractionExpressionASTNode(
                    left_arg=ExponentExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=1),
                        right_arg=RealLiteralASTNode(value=2)
                    ),
                    right_arg=RealLiteralASTNode(value=3)
                ),
            )])),
        ('1 ^ (2 + 3)', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=AdditionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=RealLiteralASTNode(value=3)
                    )
                ),
            )])),
        ('1 ^ (2 - -3)', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
                value=ExponentExpressionASTNode(
                    left_arg=RealLiteralASTNode(value=1),
                    right_arg=SubtractionExpressionASTNode(
                        left_arg=RealLiteralASTNode(value=2),
                        right_arg=NegativeExpressionASTNode(RealLiteralASTNode(value=3))
                    )
                ),
            )])),
        ('1 ** (2 --3)', SubtheoryASTNode(declarations=[
            AtomicDeclarationASTNode(
                identifier='x', identifier_type=TypeASTNode(value=None),
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
        result = parser.parseSourceCode('x = '+input_string).asList()[0]
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_function_call():
    expected_input_output_pairs = [
        ('f()', ['f', '(', ')']),
        ('f(x := 1)', ['f', '(', 'x', ':=', 1, ')']),
        ('f(x := 1, y := 2)', ['f', '(', 'x', ':=', 1, 'y', ':=', 2, ')']),
        ('f(x := 1e3, y := var)', ['f', '(', 'x', ':=', 1_000.0, 'y', ':=', 'var', ')']),
        ('f(x := 1, y := True)', ['f', '(', 'x', ':=', 1, 'y', ':=', True, ')']),
        ('f(x := 1, y := g(arg:=True))', ['f', '(', 'x', ':=', 1, 'y', ':=', 'g', '(', 'arg', ':=', True, ')', ')']),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string).asList()[0][3:]
        assert result == expected_result and all(type(r) is type(er) for r, er in zip(result, expected_result)), f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_declaration():
    expected_input_output_pairs = [
        ('x', ['x', []]),
        ('x = 1', ['x', [], '=', 1]),
        ('x: Integer', ['x', [':', 'Integer']]),
        ('x: Real = 1', ['x', [':', 'Real'], '=', 1]),
        # ('x ; x = 1 ; x: Integer ; x: Real = 1', ['x', 'x', '=', 1, 'x', ':', 'Integer', 'x', ':', 'Real', '=', 1]), # TODO make this work
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string).asList()[0]
        assert result == expected_result and all(type(r) is type(er) for r, er in zip(result, expected_result)), f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''

def test_parser_comments():
    expected_input_output_pairs = [
        ('x # comment', SubtheoryASTNode(declarations=[AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=None)])),
        ('x = 1 # comment', SubtheoryASTNode(declarations=[AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1))])),
        ('x: Integer # comment', SubtheoryASTNode(declarations=[AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Integer'), value=None)])),
        ('x: Real = 1 # comment', SubtheoryASTNode(declarations=[AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))])),
        (
            'x ; x = 1 ; x: Integer ; x: Real = 1 # y = 123',
            SubtheoryASTNode(declarations=[
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=None),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Integer'), value=None),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))
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
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=None),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Integer'), value=None),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1)),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=None),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value=None), value=RealLiteralASTNode(value=1)),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Integer'), value=None),
                AtomicDeclarationASTNode(identifier='x', identifier_type=TypeASTNode(value='Real'), value=RealLiteralASTNode(value=1))
            ])
        ),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string).asList()[0]
        assert result == expected_result, f'''
input_string: {repr(input_string)}
result: {repr(result)}
expected_result: {repr(expected_result)}
'''
