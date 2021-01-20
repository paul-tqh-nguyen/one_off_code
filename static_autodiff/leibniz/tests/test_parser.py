
import pytest

from leibniz import parser

def test_parser_boolean():
    expected_input_output_pairs = [
        ('True', True),
        ('False', False),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string).asList()[2]
        assert result == expected_result

def test_parser_real():
    expected_input_output_pairs = [ # TODO Make all these cases work
        ('123', 123),
        ('0.123', 0.123),
        ('-0.123', -0.123),
        ('.123', 0.123),
        ('-.123', -0.123),
        # ('1.e2', 100.0),
        # ('1.E2', 100.0),
        # ('1.0e2', 100.0),
        ('1e2', 100.0),
        ('-1E2', -100.0),
        # ('.23E2', 23.0),
        # ('-1.23e-2', -0.0123),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode('x = '+input_string).asList()[2]
        assert result == expected_result

def test_parser_identifier():
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
        result = parser.parseSourceCode(input_string+' = 1').asList()[0]
        print(f"result {repr(result)}")
        assert type(result) is str
        assert result == input_string.split(' ')[0]

