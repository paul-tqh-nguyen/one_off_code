
import pytest
import itertools
from typing import Generator

from tibs import parser

def generate_invalid_input_strings() -> Generator[str, None, None]:
    type_to_exemplars = {
        'Boolean': {'True', 'False'},
        'Integer': {'0', '1'},
        'Float': {'1.0', '0.0', '1e3'},
        'String': {'"string"'},
        'NothingType': {'Nothing'},
    }
    types = list(type_to_exemplars.keys())
    
    type_to_compatible_types = {type: type for type in types}
    type_to_compatible_types['Integer'] = {'Integer', 'Float'}
    type_to_compatible_types['Float'] = {'Integer', 'Float'}

    type_to_binary_operations = {
        'Boolean': {'and', 'xor', 'or'},
        'Integer': {
            '**', '^', '*', '/', '+', '-',
            '>', '>=', '<', '<=', '==', '!='
        },
        'Float': {
            '**', '^', '*', '/', '+', '-',
            '>', '>=', '<', '<=', '==', '!='
        },
        'String': {'<<'},
        'NothingType': {},
    }

    type_to_unary_operations = {
        'Boolean': {'not'},
        'Integer': {'-'},
        'Float': {'-'},
        'String': set(),
        'NothingType': set(),
    }
    type_to_unary_operations = {type: unary_operations | {''} for type, unary_operations in type_to_unary_operations.items()}

    assert len(types) == len(type_to_exemplars) == len(type_to_compatible_types) == len(type_to_binary_operations) == len(type_to_unary_operations)
    
    for type in types:
        other_types = {other_type for other_type in types if other_type not in type_to_compatible_types[type]}
        for other_type in other_types:
            unary_operations = type_to_unary_operations[type]
            type_exemplars = type_to_exemplars[type]
            binary_operations = type_to_binary_operations[type]
            other_type_exemplars = type_to_exemplars[other_type]
            for un_op1, arg1, bin_op, un_op2, arg2 in itertools.product(
                    unary_operations,
                    type_exemplars,
                    binary_operations,
                    unary_operations,
                    other_type_exemplars
            ):
                input_string = f'x = {un_op1} {arg1} {bin_op} {un_op2} {arg2}'
                yield input_string

@pytest.mark.parametrize("input_string", generate_invalid_input_strings())
def test_parser_incompatible_expression_use(input_string):
    with pytest.raises(parser.ParseError, match='Could not parse the following:'):
        parser.parseSourceCode(input_string)
