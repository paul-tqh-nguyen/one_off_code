
import pytest
import random
import datetime
import itertools

from typing import Generator

from tibs import parser, type_inference
from tibs.misc_utilities import *

random.seed(datetime.date.today())

############################
# test_parser_invalid_misc #
############################

BAD_PARSE_STRINGS = eager_map(
    pytest.param,
    (
        '''
###
init()

x: Integer

''',
        'print << if << 3 ',
        'print << return 3 ',
        'print << if True 1 else 2',
        'print << Boolean',
        'print << {1;2;3}',
        'print << ',
        'x = {}',
        'x = """',
        'x = \'123\'',
        'x = print()',
        'x = print(1, 2, 3)',
        'x = print(1, "2", 3)',
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
        'print(x:=1)',
        'not(x:=1)',
        'and(x:=1)',
        'xor(x:=1)',
        'or(x:=1)',
        'return(x:=1)',
        'function(x:=1)',
        'function',
        'function = 1',
        'x: function = 1',
        'function: Integer = 1',
        'x: Integer<??> = 1',
        'x: Integer<1, ??> = 1',
        'x: Integer<???, 1> = 1',
        'x: for = 1',
        'for = 1',
        'while 0 {return}',
        'if 0 {return}',
        'if 0 {return} then 123',
        'if print() {return} then 123',
        'if 0 {return} then if',
    )
)

@pytest.mark.parametrize('input_string', BAD_PARSE_STRINGS)
def test_parser_invalid_misc(input_string):
    with pytest.raises(parser.ParseError, match='Could not parse the following:'):
        parser.parseSourceCode(input_string)

###########################
# test_invalid_expression #
###########################

UNIFORM_INPUT_TYPE_BINARY_OPERATORS = (
    'and',
    'xor',
    'or',
    '**',
    '^',
    '*',
    '/',
    '+',
    '-',
    # TODO should we include comparison operators here?
)

def generate_example_values() -> Generator[str, None, None]:
    '''Every yielded element must be of a unique type (different shapes imply different types) and have unique value.'''
    example_vector_values = (
        'Nothing',
        '100',
        'True',
        '12.34',
        ' "blah" ',
    )
    assert len(example_vector_values) == len(set(example_vector_values)) == len(
        quadratic_unique(
            map(
                compose(type_inference.perform_type_inference, parser.parseSourceCode),
                example_vector_values
            )
        )
    )
    yield from example_vector_values
    max_rank = 2 # random.randint(2,10)
    for _ in range(max_rank):
        dimension_size = random.randint(1,3)
        example_vector_values = tuple(
            '[' + str.join(',', itertools.repeat(subvectors, dimension_size)) + ']'
            for subvectors in example_vector_values
        )
        yield from example_vector_values
    return

INVALID_EXPRESSIONS = [
    pytest.param(
        ' '.join([value_1, binary_operator, value_2]),
        id=f'invalid_expression_{i}'
    )
    for i, (value_1, binary_operator, value_2) in enumerate(itertools.product(generate_example_values(), UNIFORM_INPUT_TYPE_BINARY_OPERATORS, generate_example_values()))
    if value_1 != value_2
]
# too many slow cases, so just do a few random ones
random.shuffle(INVALID_EXPRESSIONS)
INVALID_EXPRESSIONS = INVALID_EXPRESSIONS[:100]

@pytest.mark.parametrize('input_string', INVALID_EXPRESSIONS)
def test_invalid_expression(input_string):
    with pytest.raises(
            (parser.ParseError, type_inference.TypeInferenceFailure)
            , match=r'Could not parse the following:|operates on expressions with different types.'
    ):
        result = parser.parseSourceCode(input_string)
        type_inference.perform_type_inference(result)
