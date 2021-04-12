
import pytest
from tibs import parser
from tibs.misc_utilities import *

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
