
import pytest
from tibs import parser

# TODO make sure all these imports are used

TYPES = [
    'NothingType',
    'Integer',
    'Boolean',
    'Float',
    'String',
]

LITERALS = [
    'Nothing',
    'True',
    'False',
]

OPERATORS = [
    'print',
    'not',
    'and',
    'xor',
    'or',
    '**',
    '^',
    '*',
    '/',
    '+',
    '-',
    '=',
    '>',
    '>=',
    '<',
    '<=',
    '==',
    '!=',
    '<<',
]

SYNTACTIC_CONTRUCT_KEYWORDS = [
    'function',
    'for',
    'while',
    'if',
    'else'
]

INVALID_INPUT_STRING_TEMPLATE_TO_RESERVED_KEYWORDS = [
    ('{keyword} = 1', TYPES + LITERALS + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
    ('{keyword}(x:=1)', TYPES + LITERALS + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
    ('f({keyword}:=1)', TYPES + LITERALS + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
    ('x = {{{keyword}}}', TYPES + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
    ('{keyword}', TYPES + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
    ('print << {keyword}', TYPES + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
    ('x: {keyword} = 1', LITERALS + OPERATORS + SYNTACTIC_CONTRUCT_KEYWORDS),
]

INVALID_INPUT_STRINGS = [
    pytest.param(invalid_input_string_template.format(keyword=keyword))
    for invalid_input_string_template, keywords in INVALID_INPUT_STRING_TEMPLATE_TO_RESERVED_KEYWORDS
    for keyword in keywords
]

@pytest.mark.parametrize('input_string', INVALID_INPUT_STRINGS)
def test_parser_invalid_keyword_use(input_string):
    with pytest.raises(parser.ParseError, match='Could not parse the following:'):
        parser.parseSourceCode(input_string)
