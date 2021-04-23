
import pytest
from tibs import parser, type_inference
from tibs.misc_utilities import *

INVALID_STRING_EXPRESSION = [pytest.param(*args, id=f'invalid_string_expression_{i}') for i, args in enumerate([
    (
        ' "a" << 1.2 ',
        parser.ParseError,
        'Could not parse the following'
    ),
])]

@pytest.mark.parametrize('input_string, exception_type, error_match_string', INVALID_STRING_EXPRESSION)
def test_invalid_string_expression(input_string, exception_type, error_match_string):
    with pytest.raises(exception_type, match=error_match_string):
        result = parser.parseSourceCode(input_string)
        type_inference.perform_type_inference(result)
