
import pytest
from tibs import parser, type_inference
from tibs.misc_utilities import *

INVALID_CONDITIONAL_INPUT_STRINGS = [pytest.param(*args, id=f'case_{i}') for i, args in enumerate([
    (
        '''
a = [1,2]
b = 2
if a == b {
   print "equal"
} else {
   print "not equal"
}
''',
        type_inference.TypeInferenceFailure,
        'compares expressions with different types.'
    ),
])]

@pytest.mark.parametrize('input_string, exception_type, error_match_string', INVALID_CONDITIONAL_INPUT_STRINGS)
def test_invalid_conditionals(input_string, exception_type, error_match_string):
    result = parser.parseSourceCode(input_string)
    with pytest.raises(exception_type, match=error_match_string):
        type_inference.perform_type_inference(result)
