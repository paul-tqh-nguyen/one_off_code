
import pytest
from tibs import parser, type_inference
from tibs.misc_utilities import *

INVALID_EXAMPLES = [pytest.param(*args, id=f'case_{i}') for i, args in enumerate([
    (
        '''
while [True, True]
    func(x:=1)
''',
        type_inference.TypeInferenceConsistencyError,
        'has the following inconsistent types'
    ),
])]

@pytest.mark.parametrize('input_string, exception_type, error_match_string', INVALID_EXAMPLES)
def test_invalid_while_loop(input_string, exception_type, error_match_string):
    with pytest.raises(exception_type, match=error_match_string):
        result = parser.parseSourceCode(input_string)
        type_inference.perform_type_inference(result)
