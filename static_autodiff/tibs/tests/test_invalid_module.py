
import pytest
from tibs import parser, type_inference
from tibs.misc_utilities import *

INVALID_MODULE_CASES = tuple(pytest.param(*args, id=f'invalid_module_{i}') for i, args in enumerate([
    (
        'function f(a: Integer) -> NothingType return ; f(a:=True or False)',
        type_inference.TypeInferenceConsistencyError,
        'has the following inconsistent types'
    ),
    (
        'function f(x: Integer) -> NothingType return ; f(a:=1)',
        type_inference.TypeInferenceFailure,
        'given unexpected paramter binding for'
    ),
    (
        'function f(x: Integer) -> NothingType return ; f(x:=2, a:=1)',
        type_inference.TypeInferenceConsistencyError,
        f'given [0-9]+ args when [0-9]+ were expected'
    ),
]))

@pytest.mark.parametrize('input_string, exception_type, error_match_string', INVALID_MODULE_CASES)
def test_invalid_module(input_string, exception_type, error_match_string):
    result = parser.parseSourceCode(input_string)
    with pytest.raises(exception_type, match=error_match_string):
        type_inference.perform_type_inference(result)
