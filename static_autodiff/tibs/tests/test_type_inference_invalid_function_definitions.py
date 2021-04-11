
import pytest
from tibs import parser, type_inference
from tibs.misc_utilities import *

INVALID_RETURN_STATEMENT_INPUT_STRINGS = [pytest.param(*args, id=f'invalid_return_statement_{i}') for i, args in enumerate([
    (
        '''
function f() -> Integer, Boolean {
    function g() -> Integer
        return 1, 2, 3, 4
    return g(), True
}
''',
        'g is declared to have 1 return values but attempts to return 4 values.'
    ),
    (
        '''
function f() -> Integer, Boolean {
    function g() -> String
        return 1
    return g(), True
}
''',
        'has the following inconsistent types'
    ),
    (
        '''
function f() -> Integer, Boolean {
    function g() -> String
        return "asd"
    function g() -> Float {
        return 1.1
    }
    return g(), True
}
''',
        'g defined multiple times.'
    )
])]

@pytest.mark.parametrize('input_string, error_match_string', INVALID_RETURN_STATEMENT_INPUT_STRINGS)
def test_invalid_return_statements(input_string, error_match_string):
    result = parser.parseSourceCode(input_string)
    with pytest.raises(type_inference.TypeInferenceFailure, match=error_match_string):
        type_inference.perform_type_inference(result)

def test_return_statement_outside_function_definition():
    input_string = '''
return 123
'''
    result = parser.parseSourceCode(input_string)
    with pytest.raises(type_inference.TypeInferenceFailure, match='Return statement used outside of function body.'):
        type_inference.perform_type_inference(result)
