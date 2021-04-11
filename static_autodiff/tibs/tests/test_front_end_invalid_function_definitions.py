
import pytest
from tibs import parser, type_inference
from tibs.misc_utilities import *

INVALID_FUNCTION_DEFINITION_INPUT_STRINGS = [pytest.param(*args, id=f'invalid_function_definition_{i}') for i, args in enumerate([
    (
        '''
function f() -> NothingType {
    return print "if" 
}
''',
        parser.ParseError,
        'Could not parse the following'
    ),
    (
        '''
function f() -> Integer, Boolean {
    function g() -> Integer
        return 1, 2, 3, 4
    return g(), True
}
''',
        type_inference.TypeInferenceFailure,
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
        type_inference.TypeInferenceFailure,
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
        type_inference.TypeInferenceFailure,
        'g defined multiple times.'
    )
])]

@pytest.mark.parametrize('input_string, exception_type, error_match_string', INVALID_FUNCTION_DEFINITION_INPUT_STRINGS)
def test_invalid_function_definitions(input_string, exception_type, error_match_string):
    result = parser.parseSourceCode(input_string)
    with pytest.raises(exception_type, match=error_match_string):
        type_inference.perform_type_inference(result)

def test_return_statement_outside_function_definition():
    input_string = '''
return 123
'''
    result = parser.parseSourceCode(input_string)
    with pytest.raises(type_inference.TypeInferenceFailure, match='Return statement used outside of function body.'):
        type_inference.perform_type_inference(result)
