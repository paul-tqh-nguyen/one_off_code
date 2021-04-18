
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
        type_inference.TypeInferenceFailure,
        r'given [0-9]+ args when [0-9]+ were expected'
    ),
    (
        'function f(x: Integer, y: Integer) -> NothingType return ; f(x:=2, a:=1)',
        type_inference.TypeInferenceFailure,
        r'given unexpected paramter binding for'
    ),
    (
        'function f(x: Integer, x: Integer) -> NothingType return',
        type_inference.SemanticError,
        r'defined with redundantly defined parameters.'
    ),
    (
        'function f(x: Integer, y: Integer) -> NothingType return ; f(x:=2, y:=1, y:=1)',
        type_inference.SemanticError,
        r'called with redundantly defined parameters.'
    ),
    (
        '''
reused_variable_name: Integer = 10
function reused_variable_name(x: Integer) -> NothingType return
reused_variable_name(x:=1)
''',
        type_inference.TypeInferenceFailure,
        r' defined multiple times.'
    ),
    (
        '''
start = 0
end = 2
delta = 0.5
for x:(start, end, delta) {
    True or True
}
''',
        type_inference.TypeInferenceConsistencyError,
        r'has the following inconsistent types: '
    ),
    (
        '''
start = 0.0
end = "two"
delta = 0.5
for x:(start, end, delta) {
    True or True
}
''',
        type_inference.TypeInferenceConsistencyError,
        r'has the following inconsistent types: '
    ),
    (
        '''
start = 0.0
end = 200
delta = 5
for x:(start, end, delta) {
    True or True
}
''',
        type_inference.TypeInferenceConsistencyError,
        r'has the following inconsistent types: '
    ),
    (
        '''
start = 0
end = 200
delta = 5
for x:(start, end, delta) {
    True or True
}
x + 1 
''',
        type_inference.TypeInferenceFailure,
        r' used before type declared.'
    ),
    (
        '''
start = [[1,2]]
start = 123
''',
        type_inference.TypeInferenceConsistencyError,
        r' has the following inconsistent types: '
    ),
]))

@pytest.mark.parametrize('input_string, exception_type, error_match_string', INVALID_MODULE_CASES)
def test_invalid_module(input_string, exception_type, error_match_string):
    with pytest.raises(exception_type, match=error_match_string):
        result = parser.parseSourceCode(input_string)
        type_inference.perform_type_inference(result)
