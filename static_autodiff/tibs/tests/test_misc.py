
import os
import ast
import inspect
import tibs.tests
from tibs.misc_utilities import *

from typing import Dict

def test_no_redundant_tests():
    '''Meta-test to verify that we don't accidentally have redundantly defined tests as only one of them will be run in such situations.'''
    test_files_dir = inspect.getfile(tibs.tests)
    test_files_dir = os.path.dirname(test_files_dir)
    stdout, stderr, return_code = shell(f'find {test_files_dir} -name "*.py" ')
    assert return_code == 0
    assert len(eager_filter(len, map(str.strip, stderr.split('\n')))) == 0
    function_name_to_location: Dict[str, str] = {}
    for location in filter(len, stdout.split('\n')):
        breakpoint()
        with open(location, 'r') as f:
            source = f.read()
            source_ast = ast.parse(source, filename=location)
        function_names = [function.name for function in source_ast.body if isinstance(function, ast.FunctionDef)]
        for function_name in function_names:
            assert function_name not in function_name_to_location, f'{function_name} defined in {location} and {function_name_to_location[function_name]}'
            function_name_to_location[function_name] = location
