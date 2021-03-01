
import os
import inspect

import pytest
from .test_utilities import *

import tibs
from tibs import compiler
from tibs.misc_utilities import *
""" # TODO enable this
def test_compiler_c_functions_registered():
    tibs_dir = os.path.dirname(inspect.getfile(tibs))
    search_dir = os.path.join(tibs_dir, 'compiler',  'mlir')
    grep_command = f'''
pushd {search_dir}
grep -r "extern \\\"C\\\" .*{{" . --include "*.cpp"
popd
'''
    stdout_string, stderr_string, returncode = shell(grep_command)
    lines = eager_filter(len, (''.join(line.split(':')[1:]).strip() for line in stdout_string.split('\n')))
    c_function_name_to_arg_count = {}
    for line in lines:
        assert line.startswith('extern "C" ')
        assert line.endswith(') {')
        left_of_open_paren_string, right_of_open_paren_string = line.split('(')
        c_function_name = left_of_open_paren_string.strip().split(' ')[-1]
        assert len([character for character  in right_of_open_paren_string if character == ')']) == 1
        arg_count = len(eager_filter(len, right_of_open_paren_string.split(')')[0].split(',')))
        c_function_name_to_arg_count[c_function_name] = arg_count
    assert returncode == 0
    assert stderr_string == ''
    for c_function_name, arg_count in c_function_name_to_arg_count.items():
        c_function = getattr(compiler.c, c_function_name)
        assert c_function.input_types is not BOGUS_TOKEN, f'{c_function_name} has no input types declared in tibs.compiler.c .'
        assert c_function.return_type is not BOGUS_TOKEN, f'{c_function_name} has no return type declared in tibs.compiler.c .'
        assert len(c_function.input_types) == arg_count, f'{c_function_name} has {len(c_function.input_types)} input types declared in tibs.compiler.c when {arg_count} are expected.'
    unknown_source_func_names = [func_name for func_name in compiler.c.func_name_to_func.keys() if func_name not in c_function_name_to_arg_count]
    assert len(unknown_source_func_names) == 0, f'The functions {unknown_source_func_names} have their source code definitions in unexpected places.'
    return

@subprocess_test
def test_compiler():
    # TODO update this test
    mod_gen = compiler.ModuleGenerator()
    if True: # TODO remove this
        location_pointer = mod_gen.new_location(b'/tmp/fake_file.tibs', 12, 34)
        compiler.c.generateModule(mod_gen.module_generator_pointer, location_pointer)
        mod_gen.delete_location(location_pointer)
    mod_gen.run_pass_manager()
    result = mod_gen.dump_module()
    print(result)
    assert False
    return
"""
