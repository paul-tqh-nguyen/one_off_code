
import os
import inspect

import pytest
from .test_utilities import *

import tibs
from tibs import compiler
from tibs.misc_utilities import *

def test_compiler_c_functions_registered():
    tibs_dir = os.path.dirname(inspect.getfile(tibs))
    search_dir = os.path.join(tibs_dir, 'compiler',  'mlir')
    grep_command = f'''
pushd {search_dir}
grep -r "extern \\\"C\\\" .*{{" . --include "*.cpp" | cut -d"(" -f1
popd
'''
    stdout_string, stderr_string, returncode = shell(grep_command)
    c_function_names = set(filter(len, (''.join(line.split(':')[1:]).strip().split(' ')[-1] for line in stdout_string.split('\n'))))
    assert returncode == 0
    assert stderr_string == ''
    for c_function_name in c_function_names:
        c_function = getattr(compiler.c, c_function_name)
        assert c_function.input_types is not BOGUS_TOKEN, f'{c_function_name} has no input types declared in tibs.compiler.c .'
        assert c_function.return_type is not BOGUS_TOKEN, f'{c_function_name} has no return type declared in tibs.compiler.c .'
    unknown_source_func_names = [func_name for func_name in compiler.c.func_name_to_func.keys() if func_name not in c_function_names]
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
    # assert False
    return
