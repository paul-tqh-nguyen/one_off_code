
import pytest
from .test_utilities import *

from tibs import compiler

from tibs.misc_utilities import *

def test_compiler(capsys):
    # TODO update this test
    mod_gen = compiler.ModuleGenerator()
    mod_gen.c.generateModule(mod_gen.module_generator_pointer) # TODO remove this
    mod_gen.run_pass_manager()
    with capsys.disabled():
        result = mod_gen.dump_module()
    print(f"result {repr(result)}")
    # assert False
    return
