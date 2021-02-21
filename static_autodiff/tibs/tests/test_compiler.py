
import pytest

from .test_utilities import *

from tibs import compiler

from tibs.misc_utilities import *

@subprocess_test
def test_compiler():
    # TODO update this test
    mod_gen = compiler.ModuleGenerator()
    mod_gen.c.generateModule(mod_gen.module_generator_pointer) # TODO remove this
    mod_gen.run_pass_manager()
    result = mod_gen.dump_module()
    assert False
    return
