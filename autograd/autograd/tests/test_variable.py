import pytest
import numpy as np

import autograd
from autograd import Variable
from autograd.misc_utilities import *

def test_variable_depended_on_variables():
    '''
    The computation graph is:

                  f
                 /|
                / |
               e  |
               |  |
               d  |
             / |  |
            |  |  |
            |  c  |
            | / \ |
            |/   \|
            a     b

    '''
    a = Variable(np.random.rand(8))
    b = Variable(np.random.rand(8))
    c = a.dot(b)
    d = a - c
    e = d ** 2
    f = b - e
    topologically_sorted_variables = list(f.depended_on_variables())
    assert len(topologically_sorted_variables) == len(set(topologically_sorted_variables)) == 6
    topologically_sorted_variable_ids = eager_map(id, topologically_sorted_variables)
    a_index = topologically_sorted_variable_ids.index(id(a))
    b_index = topologically_sorted_variable_ids.index(id(b))
    c_index = topologically_sorted_variable_ids.index(id(c))
    d_index = topologically_sorted_variable_ids.index(id(d))
    e_index = topologically_sorted_variable_ids.index(id(e))
    f_index = topologically_sorted_variable_ids.index(id(f))
    assert f_index < e_index
    assert f_index < b_index
    assert e_index < d_index
    assert d_index < a_index
    assert d_index < c_index
    assert c_index < a_index
    assert c_index < b_index
