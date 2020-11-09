import pytest
import numpy as np

import sys ; sys.path.append('..')
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
    a = topologically_sorted_variable_ids.index(a)
    b = topologically_sorted_variable_ids.index(b)
    c = topologically_sorted_variable_ids.index(c)
    d = topologically_sorted_variable_ids.index(d)
    e = topologically_sorted_variable_ids.index(e)
    f = topologically_sorted_variable_ids.index(f)
    assert f < e
    assert f < b
    assert e < d
    assert d < a
    assert d < c
    assert c < a
    assert c < b
