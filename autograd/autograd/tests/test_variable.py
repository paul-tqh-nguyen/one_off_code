import pytest
import numpy as np

import sys ; sys.path.append('..')
import autograd
from autograd import Variable

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
    a = topologically_sorted_variables.index(a)
    b = topologically_sorted_variables.index(b)
    c = topologically_sorted_variables.index(c)
    d = topologically_sorted_variables.index(d)
    e = topologically_sorted_variables.index(e)
    f = topologically_sorted_variables.index(f)
    assert f < e
    assert f < b
    assert e < d
    assert d < a
    assert d < c
    assert c < a
    assert c < b

def test_variable_all_any():
    var = Variable(np.arange(10))
    assert not var.all()
    assert var.any()
    
    var = Variable(np.arange(10)+12)
    assert var.all()
    assert var.any()

    var = Variable(np.zeros(10))
    assert not var.all()
    assert not var.any()

def test_variable_eq():
    a = Variable(np.arange(10))
    b = Variable(np.arange(10), dtype=float)
    assert a.eq(b).all()
    assert (a == b).all()
