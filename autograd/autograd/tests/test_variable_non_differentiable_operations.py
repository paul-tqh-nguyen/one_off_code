import pytest
import numpy as np

import sys ; sys.path.append('..')
import autograd
from autograd import Variable

def test_variable_all_any():
    for var in (
            Variable(np.arange(100)),
            Variable(np.arange(100).reshape(2, 50)),
            Variable(np.arange(100).reshape(2, 5, 10)),
    ):
        for operand in (var, var.data):
            assert not operand.all()
            assert operand.any()
            assert not np.all(operand)
            assert np.any(operand)
    
    for var in (
            Variable(np.array(1)),
            Variable(np.ones(100)),
            Variable(np.ones(100).reshape(2, 50)),
            Variable(np.ones(100).reshape(2, 5, 10)),
    ):
        for operand in (var, var.data):
            assert operand.all()
            assert operand.any()
            assert np.all(operand)
            assert np.any(operand)
    
    for var in (
            Variable(np.array(0)),
            Variable(np.zeros(100)),
            Variable(np.zeros(100).reshape(2, 50)),
            Variable(np.zeros(100).reshape(2, 5, 10)),
    ):
        for operand in (var, var.data):
            assert not operand.all()
            assert not operand.any()
            assert not np.all(operand)
            assert not np.any(operand)

    assert not Variable(0).all()
    assert not Variable(0).any()
    assert Variable(1).all()
    assert Variable(1).any()

    assert not np.all(0)
    assert not np.any(0)
    assert np.all(3)
    assert np.any(4)

def test_variable_equal():

    # 1-D Array Case
    
    var = Variable(np.arange(5))
    other_var = Variable(np.arange(5))
    # Variable + Variable
    assert var.equal(other_var).all()
    assert var.eq(other_var).all()
    assert (var == other_var).all()
    assert np.equal(var, other_var).all()
    # nupmy + numpy
    assert np.equal(np.arange(5), np.arange(5)).all()
    assert (np.arange(5) == np.arange(5)).all()
    # Variable + numpy
    assert var.equal(np.arange(5)).all()
    assert var.eq(np.arange(5)).all()
    assert (var == np.arange(5)).all()
    assert np.equal(var, np.arange(5, dtype=float)).all()
    # numpy + Variable
    assert np.equal(np.arange(5, dtype=float), var).all()
    assert (np.arange(5) == var).all()
    
    # 0-D Array Case
    
    var = Variable(np.array(9))
    other_var = Variable(np.array(9))
    # Variable + Variable
    assert var.equal(other_var)
    assert var.eq(other_var)
    assert var == other_var
    assert np.equal(var, other_var)
    # nupmy + numpy
    assert np.equal(np.array(9), np.array(9))
    assert np.all(np.array(9) == np.array(9))
    # Variable + numpy
    assert var.equal(np.array(9)).all()
    assert var.eq(np.array(9)).all()
    assert (var == np.array(9)).all()
    assert np.equal(var, np.array(9, dtype=float)).all()
    # numpy + Variable
    assert np.equal(np.array(9, dtype=float), var).all()
    assert (np.array(9) == var).all()
    
    # Python Int Case
    
    var = Variable(7)
    other_var = Variable(7)
    # Variable + Variable
    assert var.equal(other_var)
    assert var.eq(other_var)
    assert var == other_var
    assert np.equal(var, other_var)
    # nupmy + numpy
    assert np.equal(np.array(7), np.array(7))
    assert np.all(np.array(7) == np.array(7))
    # Variable + numpy
    assert var.equal(np.array(7)).all()
    assert var.eq(np.array(7)).all()
    assert (var == np.array(7)).all()
    assert np.equal(var, np.array(7, dtype=float)).all()
    # numpy + Variable
    assert np.equal(np.array(7, dtype=float), var).all()
    assert (np.array(7) == var).all()
