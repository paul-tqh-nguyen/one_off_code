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

def test_variable_not_equal():

    # 1-D Array Case
    
    var = Variable(np.array([11, 22, 33, 44, 55]))
    other_var = Variable(np.arange(5))
    # Variable + Variable
    assert var.not_equal(other_var).all()
    assert var.neq(other_var).all()
    assert var.ne(other_var).all()
    assert (var != other_var).all()
    assert np.not_equal(var, other_var).all()
    # nupmy + numpy
    assert np.not_equal(np.arange(5), np.array([11, 22, 33, 44, 55])).all()
    assert (np.arange(5) != np.array([11, 22, 33, 44, 55])).all()
    # Variable + numpy
    assert var.not_equal(np.arange(5)).all()
    assert var.neq(np.arange(5)).all()
    assert var.ne(np.arange(5)).all()
    assert (var != np.arange(5)).all()
    assert np.not_equal(var, np.arange(5, dtype=float)).all()
    # numpy + Variable
    assert np.not_equal(np.arange(5, dtype=float), var).all()
    assert (np.arange(5) != var).all()
    
    # 0-D Array Case
    
    var = Variable(np.array(21, dtype=float))
    other_var = Variable(np.array(9))
    # Variable + Variable
    assert var.not_equal(other_var)
    assert var.neq(other_var)
    assert var.ne(other_var)
    assert var != other_var
    assert np.not_equal(var, other_var)
    # nupmy + numpy
    assert np.not_equal(np.array(9), np.array(21))
    assert np.all(np.array(9) != np.array(21))
    # Variable + numpy
    assert var.not_equal(np.array(9)).all()
    assert var.neq(np.array(9)).all()
    assert var.ne(np.array(9)).all()
    assert (var != np.array(9)).all()
    assert np.not_equal(var, np.array(9, dtype=float)).all()
    # numpy + Variable
    assert np.not_equal(np.array(9, dtype=float), var).all()
    assert (np.array(9) != var).all()
    
    # Python Int Case
    
    var = Variable(37)
    other_var = Variable(84)
    # Variable + Variable
    assert var.not_equal(other_var)
    assert var.neq(other_var)
    assert var.ne(other_var)
    assert var != other_var
    assert np.not_equal(var, other_var)
    # nupmy + numpy
    assert np.not_equal(np.array(37), np.array(84))
    assert np.all(np.array(37) != np.array(84))
    # Variable + numpy
    assert var.not_equal(np.array(84)).all()
    assert var.neq(np.array(84)).all()
    assert var.ne(np.array(84)).all()
    assert (var != np.array(84)).all()
    assert np.not_equal(var, np.array(84, dtype=float)).all()
    # numpy + Variable
    assert np.not_equal(np.array(84, dtype=float), var).all()
    assert (np.array(84) != var).all()

def test_variable_greater():

    # 1-D Array Case
    
    var = Variable(np.array([11, 22, 33, 44, 55]))
    other_var = Variable(np.arange(5))
    # Variable + Variable
    assert var.greater(other_var).all()
    assert var.greater_than(other_var).all()
    assert var.gt(other_var).all()
    assert (var > other_var).all()
    assert np.greater(var, other_var).all()
    # nupmy + numpy
    assert np.greater(np.array([11, 22, 33, 44, 55]), np.arange(5)).all()
    assert (np.array([11, 22, 33, 44, 55]) > np.arange(5)).all()
    # Variable + numpy
    assert var.greater(np.arange(5)).all()
    assert var.greater_than(np.arange(5)).all()
    assert var.gt(np.arange(5)).all()
    assert (var > np.arange(5)).all()
    assert np.greater(var, np.arange(5, dtype=float)).all()
    # numpy + Variable
    assert np.greater(var, np.arange(5, dtype=float)).all()
    assert (var > np.arange(5)).all()
    
    # 0-D Array Case
    
    var = Variable(np.array(21, dtype=float))
    other_var = Variable(np.array(9))
    # Variable + Variable
    assert var.greater(other_var)
    assert var.greater_than(other_var)
    assert var.gt(other_var)
    assert var > other_var
    assert np.greater(var, other_var)
    # nupmy + numpy
    assert np.greater(np.array(21), np.array(9))
    assert np.all(np.array(21) > np.array(9))
    # Variable + numpy
    assert var.greater(np.array(9)).all()
    assert var.greater_than(np.array(9)).all()
    assert var.gt(np.array(9)).all()
    assert (var > np.array(9)).all()
    assert np.greater(var, np.array(9, dtype=float)).all()
    # numpy + Variable
    assert np.greater(var, np.array(9, dtype=float)).all()
    assert (var > np.array(9)).all()
    
    # Python Int Case
    
    var = Variable(84)
    other_var = Variable(37)
    # Variable + Variable
    assert var.greater(other_var)
    assert var.greater_than(other_var)
    assert var.gt(other_var)
    assert var > other_var
    assert np.greater(var, other_var)
    # nupmy + numpy
    assert np.greater(np.array(84) > np.array(37))
    assert np.all(np.array(84) > np.array(37))
    # Variable + numpy
    assert var.greater(np.array(84)).all()
    assert var.greater_than(np.array(84)).all()
    assert var.gt(np.array(84)).all()
    assert (var > np.array(84)).all()
    assert np.greater(var, np.array(84, dtype=float)).all()
    # numpy + Variable
    assert np.greater(np.array(84, dtype=float), var).all()
    assert (np.array(84) > var).all()

