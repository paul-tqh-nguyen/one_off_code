import pytest
import numpy as np

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

def test_variable_isclose():

    # 1-D Array Case

    value = np.array([0, 1e-16])
    other_value = np.zeros(2)
    var = Variable(np.array([0, 1e-16]))
    other_var = Variable(np.zeros(2))
    # Variable + Variable
    assert var.isclose(other_var).all()
    assert np.isclose(var, other_var).all()
    # numpy + numpy
    assert np.isclose(value, other_value).all()
    # Variable + numpy
    assert var.isclose(other_value).all()
    assert np.isclose(var, other_value).all()
    # numpy + Variable
    assert np.isclose(other_value, var).all()
    
    # 0-D Array Case

    value = np.array(1e-10)
    other_value = np.array(1e-16)
    var = Variable(1e-10)
    other_var = Variable(1e-16)
    # Variable + Variable
    assert var.isclose(other_var)
    assert np.isclose(var, other_var)
    # numpy + numpy
    assert np.isclose(value, other_value)
    # Variable + numpy
    assert var.isclose(other_value).all()
    assert np.isclose(var, other_value).all()
    # numpy + Variable
    assert np.isclose(other_value, var).all()
    
    # Python Float Case

    value = 1e-10
    other_value = 1e-16
    var = Variable(1e-10)
    other_var = Variable(1e-16)
    # Variable + Variable
    assert var.isclose(other_var)
    assert np.isclose(var, other_var)
    # numpy + Python
    assert np.isclose(value, other_value)
    # Variable + Python
    assert var.isclose(other_value).all()
    assert np.isclose(var, other_value).all()
    # Python + Variable
    assert np.isclose(other_value, var).all()

def test_variable_equal():

    # 1-D Array Case
    
    var = Variable(np.arange(5))
    other_var = Variable(np.arange(5))
    # Variable + Variable
    assert var.equal(other_var).all()
    assert var.eq(other_var).all()
    assert (var == other_var).all()
    assert np.equal(var, other_var).all()
    # numpy + numpy
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
    # numpy + numpy
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
    # Python + Python
    assert np.equal(np.array(7), np.array(7))
    assert np.all(np.array(7) == np.array(7))
    # Variable + Python
    assert var.equal(np.array(7)).all()
    assert var.eq(np.array(7)).all()
    assert (var == np.array(7)).all()
    assert np.equal(var, np.array(7, dtype=float)).all()
    # Python + Variable
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
    # numpy + numpy
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
    # numpy + numpy
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
    # Python + Python
    assert np.not_equal(np.array(37), np.array(84))
    assert np.all(np.array(37) != np.array(84))
    # Variable + Python
    assert var.not_equal(np.array(84)).all()
    assert var.neq(np.array(84)).all()
    assert var.ne(np.array(84)).all()
    assert (var != np.array(84)).all()
    assert np.not_equal(var, np.array(84, dtype=float)).all()
    # Python + Variable
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
    # numpy + numpy
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
    # numpy + numpy
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
    # Python + Python
    assert np.greater(np.array(84), np.array(37))
    assert np.all(np.array(84) > np.array(37))
    # Variable + Python
    assert var.greater(np.array(37)).all()
    assert var.greater_than(np.array(37)).all()
    assert var.gt(np.array(37)).all()
    assert (var > np.array(37)).all()
    assert np.greater(var, np.array(37, dtype=float)).all()
    # Python + Variable
    assert np.greater(var, np.array(37, dtype=float)).all()
    assert (var > np.array(37)).all()

def test_variable_less():

    # 1-D Array Case
    
    var = Variable(np.arange(5))
    other_var = Variable(np.array([11, 22, 33, 44, 55]))
    # Variable + Variable
    assert var.less(other_var).all()
    assert var.less_than(other_var).all()
    assert var.lt(other_var).all()
    assert (var < other_var).all()
    assert np.less(var, other_var).all()
    # numpy + numpy
    assert np.less(np.arange(5), np.array([11, 22, 33, 44, 55])).all()
    assert (np.arange(5) < np.array([11, 22, 33, 44, 55])).all()
    # Variable + numpy
    assert var.less(np.array([11, 22, 33, 44, 55])).all()
    assert var.less_than(np.array([11, 22, 33, 44, 55])).all()
    assert var.lt(np.array([11, 22, 33, 44, 55])).all()
    assert (var < np.array([11, 22, 33, 44, 55])).all()
    assert np.less(var, np.array([11, 22, 33, 44, 55])).all()
    # numpy + Variable
    assert np.less(var, np.array([11, 22, 33, 44, 55])).all()
    assert (var < np.array([11, 22, 33, 44, 55])).all()
    
    # 0-D Array Case
    
    var = Variable(np.array(9))
    other_var = Variable(np.array(21, dtype=float))
    # Variable + Variable
    assert var.less(other_var)
    assert var.less_than(other_var)
    assert var.lt(other_var)
    assert var < other_var
    assert np.less(var, other_var)
    # numpy + numpy
    assert np.less(np.array(9), np.array(21))
    assert np.all(np.array(9) < np.array(21))
    # Variable + numpy
    assert var.less(np.array(21)).all()
    assert var.less_than(np.array(21)).all()
    assert var.lt(np.array(21)).all()
    assert (var < np.array(21)).all()
    assert np.less(var, np.array(21, dtype=float)).all()
    # numpy + Variable
    assert np.less(var, np.array(21, dtype=float)).all()
    assert (var < np.array(21)).all()
    
    # Python Int Case
    
    var = Variable(37)
    other_var = Variable(84)
    # Variable + Variable
    assert var.less(other_var)
    assert var.less_than(other_var)
    assert var.lt(other_var)
    assert var < other_var
    assert np.less(var, other_var)
    # Python + Python
    assert np.less(np.array(37), np.array(84))
    assert np.all(np.array(37) < np.array(84))
    # Variable + Python
    assert var.less(np.array(84)).all()
    assert var.less_than(np.array(84)).all()
    assert var.lt(np.array(84)).all()
    assert (var < np.array(84)).all()
    assert np.less(var, np.array(84, dtype=float)).all()
    # Python + Variable
    assert np.less(var, np.array(84, dtype=float)).all()
    assert (var < np.array(84)).all()

def test_variable_greater_than_equal_or_equal_to():

    # 1-D Array Case
    
    var = Variable(np.array([11, 22, 33, 44, 55]))
    for other_value in (
            np.arange(5),
            np.array([11, 22, 33, 44, 55]),
            np.array([ 0, 22,  0, 44,  0]),
    ):
        other_var = Variable(other_value)
        # Variable + Variable
        assert var.greater_equal(other_var).all()
        assert var.greater_than_equal(other_var).all()
        assert var.ge(other_var).all()
        assert var.gte(other_var).all()
        assert (var >= other_var).all()
        assert np.greater_equal(var, other_var).all()
        # numpy + numpy
        assert np.greater_equal(np.array([11, 22, 33, 44, 55]), other_value.copy()).all()
        assert (np.array([11, 22, 33, 44, 55]) >= other_value.copy()).all()
        # Variable + numpy
        assert var.greater_equal(other_value.copy()).all()
        assert var.greater_than_equal(other_value.copy()).all()
        assert var.ge(other_value.copy()).all()
        assert var.gte(other_value.copy()).all()
        assert (var >= other_value.copy()).all()
        assert np.greater_equal(var, other_value.copy()).all()
        # numpy + Variable
        assert np.greater_equal(var, other_value.copy()).all()
        assert (var >= other_value.copy()).all()
        
    # 0-D Array Case
    
    var = Variable(np.array(21, dtype=float))
    for other_value in (9, 21):
        other_var = Variable(np.array(other_value))
        # Variable + Variable
        assert var.greater_equal(other_var)
        assert var.greater_than_equal(other_var)
        assert var.ge(other_var)
        assert var.gte(other_var)
        assert var >= other_var
        assert np.greater_equal(var, other_var)
        # numpy + numpy
        assert np.greater_equal(np.array(21), np.array(other_value))
        assert np.all(np.array(21) >= np.array(other_value))
        # Variable + numpy
        assert var.greater_equal(np.array(other_value)).all()
        assert var.greater_than_equal(np.array(other_value)).all()
        assert var.ge(np.array(other_value)).all()
        assert var.gte(np.array(other_value)).all()
        assert (var >= np.array(other_value)).all()
        assert np.greater_equal(var, np.array(other_value, dtype=float)).all()
        # numpy + Variable
        assert np.greater_equal(var, np.array(other_value, dtype=float)).all()
        assert (var >= np.array(other_value)).all()
        
    # Python Int Case
    
    var = Variable(84)
    for other_value in (37, 84):
        other_var = Variable(other_value)
        # Variable + Variable
        assert var.greater_equal(other_var)
        assert var.greater_than_equal(other_var)
        assert var.ge(other_var)
        assert var.gte(other_var)
        assert var >= other_var
        assert np.greater_equal(var, other_var)
        # Python + Python
        assert np.greater_equal(np.array(84), np.array(other_value))
        assert np.all(np.array(84) >= np.array(other_value))
        # Variable + Python
        assert var.greater_equal(np.array(other_value)).all()
        assert var.greater_than_equal(np.array(other_value)).all()
        assert var.ge(np.array(other_value)).all()
        assert var.gte(np.array(other_value)).all()
        assert (var >= np.array(other_value)).all()
        assert np.greater_equal(var, np.array(other_value, dtype=float)).all()
        # Python + Variable
        assert np.greater_equal(var, np.array(other_value, dtype=float)).all()
        assert (var >= np.array(other_value)).all()

def test_variable_less_than_equal_or_equal_to():

    # 1-D Array Case
    
    other_var = Variable(np.array([11, 22, 33, 44, 55]))
    for value in (
            np.arange(5),
            np.array([11, 22, 33, 44, 55]),
            np.array([ 0, 22,  0, 44,  0]),
    ):
        var = Variable(value)
        # Variable + Variable
        assert var.less_equal(other_var).all()
        assert var.less_than_equal(other_var).all()
        assert var.le(other_var).all()
        assert var.lte(other_var).all()
        assert (var <= other_var).all()
        assert np.less_equal(var, other_var).all()
        # numpy + numpy
        assert np.less_equal(value.copy(), np.array([11, 22, 33, 44, 55])).all()
        assert (value.copy() <= np.array([11, 22, 33, 44, 55])).all()
        # Variable + numpy
        assert var.less_equal(np.array([11, 22, 33, 44, 55])).all()
        assert var.less_than_equal(np.array([11, 22, 33, 44, 55])).all()
        assert var.le(np.array([11, 22, 33, 44, 55])).all()
        assert var.lte(np.array([11, 22, 33, 44, 55])).all()
        assert (var <= np.array([11, 22, 33, 44, 55])).all()
        assert np.less_equal(var, np.array([11, 22, 33, 44, 55])).all()
        # numpy + Variable
        assert np.less_equal(var, np.array([11, 22, 33, 44, 55])).all()
        assert (var <= np.array([11, 22, 33, 44, 55])).all()
        
    # 0-D Array Case
    
    other_var = Variable(np.array(21, dtype=float))
    for value in (9, 21):
        var = Variable(np.array(value))
        # Variable + Variable
        assert var.less_equal(other_var)
        assert var.less_than_equal(other_var)
        assert var.le(other_var)
        assert var.lte(other_var)
        assert var <= other_var
        assert np.less_equal(var, other_var)
        # numpy + numpy
        assert np.less_equal(np.array(value), np.array(21))
        assert np.all(np.array(value) <= np.array(21))
        # Variable + numpy
        assert var.less_equal(np.array(21)).all()
        assert var.less_than_equal(np.array(21)).all()
        assert var.le(np.array(21)).all()
        assert var.lte(np.array(21)).all()
        assert (var <= np.array(21)).all()
        assert np.less_equal(var, np.array(21, dtype=float)).all()
        # numpy + Variable
        assert np.less_equal(var, np.array(value, dtype=float)).all()
        assert (var <= np.array(value)).all()
        
    # Python Int Case
    
    other_var = Variable(84)
    for value in (37, 84):
        var = Variable(value)
        # Variable + Variable
        assert var.less_equal(other_var)
        assert var.less_than_equal(other_var)
        assert var.le(other_var)
        assert var.lte(other_var)
        assert var <= other_var
        assert np.less_equal(var, other_var)
        # Python + Python
        assert np.less_equal(np.array(value), np.array(84))
        assert np.all(np.array(value) <= np.array(84))
        # Variable + Python
        assert var.less_equal(np.array(84)).all()
        assert var.less_than_equal(np.array(84)).all()
        assert var.le(np.array(84)).all()
        assert var.lte(np.array(84)).all()
        assert (var <= np.array(84)).all()
        assert np.less_equal(var, np.array(84, dtype=float)).all()
        # Python + Variable
        assert np.less_equal(var, np.array(84, dtype=float)).all()
        assert (var <= np.array(84)).all()
