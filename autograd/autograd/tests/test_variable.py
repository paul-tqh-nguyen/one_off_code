import pytest
import numpy as np
from uuid import uuid4

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

def test_():
    unique_operation_name = f'dummy_func_{uuid4().int}'
    assert not hasattr(np, unique_operation_name)
    def dummy_func(array: np.ndarray) -> np.ndarray:
        return array*10
    setattr(np, unique_operation_name, 
    
