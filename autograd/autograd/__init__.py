
'''

Below is some terminology explanation.

In this module, the term "gradient" has the following definition:
    - Let's say we're trying to minimize some variable mv.
    - mv is a strict function of some variables (let's call them vars), i.e. directly relies on those variables.
    - Each variable var (in vars) might be a strict function of other variables (let's call them indirect_vars).
    - The gradient of var refers to d_mv_over_d_var, i.e. the partial derivative of mv w.r.t var.
    - The gradient of indirect_var (i.e. some element of indirect_vars) refers to d_mv_over_d_indirect_var.
    - Thus, what gradient refers to changes depending on the context. In particular, what gradient refers to depends on the current variable we're trying to minimize.
    - The gradient of any arbitrary variable some_var refers to d_mv_over_d_indirect_some_var. 
        - NB some_var doesn't necessarily have to be directly or indirectly relied upon by mv (in which case the gradient, i.e. d_mv_over_d_indirect_some_varm is zero).

In this module, "var is indirectly relied upon mv" means that the Variable mv directly relies on the Variable instance middle_var_1, which relies on middle_var_2, ..., which relies on middle_var_n for some n>=1.

'''

# @todo fill in doc string

###########
# Imports #
###########

from .variable import Variable, VariableOperand
from .layer import LinearLayer, LogisticRegressionLayer
from . import optimizer
