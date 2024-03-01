"""
 In this file there are defined many type of activation function with theirs derivatives
 The defined functions are:
    - identity
    - Sigmoid
    - hyperbolic tangent
    - ReLU
    - Leaky ReLU

 File: activation_functions.py
 Author: Pastore Luca N97000431
 Target: Università degli studi di Napoli Federico II
"""

import numpy as np

"""
    In every function implemented in this file: 
    -   the input parameter x rappresent the input matrix for each layer. 
        x matrix have c x N element, with c=number of output/target value and N=number of samples
    -   the "derive" parameter indicates whether the corresponding derivative function is desired
        False = no derivative, standard function / True 0 derivative function
"""


def identity(x, derive=False):
    if not derive:
        # Function
        y = x
    else:
        # Derivative
        y = 1

    return y


def sigmoid(x, derive=False):
    sig_x = 1 / (1 + np.exp(-x))

    if not derive:
        # Function
        y = sig_x
    else:
        # Derivative
        y = sig_x * (1 - sig_x)

    return y


def tanh(x, derive=False):
    if not derive:
        # Function
        y = np.tanh(x)
    else:
        # Derivative
        y = 1 - (np.tanh(x) ** 2)

    return y


def ReLU(x, derive=False):
    a = (x > 0)

    if not derive:
        # Function
        y = a * x
    else:
        # Derivative
        y = a * 1

    return y


def leaky_ReLU(x, derive=False):
    if not derive:
        # Function
        y = np.where(x > 0, x, x * 0.01)
    else:
        # Derivative
        y = np.where(x > 0, 1, 0.01)

    return y


"""
 End File activation_functions.py
"""
