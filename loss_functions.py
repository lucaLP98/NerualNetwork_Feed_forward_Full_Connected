"""
 In this file there are defined many type of loss function with theirs derivatives
 The defined functions are:
    - soft-max
    - Cross Entropy
    - Cross Entropy with soft-max
    - Sum of Square

 File: loss_functions.py
 Author: Pastore Luca N97000431
 Target: Università degli studi di Napoli Federico II
"""

import numpy as np


"""
    Function soft-max
    arg: 
        y -> matrix c x N 
"""
def softmax(y):
    y_exp = np.exp(y - np.max(y))
    z = y_exp / sum(y_exp, 0)

    return z


# ***** Functions for classification problem *****


"""
    Error Function Cross Entropy with its derivative
    
    arg:
        y -> matrix c x N which represents the output of the network
        t -> matrix c x N which represents the target value 
        derive -> indicates if the corresponding derivative function is desired 
                (False = no derivative, standard function / True 0 derivative function)

        OSS: c=number of output/target value - N=number of samples
"""
def cross_entropy(y, t, derive=False):
    if not derive:
        # Function
        error_value = -np.sum(t * np.log(y))
    else:
        # Derivative
        error_value = -t / y

    return error_value


"""
    Error Function Cross Entropy with soft-max apply to output of the network (y) with its derivative

    arg:
        y -> matrix c x N which represents the output of the network
        t -> matrix c x N which represents the target value
        derive -> indicates if the corresponding derivative function is desired 
                (False = no derivative, standard function / True 0 derivative function)
                  
        OSS: c=number of output/target value - N=number of samples
"""
def cross_entropy_sm(y, t, derive=False):
    epsilon = 1e-10

    if not derive:
        # Function
        # You need to apply softmax to the output y before calling the function
        error_value = -np.sum(t * np.log(y + epsilon))
    else:
        # Derivative
        y = softmax(y)
        error_value = y - t

    return error_value


# ***** Functions for regression problem *****


"""
    Error Function Sum of Square

    arg:
        y -> matrix c x N which represents the output of the network
        t -> matrix c x N which represents the target value 
        derive -> indicates if the corresponding derivative function is desired 
                (False = no derivative, standard function / True 0 derivative function)
                  
        OSS: c=number of output/target value - N=number of samples
"""
def sum_of_square(y, t, derive=False):
    if not derive:
        # Function
        error_value = (1/2)*np.sum(np.square(y - t))
    else:
        # Derivative
        error_value = y - t

    return error_value


"""
 End File loss_functions.py
"""