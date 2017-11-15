# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:13 2017

@author: marti
"""
import numpy as np

def pessimistic_true_error(y,n):  # Pessimistic Estimate of the True Error
    z = 0.674  # c = 25% upper confidence limit
    n = float(n)
    e = y/n
    under_sqrt = (e/n) - ((e**2)/n) + ((z**2)/(4*n**2))
    numerator = e + ((z**2)/(2*n)) + np.sqrt(under_sqrt)
    denominator = 1 + ((z**2)/n)
    return (numerator/denominator)






          