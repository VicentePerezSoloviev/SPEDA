import numpy as np


def rastrigins_function(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += (x[i]**2 - 10*np.cos(2*np.pi*x[i]) + 10)
    return result

