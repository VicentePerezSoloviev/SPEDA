import numpy as np


class Benchmarking:

    def __init__(self, dim):
        self.d = dim

    def bent_cigar_function(self, x):
        result = 0
        for i in range(1, self.d):
            result += x[i]**2
        result = result*np.power(10, 6) + x[0]**2
        return result

    def discuss_function(self, x):
        result = np.power(10, 6)*x[0]**2
        for i in range(1, self.d):
            result += x[i]**2
        return result

    def rosenbrock_function(self, x):
        result = 0
        for i in range(self.d-1):
            result += (100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2)
        return result

    def ackley_function(self, x):
        sum1 = sum([x[i]**2 for i in range(self.d)])
        sum2 = sum([np.cos(x[i]*2*np.pi) for i in range(self.d)])
        return -20*np.exp(-0.2*np.sqrt(sum1/self.d)) - np.exp(sum2/self.d) + 20 + np.e

    def weierstrass_function(self, x):
        a = 0.5
        b = 3
        k_max = 20
        f = 0
        for i in range(self.d):
            aux_f1 = 0
            aux_f2 = 0
            for k in range(k_max):
                aux_f1 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))
            for k in range(k_max):
                aux_f2 += np.power(a, k)*np.cos(2*np.pi*np.power(b, k)*0.5)

            f += aux_f1 - self.d * aux_f2

        return f

    def griewank_function(self, x):
        f = 0
        for i in range(self.d):
            f += x[i]**2/4000
        aux_ = 0
        for i in range(1, self.d):
            aux_ *= np.cos(x[i-1]/np.sqrt(i)) + 1
        return f - aux_

    def rastrigins_function(self, x):
        result = 0
        for i in range(self.d):
            result += (x[i]**2 - 10*np.cos(2*np.pi*x[i]) + 10)
        return result

    # def modified_schwefel_function(self):
