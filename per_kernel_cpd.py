import pandas as pd
import numpy as np
from SPEDA import SpEDA
from benchmarks import Benchmarking

num_vars = 30
variables = [str(i) for i in range(num_vars)]
initial_vector = pd.DataFrame(columns=variables)
initial_vector['data'] = ['mu', 'std']
initial_vector = initial_vector.set_index('data')
initial_vector.loc['mu'] = [50]*num_vars
initial_vector.loc['std'] = 100  # 200/5

benchmarking = Benchmarking(num_vars)
cost_f = ['cec8', benchmarking.cec14_8]

size_gen = 200
alpha = 0.4
l = 15
eda = SpEDA(alpha=alpha, max_it=int(300000/size_gen), dead_it=500, size_gen=size_gen,
            cost_function=cost_f[1], vector=initial_vector, l=l, per_l=alpha)
best_cost, best_ind, history, percentage_kernel = eda.run()
print(percentage_kernel)
