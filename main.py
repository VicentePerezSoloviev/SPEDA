import pandas as pd
from SPEDA import SpEDA
from benchmarks import Benchmarking

num_vars = 3
variables = [str(i) for i in range(num_vars)]
initial_vector = pd.DataFrame(columns=variables)
initial_vector['data'] = ['mu', 'std']
initial_vector = initial_vector.set_index('data')
initial_vector.loc['mu'] = [0]*num_vars
initial_vector.loc['std'] = 100

benchmarking = Benchmarking(num_vars)

eda = SpEDA(alpha=0.7, max_it=1000, dead_it=150, size_gen=200,
            cost_function=benchmarking.rastrigins_function, vector=initial_vector)
best_cost, best_ind, history = eda.run()
print(best_ind, best_cost)
print(history)

