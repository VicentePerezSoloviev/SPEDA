import pandas as pd
from SPEDA import SpEDA
from benchmarks import Benchmarking
import random

# random.seed(1234)

num_vars = 30
variables = [str(i) for i in range(num_vars)]
initial_vector = pd.DataFrame(columns=variables)
initial_vector['data'] = ['mu', 'std']
initial_vector = initial_vector.set_index('data')
initial_vector.loc['mu'] = [0]*num_vars
initial_vector.loc['std'] = 40  # 200/5

benchmarking = Benchmarking(num_vars)

dt = pd.DataFrame(columns=['it', 'alpha', 'result'])
index = 0
filename = 'results_eda_30_200_2.csv'

for it in range(15):
    for alpha in [0.6, 0.7, 0.8, 0.9, 1.0]:
        eda = SpEDA(alpha=alpha, max_it=3000, dead_it=500, size_gen=100,
                    cost_function=benchmarking.rastrigins_function, vector=initial_vector)
        best_cost, best_ind, history = eda.run()
        print(it, alpha, best_cost)
        dt.loc[index] = [it, alpha, best_cost]
        index += 1
        dt.to_csv(filename)


'''eda = SpEDA(alpha=0.5, max_it=1000, dead_it=150, size_gen=200,
            cost_function=benchmarking.rastrigins_function, vector=initial_vector)
best_cost, best_ind, history = eda.run()
print(best_cost)'''
