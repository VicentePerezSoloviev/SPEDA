import pandas as pd
from SPEDA import SpEDA
from benchmarks import Benchmarking
import random
from pybnesian import SemiparametricBN, GaussianNetwork

# random.seed(1234)

num_vars = 30
variables = [str(i) for i in range(num_vars)]
initial_vector = pd.DataFrame(columns=variables)
initial_vector['data'] = ['mu', 'std']
initial_vector = initial_vector.set_index('data')
initial_vector.loc['mu'] = [50]*num_vars
initial_vector.loc['std'] = 100  # 200/5

benchmarking = Benchmarking(num_vars)

dt = pd.DataFrame(columns=['it', 'alpha', 'result', 'size_gen', 'cost_function', 'l'])
index = 0
filename = 'results_SPeda_Funcs.csv'

for cost_f in [['cec1', benchmarking.cec14_1],
               ['cec4', benchmarking.cec14_4],
               ['cec8', benchmarking.cec14_8]]:
    for it in range(5):
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            for size_gen in [100, 200, 300]:
                for l in [5, 10, 15]:
                    eda = SpEDA(alpha=alpha, max_it=int(300000/size_gen), dead_it=500, size_gen=size_gen,
                                cost_function=cost_f[1], vector=initial_vector, l=l, per_l=alpha)
                    best_cost, best_ind, history = eda.run()
                    print(it, alpha, best_cost, size_gen, cost_f[0], l)
                    dt.loc[index] = [it, alpha, best_cost, size_gen, cost_f[0], l]
                    index += 1
                    dt.to_csv(filename)
