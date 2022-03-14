import numpy as np
from cmaes import CMA
from benchmarks import Benchmarking
import pandas as pd


def trad_cmaes(num_vars, cost_function, max_it=50):
    optimizer = CMA(mean=np.zeros(num_vars), sigma=1.3)

    for generation in range(max_it):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = cost_function(x)
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}), x3 = {x[2]})")
        optimizer.tell(solutions)
    return solutions


def ipop_cmaes(cost_function, num_vars, max_it=50, min=-100, max=100):
    bounds = np.array([[min, max], [min, max]])
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
    sigma = max * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    for generation in range(max_it):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = cost_function(x)
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}), x3 = {x[2]})")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            # popsize multiplied by 2 (or 3) before each restart.
            popsize = optimizer.population_size * 2
            mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
            optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize)
            print(f"Restart CMA-ES with popsize={popsize}")


# ipop_cmaes(num_vars, benchmarking.rastrigins_function)
dt = pd.DataFrame(columns=['it', 'sol'])
index = 0
num_vars = 30
benchmarking = Benchmarking(num_vars)
for it in range(15):
    solutions = trad_cmaes(num_vars, benchmarking.rastrigins_function)
    dt.loc[index] = [it, min([i[1] for i in solutions])]
    index += 1

dt.to_csv('results_cmaes.csv')

'''num_vars = 30
benchmarking = Benchmarking(num_vars)
solutions = trad_cmaes(num_vars, benchmarking.rosenbrock_function)
print(min([i[1] for i in solutions]))'''

