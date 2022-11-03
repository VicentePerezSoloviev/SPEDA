import numpy as np
from cmaes import CMA
from benchmarks import Benchmarking
import pandas as pd
import math
from time import process_time


def trad_cmaes(num_vars, cost_function, dead_it, max_it=50):
    optimizer = CMA(mean=np.zeros(num_vars), sigma=1.3)

    evals = 0
    print(optimizer.population_size)
    while evals <= int(300000 / optimizer.population_size):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = cost_function(x)
            evals += 1
            solutions.append((x, value))
            # print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}), x3 = {x[2]})")
        optimizer.tell(solutions)
    return solutions


def ipop_cmaes(cost_function, num_vars, max_it=50, min=-100, max=100):
    bounds = np.array([[min, max]]*num_vars)
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(num_vars) * (upper_bounds - lower_bounds))
    sigma = max * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    evaluations = 0
    print(optimizer.population_size)
    while evaluations <= int(300000 / optimizer.population_size):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = cost_function(x)
            solutions.append((x, value))
            # print(f"#{evaluations} {value} (x1={x[0]}, x2 = {x[1]}), x3 = {x[2]})")
            evaluations += 1
        optimizer.tell(solutions)

        if optimizer.should_stop():
            # popsize multiplied by 2 (or 3) before each restart.
            popsize = optimizer.population_size * 2
            mean = lower_bounds + (np.random.rand(num_vars) * (upper_bounds - lower_bounds))
            optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize)
            print(f"Restart CMA-ES with popsize={popsize}")
    return solutions


def ipop_cmaes2(cost_function, min, max, num_vars):
    seed = 0
    bounds = np.array([[min, max]]*num_vars)
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(num_vars) * (upper_bounds - lower_bounds))
    sigma = 32.768 * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    # Multiplier for increasing population size before each restart.
    inc_popsize = 2

    # print(" g    f(x1,x2)     x1      x2  ")
    # print("===  ==========  ======  ======")
    # for generation in range(200):
    evals = 0
    while evals <= num_vars*10000:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = cost_function(x)
            solutions.append((x, value))
            evals += 1
            # print(f"{generation:3d}  {value:10.5f}  {x[0]:6.2f}  {x[1]:6.2f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            seed += 1
            popsize = optimizer.population_size * inc_popsize
            mean = lower_bounds + (np.random.rand(num_vars) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                seed=seed,
                bounds=bounds,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={}".format(popsize))
    return solutions


def bipop_cemaes(cost_function, min, max, num_vars):
    bounds = np.array([[min, max]]*num_vars)
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(num_vars) * (upper_bounds - lower_bounds))
    sigma = 32.768 * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 2

    # Initial run is with "normal" population size; it is
    # the large population before first doubling, but its
    # budget accounting is the same as in case of small
    # population.
    poptype = "small"

    # for generation in range(200):
    evals = 0
    while evals <= num_vars*10000:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = cost_function(x)
            solutions.append((x, value))
            evals += 1
            # (f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = math.floor(
                    popsize0 * popsize_multiplier ** (np.random.uniform() ** 2)
                )
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)

            mean = lower_bounds + (np.random.rand(num_vars) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))
    return solutions


# ipop_cmaes(num_vars, benchmarking.rastrigins_function)
'''dt = pd.DataFrame(columns=['it', 'sol'])
index = 0
num_vars = 30
benchmarking = Benchmarking(num_vars)
for it in range(15):
    solutions = trad_cmaes(num_vars, benchmarking.rastrigins_function,
                           dead_it=1500, max_it=1500)

    dt.loc[index] = [it, min([i[1] for i in solutions])]
    print(it, min([i[1] for i in solutions]))
    index += 1

dt.to_csv('results_cmaes_300000.csv')'''

'''num_vars = 30
benchmarking = Benchmarking(num_vars)
# solutions = trad_cmaes(num_vars, benchmarking.rosenbrock_function, dead_it=1500, max_it=1500)
solutions = ipop_cmaes(benchmarking.rastrigins_function, num_vars,
                       max_it=1500, min=-100, max=100)
print(solutions)
print(min([i[1] for i in solutions]))'''

dt = pd.DataFrame(columns=['it', 'sol', 'cost_function', 'times'])
index = 0
num_vars = 30
benchmarking = Benchmarking(num_vars)
'''for cost_f in [['cec1', benchmarking.cec14_1],
               ['cec2', benchmarking.cec14_2],
               ['cec4', benchmarking.cec14_4],
               ['cec5', benchmarking.cec14_5],
               ['cec6', benchmarking.cec14_6],
               ['cec7', benchmarking.cec14_7],
               ['cec8', benchmarking.cec14_8],
               ['cec9', benchmarking.cec14_9]]:'''
for cost_f in [['cec1', benchmarking.cec14_1],
               ['cec2', benchmarking.cec14_2],
               ['cec3', benchmarking.cec14_3],
               ['cec4', benchmarking.cec14_4],
               ['cec5', benchmarking.cec14_5],
               ['cec6', benchmarking.cec14_6],
               ['cec7', benchmarking.cec14_7],
               ['cec8', benchmarking.cec14_8],
               ['cec9', benchmarking.cec14_9],
               ['cec10', benchmarking.cec14_10],
               ['cec11', benchmarking.cec14_11],
               ['cec12', benchmarking.cec14_12],
               ['cec13', benchmarking.cec14_13],
               ['cec14', benchmarking.cec14_14],
               ['cec16', benchmarking.cec14_16]]:
    for it in range(15):
        t1_start = process_time()
        solutions = trad_cmaes(num_vars, cost_f[1], dead_it=1500, max_it=1500)
        t1_stop = process_time()

        dt.loc[index] = [it, min([i[1] for i in solutions]), cost_f[0], t1_stop-t1_start]
        print(it, min([i[1] for i in solutions]), cost_f[0])
        index += 1

        dt.to_csv('times_30d_cmaes.csv')
