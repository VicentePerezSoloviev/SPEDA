import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# solution = np.array([0.5, 0.1, -0.3, 0.9, 0.4, -0.6, 0.2])
# dim_theta = solution.shape[0]

from benchmarks import Benchmarking
num_vars = 30
benchmarking = Benchmarking(num_vars)


# Cr = crossover rate
# F = mutation rate
# NP = n population
def differential_evolution(thetas_limit, target_vectors_init, objective_function, Cr=0.5, F=0.5, NP=10, max_gen=100, cr_type=''):
    n_params = len(thetas_limit)
    # Generate random target vectors
    # target_vectors = np.random.rand(NP, n_params)
    # target_vectors = np.interp(target_vectors, (0,1), (-1,1))
    target_vectors = target_vectors_init.copy()
    # Variable donor vectors
    donor_vector = np.zeros(n_params)
    # Variable trial vectors
    trial_vector = np.zeros(n_params)
    best_fitness = np.inf
    list_best_fitness = []
    for gen in range(max_gen):
        for pop in range(NP):
            # print("Target vectors :", target_vectors[pop])
            # Untuk novelti ini bisa ditambahkan selectionnya berdasarkan ranking
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = np.random.choice(index_choice, 3)
            donor_vector = target_vectors[a] - F * (target_vectors[b] - target_vectors[c])
            # print("Donor vectors :", donor_vector)
            n = np.random.randint(n_params)
            L = np.random.randint(1, n_params)
            n_end = n + L

            cross_points = np.random.rand(n_params) < Cr
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            # print("Trial vector", trial_vector)
            target_fitness = objective_function(target_vectors[pop])
            trial_fitness = objective_function(trial_vector)
            # print("Target fitness :", target_fitness)
            # print("Trial fitness :", trial_fitness)
            if trial_fitness < target_fitness:
                target_vectors[pop] = trial_vector.copy()
                best_fitness = trial_fitness
            else:
                best_fitness = target_fitness
        print("Generation:", str(gen), "Best fitness:", best_fitness)
        list_best_fitness.append(best_fitness)
    return list_best_fitness


def jade(thetas_limit, target_vectors_init, objective_function, uCR=0.5, uF=0.6, c=0.1, NP=10, max_gen=100):
    n_params = len(thetas_limit)
    # Generate random target vectors
    # target_vectors = np.random.rand(NP, n_params)
    target_vectors = target_vectors_init.copy()
    target_fitness = np.zeros(NP)
    # target_vectors = np.interp(target_vectors, (0,1), (-1,1))

    # Variable donor vectors
    donor_vector = np.zeros(n_params)
    # Variable trial vectors
    trial_vector = np.zeros(n_params)
    best_fitness = np.inf
    list_best_fitness = []
    Fi = np.zeros(NP)
    CRi = np.zeros(NP)
    onethirdNP = NP // 3
    for gen in range(max_gen):
        # Success F dan Success CR
        sCR = []
        sF = []
        random_onethird_idx = np.random.choice(np.arange(0, NP), size=onethirdNP, replace=False).tolist()
        # Generate adaptive parameter Fi dan CRi
        for pop in range(NP):
            CRi[pop] = np.random.normal(uCR, 0.1)
            if pop in random_onethird_idx:
                Fi[pop] = np.interp(np.random.rand(), (0, 1), (0, 1.2))
            else:
                Fi[pop] = np.random.normal(uF, 0.1)
        # print("Onethird ", random_onethird_idx)
        # print("CRi ", CRi)
        # print("Fi ", Fi)
        # Evaluate target vectors
        for pop in range(NP):
            target_fitness[pop] = objective_function(target_vectors[pop])

        for pop in range(NP):
            # print("Target vectors :", target_vectors[pop])
            current_best_idx = np.argmin(target_fitness)

            index_choice = [i for i in range(NP) if i != pop]
            a, b = np.random.choice(index_choice, 2)
            donor_vector = target_vectors[pop] + Fi[pop] * (target_vectors[current_best_idx] - target_vectors[pop]) + \
                           Fi[pop] * (target_vectors[a] - target_vectors[b])
            # print("Donor vectors :", donor_vector)

            cross_points = np.random.rand(n_params) <= CRi[pop]
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            # print("Trial vector", trial_vector)

            trial_fitness = objective_function(trial_vector)
            # print("Target fitness :", target_fitness)
            # print("Trial fitness :", trial_fitness)
            if trial_fitness < target_fitness[pop]:
                target_vectors[pop] = trial_vector.copy()
                sCR.append(CRi[pop])
                sF.append(Fi[pop])
                best_fitness = trial_fitness
            else:
                best_fitness = target_fitness[pop]

        # Update uCR dan uF
        # print("uCR : ", sCR)
        # print("uF ", sF)
        uCR = (1 - c) * uCR + c * np.mean(sCR)
        # print(sCR, uCR, sF)
        uF = (1 - c) * uF + c * (np.sum(np.power(sF, 2)) / np.sum(sF))
        # print("Generation:", str(gen), "Best fitness:", best_fitness)
        list_best_fitness.append(best_fitness)
    return list_best_fitness


limits = [(-100, 100)] * num_vars
# print(limits)
'''size_gen = 200
target_vectors = np.random.rand(size_gen, num_vars)
target_vectors = np.interp(target_vectors, (0, 1), (-1, 1))
print("Differential Evolution")
result_de = differential_evolution(limits, target_vectors, benchmarking.rastrigins_function, NP=size_gen)
print("JADE")
result_jade = jade(limits, target_vectors, benchmarking.rastrigins_function,
                   uCR=0.2, uF=0.2, c=0.2, NP=size_gen)
fig, ax = plt.subplots()
ax.plot(result_de, label="DE")
ax.plot(result_jade, label="JADE")
ax.legend()
plt.show()
print("Best DE :", result_de[-1])
print("Best JADE :", result_jade[-1])'''


dt_results = pd.DataFrame(columns=['uCR', 'uF', 'c', 'size_gen', 'it', 'cost'])
index = 0
filename = 'results_jade_300000_2.csv'

for uCR in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for uF in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for size_gen in [100, 200]:
                for it in range(15):
                    target_vectors = np.random.rand(size_gen, num_vars)
                    target_vectors = np.interp(target_vectors, (0, 1), (-1, 1))
                    result_jade = jade(limits, target_vectors, benchmarking.rastrigins_function,
                                       uCR=uCR, uF=uF, c=c, NP=size_gen, max_gen=int(150000/size_gen))

                    dt_results.loc[0] = [uCR, uF, c, size_gen, it, min(result_jade)]
                    print(uCR, uF, c, size_gen, it, min(result_jade))
                    index += 1
                    dt_results.to_csv(filename)
