import numpy as np
import pandas as pd
from pybnesian import SemiparametricBN, GaussianNetwork
import random


class SpEDA:

    best_cost = 999999999999999
    best_ind = {}
    history = []

    def __init__(self, alpha: float, max_it: int, dead_it: int, size_gen: int,
                 cost_function: callable, vector: pd.DataFrame):
        self.max_it = max_it
        self.dead_it = dead_it
        self.size_gen = size_gen
        self.trunc_size = int(size_gen*alpha)

        self.cost_function = cost_function
        self.vector = vector
        self.variables = list(vector.columns)

        self.generation = pd.DataFrame(columns=self.variables + ['cost'])
        self.pm = SemiparametricBN(self.variables)
        # self.pm = GaussianNetwork(self.variables)

        self.initialization()

    def initialization(self):
        # initialize each element to a normal distribution
        for col in self.generation.drop('cost', axis=1).columns:
            self.generation[col] = np.random.normal(self.vector.loc['mu', col],
                                                    self.vector.loc['std', col],
                                                    size=self.size_gen)

        '''for col in self.generation.drop('cost', axis=1).columns:
            # self.generation[col] = np.random.randint(-100, 100, self.size_gen)
            self.generation[col] = [random_float(-100, 100) for i in range(self.size_gen)]'''

    def evaluation(self):
        for i in range(len(self.generation)):
            self.generation.loc[i, 'cost'] = self.cost_function(self.generation[self.variables].loc[i].values)

    def truncation(self):
        self.generation['cost'] = self.generation['cost'].astype(float)
        self.generation = self.generation.nsmallest(self.trunc_size, 'cost').reset_index(drop=True)

    def update_pm(self):
        self.pm = SemiparametricBN(self.variables)
        self.pm.fit(self.generation.drop('cost', axis=1))

    def new_generation(self, per_elitist=0.1):
        # Elitist approach: 10% from previous generation and 90% new sampling
        bests = self.generation.head(int(self.size_gen*per_elitist))
        size_sampling = int((1-per_elitist)*self.size_gen)
        self.generation = self.pm.sample(size_sampling).to_pandas()
        # self.generation['cost'] = np.nan
        self.generation = self.generation[self.variables].append(bests[self.variables]).reset_index(drop=True)

    def run(self):
        no_improvement_it = 0
        for iteration in range(self.max_it):
            if no_improvement_it == self.dead_it:
                return self.best_cost, self.best_ind, self.history

            self.evaluation()
            self.truncation()
            self.update_pm()

            best_local_cost = float(self.generation.loc[0, 'cost'])
            if best_local_cost < self.best_cost:
                self.best_cost = best_local_cost
                self.best_ind = self.generation.loc[0].to_dict()
                no_improvement_it = 0
            else:
                no_improvement_it += 1

            self.history.append(best_local_cost)

            self.new_generation(per_elitist=0.1)
            # print('IT', str(iteration), '\tcost', self.best_cost)

        return self.best_cost, self.best_ind, self.history


def random_float(low, high):
    return random.random()*(high-low) + low
