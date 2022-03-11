import numpy as np
import pandas as pd
from pybnesian import SemiparametricBN


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
        variables = list(vector.columns)

        self.generation = pd.DataFrame(columns=variables + ['cost'])
        self.pm = SemiparametricBN(variables)

        self.initialization()

    def initialization(self):
        # initialize each element to a normal distribution
        for col in self.generation.drop('cost', axis=1).columns:
            self.generation[col] = np.random.normal(self.vector.loc['mu', col],
                                                    self.vector.loc['std', col],
                                                    size=self.size_gen)

    def evaluation(self):
        for i in range(len(self.generation)):
            self.generation.loc[i, 'cost'] = self.cost_function(self.generation.drop('cost', axis=1).loc[i].values)

    def truncation(self):
        self.generation['cost'] = self.generation['cost'].astype(float)
        self.generation = self.generation.nsmallest(self.trunc_size, 'cost').reset_index(drop=True)

    def update_pm(self):
        self.pm.fit(self.generation.drop('cost', axis=1))

    def new_generation(self, per_elitist=0.1):
        # Elitist approach: 10% from previous generation and 90% new sampling
        bests = self.generation.head(int(self.size_gen*per_elitist))
        size_sampling = int((1-per_elitist)*self.size_gen)
        self.generation = self.pm.sample(size_sampling).to_pandas()
        self.generation['cost'] = np.nan
        self.generation = self.generation.append(bests).reset_index(drop=True)

    def run(self):
        no_improvement_it = 0
        for iteration in range(self.max_it):
            if iteration == self.dead_it:
                return self.best_cost, self.best_ind, self.history

            self.evaluation()
            self.truncation()
            self.update_pm()

            best_local_cost = float(self.generation.loc[0, 'cost'])
            if best_local_cost < self.best_cost:
                self.best_cost = best_local_cost
                self.best_ind = self.generation.loc[0].to_dict()
            else:
                no_improvement_it += 1

            self.history.append(best_local_cost)

            self.new_generation()

        return self.best_cost, self.best_ind, self.history

