import numpy as np
import pandas as pd
from pybnesian import SemiparametricBN, GaussianNetwork, OperatorPool, GreedyHillClimbing, ValidatedLikelihood
from pybnesian import ArcOperatorSet, ChangeNodeTypeSet
import random


class SpEDA:

    best_cost = 999999999999999
    best_ind = {}
    history = []

    def __init__(self, alpha: float, max_it: int, dead_it: int, size_gen: int,
                 cost_function: callable, vector: pd.DataFrame, l, per_l):
        self.max_it = max_it
        self.dead_it = dead_it
        self.size_gen = size_gen
        self.trunc_size = int(size_gen*alpha)

        self.cost_function = cost_function
        self.vector = vector
        self.variables = list(vector.columns)
        self.l = l
        self.per_l = per_l

        self.generation = pd.DataFrame(columns=self.variables + ['cost'])
        self.set_bests = pd.DataFrame(columns=self.variables + ['cost'])
        self.pm = SemiparametricBN(self.variables)
        # self.pm = GaussianNetwork(self.variables)

        self.linear_initialization()

        self.pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])

    def initialization(self):
        # initialize each element to a normal distribution
        for col in self.generation.drop('cost', axis=1).columns:
            self.generation[col] = np.random.normal(self.vector.loc['mu', col],
                                                    self.vector.loc['std', col],
                                                    size=self.size_gen)

            self.set_bests[col] = np.random.normal(self.vector.loc['mu', col],
                                                   self.vector.loc['std', col],
                                                   size=1)

    def linear_initialization(self):
        for col in self.generation.drop('cost', axis=1).columns:
            self.generation[col] = np.random.randint(-80, 80, self.size_gen).astype(float)
            self.set_bests[col] = np.random.randint(-80, 80, 1).astype(float)
            # self.generation[col] = np.arange(-80, 80, 160/self.size_gen)

    def evaluation(self):
        self.generation['cost'] = np.nan
        for i in range(len(self.generation)):
            self.generation.loc[i, 'cost'] = self.cost_function(self.generation[self.variables].loc[i].values)

    def truncation(self):
        self.generation['cost'] = self.generation['cost'].astype(float)
        self.generation = self.generation.nsmallest(self.trunc_size, 'cost').reset_index(drop=True)

        size_bests_gen = int(self.size_gen * self.per_l)
        bests = self.generation.head(size_bests_gen*self.l)
        self.set_bests = self.set_bests[self.generation.columns].append(bests[self.generation.columns]).reset_index(drop=True)
        self.set_bests = self.set_bests.nsmallest(self.l * size_bests_gen, 'cost').reset_index(drop=True)
        print(len(self.set_bests), len(bests))

    def update_pm(self):
        self.pm = SemiparametricBN(self.variables)
        hc = GreedyHillClimbing()
        df = self.generation.drop('cost', axis=1)
        df = df[self.variables].append(self.set_bests[self.variables]).reset_index(drop=True)
        vl = ValidatedLikelihood(df, k=2)
        self.pm = hc.estimate(self.pool, vl, self.pm, verbose=False)

        # self.pm = SemiparametricBN(self.variables)
        self.pm.fit(df)

    def new_generation(self):
        # Elitist approach: % from previous generation
        self.generation = self.pm.sample(self.size_gen).to_pandas()
        while len(self.generation) < self.size_gen:
            self.generation = self.generation.append(self.pm.sample(self.size_gen).to_pandas()).reset_index(drop=True)
            # remove those which do not meet the search space limits
            self.generation = self.generation.T
            self.generation = self.generation[self.generation.columns[(self.generation.max() < 80) &
                                                                      (self.generation.min() > -80)]]
            self.generation = self.generation.T

        # self.generation['cost'] = np.nan
        # self.generation = self.generation[self.variables].append(bests[self.variables]).reset_index(drop=True)
        # self.add_noise()

    def add_noise(self, size=0.2):
        noise = pd.DataFrame(np.random.normal([0]*len(self.variables), [1000]*len(self.variables),
                                              [int(self.size_gen*size), len(self.variables)]),
                             columns=self.variables, dtype='float_')
        self.generation[self.variables].append(noise).reset_index(drop=True)

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
                # print('IT', str(iteration), '\tcost', self.best_cost)
            else:
                no_improvement_it += 1

            for i in self.variables:
                print(self.pm.cpd(str(i)))

            self.history.append(best_local_cost)

            self.new_generation()
            print('IT', str(iteration), '\tcost', self.best_cost)

        return self.best_cost, self.best_ind, self.history


def random_float(low, high):
    return random.random()*(high-low) + low
