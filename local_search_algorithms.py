# coding: utf-8

import numpy as np
import random
import itertools


class Env:
    """
    把环境封装成一个类，这个类有两个方法：
        ① get_next_states：
            输入：当前状态
            返回：可能的下个状态的列表
        ② evaluate：
            输入：一个状态
            返回：该状态的优度
    """

    def __init__(self):
        pass

    def get_next_states(self, state):
        raise NotImplementedError

    def evaluate(self, state):
        raise NotImplementedError

    def render(self, state):
        pass


class ClimbingMethod:
    def __init__(self):
        pass

    @staticmethod
    def search_one_step(env, current_state):
        next_states = env.get_next_states(current_state)
        next_scores = [env.evaluate(state) for state in next_states]
        best_score = max(next_scores)
        max_idx = next_scores.index(best_score)
        return next_states[max_idx], best_score

    def run(self, init_state, env, max_step, view=False):
        current_state = init_state
        best_score = env.evaluate(current_state)
        best_state = current_state

        for i in range(max_step):
            next_state, score = self.search_one_step(env, current_state)

            if view:
                env.render(current_state)

            if score > best_score:
                best_score = score
                best_state = next_state

            current_state = next_state
        return best_state, best_score


class SimulateAnneal:
    def __init__(self, schedule=lambda t: 10 * np.exp(-0.01 * t)):
        self.schedule = schedule

    def run(self, init_state, env, max_step, view=False):
        current_state = init_state
        best_score = env.evaluate(current_state)
        best_state = current_state
        t = .0
        step = 0
        while self.schedule(t) > 1e-7 and step < max_step:
            next_states = env.get_next_states(current_state)
            next_state = random.choice(next_states)
            delta_e = env.evaluate(next_state) - env.evaluate(current_state)
            if delta_e > 0:
                current_state = next_state
                best_score = env.evaluate(current_state)
                best_state = current_state
            else:
                p = np.exp(delta_e / self.schedule(t))
                if np.random.rand() < p:
                    current_state = next_state
                    score = env.evaluate(current_state)
                    if score > best_score:
                        best_score = score
                        best_state = current_state

            if view:
                env.render(current_state)

            t += 1
            step += 1
        return best_state, best_score


class GAHelper:
    def __init__(self):
        pass

    def generate_population(self):
        raise NotImplementedError

    def mutate_fn(self, state, mutate_rate):
        raise NotImplementedError

    @staticmethod
    def crossover_fn(state1, state2):
        raise NotImplementedError


class GeneAlgorithm:

    def __init__(self, ga_helper, mutate_rate, eliminate_rate_range):
        self.ga_helper = ga_helper
        self.mutate_rate = mutate_rate
        self.eliminate_rate_range = eliminate_rate_range
        self.population = self.ga_helper.generate_population()
        self.population_size = len(self.population)
        self.scores = None

    def evaluate(self, env):
        self.scores = np.array([env.evaluate(member) for member in self.population])

    def eliminate(self, env, eliminate_rate):
        population_size = len(self.population)
        eliminate_num = int(population_size * eliminate_rate)
        eliminated_population_size = population_size - eliminate_num
        if eliminated_population_size % 2 != 0:
            eliminate_num -= 1
            eliminated_population_size += 1
        self.evaluate(env)
        self.population = np.array(self.population)[np.argsort(self.scores)].tolist()
        self.population = self.population[eliminate_num:]

    def crossover(self):
        current_population_size = len(self.population)
        crossover_num = self.population_size - current_population_size
        combinations = list(itertools.combinations(range(current_population_size), 2))
        for i in range(crossover_num):
            combination = random.choice(combinations)
            new_member = self.ga_helper.crossover_fn(self.population[combination[0]], self.population[combination[1]])
            self.population.append(new_member)

    def mutate(self):
        for member in self.population:
            self.ga_helper.mutate_fn(member, self.mutate_rate)

    def run(self, env, max_step, view=False):
        best_state = None
        best_score = -float("inf")
        self.population = self.ga_helper.generate_population()
        self.population_size = len(self.population)

        for _ in range(max_step):
            self.evaluate(env)
            max_idx = self.scores.argmax()
            if self.scores[max_idx] > best_score:
                best_score = self.scores[max_idx]
                best_state = self.population[max_idx]
            for member in self.population:
                if view:
                    env.render(member)
            self.eliminate(env, np.random.uniform(self.eliminate_rate_range[0], self.eliminate_rate_range[1]))
            self.crossover()
            self.mutate()
        return best_state, best_score
