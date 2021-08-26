import matplotlib.pyplot as plt

from local_search_algorithms import *


class ClimbingEnv(Env):
    def __init__(self, delta, x_lim, y_lim, fn):
        super().__init__()
        self.delta = delta
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.fn = fn
        self.ax = None

    def get_next_states(self, state):
        res = [
            state + np.array([0, self.delta]),
            state - np.array([0, self.delta]),
            state + np.array([self.delta, 0]),
            state - np.array([self.delta, 0])
        ]
        return res

    def evaluate(self, state):
        if state[0] < self.x_lim[0] or state[0] > self.x_lim[1] or state[1] < self.y_lim[0] or state[1] > self.y_lim[1]:
            return 0
        else:
            return self.fn(state[0], state[1])

    def render(self, state):
        if not self.ax:
            fig = plt.figure()
            plt.ion()
            self.ax = fig.gca(projection='3d')
            x = np.arange(self.x_lim[0], self.x_lim[1], 1)
            y = np.arange(self.y_lim[0], self.y_lim[1], 1)
            x, y = np.meshgrid(x, y)
            z = self.fn(x, y)
            self.ax.plot_wireframe(x, y, z)
            plt.xlim([-5, 5])
            plt.ylim([-5, 20])
        self.ax.scatter(state[0], state[1], self.evaluate(state), color='red', s=10)
        plt.pause(1e-7)


class ClimbingGAHelper(GAHelper):
    def __init__(self, x_lim, y_lim, population_size):
        super().__init__()
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.population_size = population_size

    def generate_population(self):
        return (np.random.rand(self.population_size, 2) *
                np.array([[self.x_lim[1] - self.x_lim[0], self.y_lim[1] - self.y_lim[0]]]) +
                np.array([[self.x_lim[0], self.y_lim[0]]])).tolist()

    def mutate_fn(self, state, mutate_rate):
        if np.random.rand() < mutate_rate:
            state[0] = np.random.rand() * (self.x_lim[1] - self.x_lim[0]) + self.x_lim[0]
        if np.random.rand() < mutate_rate:
            state[1] = np.random.rand() * (self.y_lim[1] - self.y_lim[0]) + self.y_lim[0]

    @staticmethod
    def crossover_fn(state1, state2):
        if np.random.rand() > 0.5:
            new_state = np.array([state1[0], state2[1]])
        else:
            new_state = np.array([state2[0], state1[1]])
        return new_state


if __name__ == '__main__':
    env = ClimbingEnv(
        delta=0.2,
        x_lim=[-5, 5],
        y_lim=[-5, 20],
        fn=lambda x, y: (np.sin(0.5 * y) + 0.8) ** 2 +
                        (np.cos(0.5 * x) + 3) ** 2 +
                        (y + 5) ** 2 * 0.01,
    )

    print("爬山法")
    init_state = np.array([3, 5])
    optimizer = ClimbingMethod()
    best_state, best_score = optimizer.run(env=env, init_state=init_state, max_step=200, view=False)
    print("best_state:", best_state)
    print("best_score:", best_score)

    print("模拟退火")
    init_state = np.array([3, 5])
    optimizer = SimulateAnneal(lambda t: 10 * np.exp(-0.01 * t))
    best_state, best_score = optimizer.run(env=env, init_state=init_state, max_step=200, view=False)
    print("best_state:", best_state)
    print("best_score:", best_score)

    print("遗传算法")
    ga_helper = ClimbingGAHelper(x_lim=[-5, 5], y_lim=[-5, 20], population_size=5)
    optimizer = GeneAlgorithm(eliminate_rate_range=[0.2, 0.5], mutate_rate=0.01, ga_helper=ga_helper)
    best_state, best_score = optimizer.run(env=env, max_step=1000, view=False)
    print("best_state:", best_state)
    print("best_score:", best_score)
