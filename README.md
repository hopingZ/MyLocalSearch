# MyLocalSearch
这是人工智能课程的 Local Search 这一讲的课程实践，基于 numpy 实现了几个 Local Search 算法，如：
- 爬山法
- 模拟退火
- 简单遗传算法

# Example
## 准备
```
import matplotlib.pyplot as plt  # 如果需要画图
from local_search_algorithms import *

# 首先将要求解的问题写成一个 Env 的子类，要实现以下函数
# get_next_states(self, state) -> states，输入一组参数，输出若干组发生了微小变化的参数
# evaluate(self, state) -> 数值，输入一组参数，评估这组参数的优度
# evaluate(self, state) 可选，不实现就不画图
# 比如在一定范围内，找函数 f(x, y) 最大值的问题，可以这么写

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
        
        
if __name__ == '__main__':
    env = ClimbingEnv(
        delta=0.2,
        x_lim=[-5, 5],
        y_lim=[-5, 20],
        fn=lambda x, y: (np.sin(0.5 * y) + 0.8) ** 2 +
                        (np.cos(0.5 * x) + 3) ** 2 +
                        (y + 5) ** 2 * 0.01,
    )
```

## 爬山法
```
optimizer = ClimbingMethod()
best_state, best_score = optimizer.run(
    env=env,                # 上面那个 env
    init_state=init_state,  # 初始状态
    max_step=200,           # 最多跑几步
    view=False              # 要不要可视化
)
```

## 模拟退火
```
optimizer = SimulateAnneal(lambda t: 10 * np.exp(-0.01 * t))
# 调用跟爬山法一样
```

## 简单遗传算法
```
# 要先根据求解的问题写一个相应的 GAHelper 子类，要实现以下函数
# generate_population(self) -> states，生成若干组参数作为种群
# mutate_fn(self, state)，输入一组参数，以一定概率随机修改其中的某些参数
# crossover_fn(state1, state2) -> state，输入两组参数，生成兼具两组参数特性的一组新参数，比如某些参数跟 state1 一样，某些参数跟 state2 一样

class ClimbingGAHelper(GAHelper):
    def __init__(self, x_lim, y_lim, population_size, mutate_rate):
        super().__init__()
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.population_size = population_size
        self.mutate_rate = mutate_rate

    def generate_population(self):
        return (np.random.rand(self.population_size, 2) *
                np.array([[self.x_lim[1] - self.x_lim[0], self.y_lim[1] - self.y_lim[0]]]) +
                np.array([[self.x_lim[0], self.y_lim[0]]])).tolist()

    def mutate_fn(self, state):
        if np.random.rand() < self.mutate_rate:
            state[0] = np.random.rand() * (self.x_lim[1] - self.x_lim[0]) + self.x_lim[0]
        if np.random.rand() < self.mutate_rate:
            state[1] = np.random.rand() * (self.y_lim[1] - self.y_lim[0]) + self.y_lim[0]

    @staticmethod
    def crossover_fn(state1, state2):
        if np.random.rand() > 0.5:
            new_state = np.array([state1[0], state2[1]])
        else:
            new_state = np.array([state2[0], state1[1]])
        return new_state
        
ga_helper = ClimbingGAHelper(x_lim=[-5, 5], y_lim=[-5, 20], population_size=5, mutate_rate=0.01)
optimizer = GeneAlgorithm(
    eliminate_rate_range=[0.2, 0.5],  # 每轮种群死亡比例的范围
    ga_helper=ga_helper
)
best_state, best_score = optimizer.run(env=env, max_step=1000, view=False)
```
