# 导入numpy库，用于进行科学计算
import numpy as np


# 环境模拟类定义
class Env():
    def __init__(self, name):
        self.Name = name  # 环境的名称
        self.N = 11  # 状态总数
        self.A = np.arange(4)  # 动作集，这里有4个动作
        self.X = np.arange(self.N)  # 状态集
        self.P = None  # 转移概率矩阵
        self.R = None  # 奖励向量
        self.makeP()  # 初始化转移矩阵
        self.makeR()  # 初始化奖励向量
        self.Gamma = 1  # 折扣因子
        self.StartState = 0  # 初始状态
        self.EndStates = [6, 10]  # 终止状态列表
        # 状态到网格位置的映射，用于将状态表示为网格中的行和列
        self.X2RowCol = {
            # 键为状态，值为对应的行列位置
            0: (1, 1), 1: (2, 1), 2: (3, 1), 3: (4, 1),
            4: (1, 2), 5: (3, 2), 6: (4, 2),
            7: (1, 3), 8: (2, 3), 9: (3, 3), 10: (4, 3)
        }

    def action(self, x, a):
        # 根据当前状态x和动作a，随机选择下一个状态
        x_ = np.random.choice(self.N, p=self.P[x, a, :])
        return x_  # 返回新状态

    def makeP(self):
        # 初始化状态转移概率矩阵，三维数组：状态x动作x状态
        self.P = np.zeros((self.N, len(self.A), self.N))

        # 网格布局说明（4x3），含有障碍物的布局
        # 网格索引排列：
        # 7  8  9 10
        # 4  #  5  6
        # 0  1  2  3

        # 定义状态转移字典
        transitions = {
            # 各状态下各动作的可能结果
            0: {'U': 4, 'R': 1, 'D': 0, 'L': 0},
            1: {'U': 1, 'R': 2, 'D': 1, 'L': 0},
            2: {'U': 5, 'R': 3, 'D': 2, 'L': 1},
            3: {'U': 6, 'R': 3, 'D': 3, 'L': 2},
            4: {'U': 7, 'R': 4, 'D': 0, 'L': 4},
            5: {'U': 9, 'R': 6, 'D': 2, 'L': 5},
            6: {'U': 10, 'R': 6, 'D': 3, 'L': 5},
            7: {'U': 7, 'R': 8, 'D': 4, 'L': 7},
            8: {'U': 8, 'R': 9, 'D': 8, 'L': 7},
            9: {'U': 9, 'R': 10, 'D': 5, 'L': 8},
            10: {'U': 10, 'R': 10, 'D': 6, 'L': 9},
        }

        # 使用转移字典来填充转移概率矩阵
        for s in transitions:
            for a, direction in enumerate(['U', 'R', 'D', 'L']):
                next_state = transitions[s][direction]
                # 主要方向上的转移概率设为0.8
                if next_state != s:
                    self.P[s, a, next_state] += 0.8
                else:
                    self.P[s, a, s] += 0.8

                # 垂直方向的转移概率设为0.1
                perpendicular_directions = {
                    'U': ['L', 'R'],
                    'R': ['U', 'D'],
                    'D': ['L', 'R'],
                    'L': ['U', 'D']
                }
                for pd in perpendicular_directions[direction]:
                    perpendicular_next_state = transitions[s][pd]
                    if perpendicular_next_state != s:
                        self.P[s, a, perpendicular_next_state] += 0.1
                    else:
                        self.P[s, a, s] += 0.1

        # 归一化概率，确保每个动作的总概率为1
        for s in range(self.N):
            for a in range(len(self.A)):
                if np.sum(self.P[s, a, :]) > 0:
                    self.P[s, a, :] /= np.sum(self.P[s, a, :])

    def makeR(self):
        # 初始化奖励向量，默认为-0.04（非终止状态的奖励）
        self.R = np.full(self.N, -0.04)
        # 设定特定状态的奖励，如终止状态
        self.R[6] = -1.0  # 某个终止状态的奖励
        self.R[10] = 1.0  # 另一个终止状态的奖励


# 被动学习-时序差分方法(TD Learning)
# 通过TD Learning学习环境的状态值函数。状态值函数代表每个状态的预期价值，
# 即从该状态出发，按照某策略执行动作所能获得的累积奖励的期望值。
class TD():
    def __init__(self, E):
        self.E = E  # 环境对象，用于交互和获取状态信息
        self.Alpha = 0.5  # 学习率，用于调节状态值更新的步长
        self.Pi = [3, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0]  # 策略，指定每个状态下选择的动作
        self.U = [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]  # 状态值函数，记录每个状态的价值

    def train(self):
        # 训练函数，用于更新状态值函数
        x = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 随机选择一个初始状态
        while x not in self.E.EndStates:  # 如果当前状态不是终止状态，则继续循环
            a = self.Pi[x]  # 根据策略选择动作
            _x = self.E.action(x, a)  # 在环境中执行动作，获取新状态
            r = self.E.R[x]  # 获取当前状态的即时奖励
            # 更新状态值函数，使用时序差分方法
            self.U[x] = self.U[x] + self.Alpha * \
                (r + self.E.Gamma * self.U[_x] - self.U[x])
            x = _x  # 更新当前状态为新状态

    def generate_episode(self):
        # 生成一个序列（episode），用于评估策略的性能
        x = self.E.StartState  # 从环境的开始状态出发
        episode = []  # 初始化序列列表

        while x not in self.E.EndStates:
            # 对当前状态的每个动作评估可能的下一个状态
            action_values = []
            for a in range(len(self.E.A)):
                next_state = np.argmax(self.E.P[x, a, :])  # 找到最可能的下一个状态
                action_values.append(self.U[next_state])  # 记录该状态的价值

            best_action = np.argmax(action_values)  # 选择具有最高期望价值的动作
            _x = self.E.action(x, best_action)  # 在环境中执行动作，获取新状态
            r = self.E.R[x]  # 获取当前状态的即时奖励
            episode.append((x, best_action, r))  # 将状态、动作和奖励添加到序列中
            x = _x  # 更新当前状态

            if x in self.E.EndStates:
                episode.append((x,))  # 如果达到终止状态，将其添加到序列中

        return episode  # 返回生成的序列


# 计算softmax函数值 解决normal
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# 主动学习-Q学习
# Q学习是一种用于学习状态-动作对（即Q值）的预期回报的方法。
# 这个Q值代表从当前状态开始，执行特定动作，然后按照策略执行动作所能获得的累积奖励的期望值。
class Q_Learning():
    def __init__(self, E):
        self.E = E  # 环境对象
        self.Alpha = 0.5  # 学习率，用于调整Q值更新的步长
        # 初始化Q值矩阵，所有状态-动作对的初始值为1/4
        self.Q = np.ones((11, 4))/4
        # 为特定的终止状态设置固定Q值
        self.Q[10, :] = 1  # 某个终止状态的Q值设为正值
        self.Q[6, :] = -1  # 另一个终止状态的Q值设为负值

    def train(self):
        # 训练函数，用于更新Q值矩阵
        x = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 随机选择一个初始状态
        while x not in self.E.EndStates:  # 如果当前状态不是终止状态，则继续循环
            P = softmax(self.Q[x])  # 使用softmax函数计算当前状态下每个动作的选择概率
            a = np.random.choice(4, p=P)  # 根据概率选择动作
            _x = self.E.action(x, a)  # 执行动作，获取新状态
            r = self.E.R[x]  # 获取当前状态的即时奖励
            # 更新Q值，使用Q学习的更新公式
            self.Q[x, a] = self.Q[x, a] + self.Alpha * \
                (r + self.E.Gamma * np.max(self.Q[_x]) - self.Q[x, a])
            x = _x  # 更新当前状态为新状态


# 价值函数的线性逼近
# 此类使用线性逼近方法来逼近价值函数。通过使用特征向量和权重向量来近似状态的价值。
class F_TD():
    def __init__(self, E):
        self.w = np.array([0.5, 0.5, 0.5])  # 初始化权重向量
        self.E = E  # 环境对象
        self.Alpha = 0.001  # 学习率，用于调整权重向量的更新步长
        self.Pi = [3, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0]  # 策略，指定每个状态下选择的动作

    def U(self, x):
        # 定义状态价值函数U，用于计算状态x的价值
        if x == 10:
            return 1  # 如果是特定终止状态，则价值为1
        if x == 6:
            return -1  # 如果是另一特定终止状态，则价值为-1
        (row, col) = self.E.X2RowCol[x]  # 将状态转换为网格的行列坐标
        return np.dot(np.array([1, row, col]), self.w)  # 使用线性组合计算状态价值

    def dU(self, x):
        # 计算状态x的特征向量，用于更新权重
        (row, col) = self.E.X2RowCol[x]  # 将状态转换为网格的行列坐标
        return np.array([1, row, col])  # 返回特征向量

    def train(self):
        # 训练函数，用于更新权重向量
        x0 = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 随机选择一个初始状态
        a0 = self.Pi[x0]  # 根据策略选择初始动作

        Rsum = self.E.R[x0]  # 初始化累计奖励
        x = x0
        a = a0
        gamma = self.E.Gamma  # 初始化折扣因子
        while x not in self.E.EndStates:  # 如果当前状态不是终止状态，则继续循环
            x = self.E.action(x, a)  # 执行动作，获取新状态
            Rsum += gamma * self.E.R[x]  # 更新累计奖励
            a = self.Pi[x]  # 根据策略选择下一个动作
            gamma *= self.E.Gamma  # 更新折扣因子

        # 更新权重向量，根据累积奖励和当前估计的状态价值的差异进行调整
        self.w = self.w + self.Alpha * (Rsum - self.U(x0)) * self.dU(x0)


if __name__ == "__main__":
    # 创建环境对象
    E = Env("YYW")

    """
    在强化学习中，设置迭代次数是为了确定模型训练或策略评估过程中的迭代步骤数量。
    10次迭代：这通常用于快速测试或原型设计阶段。在这个数量级，我们不期望模型有显著的学习或性能改进，但可以用来检查代码的基本功能是否正常。
    100次迭代：用于较小的问题或初步的学习测试。对于一些简单的问题或初步的模型验证，这个数量级的迭代可能足以看到一些基本的学习效果。
    1000次迭代：这通常是更深入的测试和学习阶段。对于中等复杂度的问题，这个数量级可能足以让模型学习到有意义的策略或行为。
    10000次迭代：用于复杂问题或期望模型达到较高性能的场景。在这个数量级，模型有更多的时间来学习和调整其策略，以便在复杂的环境中实现较好的性能。
    100000次迭代及以上：这个数量级通常用于非常复杂的问题，或者在需要模型达到非常高精度的场景中。这可能需要显著的计算资源和时间，但可以使模型充分学习和优化其策略。
    """

    print("1--------------------------------------------1")
    # 被动学习-时序差分方法TD Learning
    # 创建TD Learning对象，用于被动学习
    TD_Learning = TD(E)
    num_iterations = 100  # 设置迭代次数
    for i in range(num_iterations):
        TD_Learning.train()  # 进行训练，更新状态值函数
    print(TD_Learning.U)  # 打印更新后的状态值函数

    # 生成一个序列（episode）
    episode = TD_Learning.generate_episode()  # 生成序列用于展示学习效果
    # 打印生成的序列
    print("Generated episode:")
    for step in episode:
        print(step, end="")
        print("-->")  # 打印序列中的每一步

    print("2--------------------------------------------2")
    # 主动学习-Q Learning
    # 创建Q Learning对象，用于主动学习
    Q_Learning = Q_Learning(E)
    num_iterations = 100000  # 设置迭代次数
    for _ in range(num_iterations):
        Q_Learning.train()  # 进行训练，更新Q值矩阵

    print(Q_Learning.Q)  # 打印更新后的Q值矩阵

    print("3--------------------------------------------3")
    # 价值函数的线性逼近
    # 创建线性逼近的TD Learning对象
    F_TD = F_TD(E)
    num_iterations = 100  # 设置迭代次数，可以根据需要调整
    for _ in range(num_iterations):
        F_TD.train()  # 进行训练，更新权重向量

    print(F_TD.w)  # 打印更新后的权重向量
