import numpy as np

from policy.base_policy import Policy

# # memory for nodes
# N_STATES = 36 * 2  # 杆子能获取的环境信息数

# memory for 1action
N_STATES = 36 * 3
MEMORY_CAPACITY = 50000     # 记忆库大小


def tanh(x, scale=1):
    return (np.exp(scale * x) - np.exp(-scale * x))/(np.exp(scale * x) + np.exp(- scale * x))


class Qroute(Policy):
    attrs = Policy.attrs | set(['Qtable', 'discount', 'epsilon'])

    def __init__(self, network, initQ=0, discount=0.99, epsilon=0, static=False):
        super().__init__(network)

        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))     # 初始化记忆库

        self.discount = discount
        self.epsilon = epsilon
        self.static = static

        self.Qtable = {x: np.random.normal(
            initQ, 1, (len(self.links), len(ys)))
            for x, ys in self.links.items()}

        self.check_action = {x: np.zeros(
            (len(self.links), len(ys)))
            for x, ys in self.links.items()
        }

        self.check_utilization = {x: 0 for x, _ in self.links.items()}

        for x, table in self.Qtable.items():
            # Q_x(z, x) = 0, forall z in x.neighbors
            table[x] = 0
            # Q_x(z, y) = -1 if z == y else 0
            table[self.links[x]] = -np.eye(table.shape[1])

    def choose(self, source, dest, idx=False):
        scores = self.Qtable[source][dest]
        # score_max = scores.max()
        # choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        if np.random.uniform() < self.epsilon:
            # exploration
            choice = int(np.random.randint(0,len(scores),1))
            return (choice, scores[choice]) if idx else self.links[source][choice]
        else:  # greedy
            choice = np.argmax(scores)
            self.check_action[source][dest][choice] += 1
            self.check_utilization[self.links[source][choice]] += 1
            return (choice, scores.max()) if idx else self.links[source][choice]

    def get_info(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def _extract(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = -info['q_y'] - info['t_y']

        return r, info, x, y, d

    def _update_qtable(self, r, x, y, d, max_Q_y, lr):
        y_idx = self.action_idx[x][y]
        old_score = self.Qtable[x][d][y_idx]
        self.Qtable[x][d][y_idx] += lr * \
            (r + self.discount * max_Q_y - old_score)

    def _update(self, reward, lr={'q': 0.1}):
        " update agent once/one turn "
        r, info, x, y, d = self._extract(reward)
        self._update_qtable(r, x, y, d, info['max_Q_y'], lr['q'])

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(reward, lr if lr else self._update.__defaults__[0])
        else:
            None


class CQ(Qroute):
    attrs = Qroute.attrs | set(['decay', 'confidence'])

    def __init__(self, network, decay=0.9, initQ=0, discount=0.9):
        super().__init__(network, initQ, discount=discount)
        self.decay = decay
        self.confidence = {x: np.zeros_like(table, dtype=np.float64)
                            for x, table in self.Qtable.items()}
        self.clean()

    def clean(self):
        for x, conf in self.confidence.items():
            conf.fill(0.0) # empty confidence
            # the decision of sending to the destination is undoubtedly correct
            # base case: C_x(z, y) = 1 if z == y else 0
            conf[self.links[x]] = np.eye(conf.shape[1])

    def get_info(self, source, action, packet):
        z_idx, max_Q_f = self.choose(action, packet.dest, idx=True)
        return {
            'max_Q_f': max_Q_f,
            'C_f': self.confidence[action][packet.dest][z_idx]
        }

    def _update_qtable(self, r, x, y, d, C, max_Q):
        y_idx = self.action_idx[x][y]
        old_Q = self.Qtable[x][d][y_idx]
        old_conf = self.confidence[x][d][y_idx]
        eta = max(C, 1-old_conf)
        self.Qtable[x][d][y_idx] += eta * \
            (r + self.discount * max_Q - old_Q)
        self.confidence[x][d][y_idx] += eta * (C-old_conf)
        # counteract the effect of confidence_decay()
        self.confidence[x][d][y_idx] /= self.decay

    def _update(self, reward, lr={}):
        r, info, x, y, d = self._extract(reward)
        self._update_qtable(r, x, y, d, info['C_f'], info['max_Q_f'])

    def learn(self, rewards, lr={}):
        super().learn(rewards, lr)
        self.confidence_decay()

    def confidence_decay(self):
        for table in self.confidence.values():
            table *= self.decay


class CDRQ(CQ):
    mode = 'dual'

    def get_info(self, source, action, packet):
        w_idx, max_Q_b = self.choose(source, packet.source, idx=True)
        z_idx, max_Q_f = self.choose(action, packet.dest, idx=True)
        return {
            'max_Q_b': max_Q_b,
            'max_Q_f': max_Q_f,
            'C_b': self.confidence[source][packet.source][w_idx],
            'C_f': self.confidence[action][packet.dest][z_idx],
        }

    def _update(self, reward, lr={}):
        r_f, info, x, y, dst = self._extract(reward)
        self._update_qtable(r_f, x, y, dst, info['C_f'], info['max_Q_f']) # forward
        r_b = -info['q_x']-info['t_x']
        src = reward.packet.source
        self._update_qtable(r_b, y, x, src, info['C_b'], info['max_Q_b']) # backward
