import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from policy.base_policy import Policy
from policy.config import config, get_adj_mtx


def clip_r(r, mini=-5):
    return r if r > mini else mini


# Build the policy training
class DRL(Policy):
    def __str__(self):
        print(self.config)
        print('Memory shape', self.memory[0].shape)
        print('-----------------------------------')
        print('Network shape', self.eval_net[0])
        print('-----------------------------------')
        print('Optimizer', self.optimizer)
        return '-----------------------------------'

    def __init__(self, network, net_name='6x6',
                 epsilon=0, static=False):
        super().__init__(network)
        self.config = config(net_name)
        self.config.target_replace_iter = 8
        self.static = static
        self.epsilon = epsilon
        _, self.adj, _ = get_adj_mtx(self.links.items(),
                                     self.config.number_of_node)

        # initialize the memory
        self.build_memory()

    def pre_training(self, pre_train, pre_time):
        self.learn_step_counter = 0
        for i in tqdm(range(pre_time)):
            s, a, r, s_, a_, is_dest = self._batch_memory(pre_train)
            self._update_parameter(s, a, r, s_, a_, is_dest)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose(self, source, dest, target=False, idx=False):
        x = self._get_state(source, dest)
        x = torch.tensor(x, dtype=torch.float).view(1, -1)
        scores = self.eval_net[source].forward(x, self.adj).view(-1, 1)[self.links[source]]
        if np.random.uniform() < self.epsilon:
            # exploration
            choice = int(np.random.randint(0, len(scores), 1))
            return (choice, scores[choice]) if idx else self.links[source][choice]
        else:
            # greedy
            choice = int(torch.argmax(scores))
            return (choice, scores.max()) if idx else self.links[source][choice]

    def learn(self, rewards):
        if not self.static:
            for reward in rewards:
                self._store_memory(reward)
                x = reward.source
                # immediate update
                if self.memory_counter[x] % self.config.memory_capacity == 0 and\
                   self.memory_counter[x] != 0:
                    s, a, r, s_, a_, is_dest = self._batch_memory(self.memory[reward.source])
                    self._update_parameter(reward.source, s, a, r, s_, a_, is_dest)
#             for x in range(self.config.number_of_node):
#                 if self.memory_counter[x] % self.config.memory_capacity == 0 and\
#                    self.memory_counter[x] != 0:
#                     s, a, r, s_, a_, is_dest = self._batch_memory(self.memory[x])
#                     self._update_parameter(x, s, a, r, s_, a_, is_dest)
        else:
            None

    def reset_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.glb_net.parameters(),
                                          lr=learning_rate)

    def _get_state(self, source, dest):
        x = np.zeros(self.config.n_state)
        for neighbor in self.links[source]:
            # 0 for source, 1 for action, 2 for destination
            x[3 * source], x[3 * neighbor + 1], x[3 * dest + 2] = 1, 1, 1
        return x

    def build_model(self, NN_glb, NN_eval, NN_target):
        node = self.config.number_of_node
        self.glb_net = NN_glb
        self.eval_net = {i: NN_eval[i] for i in range(node)}
        self.target_net = {i: NN_target[i] for i in range(node)}

        self.optimizer = torch.optim.Adam(self.glb_net.parameters(),
                                          lr=self.config.learning_rate)
        self.loss_func = nn.MSELoss()   # MSE loss

    def build_memory(self,):
        self.loss = []
        # for timing the change of the target network
        self.learn_step_counter = [0 for i in range(self.config.number_of_node)]
        # counter of the memory
        self.memory_counter = [0 for i in range(self.config.number_of_node)]
        self.memory = {i: np.zeros((self.config.memory_capacity,
                                    2 * self.config.n_state + 4))
                       for i in range(self.config.number_of_node)}

    def _store_memory(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']
        s = self._get_state(x, d)
        s_ = self._get_state(y, d)
        a_ = 0
        r = clip_r(r, -10)

        if y == d:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((s, y, r, s_, a_, is_dest))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter[x] % self.config.memory_capacity
        self.memory[x][index, :] = transition
        self.memory_counter[x] += 1

    def _batch_memory(self, memory):
        sample_index = np.random.choice(memory.shape[0], self.config.batch,
                                        replace=False)
        b_memory = memory[sample_index, :]
        n_s = self.config.n_state
        s = torch.FloatTensor(b_memory[:, :n_s])
        a = torch.LongTensor(b_memory[:, n_s:n_s+1].astype(int))
        r = torch.FloatTensor(b_memory[:, n_s+1:n_s+2])
        s_ = torch.FloatTensor(b_memory[:, n_s+2:2*n_s+2])
        a_ = torch.LongTensor(b_memory[:, 2*n_s+2:2*n_s+3].astype(int))
        is_dest = torch.FloatTensor(b_memory[:, n_s*2+3:])
        # a, a_ --> LongTensor
        # s, s_, r, is_dest --> FloatTensor
        return s, a, r, s_, a_, is_dest

    def _update_parameter(self, x, s, a, r, s_, a_, is_dest):
        q_eval = self.eval_net[x](s, self.adj).gather(1, a)

        ava_act_ = s_[:, 1::3]
        # q_next must be max_action
        # q_next = self.target_net[x](s_, self.adj)
        # q_next = torch.where(ava_act_ > 0, q_next, torch.Tensor([-1e12]))
        # q_next = q_next.max(1).values.view(-1, 1)

        q_next = torch.zeros((self.config.batch, 1), requires_grad=False)
        a = a.view(self.config.batch)
        for i in torch.unique(a):
            s1 = s_[a == i, :]
            ava_act_tr = ava_act_[a == i, :]
            # q_tr = self.target_net[int(i)](s1, self.adj)
            q_tr = self.eval_net[int(i)](s1, self.adj)
            q_tr = torch.where(ava_act_tr > 0, q_tr, torch.Tensor([-1e12]))
            q_next[a == i, 0] = q_tr.max(1).values.view(-1)

        q_target = r + self.config.reward_dacay * q_next * is_dest

        loss = self.loss_func(q_eval, q_target.detach())
        self.loss.append(float(loss))

        # push local gradient to global network
        self.optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(self.eval_net[x].parameters(), self.glb_net.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()

        # load the global parameter
        self.eval_net[x].load_state_dict(self.glb_net.state_dict())
        tau = self.config.tau
        for target_param, param in zip(self.target_net[x].parameters(), self.eval_net[x].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def store(self, filename):
        torch.save(self.glb_net.state_dict(), filename)

    def load(self, filename):
        para = torch.load(filename)
        self.glb_net.load_state_dict(para)
        for x in range(self.config.number_of_node):
            self.eval_net[x].load_state_dict(para)
            self.target_net[x].load_state_dict(para)
