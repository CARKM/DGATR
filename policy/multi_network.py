import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from policy.base_policy import Policy
from policy.network import sxsDQN, sxsGCN, sxsGAT
from policy.network import GEANT2DQN, GEANT2GCN, GEANT2GAT, GEANT2multi_head_GAT
from policy.network import lataDQN, lataGCN, lataGAT
from policy.config import config, get_adj_mtx
import logging


def clip_r(r, mini=-5):
    return r if r > mini else mini


# Build the policy training
class DRL(Policy):
    def __str__(self):
        print(self.config)
        print('Memory shape', self.memory.shape)
        print('Network shape', self.eval_net)
        print('Optimizer', self.optimizer)
        return '-----------------------------------'

    def __init__(self, network, net_name='6x6', model='DQN', pre_train=None,
                 pre_time=10000, epsilon=0, static=False):
        super().__init__(network)
        self.config = config(net_name)
        self.static = static
        self.epsilon = epsilon
        _, self.adj, _ = get_adj_mtx(self.links.items(),
                                     self.config.number_of_node)

        # Model
        self._build_model(model, net_name)

        self.loss = []

        # Pre_train
        if pre_train is not None:
            self.pre_training(pre_train, pre_time)

        # initialize the memory
        self.build_memory()

    def pre_training(self, pre_train, pre_time):
        self.learn_step_counter = 0
        for i in tqdm(range(pre_time)):
            s, a, r, s_, a_, is_dest = self._batch_memory(pre_train)
            self._update_parameter(s, a, r, s_, a_, is_dest, pre=True)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose(self, source, dest, target=False, idx=False):
        x = self._get_state(source, dest)
        x = torch.tensor(x, dtype=torch.float).view(1, -1)
        scores = self.eval_net.forward(x, self.adj).view(-1, 1)[self.links[source]]
        if np.random.uniform() < self.epsilon:
            # exploration
            choice = int(np.random.randint(0, len(scores), 1))
            return (choice, scores[choice]) if idx else self.links[source][choice]
        else:
            # greedy
            choice = int(torch.argmax(scores))
            if not target:
                self.check_action[source][dest][choice] += 1
                self.check_utilization[self.links[source][choice]] += 1
            return (choice, scores.max()) if idx else self.links[source][choice]

    def learn(self, rewards):
        if not self.static:
            for reward in rewards:
                self._store_memory(reward)
                # self._update_immediately(reward)
                if self.memory_counter % self.config.memory_capacity == 0 and\
                   self.memory_counter != 0:
                    s, a, r, s_, a_, is_dest = self._batch_memory(self.memory)
                    self._update_parameter(s, a, r, s_, a_, is_dest)
        else:
            None

    def reset_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=learning_rate)

    def _get_state(self, source, dest):
        x = np.zeros(self.config.n_state)
        for neighbor in self.links[source]:
            # 0 for source, 1 for action, 2 for destination
            x[3 * source], x[3 * neighbor + 1], x[3 * dest + 2] = 1, 1, 1
        return x

    def _build_model(self, model, net_name):
        node = self.config.number_of_node
        inp = self.config.node_input
        out = self.config.node_output
        if net_name == '6x6':
            if model == 'DQN':
                self.eval_net = sxsDQN(node, inp, out)
                self.target_net = sxsDQN(node, inp, out)
            elif model == 'GCN':
                self.eval_net = sxsGCN(node, inp, out)
                self.target_net = sxsGCN(node, inp, out)
            else:
                self.eval_net = sxsGAT(node, inp, out)
                self.target_net = sxsGAT(node, inp, out)
        if net_name == 'lata':
            if model == 'DQN':
                self.eval_net = lataDQN(node, inp, out)
                self.target_net = lataDQN(node, inp, out)
            elif model == 'GCN':
                self.eval_net = lataGCN(node, inp, out)
                self.target_net = lataGCN(node, inp, out)
            else:
                self.eval_net = lataGAT(node, inp, out)
                self.target_net = lataGAT(node, inp, out)
        if net_name == 'GEANT2':
            if model == 'DQN':
                self.eval_net = GEANT2DQN(node, inp, out)
                self.target_net = GEANT2DQN(node, inp, out)
            elif model == 'GCN':
                self.eval_net = GEANT2GCN(node, inp, out)
                self.target_net = GEANT2GCN(node, inp, out)
            elif model == 'GAT':
                self.eval_net = GEANT2GAT(node, inp, out)
                self.target_net = GEANT2GAT(node, inp, out)
            else:
                self.eval_net = GEANT2multi_head_GAT(node, inp, out,
                                                     self.config.number_of_head)
                self.target_net = GEANT2multi_head_GAT(node, inp, out,
                                                       self.config.number_of_head)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=self.config.learning_rate)
        self.loss_func = nn.MSELoss()   # MSE loss

        logging.debug("The network structure is: ", self.eval_net)
        logging.debug("The learning rate is now: ", self.config.learning_rate)

    def build_memory(self,):
        # for timing the change of the target network
        self.learn_step_counter = 0
        # counter of the memory
        self.memory_counter = 0
        self.memory = np.zeros((self.config.memory_capacity,
                                2 * self.config.n_state + 4))

    def _store_memory(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']
        s = self._get_state(x, d)
        s_ = self._get_state(y, d)
        a_ = self.choose(y, d, target=True)

        if y == d:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((s, y, r, s_, a_, is_dest))
        # ?????????????????????, ??????????????????
        index = self.memory_counter % self.config.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

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

    def _update_parameter(self, s, a, r, s_, a_, is_dest, pre=False):
        tau = self.config.tau
        if self.learn_step_counter % self.config.target_replace_iter == 0:
            for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#         if self.learn_step_counter % self.config.target_replace_iter == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        q_eval = self.eval_net(s, self.adj).gather(1, a)

        # q_next must be max_action
        if pre:
            q_next = self.target_net(s_, self.adj).gather(1, a_)
        else:
            ava_act = s_[:, 1::3]
            q_next = self.target_net(s_, self.adj)
            q_next = torch.where(ava_act > 0, q_next, torch.Tensor([-1e12]))
            q_next = q_next.max(1).values.view(-1, 1)

        q_target = r + self.config.reward_dacay * q_next * is_dest

        loss = self.loss_func(q_eval, q_target.detach())

        self.loss.append(float(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store(self, filename):
        torch.save(self.eval_net.state_dict(), filename)

    def load(self, filename):
        self.eval_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(torch.load(filename))
