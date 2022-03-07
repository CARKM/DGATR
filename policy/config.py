import numpy as np
import torch


# to get the adjacency matrix
def get_adj_mtx(link_info, number_of_node):
    A = np.eye(number_of_node)
    for s, ns in link_info:
        for n in ns:
            A[s][n] = 1
    D = np.array(np.sum(A, axis=0))
    D = np.matrix(np.diag(D))
    A_ = D**-1 * A
    # A_ after scaler
    # A nonscaler
    return torch.FloatTensor(A_), torch.FloatTensor(A), torch.FloatTensor(D)


class config(object):
    def __init__(self, net_name):
        super(config).__init__()
        if net_name == '6x6':
            self.sxs_parameter()
            self.sxs_hyper()
        if net_name == 'lata':
            self.lata_parameter()
            self.lata_hyper()
        if net_name == 'GEANT2':
            self.GEANT2_parameter()
            self.GEANT2_hyper()
        if net_name == 'AT&T':
            self.ATT_parameter()
            self.ATT_hyper()

    def sxs_parameter(self, n_node=36, node_input=3, node_output=8, num_head=2):
        self.number_of_node = n_node
        self.node_input = node_input
        self.node_output = node_output
        self.n_state = node_input * n_node
        self.number_of_head = num_head

    def ATT_parameter(self, n_node=27, node_input=3, node_output=8):
        self.number_of_node = n_node
        self.node_input = node_input
        self.node_output = node_output
        self.n_state = node_input * n_node

    def lata_parameter(self, n_node=116, node_input=3, node_output=8):
        self.number_of_node = n_node
        self.node_input = node_input
        self.node_output = node_output
        self.n_state = node_input * n_node

    def GEANT2_parameter(self, n_node=24, node_input=3, node_output=8, num_head=2):
        self.number_of_node = n_node
        self.node_input = node_input
        self.node_output = node_output
        self.n_state = node_input * n_node
        self.number_of_head = num_head

    def sxs_hyper(self, learning_rate_critic=4e-5, learning_rate_actor=4e-7, reward_dacay=0.99, batch=256, memory_capacity=2000, target_replace_iter=30, tau=0.05):
        self.learning_rate = learning_rate_critic
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_actor = learning_rate_actor
        self.reward_dacay = reward_dacay
        self.tau = tau

        self.batch = batch
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter

    def ATT_hyper(self, learning_rate_critic=4e-5, reward_dacay=0.9, batch=256, memory_capacity=2000, target_replace_iter=30, tau=0.05):
        self.learning_rate = learning_rate_critic
        self.learning_rate_critic = learning_rate_critic
        self.reward_dacay = reward_dacay
        self.tau = tau

        self.batch = batch
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter

    def lata_hyper(self, learning_rate=4e-7, reward_dacay=0.99, batch=256, memory_capacity=2000, target_replace_iter=30, tau=0.05):
        self.learning_rate = learning_rate
        self.reward_dacay = reward_dacay

        self.batch = batch
        self.tau = tau
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter

    def GEANT2_hyper(self, learning_rate=4e-5, reward_dacay=0.99, batch=256, memory_capacity=2000, target_replace_iter=30, tau=0.05):
        self.learning_rate = learning_rate
        self.reward_dacay = reward_dacay

        self.batch = batch
        self.tau = tau
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter

    def __str__(self,):
        print('Network Topology Information')
        print('Number of Node:', self.number_of_node)
        print('Input dimension', self.node_input)
        print('-----------------------------------')
        print('Hyper Parameter')
        print('Learning rate:', self.learning_rate)
        print('Reward_decay:', self.reward_dacay)
        print('Memory capacity:', self.memory_capacity)
        print('Batch size:', self.batch)
        print('Tau:', self.tau)

        return '-----------------------------------'
