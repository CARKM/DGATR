import torch
import torch.nn as nn
import torch.nn.functional as F


# 6x6DQN
class sxsDQN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output, dim_action=36):
        super(sxsDQN, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input

        self.fc1 = nn.Linear(number_of_node * dim_input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, dim_action)

    def forward(self, X, adj):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# 6x6GCN
class sxsGCN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output, dim_action=False):
        super(sxsGCN, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        if not dim_action:
            self.dim_action = number_of_node
        else:
            self.dim_action = dim_action
        self.node_update1 = nn.Linear(dim_input, 4)
        self.node_update2 = nn.Linear(4, dim_output)
        self.node_update3 = nn.Linear(dim_output, 3)

        self.fc1 = nn.Linear(number_of_node * dim_input, 54)
        self.out = nn.Linear(54, dim_action)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.node_update1(torch.matmul(adj, X)))
        X = F.relu(self.node_update2(torch.matmul(adj, X)))
        X = F.relu(self.node_update3(torch.matmul(adj, X)))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# 6x6GAT
class sxsGAT(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output, dim_action=False):
        super(sxsGAT, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        if not dim_action:
            self.dim_action = number_of_node
        else:
            self.dim_action = dim_action
        self.GAT1 = GraphAttentionLayer(dim_input, 4)
        self.GAT2 = GraphAttentionLayer(4, dim_output)

        self.fc1 = nn.Linear(number_of_node * dim_output, 54)
        self.out = nn.Linear(54, self.dim_action)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.GAT1(X, adj))
        X = F.relu(self.GAT2(X, adj))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


class sxsDueling(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output, dim_action=False):
        super(sxsDueling, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        if not dim_action:
            self.dim_action = number_of_node
        else:
            self.dim_action = dim_action
        self.GAT1 = GraphAttentionLayer(dim_input, 4)
        self.GAT2 = GraphAttentionLayer(4, dim_output)

        self.fc1 = nn.Linear(number_of_node * dim_output, 54)
        self.value_head = nn.Linear(54, 1)
        self.adv_head = nn.Linear(54, self.dim_action)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.GAT1(X, adj))
        X = F.relu(self.GAT2(X, adj))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))

        value = self.value_head(X)
        adv = self.adv_head(X)  # dim = (N_ACTION,1)
        return value + adv - adv.mean(1).reshape(-1, 1)


# lataDQN
class lataDQN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output):
        super(lataDQN, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input

        self.fc1 = nn.Linear(number_of_node * dim_input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, number_of_node)

    def forward(self, X, adj):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# lataGCN
class lataGCN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output):
        super(lataGCN, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        self.node_update1 = nn.Linear(dim_input, 4)
        self.node_update2 = nn.Linear(4, dim_output)

        self.fc1 = nn.Linear(number_of_node * dim_output, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, number_of_node)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.node_update1(torch.matmul(adj, X)))
        X = F.relu(self.node_update2(torch.matmul(adj, X)))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# lataGAT
class lataGAT(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output):
        super(lataGAT, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        self.GAT = GraphAttentionLayer(dim_input, dim_output)
        # self.GAT1 = GraphAttentionLayer(dim_input, 4)
        # self.GAT2 = GraphAttentionLayer(4, dim_output)

        self.fc1 = nn.Linear(number_of_node * dim_output, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, number_of_node)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.GAT(X, adj))
        # X = F.relu(self.GAT1(X, adj))
        # X = F.relu(self.GAT2(X, adj))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# GEANT2DQN
class GEANT2DQN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output):
        super(GEANT2DQN, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input

        self.fc1 = nn.Linear(number_of_node * dim_input, 64)
        self.out = nn.Linear(64, number_of_node)

    def forward(self, X, adj):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# GEANT2GCN
class GEANT2GCN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output):
        super(GEANT2GCN, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        self.node_update1 = nn.Linear(dim_input, 4)
        self.node_update2 = nn.Linear(4, dim_output)

        self.fc1 = nn.Linear(number_of_node * dim_output, 64)
        self.out = nn.Linear(64, number_of_node)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.node_update1(torch.matmul(adj, X)))
        X = F.relu(self.node_update2(torch.matmul(adj, X)))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# GEANT2GAT
class GEANT2GAT(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output):
        super(GEANT2GAT, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input
        self.GAT = GraphAttentionLayer(dim_input, dim_output)
        # self.GAT1 = GraphAttentionLayer(dim_input, 4)
        # self.GAT2 = GraphAttentionLayer(4, dim_output)

        self.fc1 = nn.Linear(number_of_node * dim_output, 64)
        self.out = nn.Linear(64, number_of_node)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)
        X = F.relu(self.GAT(X, adj))
        # X = F.relu(self.GAT1(X, adj))
        # X = F.relu(self.GAT2(X, adj))

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


# GEANT2GAT-multi-head
class GEANT2multi_head_GAT(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output, num_head):
        super(GEANT2multi_head_GAT, self).__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input

        self.GAT = [GraphAttentionLayer(dim_input, dim_output) for _ in range(num_head)]
        for i, attention in enumerate(self.GAT):
            self.add_module('attention_{}'.format(i), attention)

        self.fc1 = nn.Linear(number_of_node * dim_output * num_head, 64)
        self.out = nn.Linear(64, number_of_node)

    def forward(self, X, adj):
        X = X.reshape(X.size(0), self.number_of_node, self.dim_input)

        X = torch.cat([F.relu(att(X, adj)) for att in self.GAT], dim=1)

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=1, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu parameter
        # self.concat = concat   # if true, active through elu

        # trainable parameter
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, X, adj):
        """
        X: [K, N, in_features]
        adj: [N, N]
        """
        h = torch.matmul(X, self.W)   # [K, N, out_features]
        N = h.size()[1]    # N 图的节点数
        K = h.size()[0]

        a_input = torch.cat([h.repeat(1, 1, N).view(K, N*N, -1), h.repeat(1, N, 1)], dim=2).view(K, N, -1, 2*self.out_features)
        # [K, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [K, N, N, 1] => [K, N, N] attention (not scaler)

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)   # [K, N, N]
        # 1 means there is a link, otherwise, -inf to make the softmax = 0
        attention = F.softmax(attention, dim=2)

        h_prime = torch.matmul(attention, h)
        # [K, N, N] * [k, N, out_features] => [K, N, out_features]

        return h_prime  # [N, out]

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
