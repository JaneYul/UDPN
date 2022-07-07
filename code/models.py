import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from layers import GraphConvolution
from torch_geometric.nn import GCNConv
import time
import os
import sys
from prototype import *

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


'''
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

        self.temp = GCNConv(in_channels, 2 * out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation((self.conv[i])(x, edge_index))
        return x
'''

class Contrastive(nn.Module):
    def __init__(self, num_hidden, num_proj_hidden, tau, dropout):
        super(Contrastive, self).__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.fc = nn.Sequential(nn.Linear(num_hidden*2, int(num_hidden*1.5)),
                                            nn.Dropout(dropout),
                                            nn.ReLU(),
                                            nn.Linear(int(num_hidden*1.5), num_hidden),
                                            )

    def concat(self, z1, z2):
        z12 = torch.cat((z1, z2), dim=1)
        out = self.fc(z12)
        return out

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)                        # 每个元素除以各自所在的行的二范数，即归一化，之后每行的二范数为1
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())                 # 每行嵌入之间进行匹配

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)       # 定义了一个函数
        refl_sim = f(self.sim(z1, z1))              # 先算相似度，再经过函数f；输入是自己和自己，sim出来的结果是对称的；每个相似分数/tau再exp
        between_sim = f(self.sim(z1, z2))           # 不同视图之间的；不对称，每行可以看成z1的每行和z2的每行点积得到的结果，实际是以z1为锚点的；将其转置就是以z2为锚点
        # print(refl_sim.shape, between_sim.shape)  # A*B^T=C，两边取T。以上两个相似度矩阵都是[2708, 2708]
        
        return -torch.log(                          # 分子是相同节点在不同视图下的相似度。sum(1)是每行求和
            between_sim.diag()                      # refl_sim.sum(1)- refl_sim.diag()是视图内负样本
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))  # 注意between_sim.sum(1)包括了between_sim.diag()（正样本）和跨视图负样本
                                                    # 下面的diag()和sum(1)都是得到一个2708维的向量

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i*batch_size:(i + 1)*batch_size]
            refl_sim = f(self.sim(z1[mask], z1))     # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            losses.append(-torch.log( between_sim[:, i*batch_size:(i + 1)*batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i*batch_size:(i + 1)*batch_size].diag()))  )
            refl_sim, between_sim = None, None
            

        return torch.cat(losses)


    def forward(self, z1, z2, index, mean=True, batch_size=0):   # 这里的index是class信息
        h1 = self.projection(z1)            # 两个嵌入先经过同一个投影层
        h2 = self.projection(z2)

        h1, h2 = self.unique(h1, h2, index)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)     # 以h1为锚点 [2708]
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        # out = self.concat(z1, z2)
        # print('out', out.shape)

        return ret


    def unique(self, h1, h2, index):
        index = index.cpu()
        index_uni = np.unique(index)
        if len(index_uni) != len(index):
            h1_n, h2_n = torch.Tensor([]).cuda(), torch.Tensor([]).cuda()
            for i in index_uni:
                if i == -1:
                    continue
                idx = np.argwhere(index == i).reshape(1, -1).squeeze(0)
                h1_temp = h1[idx].sum(0)
                h2_temp = h2[idx].sum(0)
                h1_n = torch.cat((h1_n, h1_temp.unsqueeze(0)), 0)
                h2_n = torch.cat((h2_n, h2_temp.unsqueeze(0)), 0)
            for i in range(len(index)):
                if index[i] == -1:
                    h1_n = torch.cat((h1_n, h1[i].unsqueeze(0)), 0)
                    h2_n = torch.cat((h2_n, h2[i].unsqueeze(0)), 0)
            assert h2_n.size(0) == len(index_uni) + np.sum(np.array(index)==-1) - 1
            return h1_n, h2_n
        else:
            return h1, h2



class Model(torch.nn.Module):
    def __init__(self, encoder, contrastive, prototype):
        super(Model, self).__init__()
        self.encoder = encoder
        self.contrastive = contrastive
        self.prototype = prototype

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)




