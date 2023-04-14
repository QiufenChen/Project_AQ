# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/9 9:49
# @File: lrpDes.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import os
from copy import deepcopy

import torch
import torch as th

from torch import nn
from filter import relevance_filter

from dgl.nn.pytorch import GraphConv, TAGConv, ChebConv, GATConv

top_k_percent = 0.04  # Proportion of relevance scores that are allowed to pass.


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.
    Optionally modifies layer weights according to propagation rule. Here z^+-rule
    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.
    """

    def __init__(self, layer: torch.nn.Linear, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))
            # print('Linear weight is ', self.layer.weight.shape)

        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # r = relevance_filter(r, top_k_percent=top_k_percent)
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.
    Passes the relevance scores without modification. Might be of use later.
    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.
    Passes the relevance scores without modification. Might be of use later.
    """

    def __init__(self, layer: torch.nn.Dropout) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationTAGCN(nn.Module):
    def __init__(self, layer: TAGConv):
        super().__init__()

        self.layer = layer

        # From source code: dgl/nn/pytorch/conv/tagconv.py
        self.W = self.layer.lin.weight         # 获取该层的权重
        self.b = self.layer.lin.bias           # 获取该层的偏置
        self.K = self.layer._k                 # 获取该层的K阶
        # print(self.W.shape, self.b.shape, self.K)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor,  graph) -> torch.tensor:
        """
        a: represents the input for each layer
        r: represents the relevance of the latter layer
        """
        rho = lambda Wk: [W.abs() for W in Wk]
        num_nodes = len(a)   # a's shape is torch.Size([31, 50]
        dW = self.W.shape[1]//(self.K+1)
        Wk = [self.W[:, k*dW:(k+1)*dW].t() for k in range(self.K+1)]  # 把每一个阶的信息抽取出来

        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)  # 邻接矩阵标准化
        Ak = [th.matrix_power(A, k) for k in range(self.K+1)]

        rhoWk = rho(Wk)
        dimY, dimJ = rhoWk[0].shape
        U = th.zeros(num_nodes, dimJ, num_nodes, dimY)     # U's shape is torch.Size([31, 50, 31, 32])

        for i in range(num_nodes):
            for j in range(dimJ):
                U[i, j] = sum(Ak[k][i, :].unsqueeze(-1).matmul(rhoWk[k][:, j].unsqueeze(0)) for k in range(self.K + 1))

        r = th.einsum("ijxy,ij->xy", [U, r / U.sum(dim=(2, 3))])

        return r


    def Validate(self, graph):
        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)  # 邻接矩阵标准化
        # 分别计算A的k次方
        Ak = [th.matrix_power(A, k) for k in range(self.maxK+1)]
        # 分别计算 k 个多项式卷积核提取图节点的邻域信息, 计算 k 阶多项式, 将K个多项式卷积核提取的 feature_map 拼接,并以此将结果存储到 hs 中
        hs = []
        h = graph.ndata[self.net.features_str].float()
        hs.append(h)
        for layer_param in self.layer_params:
            # k个卷积核在图结构数据上提取特征并和bias进行线性组·合
            h = sum(Ak[k].matmul(h).matmul(layer_param[2][k]) for k in range(layer_param[1]+1))+layer_param[0]
            # 将输入h中每个元素的范围限制到区间 [min,max], 返回结果到一个新张量
            h = h.clamp(min=0)
            hs.append(h)
        return hs, Ak


