#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : layers.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 16:45
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) # W
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  # A
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # b, n, d
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48, leaky_c=0.2):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(leaky_c)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = y + x
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PreGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, node_n, p_dropout, leaky_c=0.2):
        super(PreGCN, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n

        self.gcn = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1d = nn.BatchNorm1d(node_n * hidden_feature)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(leaky_c)

        self.do = nn.Dropout(p_dropout)

    def forward(self, x):
        y = self.gcn(x)
        b, n, f = y.shape
        y = self.bn1d(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y


class PostGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, node_n):
        super(PostGCN, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n

        self.gcn = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        # self.act_f = nn.Sigmoid()
        # self.act_f = nn.LeakyReLU(option.leaky_c)  # 最后一层加激活不确定对不对
        # self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gcn(x)
        # y = self.act_f(y)  # 最后一层加激活不确定对不对
        return y


class SingleLeftLinear(nn.Module):
    def __init__(self, input_feature, out_features, seq_len, p_dropout, leaky_c=0.2):
        super(SingleLeftLinear, self).__init__()
        self.input_feature = input_feature
        self.out_features = out_features
        self.seq_len = seq_len

        self.linear = nn.Linear(input_feature, out_features)  # B, 35, 66 -> B, 35, 36
        self.bn = nn.BatchNorm1d(out_features * seq_len)
        # self.act = nn.Tanh()
        self.act = nn.LeakyReLU(leaky_c)
        self.do = nn.Dropout(p_dropout)

    def forward(self, input):
        '''

        :param input: B, 66, 64
        :return: y: B, 66, 35
        '''
        input = input.permute(0, 2, 1).contiguous()  # b， 64， 66
        y = self.linear(input)
        b, n, f = y.shape
        y = self.bn(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)
        y = y.permute(0, 2, 1)
        return y


class SingleRightLinear(nn.Module):
    def __init__(self, input_feature, out_features, node_n, p_dropout, leaky_c=0.2):
        super(SingleRightLinear, self).__init__()

        self.input_feature = input_feature
        self.out_features = out_features
        self.node_n = node_n

        self.linear = nn.Linear(input_feature, out_features)  # B, 66, 35 -> B, 66, 128
        self.bn = nn.BatchNorm1d(node_n * out_features)
        # self.act = nn.Tanh()
        self.act = nn.LeakyReLU(leaky_c)
        self.do = nn.Dropout(p_dropout)

    def forward(self, input):
        '''

        :param input: B, 66, 35
        :return: y: B, 66, 35
        '''
        y = self.linear(input)
        b, n, f = y.shape
        y = self.bn(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)
        return y
