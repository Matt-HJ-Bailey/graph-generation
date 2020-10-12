# -*- coding: utf-8 -*-
"""
Code that are NOT used for final version

Created on Fri Oct  9 15:13:41 2020

@author: Matt
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import (pad_packed_sequence,
                                pack_padded_sequence)

from model import sample_tensor

# RNN that updates according to graph structure, new proposed model
class Graph_RNN_structure(nn.Module):
    def __init__(
        self,
        hidden_size,
        batch_size,
        output_size,
        num_layers,
        is_dilation=True,
        is_bn=True,
    ):
        super(Graph_RNN_structure, self).__init__()
        ## model configuration
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers  # num_layers of cnn_output
        self.is_bn = is_bn

        ## model
        self.relu = nn.ReLU()
        # self.linear_output = nn.Linear(hidden_size, 1)
        # self.linear_output_simple = nn.Linear(hidden_size, output_size)
        # for state transition use only, input is null
        # self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # use CNN to produce output prediction
        # self.cnn_output = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)
        # )

        if is_dilation:
            self.conv_block = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size=3,
                        dilation=2 ** i,
                        padding=2 ** i,
                    )
                    for i in range(num_layers - 1)
                ]
            )
        else:
            self.conv_block = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1
                    )
                    for i in range(num_layers - 1)
                ]
            )
        self.bn_block = nn.ModuleList(
            [nn.BatchNorm1d(hidden_size) for i in range(num_layers - 1)]
        )
        self.conv_out = nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)

        # # use CNN to do state transition
        # self.cnn_transition = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1)
        # )

        # use linear to do transition, same as GCN mean aggregator
        self.linear_transition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )

        # GRU based output, output a single edge prediction at a time
        # self.gru_output = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # use a list to keep all generated hidden vectors, each hidden has size batch*hidden_dim*1, and the list size is expanding
        # when using convolution to compute attention weight, we need to first concat the list into a pytorch variable: batch*hidden_dim*current_num_nodes
        self.hidden_all = []

        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('linear')
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                # print(m.weight.data.size())
            if isinstance(m, nn.Conv1d):
                # print('conv1d')
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                # print(m.weight.data.size())
            if isinstance(m, nn.BatchNorm1d):
                # print('batchnorm1d')
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # print(m.weight.data.size())
            if isinstance(m, nn.GRU):
                # print('gru')
                m.weight_ih_l0.data = init.xavier_uniform(
                    m.weight_ih_l0.data, gain=nn.init.calculate_gain("sigmoid")
                )
                m.weight_hh_l0.data = init.xavier_uniform(
                    m.weight_hh_l0.data, gain=nn.init.calculate_gain("sigmoid")
                )
                m.bias_ih_l0.data = torch.ones(m.bias_ih_l0.data.size(0)) * 0.25
                m.bias_hh_l0.data = torch.ones(m.bias_hh_l0.data.size(0)) * 0.25

    def init_hidden(self, len=None):
        if len is None:
            return Variable(torch.ones(self.batch_size, self.hidden_size, 1)).cuda()
        else:
            hidden_list = []
            for i in range(len):
                hidden_list.append(
                    Variable(torch.ones(self.batch_size, self.hidden_size, 1)).cuda()
                )
            return hidden_list

    # only run a single forward step
    def forward(
        self,
        x,
        teacher_forcing,
        temperature=0.5,
        bptt=True,
        bptt_len=20,
        flexible=True,
        max_prev_node=100,
    ):
        # x: batch*1*self.output_size, the groud truth
        # todo: current only look back to self.output_size nodes, try to look back according to bfs sequence

        # 1 first compute new state
        # print('hidden_all', self.hidden_all[-1*self.output_size:])
        # hidden_all_cat = torch.cat(self.hidden_all[-1*self.output_size:], dim=2)

        # # # add BPTT, detach the first variable
        # if bptt:
        #     self.hidden_all[0] = Variable(self.hidden_all[0].data).cuda()

        hidden_all_cat = torch.cat(self.hidden_all, dim=2)
        # print(hidden_all_cat.size())

        # print('hidden_all_cat',hidden_all_cat.size())
        # att_weight size: batch*1*current_num_nodes
        for i in range(self.num_layers - 1):
            hidden_all_cat = self.conv_block[i](hidden_all_cat)
            if self.is_bn:
                hidden_all_cat = self.bn_block[i](hidden_all_cat)
            hidden_all_cat = self.relu(hidden_all_cat)
        x_pred = self.conv_out(hidden_all_cat)
        # 2 then compute output, using a gru
        # first try the simple version, directly give the edge prediction
        # x_pred = self.linear_output_simple(hidden_new)
        # x_pred = x_pred.view(x_pred.size(0),1,x_pred.size(1))

        # todo: use a gru version output
        # if sample==False:
        #     # when training: we know the ground truth, input the sequence at once
        #     y_pred,_ = self.gru_output(x, hidden_new.permute(2,0,1))
        #     y_pred = self.linear_output(y_pred)
        # else:
        #     # when validating, we need to sampling at each time step
        #     y_pred = Variable(torch.zeros(x.size(0), x.size(1), x.size(2))).cuda()
        #     y_pred_long = Variable(torch.zeros(x.size(0), x.size(1), x.size(2))).cuda()
        #     x_step = x[:, 0:1, :]
        #     for i in range(x.size(1)):
        #         y_step,_ = self.gru_output(x_step)
        #         y_step = self.linear_output(y_step)
        #         y_pred[:, i, :] = y_step
        #         y_step = F.sigmoid(y_step)
        #         x_step = sample(y_step, sample=True, thresh=0.45)
        #         y_pred_long[:, i, :] = x_step
        #     pass

        # 3 then update self.hidden_all list
        # i.e., model will use ground truth to update new node
        # x_pred_sample = gumbel_sigmoid(x_pred, temperature=temperature)
        x_pred_sample = sample_tensor(F.sigmoid(x_pred), sample=True)
        thresh = 0.5
        x_thresh = Variable(
            torch.ones(
                x_pred_sample.size(0), x_pred_sample.size(1), x_pred_sample.size(2)
            )
            * thresh
        ).cuda()
        x_pred_sample_long = torch.gt(x_pred_sample, x_thresh).long()
        if teacher_forcing:
            # first mask previous hidden states
            hidden_all_cat_select = hidden_all_cat * x
            x_sum = torch.sum(x, dim=2, keepdim=True).float()

        # i.e., the model will use it's own prediction to attend
        else:
            # first mask previous hidden states
            hidden_all_cat_select = hidden_all_cat * x_pred_sample
            x_sum = torch.sum(x_pred_sample_long, dim=2, keepdim=True).float()

        # update hidden vector for new nodes
        hidden_new = torch.sum(hidden_all_cat_select, dim=2, keepdim=True) / x_sum

        hidden_new = self.linear_transition(hidden_new.permute(0, 2, 1))
        hidden_new = hidden_new.permute(0, 2, 1)

        if flexible:
            # use ground truth to maintaing history state
            if teacher_forcing:
                x_id = torch.min(torch.nonzero(torch.squeeze(x.data)))
                self.hidden_all = self.hidden_all[x_id:]
            # use prediction to maintaing history state
            else:
                x_id = torch.min(torch.nonzero(torch.squeeze(x_pred_sample_long.data)))
                start = max(len(self.hidden_all) - max_prev_node + 1, x_id)
                self.hidden_all = self.hidden_all[start:]

        # maintaing a fixed size history state
        else:
            # self.hidden_all.pop(0)
            self.hidden_all = self.hidden_all[1:]

        self.hidden_all.append(hidden_new)

        # 4 return prediction
        # print('x_pred',x_pred)
        # print('x_pred_mean', torch.mean(x_pred))
        # print('x_pred_sample_mean', torch.mean(x_pred_sample))
        return x_pred, x_pred_sample


# batch_size = 8
# output_size = 4
# generator = Graph_RNN_structure(hidden_size=16, batch_size=batch_size, output_size=output_size, num_layers=1).cuda()
# for i in range(4):
#     generator.hidden_all.append(generator.init_hidden())
#
# x = Variable(torch.rand(batch_size,1,output_size)).cuda()
# x_pred = generator(x,teacher_forcing=True, sample=True)
# print(x_pred)


# current baseline model, generating a graph by lstm
class Graph_generator_LSTM(nn.Module):
    def __init__(
        self, feature_size, input_size, hidden_size, output_size, batch_size, num_layers
    ):
        super(Graph_generator_LSTM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear_input = nn.Linear(feature_size, input_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # initialize
        # self.hidden,self.cell = self.init_hidden()
        self.hidden = self.init_hidden()

        self.lstm.weight_ih_l0.data = init.xavier_uniform(
            self.lstm.weight_ih_l0.data, gain=nn.init.calculate_gain("sigmoid")
        )
        self.lstm.weight_hh_l0.data = init.xavier_uniform(
            self.lstm.weight_hh_l0.data, gain=nn.init.calculate_gain("sigmoid")
        )
        self.lstm.bias_ih_l0.data = torch.ones(self.lstm.bias_ih_l0.data.size(0)) * 0.25
        self.lstm.bias_hh_l0.data = torch.ones(self.lstm.bias_hh_l0.data.size(0)) * 0.25
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def init_hidden(self):
        return (
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).cuda(),
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).cuda(),
        )

    def forward(self, input_raw, pack=False, len=None):
        input = self.linear_input(input_raw)
        input = self.relu(input)
        if pack:
            input = pack_padded_sequence(input, len, batch_first=True)
        output_raw, self.hidden = self.lstm(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        output = self.linear_output(output_raw)
        return output


# a simple MLP generator output
class Graph_generator_LSTM_output_generator(nn.Module):
    def __init__(self, h_size, n_size, y_size):
        super(Graph_generator_LSTM_output_generator, self).__init__()
        # one layer MLP
        self.generator_output = nn.Sequential(
            nn.Linear(h_size + n_size, 64),
            nn.ReLU(),
            nn.Linear(64, y_size),
            nn.Sigmoid(),
        )

    def forward(self, h, n, temperature):
        y_cat = torch.cat((h, n), dim=2)
        y = self.generator_output(y_cat)
        # y = gumbel_sigmoid(y,temperature=temperature)
        return y


# a simple MLP discriminator
class Graph_generator_LSTM_output_discriminator(nn.Module):
    def __init__(self, h_size, y_size):
        super(Graph_generator_LSTM_output_discriminator, self).__init__()
        # one layer MLP
        self.discriminator_output = nn.Sequential(
            nn.Linear(h_size + y_size, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, h, y):
        y_cat = torch.cat((h, y), dim=2)
        l = self.discriminator_output(y_cat)
        return l


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # self.relu = nn.ReLU()

    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        return y


# vanilla GCN encoder
class GCN_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_encoder, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        # self.bn1 = nn.BatchNorm1d(output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        # x = x/torch.sum(x, dim=2, keepdim=True)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.conv2(x, adj)
        # x = x / torch.sum(x, dim=2, keepdim=True)
        return x


# vanilla GCN decoder
class GCN_decoder(nn.Module):
    def __init__(self):
        super(GCN_decoder, self).__init__()
        # self.act = nn.Sigmoid()

    def forward(self, x):
        # x_t = x.view(-1,x.size(2),x.size(1))
        x_t = x.permute(0, 2, 1)
        # print('x',x)
        # print('x_t',x_t)
        y = torch.matmul(x, x_t)
        return y


# GCN based graph embedding
# allowing for arbitrary num of nodes
class GCN_encoder_graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN_encoder_graph, self).__init__()
        self.num_layers = num_layers
        self.conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        # self.conv_hidden1 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        # self.conv_hidden2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.conv_block = nn.ModuleList(
            [
                GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
                for i in range(num_layers)
            ]
        )
        self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')

    def forward(self, x, adj):
        x = self.conv_first(x, adj)
        x = self.act(x)
        out_all = []
        out, _ = torch.max(x, dim=1, keepdim=True)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            out, _ = torch.max(x, dim=1, keepdim=True)
            out_all.append(out)
        x = self.conv_last(x, adj)
        x = self.act(x)
        out, _ = torch.max(x, dim=1, keepdim=True)
        out_all.append(out)
        output = torch.cat(out_all, dim=1)
        output = output.permute(1, 0, 2)
        # print(out)
        return output


# x = Variable(torch.rand(1,8,10)).cuda()
# adj = Variable(torch.rand(1,8,8)).cuda()
# model = GCN_encoder_graph(10,10,10).cuda()
# y = model(x,adj)
# print(y.size())


def preprocess(A):
    # Get size of the adjacency matrix
    size = A.size(1)
    # Get the degrees for each node
    degrees = torch.sum(A, dim=2)

    # Create diagonal matrix D from the degrees of the nodes
    D = Variable(torch.zeros(A.size(0), A.size(1), A.size(2))).cuda()
    for i in range(D.size(0)):
        D[i, :, :] = torch.diag(torch.pow(degrees[i, :], -0.5))
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    # Create A hat
    # Return A_hat
    A_normal = torch.matmul(torch.matmul(D, A), D)
    # print(A_normal)
    return A_normal


# a sequential GCN model, GCN with n layers
class GCN_generator(nn.Module):
    def __init__(self, hidden_dim):
        super(GCN_generator, self).__init__()
        # todo: add an linear_input module to map the input feature into 'hidden_dim'
        self.conv = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.act = nn.ReLU()
        # initialize
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x, teacher_force=False, adj_real=None):
        # x: batch * node_num * feature
        batch_num = x.size(0)
        node_num = x.size(1)
        adj = Variable(
            torch.eye(node_num).view(1, node_num, node_num).repeat(batch_num, 1, 1)
        ).cuda()
        adj_output = Variable(
            torch.eye(node_num).view(1, node_num, node_num).repeat(batch_num, 1, 1)
        ).cuda()

        # do GCN n times
        # todo: try if residual connections are plausible
        # todo: add higher order of adj (adj^2, adj^3, ...)
        # todo: try if norm everytim is plausible

        # first do GCN 1 time to preprocess the raw features

        # x_new = self.conv(x, adj)
        # x_new = self.act(x_new)
        # x = x + x_new

        x = self.conv(x, adj)
        x = self.act(x)

        # x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        # then do GCN rest n-1 times
        for i in range(1, node_num):
            # 1 calc prob of a new edge, output the result in adj_output
            x_last = x[:, i : i + 1, :].clone()
            x_prev = x[:, 0:i, :].clone()
            x_prev = x_prev
            x_last = x_last
            prob = x_prev @ x_last.permute(0, 2, 1)
            adj_output[:, i, 0:i] = prob.permute(0, 2, 1).clone()
            adj_output[:, 0:i, i] = prob.clone()
            # 2 update adj
            if teacher_force:
                adj = Variable(
                    torch.eye(node_num)
                    .view(1, node_num, node_num)
                    .repeat(batch_num, 1, 1)
                ).cuda()
                adj[:, 0 : i + 1, 0 : i + 1] = adj_real[:, 0 : i + 1, 0 : i + 1].clone()
            else:
                adj[:, i, 0:i] = prob.permute(0, 2, 1).clone()
                adj[:, 0:i, i] = prob.clone()
            adj = preprocess(adj)
            # print(adj)
            # print(adj.min().data[0],adj.max().data[0])
            # print(x.min().data[0],x.max().data[0])
            # 3 do graph conv, with residual connection
            # x_new = self.conv(x, adj)
            # x_new = self.act(x_new)
            # x = x + x_new

            x = self.conv(x, adj)
            x = self.act(x)

            # x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        # one = Variable(torch.ones(adj_output.size(0), adj_output.size(1), adj_output.size(2)) * 1.00).cuda().float()
        # two = Variable(torch.ones(adj_output.size(0), adj_output.size(1), adj_output.size(2)) * 2.01).cuda().float()
        # adj_output = (adj_output + one) / two
        # print(adj_output.max().data[0], adj_output.min().data[0])
        return adj_output


# #### test code ####
# print('teacher forcing')
# # print('no teacher forcing')
#
# start = time.time()
# generator = GCN_generator(hidden_dim=4)
# end = time.time()
# print('model build time', end-start)
# for run in range(10):
#     for i in [500]:
#         for batch in [1,10,100]:
#             start = time.time()
#             torch.manual_seed(123)
#             x = Variable(torch.rand(batch,i,4)).cuda()
#             adj = Variable(torch.eye(i).view(1,i,i).repeat(batch,1,1)).cuda()
#             # print('x', x)
#             # print('adj', adj)
#
#             # y = generator(x)
#             y = generator(x,True,adj)
#             # print('y',y)
#             end = time.time()
#             print('node num', i, '  batch size',batch, '  run time', end-start)


class CNN_decoder(nn.Module):
    def __init__(self, input_size, output_size, stride=2):

        super(CNN_decoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv1_1 = nn.ConvTranspose1d(
            in_channels=int(self.input_size),
            out_channels=int(self.input_size / 2),
            kernel_size=3,
            stride=stride,
        )
        self.bn1_1 = nn.BatchNorm1d(int(self.input_size / 2))
        self.deconv1_2 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 2),
            out_channels=int(self.input_size / 2),
            kernel_size=3,
            stride=stride,
        )
        self.bn1_2 = nn.BatchNorm1d(int(self.input_size / 2))
        self.deconv1_3 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 2),
            out_channels=int(self.output_size),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deconv2_1 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 2),
            out_channels=int(self.input_size / 4),
            kernel_size=3,
            stride=stride,
        )
        self.bn2_1 = nn.BatchNorm1d(int(self.input_size / 4))
        self.deconv2_2 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 4),
            out_channels=int(self.input_size / 4),
            kernel_size=3,
            stride=stride,
        )
        self.bn2_2 = nn.BatchNorm1d(int(self.input_size / 4))
        self.deconv2_3 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 4),
            out_channels=int(self.output_size),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deconv3_1 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 4),
            out_channels=int(self.input_size / 8),
            kernel_size=3,
            stride=stride,
        )
        self.bn3_1 = nn.BatchNorm1d(int(self.input_size / 8))
        self.deconv3_2 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 8),
            out_channels=int(self.input_size / 8),
            kernel_size=3,
            stride=stride,
        )
        self.bn3_2 = nn.BatchNorm1d(int(self.input_size / 8))
        self.deconv3_3 = nn.ConvTranspose1d(
            in_channels=int(self.input_size / 8),
            out_channels=int(self.output_size),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """

        :param
        x: batch * channel * length
        :return:
        """
        # hop1
        x = self.deconv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        # print(x.size())
        x_hop1 = self.deconv1_3(x)
        # print(x_hop1.size())

        # hop2
        x = self.deconv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x_hop2 = self.deconv2_3(x)
        # print(x_hop2.size())

        # hop3
        x = self.deconv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        # print(x.size())
        x_hop3 = self.deconv3_3(x)
        # print(x_hop3.size())

        return x_hop1, x_hop2, x_hop3

        # # reference code for doing residual connections
        # def _make_layer(self, block, planes, blocks, stride=1):
        #     downsample = None
        #     if stride != 1 or self.inplanes != planes * block.expansion:
        #         downsample = nn.Sequential(
        #             nn.Conv2d(self.inplanes, planes * block.expansion,
        #                       kernel_size=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(planes * block.expansion),
        #         )
        #
        #     layers = []
        #     layers.append(block(self.inplanes, planes, stride, downsample))
        #     self.inplanes = planes * block.expansion
        #     for i in range(1, blocks):
        #         layers.append(block(self.inplanes, planes))
        #
        #     return nn.Sequential(*layers)


class CNN_decoder_share(nn.Module):
    def __init__(self, input_size, output_size, stride, hops):
        super(CNN_decoder_share, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hops = hops

        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(
            in_channels=int(self.input_size),
            out_channels=int(self.input_size),
            kernel_size=3,
            stride=stride,
        )
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(
            in_channels=int(self.input_size),
            out_channels=int(self.output_size),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """

        :param
        x: batch * channel * length
        :return:
        """

        # hop1
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x_hop1 = self.deconv_out(x)
        # print(x_hop1.size())

        # hop2
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_hop2 = self.deconv_out(x)
        # print(x_hop2.size())

        # hop3
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x_hop3 = self.deconv_out(x)
        # print(x_hop3.size())

        return x_hop1, x_hop2, x_hop3


class CNN_decoder_attention(nn.Module):
    def __init__(self, input_size, output_size, stride=2):

        super(CNN_decoder_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(
            in_channels=int(self.input_size),
            out_channels=int(self.input_size),
            kernel_size=3,
            stride=stride,
        )
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(
            in_channels=int(self.input_size),
            out_channels=int(self.output_size),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.deconv_attention = nn.ConvTranspose1d(
            in_channels=int(self.input_size),
            out_channels=int(self.input_size),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn_attention = nn.BatchNorm1d(int(self.input_size))
        self.relu_leaky = nn.LeakyReLU(0.2)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """

        :param
        x: batch * channel * length
        :return:
        """
        # hop1
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop1 = self.deconv_out(x)
        x_hop1_attention = self.deconv_attention(x)
        # x_hop1_attention = self.bn_attention(x_hop1_attention)
        x_hop1_attention = self.relu(x_hop1_attention)
        x_hop1_attention = torch.matmul(
            x_hop1_attention,
            x_hop1_attention.view(
                -1, x_hop1_attention.size(2), x_hop1_attention.size(1)
            ),
        )
        # x_hop1_attention_sum = torch.norm(x_hop1_attention, 2, dim=1, keepdim=True)
        # x_hop1_attention = x_hop1_attention/x_hop1_attention_sum

        # print(x_hop1.size())

        # hop2
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop2 = self.deconv_out(x)
        x_hop2_attention = self.deconv_attention(x)
        # x_hop2_attention = self.bn_attention(x_hop2_attention)
        x_hop2_attention = self.relu(x_hop2_attention)
        x_hop2_attention = torch.matmul(
            x_hop2_attention,
            x_hop2_attention.view(
                -1, x_hop2_attention.size(2), x_hop2_attention.size(1)
            ),
        )
        # x_hop2_attention_sum = torch.norm(x_hop2_attention, 2, dim=1, keepdim=True)
        # x_hop2_attention = x_hop2_attention/x_hop2_attention_sum

        # print(x_hop2.size())

        # hop3
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop3 = self.deconv_out(x)
        x_hop3_attention = self.deconv_attention(x)
        # x_hop3_attention = self.bn_attention(x_hop3_attention)
        x_hop3_attention = self.relu(x_hop3_attention)
        x_hop3_attention = torch.matmul(
            x_hop3_attention,
            x_hop3_attention.view(
                -1, x_hop3_attention.size(2), x_hop3_attention.size(1)
            ),
        )
        # x_hop3_attention_sum = torch.norm(x_hop3_attention, 2, dim=1, keepdim=True)
        # x_hop3_attention = x_hop3_attention / x_hop3_attention_sum

        # print(x_hop3.size())

        return (
            x_hop1,
            x_hop2,
            x_hop3,
            x_hop1_attention,
            x_hop2_attention,
            x_hop3_attention,
        )


#### test code ####
# x = Variable(torch.randn(1, 256, 1)).cuda()
# decoder = CNN_decoder(256, 16).cuda()
# y = decoder(x)


class Graphsage_Encoder(nn.Module):
    def __init__(self, feature_size, input_size, layer_num):
        super(Graphsage_Encoder, self).__init__()

        self.linear_projection = nn.Linear(feature_size, input_size)

        self.input_size = input_size

        # linear for hop 3
        self.linear_3_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        self.linear_3_1 = nn.Linear(input_size * (2 ** 1), input_size * (2 ** 2))
        self.linear_3_2 = nn.Linear(input_size * (2 ** 2), input_size * (2 ** 3))
        # linear for hop 2
        self.linear_2_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        self.linear_2_1 = nn.Linear(input_size * (2 ** 1), input_size * (2 ** 2))
        # linear for hop 1
        self.linear_1_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        # linear for hop 0
        self.linear_0_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))

        self.linear = nn.Linear(input_size * (2 + 2 + 4 + 8), input_size * (16))

        self.bn_3_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))
        self.bn_3_1 = nn.BatchNorm1d(self.input_size * (2 ** 2))
        self.bn_3_2 = nn.BatchNorm1d(self.input_size * (2 ** 3))

        self.bn_2_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))
        self.bn_2_1 = nn.BatchNorm1d(self.input_size * (2 ** 2))

        self.bn_1_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))

        self.bn_0_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))

        self.bn = nn.BatchNorm1d(input_size * (16))

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, nodes_list, nodes_count_list):
        """

        :param nodes: a list, each element n_i is a tensor for node's k-i hop neighbours
                (the first nodes_hop is the furthest neighbor)
                where n_i = N * num_neighbours * features
               nodes_count: a list, each element is a list that show how many neighbours belongs to the father node
        :return:
        """

        # 3-hop feature
        # nodes original features to representations
        nodes_list[0] = Variable(nodes_list[0]).cuda()
        nodes_list[0] = self.linear_projection(nodes_list[0])
        nodes_features = self.linear_3_0(nodes_list[0])
        nodes_features = self.bn_3_0(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[0]
        # print(nodes_count,nodes_count.size())
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(
                nodes_features.size(0), nodes_count.size(1), nodes_features.size(2)
            )
        ).cuda()
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            # print(nodes_count[:,j][0],type(nodes_count[:,j][0]))
            nodes_features_farther[:, j, :] = torch.mean(
                nodes_features[:, i : i + int(nodes_count[:, j][0]), :],
                1,
                keepdim=False,
            )
            i += int(nodes_count[:, j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_3_1(nodes_features)
        nodes_features = self.bn_3_1(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(
                nodes_features.size(0), nodes_count.size(1), nodes_features.size(2)
            )
        ).cuda()
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:, j, :] = torch.mean(
                nodes_features[:, i : i + int(nodes_count[:, j][0]), :],
                1,
                keepdim=False,
            )
            i += int(nodes_count[:, j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        # print('nodes_feature',nodes_features.size())
        nodes_features = self.linear_3_2(nodes_features)
        nodes_features = self.bn_3_2(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_3 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_3.size())

        # 2-hop feature
        # nodes original features to representations
        nodes_list[1] = Variable(nodes_list[1]).cuda()
        nodes_list[1] = self.linear_projection(nodes_list[1])
        nodes_features = self.linear_2_0(nodes_list[1])
        nodes_features = self.bn_2_0(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(
                nodes_features.size(0), nodes_count.size(1), nodes_features.size(2)
            )
        ).cuda()
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:, j, :] = torch.mean(
                nodes_features[:, i : i + int(nodes_count[:, j][0]), :],
                1,
                keepdim=False,
            )
            i += int(nodes_count[:, j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_2_1(nodes_features)
        nodes_features = self.bn_2_1(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_2 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_2.size())

        # 1-hop feature
        # nodes original features to representations
        nodes_list[2] = Variable(nodes_list[2]).cuda()
        nodes_list[2] = self.linear_projection(nodes_list[2])
        nodes_features = self.linear_1_0(nodes_list[2])
        nodes_features = self.bn_1_0(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_1 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_1.size())

        # own feature
        nodes_list[3] = Variable(nodes_list[3]).cuda()
        nodes_list[3] = self.linear_projection(nodes_list[3])
        nodes_features = self.linear_0_0(nodes_list[3])
        nodes_features = self.bn_0_0(
            nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        )
        nodes_features_hop_0 = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        # print(nodes_features_hop_0.size())

        # concatenate
        nodes_features = torch.cat(
            (
                nodes_features_hop_0,
                nodes_features_hop_1,
                nodes_features_hop_2,
                nodes_features_hop_3,
            ),
            dim=2,
        )
        nodes_features = self.linear(nodes_features)
        # nodes_features = self.bn(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(
            -1, nodes_features.size(2), nodes_features.size(1)
        )
        # print(nodes_features.size())
        return nodes_features
