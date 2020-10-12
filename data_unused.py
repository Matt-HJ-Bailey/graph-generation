# -*- coding: utf-8 -*-
"""
Below are codes not used in current version
they are based on pytorch default data loader,
we should consider reimplement them in current datasets,
since they are more efficient


Created on Fri Oct  9 14:59:19 2020

@author: Matt
"""

import random

import torch
import torch.nn as nn
import networkx as nx
import numpy as np


from data import bfs_seq, encode_adj, encode_adj_flexible

class Graph_sequence_sampler_truncate:
    """
    the output will truncate according to the max_prev_node
    """

    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node=25):
        self.batch_size = batch_size
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

    def sample(self):
        # batch, length, feature
        x_batch = np.zeros(
            (self.batch_size, self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        y_batch = np.zeros(
            (self.batch_size, self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        len_batch = np.zeros(self.batch_size)
        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            len_batch[i] = adj_copy.shape[0]
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0 : adj_encoded.shape[0], :] = adj_encoded
            x_batch[i, 1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        # sort in descending order
        len_batch_order = np.argsort(len_batch)[::-1]
        len_batch = len_batch[len_batch_order]
        x_batch = x_batch[len_batch_order, :, :]
        y_batch = y_batch[len_batch_order, :, :]

        return (
            torch.from_numpy(x_batch).float(),
            torch.from_numpy(y_batch).float(),
            len_batch.astype("int").tolist(),
        )

    def calc_max_prev_node(self, iter):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 10) == 0:
                print(i)
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max(
                [len(adj_encoded[i]) for i in range(len(adj_encoded))]
            )
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-100:]
        return max_prev_node

class Graph_sequence_sampler_fast:
    """
    only output y_batch (which is needed in batch version of new model)
    """
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node=25):
        self.batch_size = batch_size
        self.G_list = G_list
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

    def sample(self):
        # batch, length, feature
        y_batch = np.zeros(
            (self.batch_size, self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('graph size',adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # get the feature for the permuted G
            # encode adj
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)

            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0 : adj_encoded.shape[0], :] = adj_encoded

        return torch.from_numpy(y_batch).float()


class Graph_sequence_sampler_flexible:
    """
    output size is flexible (using list to represent), batch size is 1
    """
    def __init__(self, G_list):
        self.G_list = G_list
        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

        self.y_batch = []

    def sample(self):
        # generate input x, y pairs
        # first sample and get a permuted adj
        adj_idx = np.random.randint(len(self.adj_all))
        adj_copy = self.adj_all[adj_idx].copy()
        # print('graph size',adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        # get the feature for the permuted G

        # encode adj
        adj_encoded = encode_adj_flexible(adj_copy.copy())

        # get x and y and adj
        # for small graph the rest are zero padded
        self.y_batch = adj_encoded
        return self.y_batch, adj_copy


def preprocess(A):
    """
     preprocess the adjacency matrix
     potential use: an encoder along with the GraphRNN decoder
    :param A: DESCRIPTION
    :type A: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = np.sum(A, axis=1) + 1

    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(np.power(degrees, -0.5).flatten())
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    A_normal = np.dot(np.dot(D, A_hat), D)
    return A_normal



class Graph_sequence_sampler_bfs_permute_truncate_multigraph:
    """
    truncate the output seqence to save representation, and allowing for infinite generation
    now having a list of graphs
    """
    def __init__(
        self, G_list, max_node_num=25, batch_size=4, max_prev_node=25, feature=None
    ):
        self.batch_size = batch_size
        self.G_list = G_list
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
        self.has_feature = feature

    def sample(self):

        # batch, length, feature
        # self.x_batch = np.ones((self.batch_size, self.n - 1, self.max_prev_node))
        x_batch = np.zeros(
            (self.batch_size, self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        # self.x_batch[:,0,:] = np.ones((self.batch_size, self.max_prev_node))  # first input is all ones
        # batch, length, feature
        y_batch = np.zeros(
            (self.batch_size, self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        # batch, length, length
        adj_batch = np.zeros(
            (self.batch_size, self.n, self.n)
        )  # here zeros are padded for small graph
        # batch, size, size
        adj_norm_batch = np.zeros(
            (self.batch_size, self.n, self.n)
        )  # here zeros are padded for small graph
        # batch, size, feature_len: degree and clustering coefficient
        if self.has_feature is None:
            feature_batch = np.zeros(
                (self.batch_size, self.n, self.n)
            )  # use one hot feature
        else:
            feature_batch = np.zeros((self.batch_size, self.n, 2))

        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # get the feature for the permuted G
            node_list = [G.nodes()[i] for i in x_idx]
            feature_degree = np.array(list(G.degree(node_list).values()))[:, np.newaxis]
            feature_clustering = np.array(
                list(nx.clustering(G, nodes=node_list).values())
            )[:, np.newaxis]

            # encode adj
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)

            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0 : adj_encoded.shape[0], :] = adj_encoded
            x_batch[i, 1 : adj_encoded.shape[0] + 1, :] = adj_encoded
            adj_batch[i, 0 : adj_copy.shape[0], 0 : adj_copy.shape[0]] = adj_copy
            adj_copy_norm = preprocess(adj_copy)

            adj_norm_batch[
                i, 0 : adj_copy.shape[0], 0 : adj_copy.shape[0]
            ] = adj_copy_norm

            if self.has_feature is None:
                feature_batch[i, 0 : adj_copy.shape[0], 0 : adj_copy.shape[0]] = np.eye(
                    adj_copy.shape[0]
                )
            else:
                feature_batch[i, 0 : adj_copy.shape[0], :] = np.concatenate(
                    (feature_degree, feature_clustering), axis=1
                )

        return (
            torch.from_numpy(x_batch).float(),
            torch.from_numpy(y_batch).float(),
            torch.from_numpy(adj_batch).float(),
            torch.from_numpy(adj_norm_batch).float(),
            torch.from_numpy(feature_batch).float(),
        )


# generate own synthetic dataset
def Graph_synthetic(seed):
    G = nx.Graph()
    np.random.seed(seed)
    base = np.repeat(np.eye(5), 20, axis=0)
    rand = np.random.randn(100, 5) * 0.05
    node_features = base + rand

    # # print('node features')
    # for i in range(node_features.shape[0]):
    #     print(np.around(node_features[i], decimals=4))

    node_distance_l1 = np.ones((node_features.shape[0], node_features.shape[0]))
    node_distance_np = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        for j in range(node_features.shape[0]):
            if i != j:
                node_distance_l1[i, j] = np.sum(
                    np.abs(node_features[i] - node_features[j])
                )
                # print('node distance', node_distance_l1[i,j])
                node_distance_np[i, j] = 1 / np.sum(
                    np.abs(node_features[i] - node_features[j]) ** 2
                )

    print("node distance max", np.max(node_distance_l1))
    print("node distance min", np.min(node_distance_l1))
    node_distance_np_sum = np.sum(node_distance_np, axis=1, keepdims=True)
    embedding_dist = node_distance_np / node_distance_np_sum

    # generate the graph
    average_degree = 9
    for i in range(node_features.shape[0]):
        for j in range(i + 1, embedding_dist.shape[0]):
            p = np.random.rand()
            if p < embedding_dist[i, j] * average_degree:
                G.add_edge(i, j)

    G.remove_nodes_from(nx.isolates(G))
    print("num of nodes", G.number_of_nodes())
    print("num of edges", G.number_of_edges())

    G_deg = nx.degree_histogram(G)
    G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
    print("average degree", sum(G_deg_sum) / G.number_of_nodes())
    print("average path length", nx.average_shortest_path_length(G))
    print("diameter", nx.diameter(G))
    G_cluster = sorted(list(nx.clustering(G).values()))
    print("average clustering coefficient", sum(G_cluster) / len(G_cluster))
    print("Graph generation complete!")
    # node_features = np.concatenate((node_features, np.zeros((1,node_features.shape[1]))),axis=0)

    return G, node_features

# return adj and features from a single graph
class GraphDataset_adj(torch.utils.data.Dataset):
    """Graph Dataset"""

    def __init__(self, G, features=None):
        self.G = G
        self.n = G.number_of_nodes()
        adj = np.asarray(nx.to_numpy_matrix(self.G))

        # permute adj
        subgraph_idx = np.random.permutation(self.n)
        # subgraph_idx = np.arange(self.n)
        adj = adj[np.ix_(subgraph_idx, subgraph_idx)]

        self.adj = torch.from_numpy(adj + np.eye(len(adj))).float()
        self.adj_norm = torch.from_numpy(preprocess(adj)).float()
        if features is None:
            self.features = torch.Tensor(self.n, self.n)
            self.features = nn.init.eye(self.features)
        else:
            features = features[subgraph_idx, :]
            self.features = torch.from_numpy(features).float()
        print("embedding size", self.features.size())

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = {"adj": self.adj, "adj_norm": self.adj_norm, "features": self.features}
        return sample

# 
class GraphDataset_adj_batch(torch.utils.data.Dataset):
    """
    Graph Dataset
    return adj and features from a list of graphs
    """

    def __init__(self, graphs, has_feature=True, num_nodes=20):
        self.graphs = graphs
        self.has_feature = has_feature
        self.num_nodes = num_nodes

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        adj_raw = np.asarray(nx.to_numpy_matrix(self.graphs[idx]))
        np.fill_diagonal(adj_raw, 0)  # in case the self connection already exists

        # sample num_nodes size subgraph
        subgraph_idx = np.random.permutation(adj_raw.shape[0])[0 : self.num_nodes]
        adj_raw = adj_raw[np.ix_(subgraph_idx, subgraph_idx)]

        adj = torch.from_numpy(adj_raw + np.eye(len(adj_raw))).float()
        adj_norm = torch.from_numpy(preprocess(adj_raw)).float()
        adj_raw = torch.from_numpy(adj_raw).float()
        if self.has_feature:
            dictionary = nx.get_node_attributes(self.graphs[idx], "feature")
            features = np.zeros((self.num_nodes, list(dictionary.values())[0].shape[0]))
            for i in range(self.num_nodes):
                features[i, :] = list(dictionary.values())[subgraph_idx[i]]
            # normalize
            features -= np.mean(features, axis=0)
            epsilon = 1e-6
            features /= np.std(features, axis=0) + epsilon
            features = torch.from_numpy(features).float()
        else:
            n = self.num_nodes
            features = torch.Tensor(n, n)
            features = nn.init.eye(features)

        sample = {
            "adj": adj,
            "adj_norm": adj_norm,
            "features": features,
            "adj_raw": adj_raw,
        }
        return sample


# 
class GraphDataset_adj_batch_1(torch.utils.data.Dataset):
    """
    Graph Dataset,
    return adj and features from a list of graphs
    batch size = 1, so that graphs can have various size each time
    """

    def __init__(self, graphs, has_feature=True):
        self.graphs = graphs
        self.has_feature = has_feature

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        adj_raw = np.asarray(nx.to_numpy_matrix(self.graphs[idx]))
        np.fill_diagonal(adj_raw, 0)  # in case the self connection already exists
        n = adj_raw.shape[0]
        # give a permutation
        subgraph_idx = np.random.permutation(n)
        # subgraph_idx = np.arange(n)

        adj_raw = adj_raw[np.ix_(subgraph_idx, subgraph_idx)]

        adj = torch.from_numpy(adj_raw + np.eye(len(adj_raw))).float()
        adj_norm = torch.from_numpy(preprocess(adj_raw)).float()

        if self.has_feature:
            dictionary = nx.get_node_attributes(self.graphs[idx], "feature")
            features = np.zeros((n, list(dictionary.values())[0].shape[0]))
            for i in range(n):
                features[i, :] = list(dictionary.values())[i]
            features = features[subgraph_idx, :]
            # normalize
            features -= np.mean(features, axis=0)
            epsilon = 1e-6
            features /= np.std(features, axis=0) + epsilon
            features = torch.from_numpy(features).float()
        else:
            features = torch.Tensor(n, n)
            features = nn.init.eye(features)

        sample = {"adj": adj, "adj_norm": adj_norm, "features": features}
        return sample


class GraphDataset(torch.utils.data.Dataset):
    """Graph Dataset,
    get one node at a time, for a single graph"""

    def __init__(
        self,
        G,
        hops=1,
        max_degree=5,
        vocab_size=35,
        embedding_dim=35,
        embedding=None,
        shuffle_neighbour=True,
    ):
        self.G = G
        self.shuffle_neighbour = shuffle_neighbour
        self.hops = hops
        self.max_degree = max_degree
        if embedding is None:
            self.embedding = torch.Tensor(vocab_size, embedding_dim)
            self.embedding = nn.init.eye(self.embedding)
        else:
            self.embedding = torch.from_numpy(embedding).float()
        print("embedding size", self.embedding.size())

    def __len__(self):
        return len(self.G.nodes())

    def __getitem__(self, idx):
        idx = idx + 1
        idx_list = [idx]
        node_list = [self.embedding[idx].view(-1, self.embedding.size(1))]
        node_count_list = []
        for i in range(self.hops):
            # sample this hop
            adj_list = np.array([])
            adj_count_list = np.array([])
            for idx in idx_list:
                if self.shuffle_neighbour:
                    adj_list_new = list(self.G.adj[idx - 1])
                    random.shuffle(adj_list_new)
                    adj_list_new = np.array(adj_list_new) + 1
                else:
                    adj_list_new = np.array(list(self.G.adj[idx - 1])) + 1
                adj_count_list_new = np.array([len(adj_list_new)])
                adj_list = np.concatenate((adj_list, adj_list_new), axis=0)
                adj_count_list = np.concatenate(
                    (adj_count_list, adj_count_list_new), axis=0
                )
            # print(i, adj_list)
            # print(i, embedding(Variable(torch.from_numpy(adj_list)).long()))
            index = torch.from_numpy(adj_list).long()
            adj_list_emb = self.embedding[index]
            node_list.append(adj_list_emb)
            node_count_list.append(adj_count_list)
            idx_list = adj_list

        # padding, used as target
        idx_list = [idx]
        node_list_pad = [self.embedding[idx].view(-1, self.embedding.size(1))]
        node_count_list_pad = []
        node_adj_list = []
        for i in range(self.hops):
            adj_list = np.zeros(self.max_degree ** (i + 1))
            adj_count_list = np.ones(self.max_degree ** (i)) * self.max_degree
            for j, idx in enumerate(idx_list):
                if idx == 0:
                    adj_list_new = np.zeros(self.max_degree)
                else:
                    if self.shuffle_neighbour:
                        adj_list_new = list(self.G.adj[idx - 1])
                        # random.shuffle(adj_list_new)
                        adj_list_new = np.array(adj_list_new) + 1
                    else:
                        adj_list_new = np.array(list(self.G.adj[idx - 1])) + 1
                start_idx = j * self.max_degree
                incre_idx = min(self.max_degree, adj_list_new.shape[0])
                adj_list[start_idx : start_idx + incre_idx] = adj_list_new[:incre_idx]
            index = torch.from_numpy(adj_list).long()
            adj_list_emb = self.embedding[index]
            node_list_pad.append(adj_list_emb)
            node_count_list_pad.append(adj_count_list)
            idx_list = adj_list
            # calc adj matrix
            node_adj = torch.zeros(index.size(0), index.size(0))
            for first in range(index.size(0)):
                for second in range(first, index.size(0)):
                    if index[first] == index[second]:
                        node_adj[first, second] = 1
                        node_adj[second, first] = 1
                    elif self.G.has_edge(index[first], index[second]):
                        node_adj[first, second] = 0.5
                        node_adj[second, first] = 0.5
            node_adj_list.append(node_adj)

        node_list = list(reversed(node_list))
        node_count_list = list(reversed(node_count_list))
        node_list_pad = list(reversed(node_list_pad))
        node_count_list_pad = list(reversed(node_count_list_pad))
        node_adj_list = list(reversed(node_adj_list))
        sample = {
            "node_list": node_list,
            "node_count_list": node_count_list,
            "node_list_pad": node_list_pad,
            "node_count_list_pad": node_count_list_pad,
            "node_adj_list": node_adj_list,
        }
        return sample
