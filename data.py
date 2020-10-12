import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp

def Graph_load_batch(
    min_num_nodes=20,
    max_num_nodes=1000,
    name="ENZYMES",
    node_attributes=True,
    graph_labels=True,
):
    """
    load many graphs, e.g. enzymes
    load ENZYMES and PROTEIN and DD dataset
    :return: a list of graphs
    """
    print("Loading graph dataset: " + str(name))
    G = nx.Graph()
    # load data
    path = "dataset/" + name + "/"
    data_adj = np.loadtxt(path + name + "_A.txt", delimiter=",").astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path + name + "_node_attributes.txt", delimiter=",")
    data_node_label = np.loadtxt(
        path + name + "_node_labels.txt", delimiter=","
    ).astype(int)
    data_graph_indicator = np.loadtxt(
        path + name + "_graph_indicator.txt", delimiter=","
    ).astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            path + name + "_graph_labels.txt", delimiter=","
        ).astype(int)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph["label"] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if (
            G_sub.number_of_nodes() >= min_num_nodes
            and G_sub.number_of_nodes() <= max_num_nodes
        ):
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print("Loaded")
    return graphs


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# load cora, citeseer and pubmed dataset
def Graph_load(dataset="cora"):
    """
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    """
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        load = pkl.load(
            open("dataset/ind.{}.{}".format(dataset, names[i]), "rb"), encoding="latin1"
        )
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


def bfs_seq(G, start_id):
    """
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    """
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output


def encode_adj(adj, max_prev_node=10, is_full=False):
    """

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    """
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def decode_adj(adj_output):
    """
    recover to adj from adj_output
    note: here adj_output have shape (n-1)*m
    """
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i, ::-1][
            output_start:output_end
        ]  # reverse order
    adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1))
    n = adj_full.shape[0]
    adj_full[1:n, 0 : n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    """
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    """
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end - len(adj_slice) + np.amin(non_zero)

    return adj_output


def decode_adj_flexible(adj_output):
    """
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    """
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i + 1 - len(adj_output[i])
        output_end = i + 1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output) + 1, len(adj_output) + 1))
    n = adj_full.shape[0]
    adj_full[1:n, 0 : n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def test_encode_decode_adj():
    ######## code test ###########
    G = nx.ladder_graph(5)
    G = nx.grid_2d_graph(20, 20)
    G = nx.ladder_graph(200)
    G = nx.karate_club_graph()
    G = nx.connected_caveman_graph(2, 3)
    print(G.number_of_nodes())

    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    #
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]

    print("adj\n", adj)
    adj_output = encode_adj(adj, max_prev_node=5)
    print("adj_output\n", adj_output)
    adj_recover = decode_adj(adj_output, max_prev_node=5)
    print("adj_recover\n", adj_recover)
    print("error\n", np.amin(adj_recover - adj), np.amax(adj_recover - adj))

    adj_output = encode_adj_flexible(adj)
    for i in range(len(adj_output)):
        print(len(adj_output[i]))
    adj_recover = decode_adj_flexible(adj_output)
    print(adj_recover)
    print(np.amin(adj_recover - adj), np.amax(adj_recover - adj))


def encode_adj_full(adj):
    """
    return a n-1*n-1*2 tensor, the first dimension is an adj matrix, the second show if each entry is valid
    :param adj: adj matrix
    :return:
    """
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]
    adj_output = np.zeros((adj.shape[0], adj.shape[1], 2))
    adj_len = np.zeros(adj.shape[0])

    for i in range(adj.shape[0]):
        non_zero = np.nonzero(adj[i, :])[0]
        input_start = np.amin(non_zero)
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        # write adj
        adj_output[i, 0 : adj_slice.shape[0], 0] = adj_slice[
            ::-1
        ]  # put in reverse order
        # write stop token (if token is 0, stop)
        adj_output[i, 0 : adj_slice.shape[0], 1] = 1  # put in reverse order
        # write sequence length
        adj_len[i] = adj_slice.shape[0]

    return adj_output, adj_len


def decode_adj_full(adj_output):
    """
    return an adj according to adj_output
    :param
    :return:
    """
    # pick up lower tri
    adj = np.zeros((adj_output.shape[0] + 1, adj_output.shape[1] + 1))

    for i in range(adj_output.shape[0]):
        non_zero = np.nonzero(adj_output[i, :, 1])[0]  # get valid sequence
        input_end = np.amax(non_zero)
        adj_slice = adj_output[i, 0 : input_end + 1, 0]  # get adj slice
        # write adj
        output_end = i + 1
        output_start = i + 1 - input_end - 1
        adj[i + 1, output_start:output_end] = adj_slice[::-1]  # put in reverse order
    adj = adj + adj.T
    return adj


def test_encode_decode_adj_full():
    ########### code test #############
    # G = nx.ladder_graph(10)
    G = nx.karate_club_graph()
    # get bfs adj
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]

    adj_output, adj_len = encode_adj_full(adj)
    print("adj\n", adj)
    print("adj_output[0]\n", adj_output[:, :, 0])
    print("adj_output[1]\n", adj_output[:, :, 1])
    # print('adj_len\n',adj_len)

    adj_recover = decode_adj_full(adj_output)
    print("adj_recover\n", adj_recover)
    print("error\n", adj_recover - adj)
    print("error_sum\n", np.amax(adj_recover - adj), np.amin(adj_recover - adj))

class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    """
    use pytorch dataloader
    """
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print(
                "calculating max previous node, total iteration: {}".format(iteration)
            )
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print("max previous node: {}".format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros(
            (self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros(
            (self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
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
        y_batch[0 : adj_encoded.shape[0], :] = adj_encoded
        x_batch[1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        return {"x": x_batch, "y": y_batch, "len": len_batch}

    def calc_max_prev_node(self, iter=20000, topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print("iter {} times".format(i))
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
        max_prev_node = sorted(max_prev_node)[-1 * topk :]
        return max_prev_node


class Graph_sequence_sampler_pytorch_nobfs(torch.utils.data.Dataset):
    """
    use pytorch dataloader
    """
    def __init__(self, G_list, max_num_node=None):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros(
            (self.n, self.n - 1)
        )  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros(
            (self.n, self.n - 1)
        )  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.n - 1)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0 : adj_encoded.shape[0], :] = adj_encoded
        x_batch[1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        return {"x": x_batch, "y": y_batch, "len": len_batch}


class Graph_sequence_sampler_pytorch_canonical(torch.utils.data.Dataset):
    """
    use pytorch dataloader
    """
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            self.max_prev_node = self.n - 1
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros(
            (self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros(
            (self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0 : adj_encoded.shape[0], :] = adj_encoded
        x_batch[1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        return {"x": x_batch, "y": y_batch, "len": len_batch}

    def calc_max_prev_node(self, iter=20000, topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print("iter {} times".format(i))
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
        max_prev_node = sorted(max_prev_node)[-1 * topk :]
        return max_prev_node


########## 
class Graph_sequence_sampler_pytorch_nll(torch.utils.data.Dataset):
    """
    use pytorch dataloader
    """
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            adj = np.asarray(nx.to_numpy_matrix(G))
            adj_temp = self.calc_adj(adj)
            self.adj_all.extend(adj_temp)
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            # print('calculating max previous node, total iteration: {}'.format(iteration))
            # self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            # print('max previous node: {}'.format(self.max_prev_node))
            self.max_prev_node = self.n - 1
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros(
            (self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros(
            (self.n, self.max_prev_node)
        )  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        # adj_copy_matrix = np.asmatrix(adj_copy)
        # G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        # start_idx = G.number_of_nodes()-1
        # x_idx = np.array(bfs_seq(G, start_idx))
        # adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0 : adj_encoded.shape[0], :] = adj_encoded
        x_batch[1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        return {"x": x_batch, "y": y_batch, "len": len_batch}

    def calc_adj(self, adj):
        max_iter = 10000
        adj_all = [adj]
        adj_all_len = 1
        for i in range(max_iter):
            adj_copy = adj.copy()
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            add_flag = True
            for adj_exist in adj_all:
                if np.array_equal(adj_exist, adj_copy):
                    add_flag = False
                    break
            if add_flag:
                adj_all.append(adj_copy)
                adj_all_len += 1
            if adj_all_len % 10 == 0:
                print("adj found:", adj_all_len, "iter used", i)
        return adj_all
