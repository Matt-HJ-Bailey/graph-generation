import pickle
import re
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community

from typing import List

from data import Graph_load, Graph_load_batch

def citeseer_ego():
    _, _, G = Graph_load(dataset="citeseer")
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    return graphs


def caveman_special(c=2, k=20, p_path=0.1, p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)), 1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1 - p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    # print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G


def perturb(graph_list, p_del, p_add=None):
    """
    Perturb the list of graphs by adding/removing edges.
    Parameters
    ----------
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns
    -------
        A list of graphs that are perturbed from the original graphs
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        trials = np.random.binomial(1, p_del, size=G.number_of_edges())
        edges = list(G.edges())
        i = 0
        for (u, v) in edges:
            if trials[i] == 1:
                G.remove_edge(u, v)
            i += 1
        if p_add is None:
            num_nodes = G.number_of_nodes()
            p_add_est = np.sum(trials) / (
                num_nodes * (num_nodes - 1) / 2 - G.number_of_edges()
            )
        else:
            p_add_est = p_add

        nodes = list(G.nodes())
        tmp = 0
        for i in range(len(nodes)):
            u = nodes[i]
            trials = np.random.binomial(1, p_add_est, size=G.number_of_nodes())
            j = 0
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                if trials[j] == 1:
                    tmp += 1
                    G.add_edge(u, v)
                j += 1

        perturbed_graph_list.append(G)
    return perturbed_graph_list


def perturb_new(graph_list, p):
    """Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_remove_count = 0
        for (u, v) in list(G.edges()):
            if np.random.rand() < p:
                G.remove_edge(u, v)
                edge_remove_count += 1
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)


def save_prediction_histogram(y_pred_data, fname_pred, max_num_node, bin_n=20):
    bin_edge = np.linspace(1e-6, 1, bin_n + 1)
    output_pred = np.zeros((bin_n, max_num_node))
    for i in range(max_num_node):
        output_pred[:, i], _ = np.histogram(
            y_pred_data[:, i, :], bins=bin_edge, density=False
        )
        # normalize
        output_pred[:, i] /= np.sum(output_pred[:, i])
    imsave(
        fname=fname_pred,
        arr=output_pred,
        origin="upper",
        cmap="Greys_r",
        vmin=0.0,
        vmax=3.0 / bin_n,
    )


# draw a single graph G
def draw_graph(G, prefix="test"):

    plt.axis("off")
    pos = nx.get_node_attributes(G, "pos")
    if not pos:  
        pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=False, node_size=10, width=2, pos=pos)
    plt.savefig(f"figures/graph_view_{prefix}.pdf", bbox_inches="tight")
    plt.close()


# draw a list of graphs [G]
def draw_graph_list(
    G_list,
    row,
    col,
    fname="figures/test",
    layout="spring",
    is_single=False,
    k=1,
    node_size=55,
    alpha=1,
    width=1.3,
):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    plt.switch_backend("agg")
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis("off")
        if layout == "spring":
            pos = nx.spring_layout(
                G, k=k / np.sqrt(G.number_of_nodes()), iterations=100
            )
            # pos = nx.spring_layout(G)

        elif layout == "spectral":
            pos = nx.spectral_layout(G)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_size,
                node_color="#336699",
                alpha=1,
                linewidths=0,
                font_size=0,
            )
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=1.5,
                node_color="#336699",
                alpha=1,
                linewidths=0.2,
                font_size=1.5,
            )
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(fname + ".png", dpi=600)
    plt.close()


def get_graph(adj):
    """
    get a graph from zero-padded adj
    :param adj:
    :return:
    """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def save_graph_list(G_list: List[nx.Graph], fname: str) -> None:
    """
    save a list of graphs
    Parameters
    ----------
        G_list: The list of graphs to save
        fname: the filename to save to as a pickle file
    Returns
    -------
        None
    """
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def pick_connected_component(G: nx.Graph) -> nx.Graph:
    """
    pick the first connected component
    Parameters
    ----------
        G_list: The graph to get the first connected component from
    Returns
    -------
        subgraph: the first connected component
    """
    node_list = nx.node_connected_component(G, 0)
    return G.subgraph(node_list)


def pick_connected_component_new(G):
    adj_list = G.adjacency_list()
    for idx, adj in enumerate(adj_list):
        idx_min = min(adj)
        if idx < idx_min and idx >= 1:
            break
    node_list = list(range(id))  # only include node prior than node "id"
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


# load a list of graphs
def load_graph_list(fname, is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    for i in range(len(graph_list)):
        edges_with_selfloops = graph_list[i].selfloop_edges()
        if len(edges_with_selfloops) > 0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(
                nx.connected_component_subgraphs(graph_list[i]), key=len
            )
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list


def export_graphs_to_txt(g_list, output_filename_prefix):
    i = 0
    for G in g_list:
        f = open(output_filename_prefix + "_" + str(i) + ".txt", "w+")
        for (u, v) in G.edges():
            idx_u = G.nodes().index(u)
            idx_v = G.nodes().index(v)
            f.write(f"{idx_u}\t{idx_v}\n")
        i += 1


def snap_txt_output_to_nx(in_fname):
    G = nx.Graph()
    with open(in_fname, "r") as f:
        for line in f:
            if not line[0] == "#":
                splitted = re.split("[ \t]", line)

                # self loop might be generated, but should be removed
                u = int(splitted[0])
                v = int(splitted[1])
                if not u == v:
                    G.add_edge(int(u), int(v))
    return G


def test_perturbed():
    graphs = []
    for i in range(100, 101):
        for j in range(4, 5):
            for k in range(500):
                graphs.append(nx.barabasi_albert_graph(i, j))
    g_perturbed = perturb(graphs, 0.9)
    print([g.number_of_edges() for g in graphs])
    print([g.number_of_edges() for g in g_perturbed])

def test_graph_load_DD():
    graphs, max_num_nodes = Graph_load_batch(
        min_num_nodes=10, name="DD", node_attributes=False, graph_labels=True
    )
    random.shuffle(graphs)
    plt.switch_backend("agg")
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig("figures/test.png")
    plt.close()
    row = 4
    col = 4
    draw_graph_list(graphs[0 : row * col], row=row, col=col, fname="figures/test")
    print("max num nodes", max_num_nodes)
    
if __name__ == "__main__":
    # test_perturbed()
    # graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_community4_4_128_train_0.dat')
    # graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_community4_4_128_pred_2500_1.dat')
    graphs = load_graph_list("eval_results/mmsb/" + "community41.dat")

    for i in range(0, 160, 16):
        draw_graph_list(graphs[i : i + 16], 4, 4, fname="figures/community4_" + str(i))
