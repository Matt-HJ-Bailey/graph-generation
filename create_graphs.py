import random

import networkx as nx
import numpy as np

import os
import pickle as pkl

from utils import caveman_special, n_community, perturb_new, Graph_load_batch, Graph_load

def generate_ladder_graphs(min_size:int = 100, max_size:int = 201):
    return [nx.ladder_graph(i) for i in range(min_size, max_size)]
   
def create(args):

    # synthetic graphs
    if args.graph_type == "ladder":
        graphs = generate_ladder_graphs(100, 201)
        args.max_prev_node = 10
    elif args.graph_type == "ladder_small":
        graphs = generate_ladder_graphs(2, 11)
        args.max_prev_node = 10
    elif args.graph_type == "tree":
        graphs = []
        for i in range(2, 5):
            for j in range(3, 5):
                graphs.append(nx.balanced_tree(i, j))
        args.max_prev_node = 256
    elif args.graph_type == "caveman":
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graphs.append(caveman_special(i, j, p_edge=0.3))
        args.max_prev_node = 100
    elif args.graph_type == "caveman_small":
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(20):
                    graphs.append(caveman_special(i, j, p_edge=0.8))  # default 0.8
        args.max_prev_node = 20
    elif args.graph_type == "caveman_small_single":
        graphs = []
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graphs.append(caveman_special(i, j, p_edge=0.5))
        args.max_prev_node = 20
    elif args.graph_type.startswith("community"):
        num_communities = int(args.graph_type[-1])
        print("Creating dataset with ", num_communities, " communities")
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        for k in range(3000):
            graphs.append(n_community(c_sizes, p_inter=0.01))
        args.max_prev_node = 80
    elif args.graph_type == "grid":
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 40
    elif args.graph_type == "grid_small":
        graphs = []
        for i in range(2, 5):
            for j in range(2, 6):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 15
    elif args.graph_type == "barabasi":
        graphs = []
        for i in range(100, 200):
            for j in range(4, 5):
                for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i, j))
        args.max_prev_node = 130
    elif args.graph_type == "barabasi_small":
        graphs = []
        for i in range(4, 21):
            for j in range(3, 4):
                for k in range(10):
                    graphs.append(nx.barabasi_albert_graph(i, j))
        args.max_prev_node = 20
    elif args.graph_type == "grid_big":
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 90

    elif "barabasi_noise" in args.graph_type:
        graphs = []
        for i in range(100, 101):
            for j in range(4, 5):
                for k in range(500):
                    graphs.append(nx.barabasi_albert_graph(i, j))
        graphs = perturb_new(graphs, p=args.noise / 10.0)
        args.max_prev_node = 99

    # real graphs
    elif args.graph_type == "enzymes":
        graphs = Graph_load_batch(min_num_nodes=10, name="ENZYMES")
        args.max_prev_node = 25
    elif args.graph_type == "enzymes_small":
        graphs_raw = Graph_load_batch(min_num_nodes=10, name="ENZYMES")
        graphs = []
        for G in graphs_raw:
            if G.number_of_nodes() <= 20:
                graphs.append(G)
        args.max_prev_node = 15
    elif args.graph_type == "protein":
        graphs = Graph_load_batch(min_num_nodes=20, name="PROTEINS_full")
        args.max_prev_node = 80
    elif args.graph_type == "DD":
        graphs = Graph_load_batch(
            min_num_nodes=100,
            max_num_nodes=500,
            name="DD",
            node_attributes=False,
            graph_labels=True,
        )
        args.max_prev_node = 230
    elif args.graph_type == "citeseer":
        _, _, G = Graph_load(dataset="citeseer")
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
        args.max_prev_node = 250
    elif args.graph_type == "citeseer_small":
        _, _, G = Graph_load(dataset="citeseer")
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        random.shuffle(graphs)
        graphs = graphs[0:200]
        args.max_prev_node = 15
    elif args.graph_type == "collagen":
        graphs = []
        for fname in os.listdir(os.path.join("./", "dataset", "Collagen")):
            base, ext = os.path.splitext(fname)
            print(fname, base, ext)
            if ext != ".pkl":
                continue
            with open(os.path.join("./", "dataset", "Collagen", fname), "rb") as fi:
                graphs.append(pkl.load(fi))
        args.max_num_node = 800
        args.max_prev_node = 8
    elif args.graph_type == "edge_graphs":
        graphs = []
        for fname in os.listdir(os.path.join("./", "dataset", "edge_graphs")):
            base, ext = os.path.splitext(fname)
            print(fname, base, ext)
            if ext != ".pkl":
                continue
            with open(os.path.join("./", "dataset", "edge_graphs", fname), "rb") as fi:
                graphs.append(pkl.load(fi))
        args.max_num_node = max(len(graph) for graph in graphs)
        args.max_prev_node = 8
    elif args.graph_type == "ring_graphs":
        graphs = []
        for fname in os.listdir(os.path.join("./", "dataset", "ring_graphs")):
            base, ext = os.path.splitext(fname)
            print(fname, base, ext)
            if ext != ".pkl":
                continue
            with open(os.path.join("./", "dataset", "ring_graphs", fname), "rb") as fi:
                graphs.append(pkl.load(fi))
        args.max_num_node = max(len(graph) for graph in graphs)
        args.max_prev_node = 20
    random.shuffle(graphs)
    return graphs
