import dgl
import pickle
import os.path as osp
import numpy as np
import random
import torch

from tqdm import tqdm
from utils import load_labels
from scipy import sparse
from collections import defaultdict

def load_data(file_dir, dataset_info, args):
    dataset, rels_type, tar_type, comm_rels, diff_rels = dataset_info

    with open(osp.join(file_dir, dataset, 'raw', 'edges.pkl'), 'rb') as f:
        edges_data = pickle.load(f)

    # get type node dict and node type dict
    node_type_dict = get_node_type_dict(edges_data, rels_type)

    # generate global graph
    for i, edge in enumerate(edges_data):
        if i == 0:
            global_A = edge
        else:
            global_A += edge

    global_G = dgl.from_scipy(sparse.csr_matrix(global_A))
    global_G = dgl.remove_self_loop(global_G)
    global_G = dgl.add_self_loop(global_G)

    # get d_graph
    d_graph = None
    d_graph_file_name = osp.join(file_dir, dataset, 'raw', 'd_graph_' + str(args.walk_p) 
                              + "walk_p_" + str(args.walk_q) + "walk_q.pkl")
    if osp.exists(d_graph_file_name):
        with open(d_graph_file_name, 'rb') as f:
            d_graph = pickle.load(f)
    else:
        d_graph = generate_d_graph(global_G, tar_type, node_type_dict, args)
        with open(d_graph_file_name, 'wb') as f:
            pickle.dump(d_graph, f)

    # get node features
    node_features_filename = osp.join(file_dir, dataset, 'raw', "node2vec_embeddings.txt")
    node_features = np.loadtxt(node_features_filename)
    node_features = node_features[np.argsort(node_features[:, 0], ), :]
    feats = node_features[:, 1:]

    # get label data
    spt_data, qry_nodes, qry_labels = load_labels(file_dir, dataset, tar_type, args.ood_type)

    # generate tasks
    tasks = []
    for _ in tqdm(range(args.n_tasks)):
        if args.ood_type == 'no_shift':
            spt_nodes, spt_labels = spt_data
        else:
            spt_nodes, spt_labels = spt_data[random.choice(list(spt_data.keys()))]

        selected_classes = np.random.choice(get_meet_labels(spt_nodes, spt_labels, args.k_spt), args.n_way, replace=False)
        tasks.append([generate_meta_set(global_G, d_graph, selected_classes, spt_nodes, node_type_dict, 
                                        feats, spt_labels, comm_rels, diff_rels, args.k_spt, args),
                      generate_meta_set(global_G, d_graph, selected_classes, qry_nodes, node_type_dict, 
                                        feats, qry_labels, comm_rels, diff_rels, args.k_qry, args)])
    return tasks, feats.shape[1]

def get_meet_labels(nodes, labels, n_sample):
    label_count = {}
    for node, label in zip(nodes, labels):
        if label not in label_count.keys():
            label_count[label] = []
        label_count[label].append(node)
    
    # return all labels that have more than n_sample nodes
    meet_labels = []
    for label, nodes in label_count.items():
        if len(nodes) >= n_sample:
            meet_labels.append(label)
    return meet_labels

def generate_meta_set(G, d_graph, classes, nodes, node_type_dict, feats, labels, comm_rels, diff_rels, n_sel_nodes, args):
    label_index = reindex_labels(nodes, classes, labels)
    x_data = {}

    # 2 hop subgraph
    res_2_hop = None
    while not res_2_hop:
        x, y = [], []
        for l_id in range(len(label_index)):
            x.extend(random.sample(label_index[l_id], n_sel_nodes))
            y.extend(np.ones(n_sel_nodes, dtype=int) * l_id)
        res_2_hop = generate_2hop_task(G, x, feats, node_type_dict, comm_rels, diff_rels, args.n_sample)

    x_data["2_hop"] = res_2_hop

    # random walk subgraph
    res_random_walk = None
    while not res_random_walk:
        x, y = [], []
        for l_id in range(len(label_index)):
            x.extend(random.sample(label_index[l_id], n_sel_nodes))
            y.extend(np.ones(n_sel_nodes, dtype=int) * l_id)
        walk_start_list = torch.LongTensor(x).repeat(args.walk_repeat)
        sample_walks = dgl.sampling.random_walk(G, walk_start_list, length=args.walk_length)[0].numpy()
        res_random_walk = generate_task_from_walks(G, x, sample_walks, feats, node_type_dict, comm_rels, diff_rels)
    x_data["random_walk"] = res_random_walk

    # d_graph subgraph
    x_data["d_graph"] = generate_dgraph_task(G, x, feats, node_type_dict, comm_rels, diff_rels, d_graph, args)

    return x_data, torch.LongTensor(y)

def generate_dgraph_task(G, x, feats, node_type_dict, comm_rels, diff_rels, d_graph, args):
    FIRST_TRAVEL_KEY = 'first_travel_key'
    NEIGHBORS_KEY = 'neighbors'
    PROBABILITIES_KEY = 'probabilities'

    walk_start_list = [x for _ in range(args.walk_repeat)]
    walk_start_list = np.array(walk_start_list).flatten()
    shuffled_nodes = list(walk_start_list)
    random.shuffle(shuffled_nodes)

    walks = list()

    for source in shuffled_nodes:
        walk = [source]
        while len(walk) < args.walk_length:
            walk_options = d_graph[walk[-1]].get(NEIGHBORS_KEY, None)
            if len(walk) == 1:  # For the first step
                probabilities = d_graph[walk[-1]][FIRST_TRAVEL_KEY]
                walk_to = random.choices(walk_options, weights=probabilities)[0]
            else:
                probabilities = d_graph[walk[-1]][PROBABILITIES_KEY][walk[-2]]
                walk_to = random.choices(walk_options, weights=probabilities)[0]
            walk.append(walk_to)
        walks.append(walk)

    return generate_task_from_walks(G, x, walks, feats, node_type_dict, comm_rels, diff_rels)

def generate_task_from_walks(G, x, sample_walks, feats, node_type_dict, comm_rels, diff_rels):
    walk_nodes = []
    walk_edges = []

    for walk in sample_walks:
        for i in range(len(walk)-1):
            walk_nodes.append(walk[i])
            walk_edges.append((walk[i], walk[i+1]))
            walk_edges.append((walk[i+1], walk[i]))
        walk_nodes.append(walk[i+1])

    # delete duplicate nodes and edges and sort them
    walk_nodes = sorted(list(set(walk_nodes)))
    walk_edges = sorted(list(set(walk_edges)))

    subg = G.subgraph(walk_nodes)
    h_c = list(subg.ndata[dgl.NID].numpy())
    mapping = dict(zip(h_c, list(range(len(h_c)))))  #old 2 new

    new_node_type_dict = {}
    for node in mapping:
        new_node_type_dict[mapping[node]] = node_type_dict[node]

    sub_feats = torch.FloatTensor(feats[h_c])

    rows, cols = [], []
    for src, dst in walk_edges:
        rows.append(mapping[src])
        cols.append(mapping[dst])

    comm_adjs = get_adjs(comm_rels, rows, cols, new_node_type_dict, len(walk_nodes))
    diff_adjs = get_adjs(diff_rels, rows, cols, new_node_type_dict, len(walk_nodes))

    if comm_adjs is None or diff_adjs is None:
        return None

    return np.array([mapping[node] for node in x]), comm_adjs, diff_adjs, sub_feats

           
def generate_2hop_task(G, nodes, feats, node_type_dict, comm_rels, diff_rels, n_sample=100):
    # get 2hop neighbors
    h_hops_neighbor = []
    for node in nodes:
        f_hop = [n.item() for n in G.in_edges(node)[0]]
        if len(f_hop) > n_sample:
            f_hop = random.sample(f_hop, n_sample)

        # 2_hop neighbor
        n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
        n_l = sum(n_l,[])
        if len(n_l) > n_sample:
            n_l = random.sample(n_l, n_sample)

        h_hops_neighbor.extend(list(set(n_l + f_hop + [node])))

    h_hops_neighbor = np.unique(np.array(h_hops_neighbor))
    subg = G.subgraph(h_hops_neighbor)
    h_c = list(subg.ndata[dgl.NID].numpy())
    mapping = dict(zip(h_c, list(range(len(h_c)))))  #old 2 new

    new_node_type_dict = {}
    for node in mapping:
        new_node_type_dict[mapping[node]] = node_type_dict[node]

    sub_feats = torch.FloatTensor(feats[h_c])
    sub_nodes = list(subg.nodes().numpy())
    rows, cols = subg.edges()
    rows, cols = rows.numpy(), cols.numpy()
    comm_adjs = get_adjs(comm_rels, rows, cols, new_node_type_dict, len(sub_nodes))
    diff_adjs = get_adjs(diff_rels, rows, cols, new_node_type_dict, len(sub_nodes))

    if comm_adjs is None or diff_adjs is None:
        return None
    
    return np.array([mapping[node] for node in nodes]), comm_adjs, diff_adjs, sub_feats

def get_adjs(rels, rows, cols, node_type_dict, n_nodes):
    if len(rels) == 0:
        adj = torch.sparse_coo_tensor(torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]), torch.ones(n_nodes), (n_nodes, n_nodes))
        adj = adj.to_dense().float()
        return [adj]

    edges = [[] for _ in range(len(rels))]
    for src, dst in zip(rows, cols):
        for rel_id in range(len(rels)):
            if node_type_dict[src] == rels[rel_id][0] and node_type_dict[dst] == rels[rel_id][-1]:
                edges[rel_id].append([src, dst])

    if min([len(i) for i in edges]) == 0:
        return None

    adjs = []
    for sub_edges in edges:
        sub_row, sub_col = torch.LongTensor(sub_edges).T
        adj = torch.sparse_coo_tensor(torch.stack([sub_row, sub_col]), torch.ones_like(sub_row), (n_nodes, n_nodes))
        adj = adj.to_dense().float()
        adjs.append(adj)

    return adjs

def get_node_type_dict(edges_data, rels_type):
    typed_node_dict = {x: [] for x in set(sum([i.split("-") for i in rels_type], []))}

    for idx, edge in enumerate(edges_data):
        type1, type2 = rels_type[idx].split("-")
        typed_node_dict[type1] = typed_node_dict[type1] + list(edge.nonzero()[0])
        typed_node_dict[type2] = typed_node_dict[type2] + list(edge.nonzero()[1])

    for key in typed_node_dict.keys():
        typed_node_dict[key] = sorted(list(set(typed_node_dict[key])))

    node_type_dict = {}
    for type, nodes in typed_node_dict.items():
        for node in nodes:
            node_type_dict[node] = type

    for i in range(edges_data[0].shape[0]):
        if i not in node_type_dict.keys():
            for key, value in typed_node_dict.items():
                if value[0] <= i <= value[-1]:
                    node_type_dict[i] = key
    
    return node_type_dict

def reindex_labels(nodes, classes, labels):
    label_dict = {}
    for label in classes:
        label_dict[label] = len(label_dict)
    
    # group nodes with new label
    label_index = [[] for _ in label_dict.keys()]
    for node_idx in range(len(nodes)):
        if labels[node_idx] in label_dict.keys():
            label_index[label_dict[labels[node_idx]]].append(nodes[node_idx])
    return label_index 


def generate_d_graph(global_G, tar_type, node_type_dict, args):
    global_G = global_G.remove_self_loop()
    global_G = global_G.add_self_loop()
    global_G = global_G.to_networkx()
    
    nodes_generator = global_G.nodes()

    value_p = args.walk_p #target type
    value_q = args.walk_q #non-target type

    PROBABILITIES_KEY = 'probabilities'
    FIRST_TRAVEL_KEY = 'first_travel_key'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'

    d_graph = defaultdict(dict)

    for source in tqdm(nodes_generator):
        if PROBABILITIES_KEY not in d_graph[source]:
            d_graph[source][PROBABILITIES_KEY] = dict()
            
        for current_node in global_G.neighbors(source):
            if PROBABILITIES_KEY not in d_graph[current_node]:
                d_graph[current_node][PROBABILITIES_KEY] = dict()

            unnormalized_weights = list()
            d_neighbors = list()

            # Calculate unnormalized weights
            for destination in global_G.neighbors(current_node):

                if destination == source:  
                    ss_weight = 1 
                elif node_type_dict[destination] == tar_type:  #goto target type
                    ss_weight = 1 / value_p
                else:  #goto other type
                    ss_weight = 1 / value_q  

                # Assign the unnormalized sampling strategy weight, normalize during random walk
                unnormalized_weights.append(ss_weight)
                d_neighbors.append(destination)

            unnormalized_weights = np.array(unnormalized_weights)
            d_graph[current_node][PROBABILITIES_KEY][source] = unnormalized_weights / unnormalized_weights.sum()

        first_travel_weights = []

        for destination in global_G.neighbors(source):
            first_travel_weights.append(global_G[source][destination].get(WEIGHT_KEY, 1))

        first_travel_weights = np.array(first_travel_weights)
        d_graph[source][FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

        # Save neighbors
        d_graph[source][NEIGHBORS_KEY] = list(global_G.neighbors(source))

    return d_graph
