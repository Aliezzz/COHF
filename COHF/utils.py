import torch
import pickle
import random
import numpy as np
import os.path as osp

from sklearn.metrics import f1_score
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def get_activation(act):
    if act == "relu":
        return torch.nn.ReLU()
    elif act == "elu":
        return torch.nn.ELU()
    elif act == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif act == "sigmoid":
        return torch.nn.Sigmoid()
    elif act == "tanh":
        return torch.nn.Tanh()
    else:
        return torch.nn.Identity()
    
def pooling_aggregation(data, tar_dim, pooling_type="mean"):
    if pooling_type == "mean":
        return torch.mean(data, dim=tar_dim)
    elif pooling_type == "max":
        return torch.max(data, dim=tar_dim)[0]
    elif pooling_type == "sum":
        return torch.sum(data, dim=tar_dim)

def get_dataset_info(args):
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset

    train_edge_types, train_tar_type = get_edge_types(train_dataset)
    test_edge_types, test_tar_type = get_edge_types(test_dataset)

    #select common relations and different relations
    comm_rels = np.intersect1d(train_edge_types, test_edge_types)
    train_diff_rels = np.setdiff1d(train_edge_types, comm_rels)
    test_diff_rels = np.setdiff1d(test_edge_types, comm_rels)

    if not args.adopt_reverse_rels:
        comm_rels = remove_recursive_rels(list(comm_rels))
        train_diff_rels = remove_recursive_rels(list(train_diff_rels))
        test_diff_rels = remove_recursive_rels(list(test_diff_rels))
    
    return [train_dataset, train_edge_types, train_tar_type, comm_rels, train_diff_rels], \
            [test_dataset, test_edge_types, test_tar_type, comm_rels, test_diff_rels]

def get_val_dataset_info(args):
    train_dataset = args.train_dataset
    val_dataset = args.val_dataset

    train_edge_types, train_tar_type = get_edge_types(train_dataset)
    val_edge_types, val_tar_type = get_edge_types(val_dataset)

    #select common relations and different relations
    comm_rels = np.intersect1d(train_edge_types, val_edge_types)
    val_diff_rels = np.setdiff1d(val_edge_types, comm_rels)

    if args.adopt_reverse_rels:
        comm_rels = remove_recursive_rels(list(comm_rels))
        val_diff_rels = remove_recursive_rels(list(val_diff_rels))
    
    return [val_dataset, val_edge_types, val_tar_type, comm_rels, val_diff_rels]

def remove_recursive_rels(rels):
    for rel in rels:
        reverse_rel = rel.split("-")[1] + "-" + rel.split("-")[0]
        if reverse_rel in rels:
            rels.remove(reverse_rel)
    return rels

def get_edge_types(dataset):
    if dataset == 'DoubanMovie':
        return ["M-A", "A-M", "M-D", "D-M", "U-G", "G-U", "U-M", "M-U", "U-U"], "M"
    elif dataset == 'MovieLens':
        return ["U-M", "M-U", "U-A", "A-U", "U-O", "O-U"], "M"
    elif dataset == 'IMDB':
        return ["M-D", "D-M", "M-A", "A-M"], "M"

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M

def eva_score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    # print("prediction: ", prediction)

    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return micro_f1, macro_f1

def kl_loss(e_dim, e2_dis):
    prior_dis = Normal(torch.zeros(e_dim).cuda(), torch.ones(e_dim).cuda())
    kl_total = kl_divergence(e2_dis, prior_dis).mean()
    
    return kl_total

def load_labels(file_dir, dataset, tar_type, ood_type):
    if ood_type == 'no_shift':
        with open(osp.join(file_dir, dataset, 'raw', 'labels.pkl'), 'rb') as f:
            labels = pickle.load(f)

        train_nodes, train_labels = np.array(labels[0]).astype(np.int64).T
        val_nodes, val_labels = np.array(labels[1]).astype(np.int64).T
        qry_nodes, qry_labels = np.array(labels[2]).astype(np.int64).T

        spt_nodes = np.concatenate((train_nodes, val_nodes))
        spt_labels = np.concatenate((train_labels, val_labels))

        return [spt_nodes, spt_labels], qry_nodes, qry_labels

    else:
        with open(osp.join(file_dir, dataset, 'raw', 'OOD_data.pkl'), 'rb') as f:
            labels = pickle.load(f)

        train_label_env, _, _, val_label, test_label = labels[ood_type]

        spt_envs = {}

        for node, label, env in train_label_env:
            if env not in spt_envs:
                spt_envs[env] = []
            spt_envs[env].append((node, label))

        for env, data in spt_envs.items():
            nodes, labels = np.array(data).astype(np.int64).T
            spt_envs[env] = (nodes, labels)

        qry_nodes, qry_labels = np.array(test_label).astype(np.int64).T

        return spt_envs, qry_nodes, qry_labels
