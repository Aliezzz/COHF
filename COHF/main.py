import argparse
import numpy as np
import torch
import torch.nn.functional as F

from dataloader import load_data
from model import COHF
from utils import euclidean_dist, eva_score, get_dataset_info, get_val_dataset_info, kl_loss, setup_seed
from collections import defaultdict

file_dir = "/Dataset-server/"

def main(args):
    device = 'cuda:0' if args.use_cuda else 'cpu'

    train_dataset_info, test_dataset_info = get_dataset_info(args)
    train_tasks, feat_dim = load_data(file_dir, train_dataset_info, args)
    test_tasks, _ = load_data(file_dir, test_dataset_info, args)

    if args.val_dataset != 'None':
        val_dataset_info = get_val_dataset_info(args)
        val_tasks, _ = load_data(file_dir, val_dataset_info, args)

    model = COHF(feat_dim, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_micro = 0.0
    best_val_macro = 0.0

    val_micro_gl = []
    val_macro_gl = []

    test_micro_gl = []
    test_macro_gl = []

    avg_micro_gl = []
    avg_macro_gl = []

    for epoch in range(args.n_epoch):    
        if args.set_seed:
            setup_seed(42)
        model.train()
        # pred_layer.train()
        optimizer.zero_grad()
        loss = 0.0
        micro_f1s, macro_f1s = [], []

        for task in train_tasks:
            [[x_spt_data, y_spt], [x_qry_data, y_qry]] = task

            spt_embeddings, comm_dist, diff_dist = model(x_spt_data[args.subg_type])
            loss += kl_loss(args.e_dim, comm_dist) * 0.4
                # loss += kl_loss(args.e_dim, diff_dist) * 0.1
            spt_embeddings = spt_embeddings.view([args.n_way, -1, args.z_dim * 2])
            proto_embeddings = torch.mean(spt_embeddings, dim=1)

            qry_embeddings, _, _ = model(x_qry_data[args.subg_type])
            y_qry = y_qry.to(device)

            dists = euclidean_dist(qry_embeddings, proto_embeddings)
            output = F.log_softmax(-dists, dim=1)
            loss += F.nll_loss(output, y_qry)

            micro_f1, macro_f1 = eva_score(output, y_qry)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)

        print('Epoch: {:04d}'.format(epoch + 1),
                'f1_micro_train: {:.4f}'.format(np.mean(micro_f1s)),
                'f1_macro_train: {:.4f}'.format(np.mean(macro_f1s)),
                'loss_train: {:.4f}'.format(loss.item()))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            model.eval()
            micro_ep, macro_ep = [], []

            with torch.no_grad():
                if args.val_dataset != 'None':
                    for task in val_tasks:
                        [[x_spt_data, y_spt], [x_qry_data, y_qry]] = task

                        spt_embeddings, _, _ = model(x_spt_data[args.subg_type])
                        spt_embeddings = spt_embeddings.view([args.n_way, -1, args.z_dim * 2])
                        proto_embeddings = torch.mean(spt_embeddings, dim=1)

                        qry_embeddings, _, _ = model(x_qry_data[args.subg_type])
                        y_qry = y_qry.to(device)

                        dists = euclidean_dist(qry_embeddings, proto_embeddings)
                        output = F.log_softmax(-dists, dim=1)

                        micro_f1, macro_f1 = eva_score(output, y_qry)

                        micro_ep.append(micro_f1)
                        macro_ep.append(macro_f1)

                    mean_micro_ep = np.mean(micro_ep)
                    mean_macro_ep = np.mean(macro_ep)
                    
                    print('Epoch: {:04d}'.format(epoch + 1),
                            'f1_micro_val: {:.4f}'.format(mean_micro_ep),
                            'f1_macro_val: {:.4f}'.format(mean_macro_ep))
                    
                    val_micro_gl.append(mean_micro_ep)
                    val_macro_gl.append(mean_macro_ep)

                micro_ep, macro_ep = [], []
                for task in test_tasks:
                    [[x_spt_data, y_spt], [x_qry_data, y_qry]] = task

                    spt_embeddings, _, _ = model(x_spt_data[args.subg_type])
                    spt_embeddings = spt_embeddings.view([args.n_way, -1, args.z_dim * 2])
                    proto_embeddings = torch.mean(spt_embeddings, dim=1)

                    qry_embeddings, _, _ = model(x_qry_data[args.subg_type])
                    y_qry = y_qry.to(device)

                    dists = euclidean_dist(qry_embeddings, proto_embeddings)
                    output = F.log_softmax(-dists, dim=1)

                    micro_f1, macro_f1 = eva_score(output, y_qry)

                    micro_ep.append(micro_f1)
                    macro_ep.append(macro_f1)

                mean_micro_ep = np.mean(micro_ep)
                mean_macro_ep = np.mean(macro_ep)

                print('Epoch: {:04d}'.format(epoch + 1),
                        'f1_micro_test: {:.4f}'.format(mean_micro_ep),
                        'f1_macro_test: {:.4f}'.format(mean_macro_ep))
                
                test_micro_gl.append(mean_micro_ep)
                test_macro_gl.append(mean_macro_ep)

                if (epoch + 1) % 10 == 0:
                    avg_micro_gl.append(mean_micro_ep)
                    avg_macro_gl.append(mean_macro_ep)

    #get test results at the best val epoch
    if args.val_dataset != 'None':
        max_iter = val_micro_gl.index(max(val_micro_gl))
        best_val_micro = test_micro_gl[max_iter]
        best_val_macro = test_macro_gl[max_iter]

    return np.mean(avg_micro_gl), np.mean(avg_macro_gl), best_val_micro, best_val_macro

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--adopt_reverse_rels', type=bool, default=False)
    parser.add_argument('--train_dataset', type=str, default='ACM')
    parser.add_argument('--val_dataset', type=str, default='None')
    parser.add_argument('--test_dataset', type=str, default='DBLP')
    parser.add_argument('--ood_type', type=str, default='no_shift')
    parser.add_argument('--n_way', type=int, default=3)
    parser.add_argument('--k_spt', type=int, default=3)
    parser.add_argument('--k_qry', type=int, default=5)

    parser.add_argument('--n_tasks', type=int, default=20)
    parser.add_argument('--subg_type', type=str, default='random_walk')

    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--walk_repeat', type=int, default=20)

    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--att_dim', type=int, default=32)

    parser.add_argument('--e2_gc_act', type=str, default='elu')
    parser.add_argument('--e2_gc_drop', type=float, default=0.2)
    parser.add_argument('--e2_mlp_act', type=str, default='elu')
    parser.add_argument('--e2_mlp_drop', type=float, default=0.4)
    parser.add_argument('--e2_comm_conv', type=str, default='gcn')
    parser.add_argument('--e2_diff_conv', type=str, default='gcn')

    #z1_layer
    parser.add_argument('--z1_layer', type=int, default=1)
    parser.add_argument('--z1_gc1_act', type=str, default='elu')
    parser.add_argument('--z1_gc1_drop', type=float, default=0.4)
    parser.add_argument('--z1_gc2_act', type=str, default='elu')
    parser.add_argument('--z1_gc2_drop', type=float, default=0.4)
    parser.add_argument('--z1_pool', type=str, default='sum')
    parser.add_argument('--z1_conv', type=str, default='gcn')

    #z2_layer
    parser.add_argument('--z2_layer', type=int, default=1)
    parser.add_argument('--z2_gc1_act', type=str, default='elu')
    parser.add_argument('--z2_gc1_drop', type=float, default=0.4)
    parser.add_argument('--z2_gc2_act', type=str, default='elu')
    parser.add_argument('--z2_gc2_drop', type=float, default=0.4)
    parser.add_argument('--z2_conv', type=str, default='gcn')

    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)  

    args = parser.parse_args()
 
    model_params_dict = defaultdict(list)

    micro = [] 
    macro = []
    
    val_micro = []
    val_macro = []

    for i in range(args.n_loop):
        micro_f1, macro_f1, val_micro_f1, val_macro_f1 = main(args)
        micro.append(micro_f1)
        macro.append(macro_f1)
    
        val_micro.append(val_micro_f1)
        val_macro.append(val_macro_f1)

        print('loop:{} micro f1={}, macro f1={}, val_micro f1={}, val_macro f1={}'.
                format(i, micro_f1, macro_f1, val_micro_f1, val_macro_f1))
        
    print('final micro f1={}, macro f1={}, best_val_micro f1={}, best_val_macro f1={}'.
            format(np.mean(micro), np.mean(macro), np.mean(val_micro), np.mean(val_macro)))
                                  