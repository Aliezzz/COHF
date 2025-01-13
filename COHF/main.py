import argparse
import numpy as np
import os
import os.path as osp
import time
import pickle
import csv
import torch
import torch.nn.functional as F

from dataloader import load_data
from model import COHF
from utils import euclidean_dist, eva_score, get_dataset_info, get_val_dataset_info, get_loss, setup_seed
from collections import defaultdict

root_dir = "/root/COHF"
file_dir = "/root/COHF/Dataset"

def main(args):
    device = 'cuda:0' if args.use_cuda else 'cpu'
    train_dataset_info, test_dataset_info = get_dataset_info(args)
    train_tasks, feat_dim = load_data(file_dir, train_dataset_info, args)
    test_tasks, _ = load_data(file_dir, test_dataset_info, args)

    model = COHF(feat_dim, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    update_count = 0

    for epoch in range(args.n_epoch):    
        model.train()
        optimizer.zero_grad()
        loss = 0.0
        micro_f1s, macro_f1s = [], []
        l_main_set, l_KLc_set, l_KLd_set, l_cons_set = [], [], [], []

        for task in train_tasks:
            [[x_spt_data, _], [x_qry_data, y_qry]] = task
            y_qry = y_qry.to(device)

            spt_embeds, _, _, _, _ = model(x_spt_data[args.subg_type])
            spt_embeds = spt_embeds.view([args.n_way, -1, args.y_gn_dim])
            proto_embeds = torch.mean(spt_embeds, dim=1)

            qry_embeds, pred_pos, pred_neg, e2_c_dis, e2_d_dis = model(x_qry_data[args.subg_type])
            
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            [loss_main, loss_KLc, loss_KLd, loss_cons], output = get_loss(proto_embeds, qry_embeds, y_qry, pred_pos, pred_neg, e2_c_dis, e2_d_dis)
            l_main_set.append(loss_main.item())
            l_KLc_set.append(loss_KLc.item())
            l_KLd_set.append(loss_KLd.item())
            l_cons_set.append(loss_cons.item())

            loss = loss_main + anneal * (loss_KLc + loss_KLd) + loss_cons + loss
            update_count += 1

            micro_f1, macro_f1 = eva_score(output, y_qry)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)


        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            model.eval()
            micro_ep, macro_ep = [], []

            with torch.no_grad():
                set_task_micro, set_task_macro = [], []
                for task in test_tasks:
                    [[x_spt_data, _], [x_qry_data, y_qry]] = task
                    y_qry = y_qry.to(device)

                    spt_embeds, _, _, _, _ = model(x_spt_data[args.subg_type])
                    spt_embeds = spt_embeds.view([args.n_way, -1, args.y_gn_dim])
                    proto_embeds = torch.mean(spt_embeds, dim=1)

                    qry_embeds, pred_pos, pred_neg, e2_c_dis, e2_d_dis = model(x_qry_data[args.subg_type])
                    
                    if args.total_anneal_steps > 0:
                        anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
                    else:
                        anneal = args.anneal_cap

                    loss, output = get_loss(proto_embeds, qry_embeds, y_qry, pred_pos, pred_neg, e2_c_dis, e2_d_dis, anneal)
                    update_count += 1

                    task_micro, task_macro = eva_score(output, y_qry)
                    set_task_micro.append(task_micro)
                    set_task_macro.append(task_macro)

                micro_ep = np.mean(set_task_micro)
                macro_ep = np.mean(set_task_macro)

    return micro_ep, macro_ep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True)
    
    parser.add_argument('--train_dataset', type=str, default='DBLP_GTN')
    parser.add_argument('--test_dataset', type=str, default='ACM_GTN')
    parser.add_argument('--ood_type', type=str, default='no_shift', help='shift, no_shift')

    parser.add_argument('--adopt_reverse_rels', type=bool, default=True)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--walk_repeat', type=int, default=20)
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--subg_type', type=str, default='random_walk')

    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--k_spt', type=int, default=1)
    parser.add_argument('--k_qry', type=int, default=3)

    #e2_layer
    parser.add_argument('--e2_mlp_act', type=str, default='tanh')
    parser.add_argument('--e2_mlp_drop', type=float, default=0.6)
    parser.add_argument("--e2_mlp_dims", default='[100, 20]', help="2 layer mlp for e2")
    parser.add_argument('--e2_gn_act', type=str, default='relu')
    parser.add_argument('--e2_gn_drop', type=float, default=0.4)
    parser.add_argument('--e2_gn_conv', type=str, default='gcn')
    parser.add_argument('--e2_gn_layer', type=int, default=2)
    parser.add_argument('--e2_gn_dim', type=int, default=20)
    parser.add_argument('--e2_adopt_comm', type=bool, default=True)
    parser.add_argument('--e2_dim', type=int, default=20)
    parser.add_argument('--e2_att_dim', type=int, default=20)
    parser.add_argument('--e2_dis_agg', type=str, default='concat', help='seprate, concat')
    parser.add_argument('--e2_adj_agg', type=str, default='adj_single', help='adj_single, adj_multi')
    parser.add_argument('--e2_diff_adj_input', type=str, default='single', help='single, add, time')

    #z1_layer
    parser.add_argument('--z1_gn_layer', type=int, default=1)
    parser.add_argument('--z1_gn_dim', type=int, default=20)
    parser.add_argument('--z1_gn_act', type=str, default='relu')
    parser.add_argument('--z1_gn_drop', type=float, default=0.6)
    parser.add_argument('--z1_pool', type=str, default='sum')
    parser.add_argument('--z1_conv', type=str, default='ori_conv')
    parser.add_argument("--z1_mlp_dims", default='[20]')
    parser.add_argument('--z1_mlp_act', type=str, default='tanh')
    parser.add_argument('--z1_mlp_drop', type=float, default=0.6)
    parser.add_argument('--z1_dim', type=int, default=10)
    parser.add_argument('--z1_dis_agg', type=str, default='seprate', help='seprate, concat')

    #z2_layer
    parser.add_argument('--z2_gn_layer', type=int, default=1)
    parser.add_argument('--z2_gn_dim', type=int, default=20)
    parser.add_argument('--z2_gn_act', type=str, default='relu')
    parser.add_argument('--z2_gn_drop', type=float, default=0.6)
    parser.add_argument('--z2_conv', type=str, default='ori_conv')
    parser.add_argument('--z2_feat_type', type=str, default='e2', help='ori, e2')
    parser.add_argument("--z2_mlp_dims", default='[10]')
    parser.add_argument('--z2_mlp_act', type=str, default='tanh')
    parser.add_argument('--z2_mlp_drop', type=float, default=0.6)
    parser.add_argument('--z2_dim', type=int, default=20)
    parser.add_argument('--z2_dis_agg', type=str, default='seprate', help='seprate, concat')
    parser.add_argument('--z2_adj_gen', type=str, default='seprate', help='seprate, concat')

    #pred_layer
    parser.add_argument('--y_conv', type=str, default='ori_conv')
    parser.add_argument('--y_gn_dim', type=int, default=20)
    parser.add_argument('--sample_freq', type=int, default=1)

    #reconstruction
    parser.add_argument('--g_e1_dim', type=int, default=20)
    parser.add_argument('--g_pos_edge_num', type=int, default=40)
    parser.add_argument('--g_neg_ratio', type=float, default=0.6)
    parser.add_argument('--g_mlp_act', type=str, default='tanh')

    #train
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--n_loop', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)  
    parser.add_argument('--total_anneal_steps', type=int, default=2000,
                    help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')

    args = parser.parse_args()

    res_micro, res_macro = main(args)

    print('final micro f1={:.4f}, macro f1={:.4f}'.format(res_micro, res_macro))

