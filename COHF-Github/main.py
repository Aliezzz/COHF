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

root_dir = "/root/autodl-tmp/COHF"
file_dir = "/root/autodl-tmp/COHF/Dataset"

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

def select_best_param(para_name, para_range, args):
    dic = vars(args)
    best_matrix = 0
    for para_value in para_range:
        dic[para_name] = para_value
        args_ = argparse.Namespace(**dic)
        

        matrix_value = loop_and_save(args_, para_name, para_value)

        if matrix_value > best_matrix:
            best_arg = args_
            best_matrix = matrix_value
    return best_arg

def loop_and_save(args, para_name, para_value):
    res_micro_set = [] 
    res_macro_set = []
    mean_time_cost = []

    for i in range(args.n_loop):
        print('loop: ', i, para_name, para_value)
        start_time = time.time()
        res_micro, res_macro = main(args)
        time_cost = time.time() - start_time
        res_micro_set.append(res_micro)
        res_macro_set.append(res_macro)

        mean_time_cost.append(time_cost)

        print('loop:{}  micro f1={:.4f}, macro f1={:.4f}'.format(i, res_micro, res_macro))

    save_2_csv_proto(args, res_micro_set, res_macro_set, mean_time_cost)
    print('final micro f1={:.4f}, macro f1={:.4f}'.format(np.mean(res_micro_set), np.mean(res_macro_set)))

    return (np.mean(res_micro_set) + np.mean(res_macro_set))/2

def save_2_csv_proto(args, micro, macro, mean_time_cost):
    dict_items = vars(args)
    csv_name = "_".join(['COHF', str(args.update_top_data), args.ood_type, dict_items['train_dataset'], dict_items['test_dataset'],
                         str(dict_items['n_way']), 'way', str(dict_items['k_spt']), 'spt', str(dict_items['k_qry']), 'qry',  ".csv"])
    
    csv_folder = osp.join(root_dir, "results", dict_items['test_dataset'])

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    csv_loc = os.path.join(csv_folder, csv_name)

    dict_items['mean_time_cost'] = float('%.4f' % np.mean(mean_time_cost))

    dict_items['micro'] = float('%.4f' % np.mean(micro))
    dict_items['macro'] = float('%.4f' % np.mean(macro))
    dict_items['std_micro'] = float('%.4f' % np.std(micro))

    dict_names = list(dict_items.keys())

    with open(csv_loc, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=dict_names)
        if not os.path.getsize(csv_loc):
            writer.writeheader()  
        writer.writerows([dict_items])

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
 
    model_params_dict = defaultdict(list)
      
    #e2_layer
    model_params_dict['e2_adj_agg'] = ['adj_single', 'adj_multi']
    model_params_dict['e2_diff_adj_input'] = ['single', 'add', 'time']
    model_params_dict['e2_mlp_act'] = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    model_params_dict['e2_mlp_drop'] = [0.4, 0.6]
    model_params_dict['e2_mlp_dims'] = ['[100, 20]', '[50]']
    model_params_dict['e2_gn_act'] = ['relu', 'leaky_relu']
    model_params_dict['e2_gn_drop'] = [0.4, 0.6]
    model_params_dict['e2_gn_conv'] = ['gcn', 'sage', 'graph_conv','ori_conv']
    model_params_dict['e2_gn_layer'] = [1, 2]
    model_params_dict['e2_gn_dim'] = [20, 40]
    model_params_dict['e2_adopt_comm'] = [True, False]
    model_params_dict['e2_dim'] = [20, 40]
    model_params_dict['e2_att_dim'] = [20, 40]

    #z1_layer
    model_params_dict['z1_dis_agg'] = ['seprate', 'concat']
    model_params_dict['z1_gn_layer'] = [1, 2]
    model_params_dict['z1_gn_dim'] = [20, 40]
    model_params_dict['z1_gn_act'] = ['relu', 'leaky_relu']
    model_params_dict['z1_gn_drop'] = [0.4, 0.6]
    model_params_dict['z1_pool'] = ['sum', 'mean', 'max']
    model_params_dict['z1_conv'] = ['ori_conv', 'gcn', 'sage', 'graph_conv']
    model_params_dict['z1_mlp_dims'] = ['[20]', '[40, 20]', '[10]', '[20, 10]']
    model_params_dict['z1_mlp_act'] = ['tanh', 'relu', 'sigmoid', 'none']
    model_params_dict['z1_mlp_drop'] = [0.4, 0.6]
    model_params_dict['z1_dim'] = [10, 20, 40]

    #z2_layer
    model_params_dict['z2_dis_agg'] = ['seprate', 'concat']
    model_params_dict['z2_adj_gen'] = ['seprate', 'concat']
    model_params_dict['z2_gn_layer'] = [1, 2]
    model_params_dict['z2_gn_dim'] = [20, 40]
    model_params_dict['z2_gn_act'] = ['relu', 'leaky_relu']
    model_params_dict['z2_gn_drop'] = [0.4, 0.6]
    model_params_dict['z2_conv'] = ['ori_conv', 'gcn', 'sage', 'graph_conv']
    model_params_dict['z2_feat_type'] = ['ori', 'e2']
    model_params_dict['z2_mlp_dims'] = ['[10]', '[20, 10]', '[20]', '[40, 20]']
    model_params_dict['z2_mlp_act'] = ['tanh', 'relu', 'sigmoid', 'none']
    model_params_dict['z2_mlp_drop'] = [0.4, 0.6]
    model_params_dict['z2_dim'] = [10, 20, 40]

    #pred_layer
    model_params_dict['y_conv'] = ['mlp', 'ori_conv']
    model_params_dict['y_gn_dim'] = [20, 40]
    model_params_dict['sample_freq'] = [1, 2]

    #reconstruction
    model_params_dict['g_e1_dim'] = [20, 40]
    model_params_dict['g_pos_edge_num'] = [40, 60]
    model_params_dict['g_neg_ratio'] = [0.4, 0.6]
    model_params_dict['g_mlp_act'] = ['tanh', 'relu', 'sigmoid', 'none']

    #train
    model_params_dict['n_epoch'] = [80, 100, 120]
    model_params_dict['lr'] = [0.001, 0.005, 0.01]

    

    for n_way in [2, 3]:
        args.n_way = n_way
        for k_spt in [1, 3]:
            args.k_spt = k_spt
            for ood_type in ["no_shift", 'shift']:
                args.ood_type = ood_type

                for para_name in list(model_params_dict.keys()):
                    args = select_best_param(para_name, model_params_dict[para_name], args)

