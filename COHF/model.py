import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_activation, pooling_aggregation, get_gnn_layer, Attention

class q_e2_gs_e1(nn.Module):
    def __init__(self, n_feat, args):
        super(q_e2_gs_e1, self).__init__()
        self.adj_agg = args.e2_adj_agg
        self.dis_agg = args.e2_dis_agg
        self.diff_adj_input = args.e2_diff_adj_input
        self.gn_dim = args.e2_gn_dim
        self.att_dim = args.e2_att_dim
        self.mlp_dims = args.e2_mlp_dims
        self.adopt_comm = args.e2_adopt_comm

        self.comm_gc_layers = nn.ModuleList()
        self.diff_gc_layers = nn.ModuleList()

        self.comm_gc_layers.append(get_gnn_layer(args.e2_gn_conv, n_feat, self.gn_dim))
        self.diff_gc_layers.append(get_gnn_layer(args.e2_gn_conv, n_feat, self.gn_dim))

        for _ in range(1, args.e2_gn_layer):
            self.comm_gc_layers.append(get_gnn_layer(args.e2_gn_conv, self.gn_dim, self.gn_dim))
            self.diff_gc_layers.append(get_gnn_layer(args.e2_gn_conv, self.gn_dim, self.gn_dim))

        self.gnn_act = get_activation(args.e2_gn_act)
        self.gnn_drop = nn.Dropout(args.e2_gn_drop)

        self.mlp_diff_dims = [self.gn_dim] + eval(self.mlp_dims) + [args.e2_dim]
        self.mlp_comm_dims = [self.gn_dim] + eval(self.mlp_dims) + [args.e2_dim]

        if self.adj_agg == 'single':
            self.comm_attention = Attention(self.gn_dim, self.att_dim)
            if self.adopt_comm:
                self.diff_attention = Attention(self.gn_dim * 2, self.att_dim)
            else:
                self.diff_attention = Attention(self.gn_dim, self.att_dim)

        if self.dis_agg == 'concat':
            temp_comm_dims = self.mlp_comm_dims[:-1] + [self.mlp_comm_dims[-1] * 2]
            self.mlp_comm_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_comm_dims[:-1], temp_comm_dims[1:])])
            
            temp_diff_dims = self.mlp_diff_dims[:-1] + [self.mlp_diff_dims[-1] * 2]
            self.mlp_diff_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_diff_dims[:-1], temp_diff_dims[1:])])
        else:
            temp_comm_dims = self.mlp_comm_dims[:-1] + [self.mlp_comm_dims[-1]]
            self.mlp_comm_mu_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_comm_dims[:-1], temp_comm_dims[1:])])
            self.mlp_comm_logvar_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_comm_dims[:-1], temp_comm_dims[1:])])
            
            temp_diff_dims = self.mlp_diff_dims[:-1] + [self.mlp_diff_dims[-1]]
            self.mlp_diff_mu_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_diff_dims[:-1], temp_diff_dims[1:])])
            self.mlp_diff_logvar_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_diff_dims[:-1], temp_diff_dims[1:])])
            
        self.mlp_act = get_activation(args.e2_mlp_act)
        self.mlp_drop = nn.Dropout(args.e2_mlp_drop)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, comm_adjs, diff_adjs, feats):
        comm_reps = []
        if self.adj_agg == 'single':
            for adj in comm_adjs:
                comm_h = feats
                for i, layer in enumerate(self.comm_gc_layers):
                    comm_h = layer(comm_h, adj)
                    if i != len(self.comm_gc_layers) - 1:
                        comm_h = self.gnn_act(comm_h)
                        comm_h = self.gnn_drop(comm_h)
                comm_reps.append(comm_h.view(-1, self.gn_dim))
            comm_agg_reps = self.comm_attention(torch.stack(comm_reps, dim=1))

            comm_adj_agg = torch.stack(comm_adjs, dim=0).max(dim=0)[0]
            # consider different intraction ways
            diff_reps = []
            for adj in diff_adjs:
                diff_h = feats
                for i, layer in enumerate(self.diff_gc_layers):
                    if self.diff_adj_input == 'single':
                        diff_h = layer(diff_h, adj)
                    elif self.diff_adj_input == 'add':
                        diff_h = layer(diff_h, adj + comm_adj_agg)
                    elif self.diff_adj_input == 'time':
                        diff_h = layer(diff_h, comm_adj_agg * adj)
            
                    if i != len(self.diff_gc_layers) - 1:
                        diff_h = self.gnn_act(diff_h)
                        diff_h = self.gnn_drop(diff_h)
                diff_reps.append(diff_h.view(-1, self.gn_dim))
            diff_agg = torch.stack(diff_reps, dim=1)
            if self.adopt_comm:
                concat_diff_agg = torch.cat([diff_agg, comm_agg_reps.view(-1, 1, self.gn_dim).expand(diff_agg.shape)], dim=2)
                diff_agg_reps = self.diff_attention(concat_diff_agg, diff_agg)
            else:
                diff_agg_reps = self.diff_attention(diff_agg)

        else:
            comm_adj_agg = pooling_aggregation(torch.stack(comm_adjs, dim=0), tar_dim=0, pooling_type='max')
            comm_h = feats
            for i, layer in enumerate(self.comm_gc_layers):
                comm_h = layer(comm_h, comm_adj_agg)
                if i != len(self.comm_gc_layers) - 1:
                    comm_h = self.gnn_act(comm_h)
                    comm_h = self.gnn_drop(comm_h)
            comm_agg_reps = comm_h.view(-1, self.gn_dim)

            diff_adj_agg = pooling_aggregation(torch.stack(diff_adjs, dim=0), tar_dim=0, pooling_type='max')
            diff_h = feats
            for i, layer in enumerate(self.diff_gc_layers):
                if self.diff_adj_input == 'single':
                    diff_h = layer(diff_h, diff_adj_agg)
                elif self.diff_adj_input == 'add':
                    diff_h = layer(diff_h, diff_adj_agg + comm_adj_agg)
                elif self.diff_adj_input == 'time':
                    diff_h = layer(diff_h, comm_adj_agg * diff_adj_agg)
                if i != len(self.diff_gc_layers) - 1:
                    diff_h = self.gnn_act(diff_h)
                    diff_h = self.gnn_drop(diff_h)
            diff_agg_reps = diff_h.view(-1, self.gn_dim)

        if self.dis_agg == 'concat':
            comm_h = self.mlp_drop(comm_agg_reps)
            for i, layer in enumerate(self.mlp_comm_layers):
                comm_h = layer(comm_h)
                if i != len(self.mlp_comm_layers) - 1:  
                    comm_h = self.mlp_act(comm_h) 
                else:
                    comm_mu = comm_h[:, :self.mlp_comm_dims[-1]]  
                    comm_logvar = comm_h[:, self.mlp_comm_dims[-1]:]  

            diff_h = self.mlp_drop(diff_agg_reps)
            for i, layer in enumerate(self.mlp_diff_layers):
                diff_h = layer(diff_h)
                if i != len(self.mlp_diff_layers) - 1:  
                    diff_h = self.mlp_act(diff_h) 
                else:
                    diff_mu = diff_h[:, :self.mlp_diff_dims[-1]]  
                    diff_logvar = diff_h[:, self.mlp_diff_dims[-1]:]
        else:
            comm_mu = self.mlp_drop(comm_agg_reps)
            comm_logvar = self.mlp_drop(comm_agg_reps)

            for i, layer in enumerate(self.mlp_comm_mu_layers):
                comm_mu = layer(comm_mu)
                if i != len(self.mlp_comm_mu_layers) - 1:  
                    comm_mu = self.mlp_act(comm_mu) 

            for i, layer in enumerate(self.mlp_comm_logvar_layers):
                comm_logvar = layer(comm_logvar)
                if i != len(self.mlp_comm_logvar_layers) - 1:  
                    comm_logvar = self.mlp_act(comm_logvar)

            diff_mu = self.mlp_drop(diff_agg_reps)
            diff_logvar = self.mlp_drop(diff_agg_reps)

            for i, layer in enumerate(self.mlp_diff_mu_layers):
                diff_mu = layer(diff_mu)
                if i != len(self.mlp_diff_mu_layers) - 1:  
                    diff_mu = self.mlp_act(diff_mu)
            
            for i, layer in enumerate(self.mlp_diff_logvar_layers):
                diff_logvar = layer(diff_logvar)
                if i != len(self.mlp_diff_logvar_layers) - 1:  
                    diff_logvar = self.mlp_act(diff_logvar)

        e2_c = self.reparameterize(comm_mu, comm_logvar)
        e2_d = self.reparameterize(diff_mu, diff_logvar)

        return e2_c, e2_d, [comm_mu, comm_logvar], [diff_mu, diff_logvar]
    
class p_z1_gs_e1(nn.Module):
    def __init__(self, n_feat, args):
        super(p_z1_gs_e1, self).__init__()
        self.gn_dim = args.z1_gn_dim
        self.mlp_dims = args.z1_mlp_dims
        self.dis_agg = args.z1_dis_agg
        self.pooling_type = args.z1_pool
        self.layer = args.z1_gn_layer

        self.gc1 = get_gnn_layer(args.z1_conv, n_feat, self.gn_dim)
        self.gc1_act = get_activation(args.z1_gn_act)
        self.gc1_drop = nn.Dropout(args.z1_gn_drop)
        
        if self.layer == 2:
            self.gc2 = get_gnn_layer(args.z1_conv, self.gn_dim, self.gn_dim)

        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=1)

        self.z1_mlp_dims = [self.gn_dim] + eval(self.mlp_dims) + [args.z1_dim]
        if self.dis_agg == 'concat':
            temp_dims = self.z1_mlp_dims[:-1] + [self.z1_mlp_dims[-1] * 2]
            self.z1_mlp_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])
        else:
            temp_dims = self.z1_mlp_dims[:-1] + [self.z1_mlp_dims[-1]]
            self.z1_mlp_mu_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])
            self.z1_mlp_logvar_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

        self.mlp_act = get_activation(args.z1_mlp_act)
        self.mlp_drop = nn.Dropout(args.z1_mlp_drop)

    def forward(self, comm_adjs, diff_adjs, feats):
        # we use pooling function to aggregate the adj for two types of relations
        comm_adj_agg = pooling_aggregation(torch.stack(comm_adjs, dim=0), tar_dim=0, pooling_type=self.pooling_type)
        diff_adj_agg = pooling_aggregation(torch.stack(diff_adjs, dim=0), tar_dim=0, pooling_type=self.pooling_type)

        A_t = torch.stack([comm_adj_agg, diff_adj_agg], dim=2)
        temp = torch.matmul(A_t, self.weight_b)
        final_A = torch.squeeze(temp, 2)

        U1 = self.gc1_drop(self.gc1_act(self.gc1(feats, final_A).squeeze(0)))
        if self.layer == 2:
            U2 = self.gc2(U1, final_A).squeeze(0)
            final_U = (U1 + U2)/2
        else:
            final_U = U1

        if self.dis_agg == 'concat':
            h = self.mlp_drop(final_U)
            for i, layer in enumerate(self.z1_mlp_layers):
                h = layer(h)
                if i != len(self.z1_mlp_layers) - 1:
                    h = self.mlp_act(h)
                else:
                    Z1_mu = h[:, :self.z1_mlp_dims[-1]]
                    Z1_logvar = h[:, self.z1_mlp_dims[-1]:]
        else:
            Z1_mu = self.mlp_drop(final_U)
            Z1_logvar = self.mlp_drop(final_U)

            for i, layer in enumerate(self.z1_mlp_mu_layers):
                Z1_mu = layer(Z1_mu)
                if i != len(self.z1_mlp_mu_layers) - 1:
                    Z1_mu = self.mlp_act(Z1_mu)

            for i, layer in enumerate(self.z1_mlp_logvar_layers):
                Z1_logvar = layer(Z1_logvar)
                if i != len(self.z1_mlp_logvar_layers) - 1:
                    Z1_logvar = self.mlp_act(Z1_logvar)

        return [Z1_mu, Z1_logvar]
    
class Adj_Generator(nn.Module):
    def __init__(self, in_dim, n_per=16):  # n_per: dim of weight vector
        super(Adj_Generator, self).__init__()
        self.weight_tensor = torch.Tensor(n_per, in_dim)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def reset_parameters(self):
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, context):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)  # [n_per, 1, input_size]
        context_fc = context.unsqueeze(0) * expand_weight_tensor # [n_per, batch_size, input_size]
        context_norm = F.normalize(context_fc, p=2, dim=-1) # [n_per, batch_size, input_size]
        attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0) # [batch_size, batch_size]
        mask = (attention > 0).detach().float()  #set value to 1 if attention > 0, else 0
        attention = attention * mask + 0 * (1 - mask) # [batch_size, batch_size] delete the negative value
        adj = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=1e-12)

        return adj
    
class p_z2_e2(nn.Module):
    def __init__(self, n_feat, args):
        super(p_z2_e2, self).__init__()
        self.z2_layer = args.z2_gn_layer
        self.feat_type = args.z2_feat_type
        self.gn_dim = args.z2_gn_dim
        self.mlp_dims = args.z2_mlp_dims
        self.mlp_act = get_activation(args.z2_mlp_act)
        self.mlp_drop = nn.Dropout(args.z2_mlp_drop)
        self.dis_agg = args.z2_dis_agg
        self.adj_gen = args.z2_adj_gen

        if self.adj_gen == 'seperate':
            self.comm_adj_generator = Adj_Generator(args.e2_dim)
            self.diff_adj_generator = Adj_Generator(args.e2_dim)
        else:
            self.adj_generator = Adj_Generator(args.e2_dim * 2)

        if self.feat_type == 'e2':
            in_dim = args.e2_dim * 2
        else:
            in_dim = n_feat

        self.gc1 = get_gnn_layer(args.z2_conv, in_dim, self.gn_dim)
        self.gc1_act = get_activation(args.z2_gn_act)
        self.gc1_drop = nn.Dropout(args.z2_gn_drop)

        if args.z2_gn_layer == 2:
            self.gc2 = get_gnn_layer(args.z2_conv, self.gn_dim, self.gn_dim)

        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=1)

        self.z2_mlp_dims = [self.gn_dim] + eval(self.mlp_dims) + [args.z2_dim]
        if self.dis_agg == 'concat':
            temp_dims = self.z2_mlp_dims[:-1] + [self.z2_mlp_dims[-1] * 2]
            self.z2_mlp_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])
        else:
            temp_dims = self.z2_mlp_dims[:-1] + [self.z2_mlp_dims[-1]]
            self.z2_mlp_mu_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])
            self.z2_mlp_logvar_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

    def forward(self, e2_c, e2_d, feats):
        if self.adj_gen == 'seperate':
            comm_adj = self.comm_adj_generator(e2_c)
            diff_adj = self.diff_adj_generator(e2_d)

            A_t = torch.stack([comm_adj, diff_adj], dim=2)
            temp = torch.matmul(A_t, self.weight_b)
            final_A = torch.squeeze(temp, 2)
        else:
            final_A = self.adj_generator(torch.cat([e2_c, e2_d], dim=1))

        # check whether use original feats or feats of e2
        if self.feat_type == 'e2':
            feats = torch.cat([e2_c, e2_d], dim=1)

        U1 = self.gc1_drop(self.gc1_act(self.gc1(feats, final_A).squeeze(0)))
        if self.z2_layer == 2:
            U2 = self.gc2(U1, final_A).squeeze(0)
            final_U = (U1 + U2)/2
        else:
            final_U = U1

        if self.dis_agg == 'concat':
            h = self.mlp_drop(final_U)
            for i, layer in enumerate(self.z2_mlp_layers):
                h = layer(h)
                if i != len(self.z2_mlp_layers) - 1:
                    h = self.mlp_act(h)
                else:
                    Z2_mu = h[:, :self.z2_mlp_dims[-1]]
                    Z2_logvar = h[:, self.z2_mlp_dims[-1]:]
        else:
            Z2_mu = self.mlp_drop(final_U)
            Z2_logvar = self.mlp_drop(final_U)

            for i, layer in enumerate(self.z2_mlp_mu_layers):
                Z2_mu = layer(Z2_mu)
                if i != len(self.z2_mlp_mu_layers) - 1:
                    Z2_mu = self.mlp_act(Z2_mu)

            for i, layer in enumerate(self.z2_mlp_logvar_layers):
                Z2_logvar = layer(Z2_logvar)
                if i != len(self.z2_mlp_logvar_layers) - 1:
                    Z2_logvar = self.mlp_act(Z2_logvar)

        return [Z2_mu, Z2_logvar]
    
class p_y_gs_z1_z2(nn.Module):
    def __init__(self, args):
        super(p_y_gs_z1_z2, self).__init__()
        self.sample_freq = args.sample_freq
        self.conv = args.y_conv
        self.layer = get_gnn_layer(self.conv, args.z1_dim + args.z2_dim, args.y_gn_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, z1_dis, z2_dis, comm_adjs, diff_adjs):
        Z1_mu, Z1_logvar = z1_dis
        Z2_mu, Z2_logvar = z2_dis

        #sum all adjs
        adj = comm_adjs[0]
        for i in range(1, len(comm_adjs)):
            adj += comm_adjs[i]
        for i in range(len(diff_adjs)):
            adj += diff_adjs[i]

        #sample from z1 and z2
        for i in range(self.sample_freq):
            if i == 0:
                Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                Z2 = self.reparameterize(Z2_mu, Z2_logvar)
                Z1 = torch.unsqueeze(Z1, 0)
                Z2 = torch.unsqueeze(Z2, 0)
            else:
                Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                Z2_ = self.reparameterize(Z2_mu, Z2_logvar)
                Z1_ = torch.unsqueeze(Z1_, 0)
                Z2_ = torch.unsqueeze(Z2_, 0)
                Z1 = torch.cat([Z1, Z1_], 0)
                Z2 = torch.cat([Z2, Z2_], 0)
        Z1 = torch.mean(Z1, 0)
        Z2 = torch.mean(Z2, 0)

        if self.conv == 'mlp':
            h_y = self.layer(torch.cat([Z1, Z2], dim=1))
        else:
            h_y = self.layer(torch.cat([Z1, Z2], dim=1), adj)

        return h_y

class p_gs_e1_e2(nn.Module):
    def __init__(self, n_feat, args):
        super(p_gs_e1_e2, self).__init__()
        self.pos_edge_num = args.g_pos_edge_num
        self.neg_ratio = args.g_neg_ratio
        self.e1_dim = args.g_e1_dim
        self.e2_dim = args.e2_dim

        self.mlp_e1 = nn.Linear(n_feat, self.e1_dim)
        self.mlp_act = get_activation(args.g_mlp_act)

        self.p_e1_e2 = nn.Linear(2 * (self.e1_dim + self.e2_dim * 2), 1)

    def forward(self, feats, e2_c, e2_d, comm_adjs, diff_adjs):
        adj = comm_adjs[0]
        for i in range(1, len(comm_adjs)):
            adj += comm_adjs[i]
        for i in range(len(diff_adjs)):
            adj += diff_adjs[i]

        #transform adj to edge_index
        edge_index = adj.nonzero().T

        edge_num = edge_index.size(1)
        if edge_num > self.pos_edge_num:
            perm = torch.randperm(edge_num)
            idx = perm[:self.pos_edge_num]
            edge_index = edge_index[:, idx]

        e1 = self.mlp_act(self.mlp_e1(feats))
        e2 = torch.stack([e2_c, e2_d], dim=1).view(-1, self.e2_dim * 2)

        # positive edges
        e1_i = F.embedding(edge_index[0], e1)
        e1_j = F.embedding(edge_index[1], e1)
        e2_i = F.embedding(edge_index[0], e2)
        e2_j = F.embedding(edge_index[1], e2)

        b_ij = torch.cat([e1_i, e1_j, e2_i, e2_j], dim=1)
        e_pred_pos = self.p_e1_e2(b_ij)

        # Negative edges
        e_pred_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = edge_index.size(1)
            num_nodes = feats.size(0)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes, (2, num_edges_neg)).cuda()
            e1_i_neg = F.embedding(edge_index_neg[0], e1)
            e1_j_neg = F.embedding(edge_index_neg[1], e1)
            e2_i_neg = F.embedding(edge_index_neg[0], e2)
            e2_j_neg = F.embedding(edge_index_neg[1], e2)
            b_ij_neg = torch.cat([e1_i_neg, e1_j_neg, e2_i_neg, e2_j_neg], dim=1)
            e_pred_neg = self.p_e1_e2(b_ij_neg)

        return e_pred_pos, e_pred_neg
        
class COHF(nn.Module):
    def __init__(self, n_feat, args):
        super(COHF, self).__init__()
        self.device = 'cuda:0' if args.use_cuda else 'cpu'
        self.e2_gs_e1 = q_e2_gs_e1(n_feat, args)
        self.z1_gs_e1 = p_z1_gs_e1(n_feat, args)
        self.z2_e2 = p_z2_e2(n_feat, args)
        self.y_gs_z1_z2 = p_y_gs_z1_z2(args)
        self.gs_e1_e2 = p_gs_e1_e2(n_feat, args)

    def forward(self, data):
        xs, comm_adjs, diff_adjs, feats = data

        comm_adjs = [adj.to(self.device) for adj in comm_adjs]
        diff_adjs = [adj.to(self.device) for adj in diff_adjs]
        feats = feats.to(self.device)

        e2_c, e2_d, e2_c_dis, e2_d_dis = self.e2_gs_e1(comm_adjs, diff_adjs, feats)
        z1_dis = self.z1_gs_e1(comm_adjs, diff_adjs, feats)
        z2_dis = self.z2_e2(e2_c, e2_d, feats)

        pred_pos, pred_neg = self.gs_e1_e2(feats, e2_c, e2_d, comm_adjs, diff_adjs)

        h = self.y_gs_z1_z2(z1_dis, z2_dis, comm_adjs, diff_adjs)

        return h[xs], pred_pos, pred_neg, e2_c_dis, e2_d_dis