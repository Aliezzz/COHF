import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_activation, pooling_aggregation
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, DenseGraphConv

class Attention(nn.Module):
    def __init__(self, in_dim, att_dim=128):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1, bias=False))

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)   
    
class q_e2_gs_e1(nn.Module):
    def __init__(self, n_feat, args):
        super(q_e2_gs_e1, self).__init__()
        if args.e2_comm_conv == 'gcn':
            self.comm_gnn = DenseGCNConv(n_feat, args.h_dim)
        elif args.e2_comm_conv == 'sage':
            self.comm_gnn = DenseSAGEConv(n_feat, args.h_dim)
        elif args.e2_comm_conv == 'graph_conv':
            self.comm_gnn = DenseGraphConv(n_feat, args.h_dim)

        if args.e2_diff_conv == 'gcn':
            self.diff_gnn = DenseGCNConv(n_feat, args.h_dim)
        elif args.e2_diff_conv == 'sage':
            self.diff_gnn = DenseSAGEConv(n_feat, args.h_dim)
        elif args.e2_diff_conv == 'graph_conv':
            self.diff_gnn = DenseGraphConv(n_feat, args.h_dim)

        self.gnn_act = get_activation(args.e2_gc_act)
        self.gnn_drop = nn.Dropout(args.e2_gc_drop)

        # attention
        self.comm_attention = Attention(args.h_dim, args.att_dim)
        self.diff_attention = Attention(args.h_dim, args.att_dim)
    
        self.mlp_act = get_activation(args.e2_mlp_act)
        self.mlp_drop = nn.Dropout(args.e2_mlp_drop)
        self.mlp_mean_comm = nn.Sequential(nn.Linear(args.h_dim, args.e_dim), self.mlp_act, self.mlp_drop)
        self.mlp_var_comm = nn.Sequential(nn.Linear(args.h_dim, args.e_dim), self.mlp_act, nn.Sigmoid())

        self.mlp_mean_diff = nn.Sequential(nn.Linear(args.h_dim, args.e_dim), self.mlp_act, self.mlp_drop)
        self.mlp_var_diff = nn.Sequential(nn.Linear(args.h_dim, args.e_dim), self.mlp_act, nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, comm_adjs, diff_adjs, feats):
        comm_reps = [self.gnn_drop(self.gnn_act(self.comm_gnn(feats, adj).squeeze(0))) for adj in comm_adjs]

        comm_adj_mean = torch.stack(comm_adjs, dim=0).mean(dim=0)
        diff_reps = [self.gnn_drop(self.gnn_act(self.diff_gnn(feats, adj + comm_adj_mean).squeeze(0))) for adj in diff_adjs]

        comm_agg_reps = self.comm_attention(torch.stack(comm_reps, dim=1))
        diff_agg_reps = self.diff_attention(torch.stack(comm_reps + diff_reps, dim=1))

        mu_comm = self.mlp_mean_comm(comm_agg_reps)
        logvar_comm = self.mlp_var_comm(comm_agg_reps)
        comm_dist = Normal(mu_comm, logvar_comm)
        comm_samples = self.reparameterize(mu_comm, logvar_comm)

        mu_diff = self.mlp_mean_diff(diff_agg_reps)
        logvar_diff = self.mlp_var_diff(diff_agg_reps)
        diff_dist = Normal(mu_diff, logvar_diff)
        diff_samples = self.reparameterize(mu_diff, logvar_diff)

        return comm_dist, comm_samples, diff_dist, diff_samples
    
class p_z1_gs_e1(nn.Module):
    def __init__(self, n_feat, args):
        super(p_z1_gs_e1, self).__init__()
        self.pooling_type = args.z1_pool
        self.z1_layer = args.z1_layer

        if args.z1_conv == 'gcn':
            self.gc1 = DenseGCNConv(n_feat, args.z_dim)
        elif args.z1_conv == 'sage':
            self.gc1 = DenseSAGEConv(n_feat, args.z_dim)
        elif args.z1_conv == 'graph_conv':
            self.gc1 = DenseGraphConv(n_feat, args.z_dim)

        self.gc1_act = get_activation(args.z1_gc1_act)
        self.gc1_drop = nn.Dropout(args.z1_gc1_drop)
        
        if args.z1_layer == 2:
            if args.z1_conv == 'gcn':
                self.gc2 = DenseGCNConv(args.z_dim, args.z_dim)
            elif args.z1_conv == 'sage':
                self.gc2 = DenseSAGEConv(args.z_dim, args.z_dim)
            elif args.z1_conv == 'graph_conv':
                self.gc2 = DenseGraphConv(args.z_dim, args.z_dim)

            self.gc2_act = get_activation(args.z1_gc2_act)
            self.gc2_drop = nn.Dropout(args.z1_gc2_drop)

        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def forward(self, comm_adjs, diff_adjs, feats):
        comm_adj_agg = pooling_aggregation(torch.stack(comm_adjs, dim=0), tar_dim=0, pooling_type=self.pooling_type)
        diff_adj_agg = pooling_aggregation(torch.stack(diff_adjs, dim=0), tar_dim=0, pooling_type=self.pooling_type)

        A_t = torch.stack([comm_adj_agg, diff_adj_agg], dim=2)
        temp = torch.matmul(A_t, self.weight_b)
        temp = torch.squeeze(temp, 2)
        final_A = temp + temp.transpose(0, 1)

        U1 = self.gc1_drop(self.gc1_act(self.gc1(feats, final_A).squeeze(0)))
        if self.z1_layer == 2:
            U2 = self.gc2_drop(self.gc2_act(self.gc2(U1, final_A).squeeze(0)))
            final_U = (U1 + U2)/2
        else:
            final_U = U1

        return final_U
    
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

        return attention
    
class p_z2_e2(nn.Module):
    def __init__(self, n_feat, args):
        super(p_z2_e2, self).__init__()
        self.z2_layer = args.z2_layer
        self.comm_adj_generator = Adj_Generator(args.e_dim)
        self.diff_adj_generator = Adj_Generator(args.e_dim)

        if args.z2_conv == 'gcn':
            self.gc1 = DenseGCNConv(n_feat, args.z_dim)
        elif args.z2_conv == 'sage':
            self.gc1 = DenseSAGEConv(n_feat, args.z_dim)
        elif args.z2_conv == 'graph_conv':
            self.gc1 = DenseGraphConv(n_feat, args.z_dim)

        self.gc1_act = get_activation(args.z2_gc1_act)
        self.gc1_drop = nn.Dropout(args.z2_gc1_drop)

        if args.z2_layer == 2:
            if args.z2_conv == 'gcn':
                self.gc2 = DenseGCNConv(args.z_dim, args.z_dim)
            elif args.z2_conv == 'sage':
                self.gc2 = DenseSAGEConv(args.z_dim, args.z_dim)
            elif args.z2_conv == 'graph_conv':
                self.gc2 = DenseGraphConv(args.z_dim, args.z_dim)
            self.gc2_act = get_activation(args.z2_gc2_act)
            self.gc2_drop = nn.Dropout(args.z2_gc2_drop)

        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def forward(self, comm_samples, diff_samples, feats):
        comm_adj = self.comm_adj_generator(comm_samples)
        diff_adj = self.diff_adj_generator(diff_samples)

        A_t = torch.stack([comm_adj, diff_adj], dim=2)
        temp = torch.matmul(A_t, self.weight_b)
        temp = torch.squeeze(temp, 2)
        final_A = temp + temp.transpose(0, 1)

        U1 = self.gc1_drop(self.gc1_act(self.gc1(feats, final_A).squeeze(0)))
        if self.z2_layer == 2:
            U2 = self.gc2_drop(self.gc2_act(self.gc2(U1, final_A).squeeze(0)))
            final_U = (U1 + U2)/2
        else:
            final_U = U1
    
        return final_U

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class Pred_layer(nn.Module):
    def __init__(self, n_hid, n_class):
        super(Pred_layer, self).__init__()
        self.fc = nn.Linear(n_hid, n_class)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
        
class COHF(nn.Module):
    def __init__(self, n_feat, args):
        super(COHF, self).__init__()
        self.device = 'cuda:0' if args.use_cuda else 'cpu'
        self.e2_gs_e1 = q_e2_gs_e1(n_feat, args)
        self.z1_gs_e1 = p_z1_gs_e1(n_feat, args)
        self.z2_e2 = p_z2_e2(n_feat, args)

    def forward(self, data):
        xs, comm_adjs, diff_adjs, feats = data

        comm_adjs = [adj.to(self.device) for adj in comm_adjs]
        diff_adjs = [adj.to(self.device) for adj in diff_adjs]
        feats = feats.to(self.device)

        comm_dist, comm_samples, diff_dist, diff_samples = self.e2_gs_e1(comm_adjs, diff_adjs, feats)
        z1 = self.z1_gs_e1(comm_adjs, diff_adjs, feats)
        z2 = self.z2_e2(comm_samples, diff_samples, feats)

        h = torch.cat([z1, z2], dim=1)

        return h[xs], comm_dist, diff_dist