import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
import numpy as np

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from torch_geometric.nn import GCNConv


class TemporalConv(MessagePassing):
    def __init__(self, dim, num_relation, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu"):
        super(TemporalConv, self).__init__()
        self.dim = dim
        self.num_relation = num_relation
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        
        self.layer_norm = nn.LayerNorm(dim) if layer_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

        self.neighbor_linear = nn.Linear(dim, dim)
        self.self_linear = nn.Linear(dim, dim)

        if self.aggregate_func == "pna":
            self.agglinear = nn.Linear(dim * 13, dim)
        else:
            self.agglinear = nn.Linear(dim * 2, dim)

        # init the transform matrix for layer input
        self.input_linear_U = nn.Linear(dim, dim)

        # obtain relation embeddings as a projection of the origin relation embedding
        self.relation_linear = nn.Linear(dim, dim)
        self.eps = 1e-6
    
    def forward(self, curr_g, node_emb, rel_emb, pre_node_stat):
        edge_index = curr_g.edge_index
        edge_type = curr_g.edge_type
        edge_rel_emb = self.relation_linear(rel_emb) # (edge_num, dim)
        node_emb = node_emb + self.input_linear_U(pre_node_stat)
        output = self.propagate(edge_index=edge_index, edge_type=edge_type, 
                                node_emb=node_emb, edge_rel_emb=edge_rel_emb, 
                                pre_node_stat=pre_node_stat)
        return output
    
    def propagate(self, edge_index, size=None, **kwargs):
        return super(TemporalConv, self).propagate(edge_index, size, **kwargs)
    
    def message(self, node_emb_j, edge_rel_emb, node_emb):
        if self.message_func == "transe":
            message = node_emb_j + edge_rel_emb
        elif self.message_func == "distmult":
            message = node_emb_j * edge_rel_emb
        elif self.message_func == "rotate":
            x_j_re, x_j_im = node_emb_j.chunk(2, dim=-1)
            r_j_re, r_j_im = edge_rel_emb.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        
        # neighbor linear
        nb_message = self.neighbor_linear(message)
        # self-loop
        sf_message = self.self_linear(node_emb)

        # concat
        msg = torch.cat([nb_message, sf_message], dim=self.node_dim)
        return msg
    
    def aggregate(self, inputs, index, dim_size):
        # augment aggregation index with self-loops
        index = torch.cat([index, torch.arange(dim_size, device=inputs.device)])

        if self.aggregate_func == "pna":
            mean = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(inputs ** 2, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2).squeeze(0)
        else:
            output = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)
        return output
    
    def update(self, update, node_emb):
        # update: to-update
        # node_emb: old states
        # node update as a function of old states (node_emb) and this layer output (update)
        output = self.agglinear(torch.cat([node_emb, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
class rgraphConv(MessagePassing):
    def __init__(self, dim, num_relation, num_node, layer_norm=False, activation="relu"):
        super(rgraphConv, self).__init__()
        self.dim = dim
        self.num_relation = num_relation
        self.num_node = num_node
        
        self.layer_norm = nn.LayerNorm(dim) if layer_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

        self.conv = GCNConv(dim, dim)
    

    def create_adj(self, history_list):
        num_ent, num_rel = self.num_node, self.num_relation
        
        # concatenate all of the edge_index in history_list graph
        triplet = []
        for hist in history_list:
            triplet.append(torch.stack([hist.edge_index[0], hist.edge_type, hist.edge_index[1]]))
        triplet = torch.cat(triplet, dim=-1).t().cpu().numpy()

        ind_h = triplet[:,:2]
        ind_t = triplet[:,1:]
        
        E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, num_rel))
        E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, num_rel))

        # D_h^-2
        diag_vals_h = E_h.sum(axis=1).A1
        diag_vals_h[diag_vals_h!=0] = 1/(diag_vals_h[diag_vals_h!=0]**2)

        # D_t^-2
        diag_vals_t = E_t.sum(axis=1).A1
        diag_vals_t[diag_vals_t!=0] = 1/(diag_vals_t[diag_vals_t!=0]**2)

        # D_h^-2 and D_t^-2
        D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
        D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))

        # A_h and A_t
        A_h = E_h.transpose() @ D_h_inv @ E_h
        A_t = E_t.transpose() @ D_t_inv @ E_t
        A = A_h + A_t

        return A



    def gen_graph(self, history_list, rel_emb):
        A = self.create_adj(history_list)
        A = torch.tensor(A.toarray(), dtype=torch.float32)
        edge_index = torch.nonzero(A, as_tuple=False).t().long()
        edge_weight = A[A != 0]

        rgraph = Data(x=rel_emb, edge_index=edge_index, edge_weight=edge_weight, num_nodes=self.num_relation)

        return rgraph  
      
    def forward(self, rgraph):
        x = self.conv(rgraph.x, rgraph.edge_index, edge_weight=rgraph.edge_weight)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.activation:
            x = self.activation(x)
  
        x = F.dropout(x, p=0.2, training=self.training)
        return x+ rgraph.x
