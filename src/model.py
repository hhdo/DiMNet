from collections.abc import Sequence
import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter
import layers
import decoder
from torch_geometric.data import Data


class DiMNet(nn.Module):
    def __init__(self, dim, num_layer, num_relation, num_node, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", 
                 history_len=10, topk=30,
                 input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2, num_head=1):
        super(DiMNet, self).__init__()
        self.dim = dim
        self.num_layer = num_layer
        self.num_relation = num_relation *2 # reverse rel type should be added
        self.num_node = num_node
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.short_cut = short_cut
        self.layer_norm = layer_norm
        self.activation = activation
        self.history_len = history_len

        self.gen_edge_topk = topk
        
        self.dynGRUcell = nn.GRUCell(self.dim, self.dim)

        # model pre-define params
        self.relemb = nn.Embedding(self.num_relation, self.dim)
        self.nodeemb = nn.Embedding(self.num_node, self.dim)
        self.selfrel = nn.Embedding(1, self.dim)
        
        self.layers = nn.ModuleList()
        self.hiddenGates = nn.ModuleList()
        for _ in range(self.num_layer):
            self.layers.append(layers.TemporalConv(self.dim, self.num_relation, self.message_func, self.aggregate_func,
                                                   self.layer_norm, self.activation))
            self.hiddenGates.append(nn.Linear(self.dim, self.dim))
        self.rgraphConv = layers.rgraphConv(self.dim, self.num_relation, self.num_node, self.layer_norm, self.activation)

        self.stainit = nn.Sequential(
            nn.Linear(self.dim*2, self.dim*2),
            nn.ReLU(),
            nn.Linear(self.dim*2, self.dim)
        )
        

        self.gateW = nn.Linear(self.dim, self.dim)


        self.q_linear = nn.Linear(2*dim, dim)
        self.k_linear = nn.Linear(2*dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.dyn_weight = nn.Parameter(torch.Tensor(dim, dim))

        self.decoder = decoder.ConvTransE(dim, input_dropout, hidden_dropout, feat_dropout)
        self.cl_margin = 0.5
        self.clloss = nn.CosineEmbeddingLoss(margin=self.cl_margin, reduction="mean")
        self.lossfun = nn.CrossEntropyLoss()

    def disentangleFeat(self, curr_stat, last_stat, last_g):

        # add self-loop edge in last_g
        all_node = torch.arange(last_g.num_nodes, device=last_g.edge_index.device)
        self_loop_edge = torch.stack([all_node, all_node])
        edge_index = torch.cat([last_g.edge_index, self_loop_edge], dim=-1)
        common_type = last_g.edge_type
        type_emb = torch.cat([self.relemb(common_type), 
                              self.selfrel(torch.zeros(last_g.num_nodes, dtype=torch.long, device=last_g.edge_index.device))], dim=0)
        
        q = self.q_linear(torch.cat([curr_stat[edge_index[1]], type_emb], dim=-1)).view(-1, self.num_head, self.head_dim) # Edge * H * Head_dim
        k = self.k_linear(torch.cat([last_stat[edge_index[0]], type_emb], dim=-1)).view(-1, self.num_head, self.head_dim) # Edge * H * Head_dim
        v = self.v_linear(last_stat[edge_index[0]]).view(-1, self.num_head, self.head_dim) # Edge * H * Head_dim

        attn_score = torch.sum(q * k, dim=-1) / self.head_dim**0.5  # Edge * H

        # dynamic attention   
        dyn_prob = softmax(attn_score, edge_index[1]) # Edge * H
        feat_dyn = scatter((dyn_prob.unsqueeze(-1) * v).view(-1, self.dim), edge_index[1].unsqueeze(-1).expand(-1, self.dim), 
                           dim=0, dim_size=last_g.num_nodes, reduce="mean")
        
        # stable attention
        sta_prob = softmax(-attn_score, edge_index[1])
        feat_sta = scatter((sta_prob.unsqueeze(-1) * v).view(-1, self.dim), edge_index[1].unsqueeze(-1).expand(-1, self.dim),
                           dim=0, dim_size=last_g.num_nodes, reduce="mean")

        return feat_sta, feat_dyn@self.dyn_weight
    

    def forward(self, history_list, query_triple):
        first_g = history_list[0]
        mod_device = first_g.edge_index.device
        
        # init a empty graph
        ept_g = Data(edge_index=torch.zeros((2, 0), dtype=torch.long, device=mod_device), 
                     edge_type=torch.tensor([], device=mod_device, dtype=torch.long), 
                     num_nodes=first_g.num_nodes)
        
        # (w+1, node_emb_matrix) as the inital last timestamp state list, where ind=0 is the init state
        hist_embly_list = (self.num_layer + 1) * [self.nodeemb.weight]  
        
        # the initial dynamic and stable feature
        tempo_feat_dyn = torch.zeros((first_g.num_nodes,self.dim), device=mod_device)
        tempo_feat_sta = torch.zeros((first_g.num_nodes,self.dim), device=mod_device)

        # the initial disentangle results
        last_feat_sta, last_feat_dyn = None, None
        cl_loss = 0

        rgraph = self.rgraphConv.gen_graph(history_list, self.relemb.weight).to(mod_device)
        rel_emb = self.rgraphConv.forward(rgraph)
        
        for idx_ts, g in enumerate(history_list): 
            # temporal layer caculation
            evo_state = self.evo_module(g, rel_emb, hist_embly_list, tempo_feat_sta, tempo_feat_dyn)
            
            # caculate the dynamic feature and stable feature
            last_g = history_list[idx_ts-1] if idx_ts > 0 else ept_g
            feat_sta, feat_dyn = self.disentangleFeat(curr_stat=evo_state[-1], 
                                                      last_stat=hist_embly_list[-1], 
                                                      last_g=last_g)
            if self.training:
                if idx_ts > 0 and last_feat_sta is not None and last_feat_dyn is not None:
                    loss_ss = self.clloss(feat_sta, last_feat_sta, torch.ones(feat_sta.size(0), device=mod_device))                    
                    cl_loss += loss_ss
                last_feat_sta, last_feat_dyn = feat_sta, feat_dyn

            # update the dynGRUcell and stable feature
            tempo_feat_dyn = self.dynGRUcell(feat_dyn, tempo_feat_dyn)            
            tempo_feat_sta = feat_sta
            
            # update the hist_embly_list
            hist_embly_list = evo_state

        # score funtion
        score = self.decoder.forward(hist_embly_list[-1], self.relemb.weight, query_triple[:,[0,2,1]], mode="test")
        

        # generae a virtual graph based on the reasoning score matrix
        virtual_g = self.gen_virtual_graph(score, query_triple[:,[0,2]])
        # evo once again to generate the final state
        virtual_state = self.evo_module(virtual_g, rel_emb, hist_embly_list, tempo_feat_sta, tempo_feat_dyn)
        # score funtion
        score = self.decoder.forward(virtual_state[-1], self.relemb.weight, query_triple[:,[0,2,1]], mode="test")

        return score, cl_loss+ 0.1* torch.norm(self.dyn_weight, p=2) if cl_loss > 0 else cl_loss
    
    def evo_module(self, g, rel_emb, hist_embly_list, tempo_feat_sta, tempo_feat_dyn):
        # MeanPool to get the  node embedding at one timestamp
        node_emb_mp = scatter(rel_emb[g.edge_type], g.edge_index[1].unsqueeze(-1).expand(-1, self.dim), 
                dim=0, dim_size=g.num_nodes, reduce="mean") 
        
        initGate = torch.sigmoid(self.gateW(tempo_feat_sta))
        initEnt = initGate * hist_embly_list[-1] + (1-initGate) * self.nodeemb.weight
        # current init state is derived from the [last timestamp state] and [the current pooling of node embedding] 
        layer_input = self.stainit(torch.cat([initEnt, node_emb_mp], dim=-1))
        layer_input = F.normalize(layer_input) if self.layer_norm else layer_input
        
        update_emb_list  = list()
        update_emb_list.append(layer_input)
        for idx_l, layer in enumerate(self.layers):
            hidden = layer(g, layer_input, rel_emb[g.edge_type], hist_embly_list[idx_l+1])
            if self.short_cut:
                hidden = hidden + layer_input
                hidden_gate = torch.sigmoid(self.hiddenGates[idx_l](tempo_feat_dyn))
                hidden = hidden_gate * hidden + (1-hidden_gate) * hist_embly_list[idx_l+1]
            update_emb_list.append(hidden)
            layer_input = hidden

        return update_emb_list
    
    def get_loss(self, pred, trueValue):
        return self.lossfun(pred, trueValue)
    
    def gen_virtual_graph(self, score, triplet_hr):
        # get the topk edge
        topk_score, topk_ind = torch.topk(score, self.gen_edge_topk, dim=-1)
        edge_h = triplet_hr[:,0].unsqueeze(-1).expand(-1, self.gen_edge_topk).reshape(-1)
        edge_r = triplet_hr[:,1].unsqueeze(-1).expand(-1, self.gen_edge_topk).reshape(-1)
        edge_t = topk_ind.reshape(-1)
        virtual_g = Data(edge_index=torch.stack([edge_h, edge_t]), 
                         edge_type=edge_r, num_nodes=score.size(1))
        virtual_g = virtual_g.to(score.device)
        return virtual_g


