import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv,GINConv,SAGEConv
from torch_geometric.nn import GraphConv as GConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from torch.nn import Linear
import brevitas.nn as qnn
from brevitas.quant import Int32Bias
    
class GIN_quan(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3,JK="last", drop_ratio=0, pool='mean'):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set
            
        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """
        
        
        GraphConv = lambda i, h: GINConv(nn.Sequential(qnn.QuantLinear(i, h, bias=True, weight_bit_width=4),
             nn.ReLU(),
             qnn.QuantLinear(h, h, bias=True, weight_bit_width=4)))
      

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        # self.lin1 = qnn.QuantLinear(hid_dim, hid_dim, bias=True, weight_bit_width=4)
        # self.lin2 = qnn.QuantLinear(hid_dim, out_dim, bias=True, weight_bit_width=4)
        # self.lin1 = Linear(hid_dim, hid_dim)
        # self.lin2 = Linear(hid_dim, out_dim)
     

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
      
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        
        graph_emb = self.pool(node_emb, batch.long())
        # graph_emb = self.relu(self.lin1(graph_emb))
        # graph_emb = F.dropout(graph_emb, p=0.5, training=self.training)
        # NOTE: This is a quantized final layer. You probably don't want to be
        # this aggressive in practice.
        # graph_emb = self.lin2(graph_emb)
        
        return graph_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()