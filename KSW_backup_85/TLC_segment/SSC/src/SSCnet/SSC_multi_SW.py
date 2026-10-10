import torch
import torch.nn as nn

import dgl
from dgl.nn import SumPooling, AvgPooling
from D4CMPP_TLC.networks.src.Linear import Linears
from D4CMPP_TLC.networks.src.GAT import GATs, GAT_layer
from D4CMPP_TLC.networks.src.MPNN import MPNN_layer
from D4CMPP_TLC.networks.src.distGCN import distGCN_layer

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()

        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.1)
        linear_layers = config.get('linear_layers', 3)
        target_dim = config['target_dim']
        solvent_dim = config.get('solvent_dim', 64)
        solv_gcn_layers = config.get('solvent_conv_layers', 4)

        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(config['edge_dim'], hidden_dim, bias=False)
        )

        self.embedding_solv_lin = nn.Sequential(
            nn.Linear(config['node_dim'], solvent_dim, bias=False)
        )

        # self.ISATconv = ISATconvolution(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers,dropout, False, True, 0.1)
        self.main_conv = GATs(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers, dropout, False, True)
        self.a0_linear = Linears( hidden_dim, 1, nn.ReLU(), linear_layers, dropout, False)

        self.reduce = SumPooling()
        self.GATs_solv = GATs(solvent_dim, solvent_dim, solvent_dim, nn.ReLU(), solv_gcn_layers, dropout, False, True) 

        self.SElayer0 = nn.Sequential(
            nn.Linear(hidden_dim+solvent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), 1),
        )        

    
    def forward(self, **kargs):
        graph =  kargs['Smiles_graphs']
        r_node = kargs['Smiles_r_node']
        i_node = kargs['Smiles_i_node']
        r_edge = kargs['Smiles_r2r_edge']
        d_edge = kargs['Smiles_d2d_edge']

        graph_elu1 =  kargs['eluent1_smiles_graphs']
        r_node_elu1 = kargs['eluent1_smiles_r_node']
        i_node_elu1 = kargs['eluent1_smiles_i_node']
        r_edge_elu1 = kargs['eluent1_smiles_r2r_edge']
        d_edge_elu1 = kargs['eluent1_smiles_d2d_edge']

        graph_elu2 =  kargs['eluent2_smiles_graphs']
        r_node_elu2 = kargs['eluent2_smiles_r_node']
        i_node_elu2 = kargs['eluent2_smiles_i_node']
        r_edge_elu2 = kargs['eluent2_smiles_r2r_edge']
        d_edge_elu2 = kargs['eluent2_smiles_d2d_edge']

        ratio1 = kargs['ratio1_var']
        ratio2 = kargs['ratio2_var']

        # solute convolution
        r_node = r_node.float()
        r_node2 = self.embedding_rnode_lin(r_node)

        i_node = self.embedding_inode_lin(r_node) # i_node = r_node
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))

        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        # r_node3, d_node1, d_node2 = self.ISATconv(graph, r_node2, r_edge, i_node, d_edge)
        r_node3 = self.main_conv(real_graph, r_node2)
        h = self.reduce(real_graph, r_node3)
        # h_expanded = torch.repeat_interleave(h, dot_graph.batch_num_nodes().int(), dim=0)

        # a0s = self.a0_linear(d_node2)
        # dot_graph.nodes['d_nd'].data['a0s'] = a0s
        # a0 = dgl.nn.MaxPooling()(dot_graph, a0s)

        # eluent1 convolution
        elu1_graph = graph_elu1.node_type_subgraph(['r_nd'])
        elu1_graph.set_batch_num_nodes(graph_elu1.batch_num_nodes('r_nd'))
        elu1_graph.set_batch_num_edges(graph_elu1.batch_num_edges('r2r'))

        elu1_node_feats = r_node_elu1.float()
        elu1_node_feats = self.embedding_solv_lin(elu1_node_feats)

        solv_h1 = self.GATs_solv(elu1_graph, elu1_node_feats)
        solv_h1 = self.reduce(elu1_graph, solv_h1)

        # eluent2 convolution
        elu2_graph = graph_elu2.node_type_subgraph(['r_nd'])
        elu2_graph.set_batch_num_nodes(graph_elu2.batch_num_nodes('r_nd'))
        elu2_graph.set_batch_num_edges(graph_elu2.batch_num_edges('r2r'))

        elu2_node_feats = r_node_elu2.float()
        elu2_node_feats = self.embedding_solv_lin(elu2_node_feats)
        
        solv_h2 = self.GATs_solv(elu2_graph, elu2_node_feats)
        solv_h2 = self.reduce(elu2_graph, solv_h2)

        solv_h = solv_h1*ratio1 + solv_h2*ratio2
        
        # Weight sum of the eluent contributions
        # expanded_indices = torch.repeat_interleave( torch.arange(solv_h1.shape[0]).to(device=solv_h1.device), dot_graph.batch_num_nodes().int() )
        y = self.SElayer0( torch.cat([h, solv_h], dim=-1) )
        # y += ratio1**0.5*a2 + ratio2**0.5*b2
        # y += ratio1**2*a3 + ratio2**2*b3

        if kargs.get('get_score',False):
            return {'a0': a0, 'a1': a1, 'b1': b1, 'y': y}
        return y
        
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])


class ISATconvolution(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, alpha=0.1, max_dist = 4):
        super().__init__()        
        # Message Passing
        self.r2r = nn.ModuleList([GAT_layer(in_node_feats, out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])
        self.i2i = nn.ModuleList([GAT_layer(out_feats, out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])

        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        self.d2d = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)
        
        self.reduce = SumPooling()
    
    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge):
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))
        
        image_graph=graph.node_type_subgraph(['i_nd'])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes('i_nd'))
        image_graph.set_batch_num_edges(graph.batch_num_edges('i2i'))
                
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        for i in range(len(self.r2r)):
            r_node = self.r2r[i](real_graph, r_node)
            i_node = self.i2i[i](image_graph, i_node)
        d_node1 = self.i2d(graph, i_node)
        d_node2 = self.d2d(dot_graph, d_node1, d2d_edge)
        return r_node, d_node1, d_node2
        

class r2i_layer(nn.Module):
    def forward(self,graph, r_node, i_node):
        return i_node+r_node
    
class i2s_layer(nn.Module):
    def forward(self,graph, i_node):
        with graph.local_scope():
            graph=graph.edge_type_subgraph([('i_nd','i2d','d_nd')])
            graph.nodes['i_nd'].data['h']= i_node
            graph.update_all(dgl.function.copy_u('h', 'mail'), dgl.function.sum('mail', 'h'))
            d_node = graph.nodes['d_nd'].data['h']
        return d_node    


class s2r_Layer(nn.Module):    
    def forward(self, graph, node):
        with graph.local_scope():
            graph=graph.edge_type_subgraph([('d_nd','d2r','r_nd')])
            graph.nodes['d_nd'].data['h']= node
            graph.update_all(dgl.function.copy_u('h', 'mail'), dgl.function.sum('mail', 'h'))
            score = graph.nodes['r_nd'].data['h']
        return score
