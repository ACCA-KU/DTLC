from networkx import dfs_edges
from rdkit import Chem

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import SimpleConv
from torch_geometric.data import Batch


class GCN(torch.nn.Module):
    def __init__(self, bias = False):
        super().__init__()
        neuron_num = [48,128,128,128,128,128,128]
        self.Linear_for_dim1 = pyg_nn.Linear(in_channels = 48, out_channels = 128, bias = bias)
        self.sc = SimpleConv(aggr='sum', combine_root='sum')
        self.Linear1 = pyg_nn.Linear(neuron_num[0], neuron_num[1], bias = bias)
        self.Linear2 = pyg_nn.Linear(neuron_num[1], neuron_num[2], bias = bias)
        self.Linear3 = pyg_nn.Linear(neuron_num[2], neuron_num[3], bias = bias)
        self.Linear4 = pyg_nn.Linear(neuron_num[3], neuron_num[4], bias = bias)
        self.Linear5 = pyg_nn.Linear(neuron_num[4], neuron_num[5], bias = bias)
        self.Linear6 = pyg_nn.Linear(neuron_num[5], neuron_num[6], bias = bias)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        
    def forward(self, data):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:,0]
        
        x1 = self.sc(x, edge_index)#, edge_weight)
        x1 = self.Linear1(x1)
        x1 = F.relu(x1)
        x1 = x1 + self.Linear_for_dim1(x)
        
        x2 = self.sc(x1, edge_index)#, edge_weight)
        x2 = self.Linear2(x2) 
        x2 = F.relu(x2)
        x2 = x2 + x1 
        
        x3 = self.sc(x2, edge_index)#, edge_weight)
        x3 = self.Linear3(x3)
        x3 = F.relu(x3)
        x3 = x3 + x2
        
        x4 = self.sc(x3, edge_index)#, edge_weight)
        x4 = self.Linear4(x4)
        x4 = F.relu(x4)
        x4 = x4 + x3
        
        x5 = self.sc(x4, edge_index)#, edge_weight)
        x5 = self.Linear5(x5)
        x5 = F.relu(x5)
        x5 = x5 + x4
        
        x6 = self.sc(x5, edge_index)#, edge_weight)
        x6 = self.Linear6(x6)
        x6 = F.relu(x6)
        x6 = x6 + x5

        return x6


class GCN_eluent_interaction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.i = 0
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512, bias = False),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.2),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        #self.i+=1
        #print(self.i)
        #print(mol_GC_pooled.shape)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        
        #print(sol_GC_pooled.shape)

        concat1_GC_data = torch.cat([compund_pooled, elu1_pooled],dim = -1)
        concat2_GC_data = torch.cat([compund_pooled, elu2_pooled],dim = -1)

        interaction1_GC_data = concat1_GC_data
        interaction2_GC_data = concat2_GC_data      
        for layer in self.interaction_linear_list:
            interaction1_GC_data = layer(interaction1_GC_data)
        for layer in self.interaction_linear_list:
            interaction2_GC_data = layer(interaction2_GC_data)

        retention1 = interaction1_GC_data
        retention2 = interaction2_GC_data
        for layer in self.pred_list:
            retention1 = layer(retention1)
        for layer in self.pred_list:
            retention2 = layer(retention2)

        retention = (retention1*(compound_data.ratio[:,0].view(-1,1))) + (retention2*(compound_data.ratio[:,1].view(-1,1)))
        
        return retention



class GCN_eluent_interaction_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        

        concat1_GC_data = torch.cat([compund_pooled, elu1_pooled],dim = -1)
        concat2_GC_data = torch.cat([compund_pooled, elu2_pooled],dim = -1)

        interaction1_GC_data = concat1_GC_data
        interaction2_GC_data = concat2_GC_data      
        for layer in self.interaction_linear_list:
            interaction1_GC_data = layer(interaction1_GC_data)
        for layer in self.interaction_linear_list:
            interaction2_GC_data = layer(interaction2_GC_data)

        w_ratio = interaction1_GC_data*(compound_data.ratio[:,0].view(-1,1))+interaction2_GC_data*(compound_data.ratio[:,1].view(-1,1))

        retention = w_ratio
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention


class GCN_eluent_interaction_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.i = 0
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),#, bias = False),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),#, bias = False),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.2),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        #self.i+=1
        #print(self.i)
        #print(mol_GC_pooled.shape)
        
        #elu1_pooled = pyg_nn.global_mean_pool(elu1_GC_data, elu1_data.batch)
        #elu2_pooled = pyg_nn.global_mean_pool(elu2_GC_data, elu2_data.batch)

        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        
        #print(sol_GC_pooled.shape)

        concat1_GC_data = torch.cat([compund_pooled, elu1_pooled],dim = -1)
        concat2_GC_data = torch.cat([compund_pooled, elu2_pooled],dim = -1)

        interaction1_GC_data = concat1_GC_data
        interaction2_GC_data = concat2_GC_data      
        for layer in self.interaction_linear_list:
            interaction1_GC_data = layer(interaction1_GC_data)
        for layer in self.interaction_linear_list:
            interaction2_GC_data = layer(interaction2_GC_data)

        # united_interactions = torch.cat([interaction1_GC_data*(compound_data.ratio[:,0].view(-1,1)),
        #                                  interaction2_GC_data*(compound_data.ratio[:,1].view(-1,1))], dim = -1)

        united_interactions = interaction1_GC_data*(compound_data.ratio[:,0].view(-1,1)) + interaction2_GC_data*(compound_data.ratio[:,1].view(-1,1))

        retention = united_interactions
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention



class GCN_eluent_interaction_4(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 257, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =1024 , out_channels = 512),#, bias = False),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(in_channels =512 , out_channels = 256),#, bias = False),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),#, bias = False),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.2),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        

        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        

        ##################
        concat1_GC_data = torch.cat([compund_pooled, elu1_pooled],dim = -1)
        w_ratio1 = torch.cat([concat1_GC_data, compound_data.ratio[:,0].view(-1,1)], dim = -1)
        concat2_GC_data = torch.cat([compund_pooled, elu2_pooled],dim = -1)
        w_ratio2 = torch.cat([concat2_GC_data, compound_data.ratio[:,1].view(-1,1)], dim = -1)

        interaction1_GC_data = w_ratio1
        interaction2_GC_data = w_ratio2      
        for layer in self.interaction_linear_list:
            interaction1_GC_data = layer(interaction1_GC_data)
        for layer in self.interaction_linear_list:
            interaction2_GC_data = layer(interaction2_GC_data)

        united_interactions = torch.cat([interaction1_GC_data, interaction2_GC_data], dim = -1)

        retention = united_interactions
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention


# ================================================================================

class GCN_eluent_interaction_5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        mixed_eluent_compond_data = torch.cat([compund_pooled, mixed_eluent],dim = -1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention



class GCN_eluent_interaction_5_gaussNLL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 2)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        mixed_eluent_compond_data = torch.cat([compund_pooled, mixed_eluent],dim = -1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention


class GCN_data_corrector_5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256+1, out_channels = 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(512, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([
            pyg_nn.Linear(in_channels =512 , out_channels = 256),
             nn.LeakyReLU(0.1),
             nn.BatchNorm1d(256),
             pyg_nn.Linear(in_channels =256 , out_channels = 128),
             nn.LeakyReLU(0.1),
             nn.BatchNorm1d(128),
             #nn.Dropout(0.3),
             pyg_nn.Linear(128, 1)
             ])

        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        mixed_eluent_compond_data = torch.cat([compound_data.Q_Rf.view(-1,1), compund_pooled, mixed_eluent],dim = -1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention



class GCN_eluent_interaction_5_GibbsE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.R = 8.314462618
        self.T = 298.15

        self.a_ads_raw  = nn.Parameter(torch.tensor(0.0))  # ads scale (>=0)
        self.b_solv_raw = nn.Parameter(torch.tensor(0.0))  # solv scale (>=0)

        self.HBA_raw_param = nn.Parameter(torch.tensor(0.0))
        self.HBD_raw_param = nn.Parameter(torch.tensor(0.0))
        self.TPSA_raw_param = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.0))  # 초기값: sigmoid(0)=0.5


        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()

        self.comp_elu_MLP = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(512 , 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(256 , 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)
                                        ])


        self.comp_elu_MLP.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compound_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))
        
        mixed_eluent_compond_data = torch.cat([compound_pooled, mixed_eluent],dim = -1)
        Gibbs_solvation = mixed_eluent_compond_data
        for layer in self.comp_elu_MLP:
            Gibbs_solvation = layer(Gibbs_solvation)

        HBA_param  = -F.softplus(self.HBA_raw_param)
        HBD_param  = -F.softplus(self.HBD_raw_param)
        TPSA_param = -F.softplus(self.TPSA_raw_param)

        Gibbs_comp_ads = (HBA_param*(compound_data.compound_param[:,0].view(-1,1)))+(HBD_param*(compound_data.compound_param[:,1].view(-1,1)))+TPSA_param*(compound_data.compound_param[:,2].view(-1,1))
        Gibbs_elu1_ads = (HBA_param*(elu1_data.eluent1_param[:,0].view(-1,1)))+(HBD_param*(elu1_data.eluent1_param[:,1].view(-1,1)))+TPSA_param*(elu1_data.eluent1_param[:,2].view(-1,1))
        Gibbs_elu2_ads = (HBA_param*(elu2_data.eluent2_param[:,0].view(-1,1)))+(HBD_param*(elu2_data.eluent2_param[:,1].view(-1,1)))+TPSA_param*(elu2_data.eluent2_param[:,2].view(-1,1))
        Gibbs_elu_ads = (Gibbs_elu1_ads*(compound_data.ratio[:,0].view(-1,1))) + (Gibbs_elu2_ads*(compound_data.ratio[:,1].view(-1,1)))

        a_ads  = F.softplus(self.a_ads_raw) 
        b_solv = F.softplus(self.b_solv_raw) 
        beta = torch.sigmoid(self.beta_raw)

        sigmoid_input = ((a_ads*(Gibbs_comp_ads - Gibbs_elu_ads) - b_solv*(Gibbs_solvation))/(self.R * self.T)) + beta

            
        return sigmoid_input, Gibbs_comp_ads, Gibbs_elu_ads, Gibbs_solvation, HBA_param, HBD_param, TPSA_param, a_ads, b_solv, beta




class GCN_eluent_interaction_5_area(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()

        self.area_param = nn.Parameter(torch.tensor(0.0))
        self.kamlet_thr = 0.1
        self.step = 42.0

        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        mixed_eluent_compond_data = torch.cat([compund_pooled, mixed_eluent],dim = -1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)

        # contribution : kamlet_taft_pi * ln(1 + rot_bond_num)
        # average_contirbution : contribution * ratio
        elu1_contribution = torch.sigmoid(self.step*(elu1_data.eluent1_param[:,2].view(-1,1)-self.kamlet_thr)) * torch.log(1 + elu1_data.eluent1_param[:,3].view(-1,1))
        elu2_contribution = torch.sigmoid(self.step*(elu2_data.eluent2_param[:,2].view(-1,1)-self.kamlet_thr)) * torch.log(1 + elu2_data.eluent2_param[:,3].view(-1,1)) 
        average_contribution = elu1_contribution*(compound_data.ratio[:,0].view(-1,1))  +elu2_contribution*(compound_data.ratio[:,1].view(-1,1))
        
        compound_area_factor = self.area_param*(compound_data.compound_param[:,3].view(-1,1))*average_contribution

        Rf = retention + compound_area_factor
            
        return Rf



# =================================================================

class GCN_eluent_interaction_6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_affinity = GCN()

        self.pred1_list = nn.ModuleList([pyg_nn.Linear(in_channels = 128, out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(256, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(512, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])


        
        self.pred2_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.pred1_list.apply(self._init_weights)
        self.pred2_list.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, compound_data, elu1_data, elu2_data):

        compound_GC_data = self.GCN_compound(compound_data)
        compound_GC_affinity = self.GCN_affinity(compound_data)
        elu1_GC_affinity = self.GCN_affinity(elu1_data)
        elu2_GC_affinity = self.GCN_affinity(elu2_data)

        compound_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        compound_aff_pooled = pyg_nn.global_add_pool(compound_GC_affinity, compound_data.batch)
        elu1_aff_pooled = pyg_nn.global_add_pool(elu1_GC_affinity, elu1_data.batch)
        elu2_aff_pooled = pyg_nn.global_add_pool(elu2_GC_affinity, elu2_data.batch)
        mixed_eluent = elu1_aff_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_aff_pooled*(compound_data.ratio[:,1].view(-1,1))

        affinity = torch.sum(mixed_eluent * compound_aff_pooled, dim=1, keepdim=True)

        tlc_representation = compound_pooled * affinity


        pred1 = tlc_representation
        for layer in self.pred1_list:
            pred1 = layer(pred1)

        retention = pred1
        for layer in self.pred2_list:
            retention = layer(retention)

        return retention


class GCN_eluent_interaction_7_light(torch.nn.Module):
    def __init__(self, version):
        # version : 'only_interaction' or 'all'
        super().__init__()
        self.version = version
        self.GCN = GCN()

        if self.version == 'all':
            n = int(512)
        elif self.version =='only_interaction':
            n = int(256)
        self.pred1_list = nn.ModuleList([pyg_nn.Linear(in_channels = n, out_channels = n*2),
                                            nn.LeakyReLU(0.1),
                                            nn.BatchNorm1d(n*2),
                                            pyg_nn.Linear(n*2, n*4),
                                            nn.LeakyReLU(0.1),
                                            nn.BatchNorm1d(n*4),
                                            pyg_nn.Linear(n*4, n*2),
                                            nn.LeakyReLU(0.1),
                                            nn.BatchNorm1d(n*2),
                                            pyg_nn.Linear(n*2, n),
                                            nn.LeakyReLU(0.1),
                                            nn.BatchNorm1d(n)])

        
        self.pred2_list = nn.ModuleList([pyg_nn.Linear(in_channels =n , out_channels = int(n/2)),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(int(n/2)),
                                        pyg_nn.Linear(in_channels =int(n/2) , out_channels = int(n/4)),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(int(n/4)),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(int(n/4), 1)])


        self.pred1_list.apply(self._init_weights)
        self.pred2_list.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, compound_data, elu1_data, elu2_data):

        compound_GC_data = self.GCN(compound_data)
        elu1_GC_data = self.GCN(elu1_data)
        elu2_GC_data = self.GCN(elu2_data)

        compound_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        p_interaction = torch.abs(compound_pooled - mixed_eluent)
        n_interaction = mixed_eluent * compound_pooled

        if self.version == 'only_interaction':
            total_interaction = torch.concat([p_interaction, n_interaction], dim=-1)
        elif self.version == 'all':
            total_interaction = torch.concat([compound_pooled, mixed_eluent, p_interaction, n_interaction], dim=-1)

        pred1 = total_interaction
        for layer in self.pred1_list:
            pred1 = layer(pred1)

        retention = pred1
        for layer in self.pred2_list:
            retention = layer(retention)

        return retention




class GCN_eluent_interaction_7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_polar = GCN()
        self.GCN_non_polar = GCN()

        self.pred1_list = nn.ModuleList([pyg_nn.Linear(in_channels = 512, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 2048),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(2048),
                                        pyg_nn.Linear(2048, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])


        
        self.pred2_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.pred1_list.apply(self._init_weights)
        self.pred2_list.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, compound_data, elu1_data, elu2_data):

        comp_list = compound_data.to_data_list()   
        e1_list   = elu1_data.to_data_list()       
        e2_list   = elu2_data.to_data_list()       
        all_list  = comp_list + e1_list + e2_list
        batch_total = Batch.from_data_list(all_list)

        B  = len(comp_list)
        B1 = len(e1_list)
        B2 = len(e2_list)
        assert B == B1 == B2, f"Batch-size mismatch: compound={B}, elu1={B1}, elu2={B2}"

        GC_polar = self.GCN_polar(batch_total)
        GC_non_polar = self.GCN_non_polar(batch_total)

        pooled_p = pyg_nn.global_add_pool(GC_polar, batch_total.batch)
        pooled_n = pyg_nn.global_add_pool(GC_non_polar, batch_total.batch)


        compound_pooled = pooled_p[:B]
        
        elu1_pooled = pooled_p[B:2*B]
        elu2_pooled = pooled_p[2*B:3*B]

        compound_n_pooled = pooled_n[:B]
        elu1_n_pooled = pooled_n[B:2*B]
        elu2_n_pooled = pooled_n[2*B:3*B]

        
        mixed_eluent_p = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))
        mixed_eluent_n = elu1_n_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_n_pooled*(compound_data.ratio[:,1].view(-1,1))

        p_interaction = torch.abs(compound_pooled - mixed_eluent_p)
        n_interaction = mixed_eluent_n * compound_n_pooled

        total_interaction = torch.concat([compound_pooled, mixed_eluent_p, p_interaction, n_interaction], dim=-1)

        pred1 = total_interaction
        for layer in self.pred1_list:
            pred1 = layer(pred1)

        retention = pred1
        for layer in self.pred2_list:
            retention = layer(retention)

        return retention


class GCN_eluent_interaction_8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_polar = GCN()
        self.GCN_non_polar = GCN()

        self.pred1_list = nn.ModuleList([pyg_nn.Linear(in_channels = 260, out_channels = 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(512, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])


        self.pred2_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.pred1_list.apply(self._init_weights)
        self.pred2_list.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, compound_data, elu1_data, elu2_data):

        comp_list = compound_data.to_data_list()   
        e1_list   = elu1_data.to_data_list()       
        e2_list   = elu2_data.to_data_list()       
        all_list  = comp_list + e1_list + e2_list
        batch_total = Batch.from_data_list(all_list)

        B  = len(comp_list)
        B1 = len(e1_list)
        B2 = len(e2_list)
        assert B == B1 == B2, f"Batch-size mismatch: compound={B}, elu1={B1}, elu2={B2}"

        GC_polar = self.GCN_polar(batch_total)
        GC_non_polar = self.GCN_non_polar(batch_total)

        pooled_p = pyg_nn.global_add_pool(GC_polar, batch_total.batch)
        pooled_n = pyg_nn.global_add_pool(GC_non_polar, batch_total.batch)


        compound_pooled = pooled_p[:B]
        
        elu1_pooled = pooled_p[B:2*B]
        elu2_pooled = pooled_p[2*B:3*B]

        compound_n_pooled = pooled_n[:B]
        elu1_n_pooled = pooled_n[B:2*B]
        elu2_n_pooled = pooled_n[2*B:3*B]

        mixed_eluent_p = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))
        mixed_eluent_n = elu1_n_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_n_pooled*(compound_data.ratio[:,1].view(-1,1))

        p_interaction = torch.abs(compound_pooled - mixed_eluent_p)
        n_interaction = mixed_eluent_n * compound_n_pooled

        total_interaction = torch.concat([compound_data.snyder_param.view(-1,4), p_interaction, n_interaction], dim=-1)

        pred1 = total_interaction
        for layer in self.pred1_list:
            pred1 = layer(pred1)

        retention = pred1
        for layer in self.pred2_list:
            retention = layer(retention)

        return retention
        


#########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class SnyderLinear(nn.Module):
    """
    Snyder 입력(앞부분 몇 개 feature)에 대해서
    weight 절댓값을 항상 최소 min_abs 이상으로 보장하는 Linear Layer.
    나머지 입력 weight는 자유롭게 학습됨.
    """
    def __init__(self, in_features, out_features, snyder_dim=4, min_abs=0.001, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.snyder_dim = snyder_dim
        self.min_abs = min_abs
        
        # 기본 weight, bias 정의
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        

    def forward(self, x):
        with torch.no_grad():
            w = self.weight.clone()
            # Snyder 부분에만 절댓값 하한(min_abs) 보장
            w[:, :self.snyder_dim] = (
                torch.sign(w[:, :self.snyder_dim]) *
                (torch.abs(w[:, :self.snyder_dim]) + self.min_abs)
            )
        
        return F.linear(x, w, self.bias)




class GCN_eluent_interaction_9(torch.nn.Module):

    # LeakyReLU --> 다른 함수로 바꿔야 할 수 도 있음
    # (piecewise linear쓰면 gradient를 손실로 못씀 : 2차 도함수가 0이 안되게하기 위해 linear하면 안됨)
    # 아마 GELU 또는 softplus



    def __init__(self):
        super().__init__()
        self.GCN_polar = GCN()
        self.GCN_non_polar = GCN()

        self.pred1_list = nn.ModuleList([SnyderLinear(in_features = 260, out_features = 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(512, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])


        self.pred2_list = nn.ModuleList([SnyderLinear(in_features =512 , out_features = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(256 , 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.pred1_list.apply(self._init_weights)
        self.pred2_list.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear, SnyderLinear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, compound_data, elu1_data, elu2_data):

        comp_list = compound_data.to_data_list()   
        e1_list   = elu1_data.to_data_list()       
        e2_list   = elu2_data.to_data_list()       
        all_list  = comp_list + e1_list + e2_list
        batch_total = Batch.from_data_list(all_list)

        B  = len(comp_list)
        B1 = len(e1_list)
        B2 = len(e2_list)
        assert B == B1 == B2, f"Batch-size mismatch: compound={B}, elu1={B1}, elu2={B2}"

        GC_polar = self.GCN_polar(batch_total)
        GC_non_polar = self.GCN_non_polar(batch_total)

        pooled_p = pyg_nn.global_add_pool(GC_polar, batch_total.batch)
        pooled_n = pyg_nn.global_add_pool(GC_non_polar, batch_total.batch)


        compound_pooled = pooled_p[:B]
        
        elu1_pooled = pooled_p[B:2*B]
        elu2_pooled = pooled_p[2*B:3*B]

        compound_n_pooled = pooled_n[:B]
        elu1_n_pooled = pooled_n[B:2*B]
        elu2_n_pooled = pooled_n[2*B:3*B]

        mixed_eluent_p = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))
        mixed_eluent_n = elu1_n_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_n_pooled*(compound_data.ratio[:,1].view(-1,1))

        p_interaction = torch.abs(compound_pooled - mixed_eluent_p)
        n_interaction = mixed_eluent_n * compound_n_pooled

        total_interaction = torch.concat([compound_data.snyder_param.view(-1,4), p_interaction, n_interaction], dim=-1)

        pred1 = total_interaction
        for layer in self.pred1_list:
             pred1 = layer(pred1)

        retention = pred1
        for layer in self.pred2_list:
            retention = layer(retention)

        return retention



########################################################################


class GCN_eluent_interaction_5s(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 264, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        mixed_eluent_compond_data = torch.cat([compound_data.eluent_param.view(-1,8), compund_pooled, mixed_eluent], dim=-1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention

##################################################################################

class GCN_eluent_interaction_88(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_polar = GCN()
        self.GCN_non_polar = GCN()

        self.pred1_list = nn.ModuleList([pyg_nn.Linear(in_channels = 264, out_channels = 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512),
                                        pyg_nn.Linear(512, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])


        self.pred2_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.pred1_list.apply(self._init_weights)
        self.pred2_list.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, compound_data, elu1_data, elu2_data):

        comp_list = compound_data.to_data_list()   
        e1_list   = elu1_data.to_data_list()       
        e2_list   = elu2_data.to_data_list()       
        all_list  = comp_list + e1_list + e2_list
        batch_total = Batch.from_data_list(all_list)

        B  = len(comp_list)
        B1 = len(e1_list)
        B2 = len(e2_list)
        assert B == B1 == B2, f"Batch-size mismatch: compound={B}, elu1={B1}, elu2={B2}"

        GC_polar = self.GCN_polar(batch_total)
        GC_non_polar = self.GCN_non_polar(batch_total)

        pooled_p = pyg_nn.global_add_pool(GC_polar, batch_total.batch)
        pooled_n = pyg_nn.global_add_pool(GC_non_polar, batch_total.batch)


        compound_pooled = pooled_p[:B]
        
        elu1_pooled = pooled_p[B:2*B]
        elu2_pooled = pooled_p[2*B:3*B]

        compound_n_pooled = pooled_n[:B]
        elu1_n_pooled = pooled_n[B:2*B]
        elu2_n_pooled = pooled_n[2*B:3*B]

        mixed_eluent_p = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))
        mixed_eluent_n = elu1_n_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_n_pooled*(compound_data.ratio[:,1].view(-1,1))

        p_interaction = torch.abs(compound_pooled - mixed_eluent_p)
        n_interaction = mixed_eluent_n * compound_n_pooled

        total_interaction = torch.concat([compound_data.eluent_param.view(-1,8), p_interaction, n_interaction], dim=-1)

        pred1 = total_interaction
        for layer in self.pred1_list:
            pred1 = layer(pred1)

        retention = pred1
        for layer in self.pred2_list:
            retention = layer(retention)

        return retention



###################################################################
