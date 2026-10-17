import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GAT(torch.nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.Linear_for_dim1 = pyg_nn.Linear(in_channels=48, out_channels=128, bias=bias)

        # Keep the original residual/MLP structure and only swap message passing op to GAT.
        self.gat1 = pyg_nn.GATConv(in_channels=48, out_channels=128, heads=1, concat=True, bias=bias)
        self.gat2 = pyg_nn.GATConv(in_channels=128, out_channels=128, heads=1, concat=True, bias=bias)
        self.gat3 = pyg_nn.GATConv(in_channels=128, out_channels=128, heads=1, concat=True, bias=bias)
        self.gat4 = pyg_nn.GATConv(in_channels=128, out_channels=128, heads=1, concat=True, bias=bias)
        self.gat5 = pyg_nn.GATConv(in_channels=128, out_channels=128, heads=1, concat=True, bias=bias)
        self.gat6 = pyg_nn.GATConv(in_channels=128, out_channels=128, heads=1, concat=True, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.gat1(x, edge_index)
        x1 = F.relu(x1)
        x1 = x1 + self.Linear_for_dim1(x)

        x2 = self.gat2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = x2 + x1

        x3 = self.gat3(x2, edge_index)
        x3 = F.relu(x3)
        x3 = x3 + x2

        x4 = self.gat4(x3, edge_index)
        x4 = F.relu(x4)
        x4 = x4 + x3

        x5 = self.gat5(x4, edge_index)
        x5 = F.relu(x5)
        x5 = x5 + x4

        x6 = self.gat6(x5, edge_index)
        x6 = F.relu(x6)
        x6 = x6 + x5

        return x6

# ==============================================================================

class GAT_CLI(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.GAT_compound = GAT()
        self.GAT_eluent = GAT()
        self.dropout = dropout

        self.interaction_linear_list = nn.ModuleList(
            [
                pyg_nn.Linear(in_channels=256, out_channels=1024),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(1024),
                nn.Dropout(self.dropout),
                pyg_nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(512),
            ]
        )

        self.pred_list = nn.ModuleList(
            [
                pyg_nn.Linear(in_channels=512, out_channels=256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                pyg_nn.Linear(in_channels=256, out_channels=128),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(128),
                pyg_nn.Linear(128, 1),
            ]
        )

        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GAT_compound(compound_data)
        elu1_GC_data = self.GAT_eluent(elu1_data)
        elu2_GC_data = self.GAT_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)

        concat1_GC_data = torch.cat([compund_pooled, elu1_pooled], dim=-1)
        concat2_GC_data = torch.cat([compund_pooled, elu2_pooled], dim=-1)

        interaction1_GC_data = concat1_GC_data
        interaction2_GC_data = concat2_GC_data
        for layer in self.interaction_linear_list:
            interaction1_GC_data = layer(interaction1_GC_data)
        for layer in self.interaction_linear_list:
            interaction2_GC_data = layer(interaction2_GC_data)

        united_interactions = interaction1_GC_data * (compound_data.ratio[:, 0].view(-1, 1)) + interaction2_GC_data * (compound_data.ratio[:, 1].view(-1, 1))

        retention = united_interactions
        for layer in self.pred_list:
            retention = layer(retention)

        return retention

# ==============================================================================

class GAT_BLI(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.GAT_compound = GAT()
        self.GAT_eluent = GAT()
        self.dropout = dropout

        self.interaction_linear_list = nn.ModuleList(
            [
                pyg_nn.Linear(in_channels=256, out_channels=1024),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(1024),
                nn.Dropout(self.dropout),
                pyg_nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(512),
            ]
        )

        self.pred_list = nn.ModuleList(
            [
                pyg_nn.Linear(in_channels=512, out_channels=256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                pyg_nn.Linear(in_channels=256, out_channels=128),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(128),
                pyg_nn.Linear(128, 1),
            ]
        )

        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GAT_compound(compound_data)
        elu1_GC_data = self.GAT_eluent(elu1_data)
        elu2_GC_data = self.GAT_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled * (compound_data.ratio[:, 0].view(-1, 1)) + elu2_pooled * (compound_data.ratio[:, 1].view(-1, 1))

        mixed_eluent_compond_data = torch.cat([compund_pooled, mixed_eluent], dim=-1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)

        return retention


