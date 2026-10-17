import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class DMPNN(torch.nn.Module):
    """
    chemprop의 bond message passing(Directed edges, reverse-edge 제외)을
    torch_geometric `Data(edge_index, edge_attr, x)`에 맞춰 구현한 분자 임베딩 모듈.

    - 입력: data.x (V x 48), data.edge_index (2 x E), data.edge_attr (E x 1)
    - 출력: node embeddings (V x 128)
    """

    def __init__(
        self,
        d_v: int = 48,
        d_e: int = 1,
        d_h: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth

        # Initial directed-bond hidden state: W_i([V[src], E[e]])
        self.W_i = nn.Linear(d_v + d_e, d_h, bias=bias)
        # Directed-bond message transform: W_h(M_t)
        self.W_h = nn.Linear(d_h, d_h, bias=bias)
        # Atom/vertex finalization: W_vo([V, M_atom])
        self.W_vo = nn.Linear(d_v + d_h, d_h, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.tau = F.relu

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _compute_rev_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        edge_index: (2, E) with edge_index[0]=src, edge_index[1]=dst
        return: rev_edge_index[e] = index of reverse directed edge (dst->src)
        """
        src = edge_index[0]
        dst = edge_index[1]
        E = edge_index.size(1)

        # key encodes (src, dst) pair in the global (batched) node index space
        keys = src * num_nodes + dst
        rev_keys = dst * num_nodes + src

        # Sort keys for fast reverse lookup
        idx_sorted = torch.argsort(keys)
        keys_sorted = keys[idx_sorted]

        pos = torch.searchsorted(keys_sorted, rev_keys)

        # torch.searchsorted can return len(keys_sorted) (insertion at end)
        within = pos < keys_sorted.numel()
        pos_clamped = pos.clamp(max=max(keys_sorted.numel() - 1, 0))

        rev_edge_index = idx_sorted[pos_clamped]

        # Validate mapping
        if not bool(torch.all(within & (keys_sorted[pos_clamped] == rev_keys))):
            # Fallback: build mapping using python dict (robust for rare edge-dup cases)
            k_list = keys.tolist()
            rev_k_list = rev_keys.tolist()
            key_to_idx = {k: i for i, k in enumerate(k_list)}
            rev_edge_index_list = [key_to_idx[rk] for rk in rev_k_list]
            rev_edge_index = torch.tensor(rev_edge_index_list, device=edge_index.device, dtype=torch.long)

        if rev_edge_index.numel() != E:
            raise RuntimeError("rev_edge_index size mismatch")
        return rev_edge_index

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        num_nodes = x.size(0)
        E = edge_index.size(1)

        # Edge feature: expected shape (E, 1)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_attr = data.edge_attr
            if edge_attr.dim() == 1:
                edge_feat = edge_attr.view(-1, 1)
            else:
                edge_feat = edge_attr[:, :1]
        else:
            edge_feat = torch.zeros((E, 1), device=x.device, dtype=x.dtype)

        if E == 0:
            # No bonds: M_atom=0, but keep the same W_vo([V,M]) structure.
            M_atom = torch.zeros(num_nodes, self.d_h, device=x.device, dtype=x.dtype)
            H_v = self.W_vo(torch.cat([x, M_atom], dim=1))
            H_v = self.tau(H_v)
            H_v = self.dropout(H_v)
            return H_v

        src = edge_index[0]
        dst = edge_index[1]

        # 1) Initialize directed bond hidden states
        H0 = self.W_i(torch.cat([x[src], edge_feat], dim=1))  # (E, d_h)
        H = self.tau(H0)

        rev_edge_index = self._compute_rev_edge_index(edge_index, num_nodes)

        # 2) Directed bond message passing
        for _ in range(1, self.depth):
            # M_all[v] = sum_{e: dst[e]=v} H[e]
            M_all = torch.zeros(num_nodes, self.d_h, device=x.device, dtype=x.dtype)
            index_torch = dst.unsqueeze(1).repeat(1, self.d_h)  # (E, d_h)
            M_all.scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)

            # chemprop's bond message: message[e] = M_all[src[e]] - H[rev[e]]
            M = M_all[src] - H[rev_edge_index]

            # Update: H = tau(H0 + W_h(M))
            H = self.W_h(M)
            H = self.tau(H0 + H)
            H = self.dropout(H)

        # 3) Vertex (atom) embeddings via final aggregation
        M_atom = torch.zeros(num_nodes, self.d_h, device=x.device, dtype=x.dtype)
        index_torch = dst.unsqueeze(1).repeat(1, self.d_h)
        M_atom.scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)

        H_v = self.W_vo(torch.cat([x, M_atom], dim=1))  # (V, d_h)
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)
        return H_v


# ================================================================================


class DMPNN_eluent_interaction_3(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.DMPNN_compound = DMPNN()
        self.DMPNN_eluent = DMPNN()
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
        compound_emb = self.DMPNN_compound(compound_data)
        elu1_emb = self.DMPNN_eluent(elu1_data)
        elu2_emb = self.DMPNN_eluent(elu2_data)

        compound_pooled = pyg_nn.global_add_pool(compound_emb, compound_data.batch)
        elu1_pooled = pyg_nn.global_add_pool(elu1_emb, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_emb, elu2_data.batch)

        concat1 = torch.cat([compound_pooled, elu1_pooled], dim=-1)
        concat2 = torch.cat([compound_pooled, elu2_pooled], dim=-1)

        interaction1 = concat1
        interaction2 = concat2
        for layer in self.interaction_linear_list:
            interaction1 = layer(interaction1)
        for layer in self.interaction_linear_list:
            interaction2 = layer(interaction2)

        united_interactions = (
            interaction1 * (compound_data.ratio[:, 0].view(-1, 1))
            + interaction2 * (compound_data.ratio[:, 1].view(-1, 1))
        )

        retention = united_interactions
        for layer in self.pred_list:
            retention = layer(retention)
        return retention


# ================================================================================


class DMPNN_eluent_interaction_5(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.DMPNN_compound = DMPNN()
        self.DMPNN_eluent = DMPNN()
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
        compound_emb = self.DMPNN_compound(compound_data)
        elu1_emb = self.DMPNN_eluent(elu1_data)
        elu2_emb = self.DMPNN_eluent(elu2_data)

        compound_pooled = pyg_nn.global_add_pool(compound_emb, compound_data.batch)
        elu1_pooled = pyg_nn.global_add_pool(elu1_emb, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_emb, elu2_data.batch)

        mixed_eluent = (
            elu1_pooled * (compound_data.ratio[:, 0].view(-1, 1))
            + elu2_pooled * (compound_data.ratio[:, 1].view(-1, 1))
        )

        interaction_input = torch.cat([compound_pooled, mixed_eluent], dim=-1)

        interaction = interaction_input
        for layer in self.interaction_linear_list:
            interaction = layer(interaction)

        retention = interaction
        for layer in self.pred_list:
            retention = layer(retention)
        return retention

