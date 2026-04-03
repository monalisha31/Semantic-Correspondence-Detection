import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProjectionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.3,
                 use_gat: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self._use_pyg = False

        try:
            from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
            self._use_pyg = True

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            if use_gat:
                self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            for _ in range(num_layers - 2):
                if use_gat:
                    self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
                else:
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            if use_gat:
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, output_dim))

            self.pool_mean = global_mean_pool
            self.pool_max = global_max_pool

        except ImportError:
            print("Warning: torch_geometric not available. Using MLP fallback for GNN.")
            self.fallback = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.projection = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_pyg:
            for i in range(self.num_layers - 1):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x, inplace=True)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.convs[-1](x, edge_index)

            h_mean = self.pool_mean(x, batch)
            h_max = self.pool_max(x, batch)
            h = torch.cat([h_mean, h_max], dim=1)
        else:
            h = self.fallback(x)
            unique_batches = batch.unique()
            pooled = []
            for b in unique_batches:
                mask = (batch == b)
                pooled.append(h[mask].mean(dim=0))
            h = torch.stack(pooled)
            h = torch.cat([h, h], dim=1)

        z = self.projection(h)
        return h, z


class GatedFusion(nn.Module):

    def __init__(self, text_dim: int, graph_dim: int, fused_dim: int,
                 gate_hidden: int = 128):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Linear(text_dim + graph_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, fused_dim),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        g = self.graph_proj(graph_emb)

        combined = torch.cat([text_emb, graph_emb], dim=1)
        alpha = self.gate(combined)

        fused = alpha * t + (1 - alpha) * g

        return self.output(fused)

    def get_gate_values(self, text_emb: torch.Tensor,
                        graph_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([text_emb, graph_emb], dim=1)
        return self.gate(combined)


class ModelEquivPipeline(nn.Module):

    def __init__(self, text_projector: ProjectionHead,
                 gnn_encoder: Optional[GNNEncoder] = None,
                 fusion: Optional[GatedFusion] = None,
                 mode: str = "text_only"):
        super().__init__()
        self.text_projector = text_projector
        self.gnn_encoder = gnn_encoder
        self.fusion = fusion
        self.mode = mode

    def forward(self, text_emb: torch.Tensor,
                graph_data: Optional[dict] = None) -> torch.Tensor:
        z_text = self.text_projector(text_emb)

        if self.mode == "text_only" or self.gnn_encoder is None:
            return z_text

        if graph_data is not None and self.mode in ("graph_only", "fused"):
            _, z_graph = self.gnn_encoder(
                graph_data['x'], graph_data['edge_index'], graph_data['batch']
            )

            if self.mode == "graph_only":
                return z_graph

            if self.fusion is not None:
                return self.fusion(z_text, z_graph)

        return z_text
