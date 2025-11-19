# utils_rehub.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReHubLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4):
        super(ReHubLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=out_features, num_heads=n_heads, batch_first=True)
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, h):
        """
        h: (batch_size, num_nodes, in_features)
        """
        # Project to out_features
        h = self.linear(h)  # (batch_size, num_nodes, out_features)

        # Multi-head attention expects 3D tensor: (batch_size, seq_len, embed_dim) with batch_first=True
        h2, _ = self.attn(h, h, h)  # shape: (batch_size, num_nodes, out_features)

        # Residual connection + layer norm
        h = self.norm(h + h2)
        return h


class ReHubNet(nn.Module):
    def __init__(self, in_features, hidden_dim=32, hidden_mlp=64, num_layers=3, num_classes=2):
        super(ReHubNet, self).__init__()
        self.layers = nn.ModuleList()
        # First layer: input -> hidden
        self.layers.append(ReHubLayer(in_features, hidden_dim))
        # Hidden layers
        for _ in range(num_layers-1):
            self.layers.append(ReHubLayer(hidden_dim, hidden_dim))
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, num_classes)
        )

    def forward(self, data):
        """
        data: torch_geometric batch
        data.x: node features (num_nodes_in_batch, num_features)
        data.batch: batch vector mapping nodes to graphs
        """
        x, batch = data.x, data.batch

        # Prepare node features for ReHub layers
        batch_size = int(batch.max().item()) + 1
        num_nodes = [int((batch==i).sum()) for i in range(batch_size)]
        
        # Split into a padded tensor: (batch_size, max_nodes, in_features)
        max_nodes = max(num_nodes)
        in_features = x.size(1)
        x_padded = x.new_zeros((batch_size, max_nodes, in_features))
        mask = x.new_zeros((batch_size, max_nodes), dtype=torch.bool)
        for i in range(batch_size):
            idx = (batch==i).nonzero(as_tuple=True)[0]
            n = idx.size(0)
            x_padded[i, :n, :] = x[idx]
            mask[i, :n] = True

        # Pass through ReHub layers
        h = x_padded
        for layer in self.layers:
            h = layer(h)

        # Mean pool over nodes (only valid nodes)
        h_masked = h * mask.unsqueeze(-1)  # mask invalid nodes
        node_counts = mask.sum(dim=1).unsqueeze(-1)
        h_graph = h_masked.sum(dim=1) / node_counts  # (batch_size, hidden_dim)

        # MLP head
        out = self.mlp(h_graph)
        return out
