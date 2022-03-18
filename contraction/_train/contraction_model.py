import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class ContractionModel(nn.Module):
    def __init__(self, n_node_features: int):
        super().__init__()

        self.conv1 = GCNConv(n_node_features, 16)
        self.conv2 = GCNConv(16, 5)
        self.linear = nn.Linear(5, 1)

    def forward(self, data: Data):
        x: torch.Tensor = data.x.float()

        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.linear(x)
        x = x.flatten()

        return x
