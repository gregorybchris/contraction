import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data


class ContractionModel(nn.Module):
    def __init__(self, x_size: int):
        super().__init__()

        self.x_size = x_size

        self.conv1 = GCNConv(x_size, 16)
        self.conv2 = GCNConv(16, 5)
        self.linear = nn.Linear(5, 1)

    def forward(self, data: Data):
        x: torch.Tensor = data.x.long()
        x = F.one_hot(x.squeeze(), self.x_size).float()

        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = global_max_pool(x, data.batch)
        x = self.linear(x)
        x = x.flatten()

        return x
