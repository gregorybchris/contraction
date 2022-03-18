import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class ContractionModel(nn.Module):
    def __init__(self, n_node_features: int, n_classes: int):
        super().__init__()
        self.conv1 = GCNConv(n_node_features, 16)
        self.conv2 = GCNConv(16, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
