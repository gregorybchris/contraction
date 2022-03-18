import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

N_EPOCHS = 200


class GCN(torch.nn.Module):
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
        x = F.log_softmax(x, dim=1)
        return x


def train_model():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(N_EPOCHS):
        if epoch % 20 == 0:
            print(f"Training.. epoch {epoch}/{N_EPOCHS}")
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # if epoch % 20 == 0:
        #     print(f"Loss: {loss}")

    model.eval()
    y_pred = model(data).argmax(dim=1)
    n_correct = (y_pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(n_correct) / int(data.test_mask.sum())
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    train_model()
