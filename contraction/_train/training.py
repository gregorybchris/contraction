import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from contraction._train.contraction_model import ContractionModel

dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch.num_graphs)
    print(batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContractionModel(dataset.num_node_features, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f"Accuracy: {acc:.4f}")
