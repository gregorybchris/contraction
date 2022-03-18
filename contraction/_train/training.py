from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from contraction._train.contraction_model import ContractionModel
from contraction._train.contraction_dataset import ContractionDataset

N_EPOCHS = 200

if __name__ == '__main__':
    data_dirpath = Path(__file__).parent.parent.parent / 'data' / 'training'
    dataset = ContractionDataset(data_dirpath=data_dirpath)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in loader:
        print(batch.num_graphs)
        print(batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContractionModel(dataset.num_node_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(N_EPOCHS):
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
