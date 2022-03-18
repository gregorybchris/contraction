from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from contraction._train.contraction_model import ContractionModel
from contraction._train.contraction_dataset import ContractionDataset


def train_model(data_dirpath: Path):
    torch.manual_seed(42)

    dataset = ContractionDataset(data_dirpath=data_dirpath)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContractionModel(dataset.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    n_epochs = 10
    for epoch in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = F.mse_loss(y_pred, batch.y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, loss={loss}")

    model.eval()
    # TODO: Create test data splits
    loss = 0
    for batch in loader:
        y_pred = model(batch)
        loss += F.mse_loss(y_pred, batch.y)
    loss /= len(loader)
    print(f"MSE: {loss:.4f}")
