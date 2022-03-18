from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from contraction._solve.color import Color
from contraction._train.contraction_model import ContractionModel
from contraction._train.contraction_dataset import ContractionDataset


def train_model(data_dirpath: Path):
    torch.manual_seed(42)

    dataset = ContractionDataset(data_dirpath=data_dirpath)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContractionModel(x_size=len(Color)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    n_epochs = 30
    losses = []
    for epoch in range(n_epochs):
        loss_sum = 0
        for batch in loader:
            batch: Data
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = F.mse_loss(y_pred, batch.y)
            loss.backward()
            loss_sum += float(loss)
            optimizer.step()
        total_loss = loss_sum / len(loader)
        losses.append(total_loss)

        print(f"Epoch: {epoch}, loss={loss}")

    plot_loss(losses)

    model.eval()
    # TODO: Create test data splits
    loss = 0
    for batch in loader:
        y_pred = model(batch)
        loss += F.mse_loss(y_pred, batch.y)
    loss /= len(loader)
    print(f"MSE: {loss:.4f}")


def plot_loss(losses: List[float]):
    epochs = range(len(losses))
    plt.plot(epochs, losses)
    plt.show()
