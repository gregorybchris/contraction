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

RANDOM_SEED = 42
N_EPOCHS = 100


def train_model(data_dirpath: Path):
    torch.manual_seed(RANDOM_SEED)

    train_dataset = ContractionDataset(data_dirpath=data_dirpath, split=ContractionDataset.TRAIN_SPLIT)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = ContractionDataset(data_dirpath=data_dirpath, split=ContractionDataset.TEST_SPLIT)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContractionModel(x_size=len(Color)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_losses = []
    test_losses = []

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch: Data
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = F.mse_loss(y_pred, batch.y)
            loss.backward()
            train_loss += float(loss)
            optimizer.step()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        for batch in test_loader:
            y_pred = model(batch)
            loss = F.mse_loss(y_pred, batch.y)
            test_loss += float(loss)
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch: {epoch}, training loss={train_loss}, test loss={test_loss}")

    plt.plot(range(N_EPOCHS), train_losses, label="Train loss")
    plt.plot(range(N_EPOCHS), test_losses, label="Test loss")
    plt.legend(loc="upper right")
    plt.show()
