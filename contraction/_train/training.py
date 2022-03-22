from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from contraction._solve.color import Color
from contraction._train.contraction_model import ContractionModel
from contraction._train.contraction_dataset import ContractionDataset
from contraction._train.metrics import Metrics
from contraction._train.plotting import plot_metrics

RANDOM_SEED = 42


def train_model(data_dirpath: Path, save: bool, plot: bool, n_epochs: int):
    torch.manual_seed(RANDOM_SEED)

    solutions_dirpath = data_dirpath / 'solutions'
    metrics_dirpath = data_dirpath / 'metrics'
    models_dirpath = data_dirpath / 'models'

    train_dataset = ContractionDataset(solutions_dirpath, split=ContractionDataset.TRAIN_SPLIT)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = ContractionDataset(solutions_dirpath, split=ContractionDataset.TEST_SPLIT)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContractionModel(len(Color)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    metrics = Metrics()
    for epoch in range(n_epochs):
        metrics.log('epoch', epoch)

        # Train model on train data
        model.train()
        train_loss = 0
        n_correct = 0
        for batch in train_loader:
            batch: Data
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = F.mse_loss(y_pred, batch.y)
            loss.backward()
            train_loss += float(loss)
            y_pred_int = torch.round(y_pred)
            n_correct += 1 if y_pred_int == batch.y else 0
            optimizer.step()
        train_loss /= len(train_loader)
        metrics.log('train-loss', train_loss)
        train_accuracy = n_correct / len(train_loader)
        metrics.log('train-accuracy', train_accuracy)

        # Evaluate model on test data
        model.eval()
        test_loss = 0
        n_correct = 0
        for batch in test_loader:
            y_pred = model(batch)
            loss = F.mse_loss(y_pred, batch.y)
            test_loss += float(loss)
            y_pred_int = torch.round(y_pred)
            n_correct += 1 if y_pred_int == batch.y else 0
            # print(f"{int(batch.y)}, {int(y_pred_int)}, {float(y_pred):0.2f}")
        test_loss /= len(test_loader)
        metrics.log('test-loss', test_loss)
        test_accuracy = n_correct / len(test_loader)
        metrics.log('test-accuracy', test_accuracy)

        print(f"Epoch: {epoch}, "
              f"train loss={train_loss:0.3f}, "
              f"test loss={test_loss:0.3f}, "
              f"train accuracy={train_accuracy * 100:0.1f}%, "
              f"test accuracy={test_accuracy * 100:0.1f}%")

    if save:
        save_model(models_dirpath, model)
        save_metrics(metrics_dirpath, metrics)

    if plot:
        plot_metrics(metrics)


def save_model(models_dirpath: Path, model: ContractionModel):
    models_dirpath.mkdir(parents=True, exist_ok=True)
    model_filepath = models_dirpath / 'model.pt'
    torch.save(model.state_dict(), model_filepath)


def save_metrics(metrics_dirpath: Path, metrics: Metrics):
    metrics_dirpath.mkdir(parents=True, exist_ok=True)
    metrics.save(metrics_dirpath / 'metrics.json')
