from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from contraction._solve.color import Color
from contraction._train.contraction_model import ContractionModel
from contraction._train.contraction_dataset import ContractionDataset
from contraction._train.metrics import Metrics

RANDOM_SEED = 42
N_EPOCHS = 100


def train_model(data_dirpath: Path, save: bool, plot: bool):
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
    for epoch in range(N_EPOCHS):
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


def plot_metrics(metrics: Metrics):
    # Plot loss and accuracy
    _, axis_1 = plt.subplots()
    x_series = range(N_EPOCHS)

    axis_1.set_xlabel('Epoch')
    axis_1.set_ylabel('Loss (MSE)')
    train_loss_plot = axis_1.plot(x_series, metrics.get('train-loss'), color='red', label="Train loss")
    test_loss_plot = axis_1.plot(x_series, metrics.get('test-loss'), color='blue', label="Test loss")
    axis_1.tick_params(axis='y')

    axis_2 = axis_1.twinx()
    axis_2.set_ylabel('Accuracy')
    train_accuracy_plot = axis_2.plot(x_series, metrics.get('train-accuracy'), color='green', label="Train accuracy")
    test_accuracy_plot = axis_2.plot(x_series, metrics.get('test-accuracy'), color='orange', label="Test accuracy")
    axis_2.tick_params(axis='y')

    plots = train_loss_plot + test_loss_plot + train_accuracy_plot + test_accuracy_plot
    labels = [plot.get_label() for plot in plots]
    plt.legend(plots, labels, loc=0)

    plt.show()


def save_model(models_dirpath: Path, model: ContractionModel):
    models_dirpath.mkdir(parents=True, exist_ok=True)
    model_filepath = models_dirpath / 'model.pt'
    torch.save(model.state_dict(), model_filepath)


def save_metrics(metrics_dirpath: Path, metrics: Metrics):
    metrics_dirpath.mkdir(parents=True, exist_ok=True)
    metrics.save(metrics_dirpath / 'metrics.json')
