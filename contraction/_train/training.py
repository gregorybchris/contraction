from pathlib import Path

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

    training_dirpath = data_dirpath / 'training'
    train_dataset = ContractionDataset(training_dirpath, split=ContractionDataset.TRAIN_SPLIT)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = ContractionDataset(training_dirpath, split=ContractionDataset.TEST_SPLIT)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContractionModel(x_size=len(Color)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(N_EPOCHS):
        # Train model on train data
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
        test_losses.append(test_loss)
        test_accuracy = n_correct / len(test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch: {epoch}, "
              f"training loss={train_loss}, "
              f"test loss={test_loss}, "
              f"accuracy={test_accuracy * 100:0.1f}%")

    # Plot loss and accuracy
    _, axis_1 = plt.subplots()

    axis_1.set_xlabel('Epochs')
    axis_1.set_ylabel('Loss')
    train_loss_plot = axis_1.plot(range(N_EPOCHS), train_losses, color='red', label="Train loss")
    test_loss_plot = axis_1.plot(range(N_EPOCHS), test_losses, color='blue', label="Test loss")
    axis_1.tick_params(axis='y')

    axis_2 = axis_1.twinx()
    axis_2.set_ylabel('Accuracy')
    accuracy_plot = axis_2.plot(range(N_EPOCHS), test_accuracies, color='green', label="Accuracy")
    axis_2.tick_params(axis='y')

    plots = train_loss_plot + test_loss_plot + accuracy_plot
    labels = [plot.get_label() for plot in plots]
    plt.legend(plots, labels, loc=0)

    plt.show()

    models_dirpath = data_dirpath / 'models'
    models_dirpath.mkdir(parents=True, exist_ok=True)
    model_filepath = models_dirpath / 'model.pt'
    torch.save(model.state_dict(), model_filepath)
