from contraction._train.metrics import Metrics

HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MATPLOTLIB = False


def plot_metrics(metrics: Metrics):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required to plot metrics")

    _, axis_1 = plt.subplots()
    x_series = range(len(metrics.get('epoch')))

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
