import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_graph(path: str, y_label: str, x_label: str, **kwargs):
    plt.clf()
    colors = ["b", "r", "c", "y", "k"]
    for i, (k, v) in enumerate(kwargs.items()):
        plt.plot(v, c=colors[i], label=k)
    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(path)
    print(f"Graph saved in {path}")
    plt.clf()


def get_confusion_matrix(cm, classes, path: str, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    ticks_marks = np.arange(len(classes))
    plt.xticks(ticks_marks, classes)
    plt.yticks(ticks_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(path)
