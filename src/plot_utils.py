""" Helper functions to plot figures. """

import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from itertools import combinations_with_replacement, combinations
from numpy import zeros_like, triu_indices_from, nan
from husl import hex_to_husl
from matplotlib.colors import LinearSegmentedColormap


def plot_loss(loss_train, loss_val, title):
    assert len(loss_train) == len(loss_val), "Losses should have the same lengths"
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=loss_train, ax=ax)
    ax = sns.lineplot(data=loss_val, ax=ax)
    ax.set_title(title)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.plot()
    return fig


def plot_confusion(confusion, title):
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion,
        annot=True,
        xticklabels=["Plain", "Cipher"],
        yticklabels=["Plain", "Cipher"],
        ax=ax,
    )
    ax.set_xlabel("Prediction", fontsize=20)
    ax.set_ylabel("Ground Truth", fontsize=20)
    ax.set_title(title)
    plt.plot()
    return fig
