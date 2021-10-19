""" Helper functions to evaluate classifiers and key search strategies. """

from time import perf_counter
from src.data_utils import *
from src.model_utils import load_model
from random import sample
import src.cracker as Cracker
import src.cipher as Cipher
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from collections import Counter
from itertools import chain
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from src.trainer import Trainer

import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)



def get_rank_correct(crackers, cipher, enc, correct_key, order, use_loss=False):
    """ Compute the rank of correct keys across all partial decryptions """
    
    #Compute correct partial keys
    prev = 0
    correct_keys = [correct_key]
    key = correct_key
    for l, _ in crackers.items():
        for j in range(0, l-prev):
            key = list(key)
            idx = cipher.alphabet.index(order[-(l-j)])
            key[idx] = cipher.alphabet[0]
            key = ''.join(key)
        correct_keys.append(key)
        prev = l
    # Compute correct partial decryptions
    correct_decs = []
    for k in correct_keys:
        dec = cipher.decrypt(enc, k)
        correct_decs.append(dec)
    
    # Generate and evaluate decryption attempts
    prefixes = ['']
    ranks = []
    for i, (l, cracker) in enumerate(crackers.items()):
        correct_dec = correct_decs[-1-i]
        correct_key = correct_keys[-2-i]
        correct_p = cracker.model_dict[l].predict(correct_dec, cipher.alphabet, use_loss)
        df_sol = cracker.key_search(enc, order, use_loss=use_loss, prefixes=prefixes, strategy='iterative', n_top=0, n_candidates=10)
        
        # Set prefix to correct key
        prefixes = [correct_key]
        df_sol = df_sol.sort_values(by='P', ascending=False).reset_index()
        rank_correct = df_sol[df_sol['Key'] == correct_key].index
        ranks.append(rank_correct)
    return ranks
    
    
def evaluate_cracker(
    model_path,
    samples,
    order,
    cipher,
    cracker,
    key_rot=1,
    lower=False,
    balanced=False
):
    """Evaluate the accuracy of recovered keys for a specific cracker"""
    
    encs, keys = cipher.encrypt_all(samples, order, key_rot=key_rot)

    # Break keys
    time1 = perf_counter()
    key_pred = cracker.key_search_all(encs)
    time2 = perf_counter()

    # Evaluate
    acc = accuracy_score(keys, key_pred)
    if balanced:
        acc = balanced_accuracy_score(keys, key_pred)

    print(f"{acc*100}% of keys were recovered correctly.")
    print(f"in {time2 - time1} seconds")
    return acc


def evaluate_cracker_across_bins(
    model_path,
    data_id,
    alphabet,
    bin_size=10,
    max_length=100,
    max_samples=500,
    key_rot=1,
    lower=False,
    balanced=False,
):
    """Get the accuracies of breaking the cipher across different message lengths."""
    
    model = load_model(model_path, len(alphabet))

    cipher = Cipher.Caesar(alphabet)
    cracker = Cracker.Caesar(alphabet, model)

    accs = []

    for b in tqdm(range(1, max_length, bin_size)):
        # Sample data
        msgs = load_msgs(data_id, alphabet, b, b + bin_size, lower=lower)
        order = get_char_order(msgs, alphabet)
        n_samples = min(max_samples, len(msgs))
        samples = sample(msgs, n_samples)
        acc = evaluate_cracker(
            model_path,
            samples,
            order,
            cipher,
            cracker,
            key_rot=key_rot,
            lower=lower,
            balanced=balanced,
        )
        accs.append(acc)
    return accs


def evaluate_model_across_bins(
    model_path,
    data_id,
    alphabet,
    bin_size=10,
    max_length=100,
    lower=False,
    ciphers=None,
    balanced=False,
    max_samples=500,
):
    """Get the accuracy of the specified model on predicting plain / cipher over different message lengths."""

    model = load_model(model_path, len(alphabet))
    criterion = torch.nn.CrossEntropyLoss()

    lengths = range(0, max_length, bin_size)
    accs = []
    trainer = Trainer(nb_epochs=1)
    for l in lengths:
        dl_test, _ = gen_dl(
            data_id,
            alphabet,
            l + 1,
            l + bin_size,
            val_size=0.01,
            lower=lower,
            ciphers=ciphers,
            max_samples=max_samples,
        )
        if balanced:
            measure = balanced_accuracy_score
        else:
            measure = accuracy_score

        score = trainer.score(model, dl_test, measure)
        accs.append(score)

    return accs


def n_high_prob(probs, thresh):
    """Return propbabilites above the specified threshold."""
    
    return len([p for p in probs if thresh <= p])


def _set_notes(ax, notes):
    """Helper methods for plots. Allows to place notes next to the plot."""
    
    ax.text(1.03, 0.85, notes, transform=ax.transAxes, verticalalignment="top", size=12)
    return ax


def plot_dist(values, bins, title, notes, log=False):
    """Plot a distribution of values grouped by the specified bins."""
    
    ax = sns.barplot(x=values, y=bins)
    ax.set_title(title, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax = _set_notes(ax, notes)
    if log:
        ax.set_yscale("log")
    return ax


def get_prob_dist(probs, n_bins, focus):
    """Group probabilities in the specified number of bins."""
    
    _, bins = pd.cut(focus, n_bins, retbins=True)
    prob_dist = probs.groupby(pd.cut(probs, bins)).count()
    return prob_dist


def plot_key_prob_dist(
    msgs,
    cipher,
    cracker,
    n_keys,
    n_msgs,
    order,
    title,
    notes,
    strategy="random",
    partial_size=0,
    n_bins=10,
    focus=(0, 1),
    log=False,
):
    """Plot the distribution of likelyhood for random decryptions
    Args:
        msgs: list(string). Encrypted input messages
        cipher: Cipher. The cipher to generate the random decryptions
        cracker: Cracker. The cracker to compute the likelyhods of decryptions
        n_keys: int. The number of keys of which to generate the distribution
        n_msgs: int. The number of messages across which the distributions are averaged
        title: string. The title of the plot
        notes: string. Additional notes to the plot
        n_bins: int. Number of bins used for the distribution
        focus: tuple(int). Specifies the interval of the distribution to plot.
        log: bool. Flag to specify whether to set the yscale to log.
    """
    
    plt.figure(figsize=(10, 5))
    keys = []
    _, bins = pd.cut(focus, n_bins, retbins=True)
    samples = sample(msgs, n_msgs)
    for msg in samples:
        new_keys = top_random_keys(
            msg, cipher, cracker, n_keys, n_keys, strategy, partial_size, order
        )
        new_keys = new_keys.drop(index=0)
        new_keys = new_keys.P.groupby(pd.cut(new_keys.P, bins)).count()
        keys.append(new_keys)
    keys = pd.concat(keys).groupby(level=0).mean()
    ax = sns.barplot(x=keys.index, y=keys.values)
    ax.set_title(title, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax = _set_notes(ax, notes)
    if log:
        ax.set_yscale("log")
    return ax


def stats_top_correct_keys(samples, cipher, cracker, partial_size, order, thresh=0.9):
    """Get statistics of correct keys"""
    
    cipher_plain = Cipher.Plain(cipher.alphabet)
    samples_masked, _ = cipher_plain.encrypt_all(
        samples, order, 1, "random", partial_size
    )
    likely_hood = cracker.likely_hood(samples_masked, partial_size=partial_size)
    med = likely_hood.median()
    avg = likely_hood.mean()
    n_top = n_high_prob(likely_hood, thresh)
    return n_top / len(samples), avg, med


def n_top_random_keys(
    samples, cipher, cracker, n_keys, key_gen_strategy, partial_size, order, thresh=0.9
):
    """Get the number of keys with high probability over random keys averaged over all samples"""
    
    keys = pd.DataFrame({"Key": [], "P": []})
    n_top_keys = []
    for msg in samples:
        keys = keys.append(
            top_random_keys(
                msg,
                cipher,
                cracker,
                n_keys,
                n_keys,
                key_gen_strategy,
                partial_size,
                order,
            )
        )
    keys["likely"] = keys.P.map(lambda x: thresh < x)
    return keys.likely.sum() / len(samples)


def top_random_keys(
    msg,
    cipher,
    cracker,
    n_top_keys,
    n_random_keys,
    key_gen_strategy,
    partial_size,
    order,
):
    """Get the keys of the most likely random decryptions of msg"""
    
    encs, keys = cipher.encrypt_all(
        [msg], order, key_gen_strategy=key_gen_strategy, partial_size=partial_size
    )
    enc, key = encs[0], keys[0]

    # Generate random decryption keys
    random_keys = cipher.gen_keys(
        n_random_keys,
        key_gen_strategy,
        order,
        partial_size=partial_size,
        char_space=enc,
    )

    # Remove duplicates
    random_keys = list(set(random_keys))
    top_keys = cracker.top_keys(enc, random_keys, n_top_keys, partial_size)
    # true_key = cracker.top_keys(enc,[key], 1, partial_size)
    # top_keys = true_key.append(top_keys, ignore_index=True)
    top_keys["Distance"] = top_keys.Key.map(lambda x: hamming_distance(x, key))
    top_keys["Dec"] = top_keys.Key.map(lambda x: cipher.decrypt(enc, x))
    return top_keys


def plot_accuracies(accs, title, n_samples, bin_size=10, max_length=100):
    """Plot accuracies in a bar plot."""
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(y=accs, x=[f"[{i}, {i + 10}]" for i in range(0, max_length, 10)])
    ax.set_title(f"{title}", fontsize=20)
    ax.set_xlabel("Message length interval")
    ax.set_ylabel("Accuracy")
    for i, b in enumerate(ax.patches):
        ax.annotate(
            f"n={n_samples[i]}",
            xy=(b.get_x() + b.get_width() / 2, 10),
            size=10,
            ha="center",
            va="center",
            xytext=(0, -12),
            textcoords="offset points",
        )

    return ax


def compare_measure(measure, x_axis, title, y_label, notes="", log=False):
    """Compare different groups of the specified measure
    Args:
        measure: list(float). The measure to plot
        x_axis: list(string). The different groups to compare
        y_label: string. The description of the measure
    """

    ax = sns.barplot(y=measure, x=x_axis)
    ax = _set_notes(ax, notes)
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if log:
        ax.set_yscale("log")
    return ax


def compare_accuracies(accs, group, title, bin_size=10, max_length=100, notes=""):
    """Compare accuracies of different groups across different message lengths."""

    col_groups = []
    col_axis = []
    col_accs = []
    for i, acc in enumerate(accs):
        col_groups.extend([group[i] for _ in acc])
        col_axis.extend([f"[{i}, {i + 10}]" for i in range(0, max_length, 10)])
        col_accs.extend(acc)
    df = pd.DataFrame({"acc": col_accs, "group": col_groups, "x_axis": col_axis})
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, y="acc", hue="group", x="x_axis")
    ax.set_title(f"{title}", fontsize=20)
    ax.set_xlabel("Message length interval")
    ax.set_ylabel("Balanced accuracy")
    ax.text(1.03, 0.85, notes, transform=ax.transAxes, size=12)
    return ax


def plot_msg_length(msgs, title, bin_size=10, max_length=5000):
    """Plot the distribution of message lengths"""

    df = pd.DataFrame(msgs, columns=["Txt"])
    df["Len"] = df.Txt.map(lambda x: len(x))
    df = df.drop(columns=["Txt"])
    df = df.groupby(pd.cut(df["Len"], np.arange(0, max_length, bin_size))).count()
    ax = sns.barplot(x=df.index, y=df.Len)
    ax.set_title(f"{title} msg lengths")
    return ax


def _group_char(c, groups):
    """ Find group for character """
    
    for key in groups.keys():
        if c in groups[key]:
            return key

    return "unknown char"


def plot_msg_dist(data_id, alphabet, title, bin_size=10, max_length=100):
    """Plot the distribution of characters over a specific data set."""

    lengths = range(0, max_length, bin_size)
    bins = {
        "special keys": list(chain(range(0, 32), range(127, 128))),
        "special chars": list(
            chain(range(33, 48), range(58, 65), range(91, 97), range(123, 127))
        ),
        "lower case": range(97, 123),
        "upper case": range(65, 91),
        "numbers": range(48, 58),
        "spaces": range(32, 33),
    }

    df = pd.DataFrame(bins.keys(), columns=["Group"])
    for i, l in enumerate(lengths):
        msgs = load_msgs(data_id, alphabet, l, l + bin_size)
        msgs = "".join(msgs)
        c = Counter(alphabet)
        for elem in alphabet:
            c[elem] -= 1
        char_count = Counter(msgs)
        c.update(char_count)
        num_char = sum(c.values())

        df_temp = pd.DataFrame(c.values(), columns=[f"[{l}, {l + bin_size}]"])
        df_temp["Group"] = df_temp.index.map(lambda x: _group_char(x, bins))

        df_temp = df_temp.groupby("Group").sum() / num_char
        df = pd.merge(df, df_temp, on="Group", how="left")

    df = df.fillna(0)
    df = df.set_index("Group")
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(df, square=True, annot=True)
    ax.set_xlabel("Message length interval")
    ax.set_title(title, fontsize=20)
    return df
