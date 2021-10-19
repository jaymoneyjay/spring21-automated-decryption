""" Helper functions for data manipulation and processing. """

from typing import List
from pathlib import Path
from decouple import config

import torch, os
import numpy as np
from cipher import gen_cipher_dict
from torch.utils.data import TensorDataset, DataLoader
from pandas import read_pickle, DataFrame
from pickle5 import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

FILES = {
    "alice": "data/text/alice.txt",
    "bible": "data/text/bible.txt",
    "alice-train": "data/alice_train_10-100.pkl",
    "alice-test": "data/alice_test_10-100.pkl",
    "bible-train": "data/bible_train_10-100.pkl",
    "bible-test": "data/bible_test_10-100.pkl",
    "acars-manual-train": "data/acars_manual_train.pkl",
    "acars-manual-test": "data/acars_manual_test.pkl",
    "acars-sensor-train": "data/acars_sensor_stripped_full_train.pkl",
    "acars-sensor-test": "data/acars_sensor_stripped_full_test.pkl",
    "sms-en-train": "data/sms_en_train.pkl",
    "sms-en-test": "data/sms_en_test.pkl",
    "sms-zh-train": "data/sms_zh_train.pkl",
    "sms-zh-test": "data/sms_zh_test.pkl",
}


def gen_dl(
    data_id,
    alphabet,
    msg_len_min=40,
    msg_len_max=500,
    ciphers=None,
    shuffle=True,
    max_samples=10000,
    batch_size=20,
    key_rot=1,
    delimiter=None,
    val_size=0.2,
    keep_ws=True,
    save_spaces=True,
    lower=False,
    key_gen_strategy="random",
    partial_lengths=None,
):
    """Generate a DataLoader from the specified data source

    The texts found in the data source specified by data_id are parsed, cleaned, split into messages and encrypted
        based on the specified parameters.

    Args:
        data_id: string. ID to specify the dataset to use. Values=['alice', 'bible']
        alphabet: string. ID to specify the alphabet to use.
            Values=['basic', 'basic-nowhite', 'ascii', 'ascii-nowhite']
        msg_len_min: int (optional). Minimal length for the messages to have.
        msg_len_max: int (optional). Maximal length for the messages to have.
        ciphers: list[string] (optional). List of ciphers to generate the encryptions.
            Values=['caesar', 'substitution', 'vigenere', 'columnar']
        shuffle: bool (optional). Whether to shuffle the DataLoader or not.
        max_samples: int (optional). Maximum number of samples to pack in DataLoader.
        batch_size: int (optional). Batch size.
        key_rot: int (optional). The frequency to rotate the key when generating the encryptions.
        delimiter: char (optional).
            If set: messages are created by separating the text at this character.
            If None: msg_len_min and msg_len_max are used to separate the text.
        val_size: float (optional). Size of validation set.
        keep_ws: bool (optional). Whether to keep whitespaces when parsing the text.
        save_spaces: bool (optional). Whether to keep whitespaces when encrypting the messages.
        lower: bool (optional). Whether to transform all characters to lower case.
        key_gen_strategy: string (optional). Specify strategy used to generate encryption keys.
            'random': Keys are generated randomly.
            'iterative': Keys are generated iteratively.
        partial_lengths: list(int) (optional).
            [0]: Do not encrypt partially.
            else: The lengths of partial encryption.


    Returns:
        dl_train: DataLoader.
        dl_val: DataLoader.
    """

    # Set default value of parameters
    if ciphers is None:
        ciphers = {"plain": 1, "caesar": 1, "vigenere": 1, "substitution": 1}

    if partial_lengths is None:
        partial_lengths = [0]

    df, order = _gen_df(
        data_id,
        alphabet,
        msg_len_min,
        msg_len_max,
        ciphers,
        delimiter,
        key_rot,
        keep_ws=keep_ws,
        save_spaces=save_spaces,
        lower=lower,
        key_gen_strategy=key_gen_strategy,
        partial_lengths=partial_lengths,
    )

    if max_samples < len(df):
        df = df.sample(max_samples)

    # Drop rows with placeholder entries
    df = df[df.Enc != alphabet[0]]

    # Stratified train test split
    df_train, df_val = train_test_split(df, test_size=val_size, stratify=df["Label"])

    # One hot encode and label the data
    features_train = encode_msgs(df_train.Enc, alphabet)
    features_val = encode_msgs(df_val.Enc, alphabet)
    labels_train = torch.from_numpy(
        df_train.Label.map(lambda x: 0 if x == "plain" else 1).values
    )
    labels_val = torch.from_numpy(
        df_val.Label.map(lambda x: 0 if x == "plain" else 1).values
    )

    # Generate the DataLoader
    data_train = TensorDataset(features_train, labels_train)
    data_val = TensorDataset(features_val, labels_val)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=shuffle)

    return loader_train, loader_val


def load_msgs(
    data_id,
    alphabet,
    msg_len_min,
    msg_len_max,
    delimiter=None,
    keep_ws=True,
    lower=False,
):
    """Load messages from a specified data source
    Args:
        data_id: string. ID to specify the dataset to use. Values=['alice', 'bible']
        alphabet: string. ID to specify the alphabet to use.
            Values=['basic', 'basic-nowhite', 'ascii', 'ascii-nowhite']
        msg_len_min: int (optional). Minimal length for the messages to have.
        msg_len_max: int (optional). Maximal length for the messages to have.
        delimiter: char (optional).
            If set: messages are created by separating the text at this character.
            If None: msg_len_min and msg_len_max are used to separate the text.
        keep_ws: bool (optional). Whether to keep whitespaces when parsing the text.
        lower: bool (optional). Whether to transform all characters to lower case.

    Returns:
        msgs: list(string)
    """

    if delimiter is None:
        split_fn = _splitter_by_len(msg_len_min, msg_len_max)
    else:
        split_fn = _splitter_by_del(delimiter)

    extra_chars = [delimiter]
    if keep_ws:
        extra_chars.append(" ")

    fpath = FILES[data_id]
    fname, fex = os.path.splitext(fpath)
    if fex == ".pkl":
        df = load_df(fpath)
        if msg_len_max > 0:
            assert (
                msg_len_min < msg_len_max
            ), "Msg length max must be larger than msg length min"
            df["Len"] = df.Txt.map(lambda x: len(x))
            df = df[df.Len <= msg_len_max]
            df = df[df.Len >= msg_len_min]
        msgs = list(df.Txt)
        if lower:
            msgs = [m.lower() for m in msgs]
        msgs = _remove_unknown_chars(msgs, alphabet, extra_chars)
    elif fex == ".txt":
        txt = _load_txt(fpath)
        if lower:
            txt = txt.lower()
        txt = _remove_unknown_chars(txt, alphabet, extra_chars)
        msgs = split_fn(txt)
    else:
        raise Exception(f"file extension {fex} unknown")

    return msgs


def _gen_df_cipher(
    msgs,
    alphabet,
    cipher_ID,
    key_rot,
    save_spaces,
    key_gen_strategy,
    partial_lengths,
    order,
    low_dist_data=False,
):
    """Generate a DataFrame with encrypted data

    Args:
        msgs: list[string]. List of input messages.
        alphabet: string. Alphabet to encrypt messages with.
        cipher_ID: string. Cipher to encrypt the messages with.
        key_rot: int. The frequency to rotate the key when generating the encryptions.
        save_spaces: bool. Whether to keep whitespaces when encrypting the messages.
        key_gen_strategy: string. Specify strategy used to generate encryption keys.
        partial_lengths: list(int).
            [0]: Do not encrypt partially.
            else: The lengths of partial encryption.
        order: string. Order in which to encrypt characters. Important for partial encryption.
        low_dist_data: bool (optional). Whether to generate the encryptions to be close to the correct decryption.

    Returns:
        df: DataFrame.
    """

    txt_store = []
    enc_store = []
    key_store = []
    label_store = []
    cipher_store = []

    cipher_dict = gen_cipher_dict(alphabet=alphabet, save_spaces=save_spaces)

    for l in partial_lengths:
        cipher = cipher_dict[cipher_ID]
        enc, key = cipher.gen_data(
            msgs,
            order,
            key_rot=key_rot,
            key_gen_strategy=key_gen_strategy,
            partial_size=l,
            low_dist_data=low_dist_data,
        )
        txt_store.extend(msgs)
        enc_store.extend(enc)
        key_store.extend(key)
        if cipher_ID == "plain":
            label = ["plain" for _ in msgs]
        else:
            label = ["cipher" for _ in msgs]
        label_store.extend(label)
        cipher_store.extend(cipher_ID for _ in msgs)

    df = DataFrame(
        {
            "Txt": txt_store,
            "Enc": enc_store,
            "Key": key_store,
            "Label": label_store,
            "Cipher": cipher_store,
        }
    )
    return df


def _gen_df(
    data_id,
    alphabet,
    msg_len_min,
    msg_len_max,
    ciphers,
    delimiter,
    key_rot,
    keep_ws,
    save_spaces,
    lower,
    key_gen_strategy,
    partial_lengths,
):
    """Generate a DataFrame with plain and ciphertext data.

    Returns:
        df: pandas.DataFrame. DataFrame with the columns ['Txt', 'Enc', 'Key', 'Label', 'Cipher']
    """

    # Set default value of parameters
    if ciphers is None:
        ciphers = {"plain": 1, "caesar": 1, "vigenere": 1, "substitution": 1}

    msgs = load_msgs(
        data_id,
        alphabet,
        msg_len_min,
        msg_len_max,
        delimiter=delimiter,
        keep_ws=keep_ws,
        lower=lower,
    )
    order = get_char_order(msgs, alphabet)
    cipher_dict = gen_cipher_dict(alphabet=alphabet, save_spaces=save_spaces)

    df = DataFrame({"Txt": [], "Enc": [], "Key": [], "Label": [], "Cipher": []})

    for cipher_ID, rep in zip(ciphers.keys(), ciphers.values()):
        for _ in range(rep):
            df_cipher = _gen_df_cipher(
                msgs,
                alphabet,
                cipher_ID,
                key_rot,
                save_spaces,
                key_gen_strategy,
                partial_lengths,
                order,
            )
            df = join(df_cipher, df)
            # Generate low distance data
            if cipher_ID != "plain":
                df_low_dist = _gen_df_cipher(
                    msgs,
                    alphabet,
                    cipher_ID,
                    key_rot,
                    save_spaces,
                    key_gen_strategy,
                    partial_lengths,
                    order,
                    low_dist_data=True,
                )
                df = join(df_low_dist, df)
    return df, order


def join(df1, df2):
    """Join two DataFrames.
    Args:
        df1: pandas.DataFrame. First df
        df2: pandas.DataFrame. Other df
    """
    
    assert (df1.columns == df2.columns).all(), "Columns have to match"

    return df1.append(df2).reset_index(drop=True)


def split_df(df, frac):
    """Split dataframe.
    Args:
        df: pandas.DataFrame. Input data.
        frac: float. Fraction of data to split
    Returns:
        df: pandas.DataFrame. DataFrame without the subset
        subset: pandas.DataFrame. The subset split from the DataFrame
    """
    
    subset = df.sample(frac=frac)
    df = df.drop(subset.index)
    subset = subset.reset_index(drop=True)
    df = df.reset_index(drop=True)
    return df, subset


def get_char_order(msgs, alphabet):
    """Compute the character order by decreasing frequency in msgs.

    Args
        msgs: list. Input messages to compute the character frequencies over.
        alphabet: string. Alphabet
    """

    corpus = "".join(msgs)
    counts = Counter(corpus)
    if " " in counts.keys():
        counts.pop(" ")
    counts = counts.most_common()
    order = [item[0] for item in counts if item[0] != alphabet[0]]
    for c in alphabet[1:]:
        if c not in order:
            order.append(c)
    return "".join(order)


def encode_msgs(msgs, alphabet):
    """One-hot-encode a list of messages.

    Args:
        msgs: list(string). List of messages to encode
        alphabet: string. Alphabet

    Returns:
        one_hot: torch.tensor. Binary tensor of one-hot encoded messages"""

    alphabet_dict = _get_alphabet_dict(alphabet)

    num_char = len(alphabet_dict)

    one_hot = np.zeros(
        (len(msgs), num_char, config("MAX_SEQ_LEN", cast=int)), dtype=np.float32
    )
    for i, m in enumerate(msgs):
        for j, c in enumerate(str(m[: config("MAX_SEQ_LEN", cast=int)])[::-1]):
            try:
                one_hot[i, alphabet_dict[c], j] = 1.0
            except:
                pass  # unknown characters will be encoded as all zeros

    return torch.from_numpy(one_hot)


def _split_string(txt, msg_size):
    """Split string into chunks of msg_size"""
    
    chunks = len(txt) // msg_size
    msgs = [
        txt[i * msg_size : min(len(txt), (i + 1) * msg_size)] for i in range(0, chunks)
    ]
    return msgs


def _splitter_by_del(delimiter):
    """Split txt into messages using a delimiter"""

    def split_txt(txt):
        msgs = txt.split(delimiter)
        return msgs

    return split_txt


def _splitter_by_len(len_min, len_max):
    """Create splitter function to split txt into messages of random length in the specified interval"""

    def split_txt(txt):
        sizes = np.random.randint(len_min, len_max, size=(len(txt) // len_min + 1))
        acc_len = 0
        msgs = []
        for size in sizes:
            if acc_len + size > len(txt):
                break

            msg = txt[acc_len : acc_len + size]
            msgs.append(msg)
            acc_len += size

        return msgs

    return split_txt


def _load_txt(fpath):
    """Reads in txt from a txt file"""
    
    with open(fpath, "r") as f:
        content = f.read()

    return content


def load_df(fpath):
    """Load dataframe"""
    
    return pickle.load(open(fpath, "rb"))


def dump_df(df, fpath):
    """Dump dataframe"""
    
    df.to_pickle(fpath)


def load_full_acars_manual():
    """Load the acars txt data and filter for manual"""
    
    data = load_df("/home/jvieli/data/full_acars.pkl")
    return data[
        ((data["Label"] == "C1") | (data["Label"] == "RA"))
        & (data["db_channel"] == "sat")
    ]


def load_full_acars_sensor():
    """Load the acars txt data and filter for sensor"""
    
    data = load_df("/home/jvieli/data/full_acars.pkl")
    return data[(data["Label"] == "H1") & (data["db_channel"] == "sat")]


def _get_alphabet_dict(alphabet):
    """Assign char to index mapping."""
    
    char_to_token = {c: i for i, c in enumerate(alphabet)}

    return char_to_token


def _remove_unknown_chars(txt, alphabet_name, extra_chars):
    """Removes all characters from a text or a list of messages which are not in alphabet nor in extra_chars."""
    
    if type(txt) == list:
        return [
            _remove_unknown_chars_in_msg(msg, alphabet_name, extra_chars) for msg in txt
        ]
    else:
        return _remove_unknown_chars_in_msg(txt, alphabet_name, extra_chars)


def _remove_unknown_chars_in_msg(msg, alphabet, extra_chars):
    """Removes all characters from a single message which are not in alphabet nor in extra_chars."""
    
    return "".join([c for c in msg if c in alphabet or c in extra_chars])