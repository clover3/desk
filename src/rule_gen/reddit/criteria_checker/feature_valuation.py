import logging

import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def feature_valuation_over_train_data2(extract_feature, n_item, n_train, sb):
    data_name = "train_data2"
    items = read_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    items = items[:n_item]
    #   Load features
    # feature_list = ["my mom", "my friend", "r/askscience"]
    #   Make holdout from train
    # Check if the given feature alone can predict labels.
    #   Check IG, and filter
    #   evaluate on the holdout set
    #   import tqdm
    arr = []
    labels = []
    for text, label_s in tqdm.tqdm(items):
        row = []
        x_i = extract_feature(text)
        row.append(x_i)
        labels.append(int(label_s))
        arr.append(row)
    arr = np.array(arr)
    labels = np.array(labels)
    X_train = arr[:n_train]
    y_train = labels[:n_train]
    X_val = arr[n_train:]
    y_val = labels[n_train:]

    def get_value(y_true, y_pred):
        m = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fp, tp = m[:, 1]
        return tp / (tp + fp), tp

    # askscience: Based on personal experience.
    v = get_value(y_train, X_train[:, 0])
    print("Train", v)
    v = get_value(y_val, X_val[:, 0])
    print("Val", v)


def feature_valuation_inner(extract_feature, items, use_tqdm=True):
    arr = []
    labels = []
    if use_tqdm:
        itr = tqdm.tqdm(items)
    else:
        itr = items
    for text, label_s in itr:
        row = []
        x_i = extract_feature(text)
        row.append(x_i)
        labels.append(int(label_s))
        arr.append(row)
    X = np.array(arr)
    y = np.array(labels)
    def get_value(y_true, y_pred):
        m = confusion_matrix(y_true, y_pred)
        fp, tp = m[:, 1]
        return tp / (tp + fp), tp

    return get_value(y, X[:, 0])
