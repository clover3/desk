from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.s9.clf_for_one import compute_f_k
from rule_gen.reddit.s9.j_res_loader import load_llg_toxic, load_s9
from rule_gen.reddit.single_run2.info_gain import information_gain


def main():
    s9_run_name = "llama_s9nosb"
    print("load_run_name_fmt", s9_run_name)
    # sb_list = get_split_subreddit_list("train")
    sb_list = get_split_subreddit_list("val")


    def load_fn(sb, split):
        dataset = f"{sb}_2_{split}_100"
        X_s9 = load_s9(dataset, s9_run_name)
        X_t = load_llg_toxic(dataset, "llg_toxic")
        X_llg = load_llg_default(dataset, "llg_default")
        X = np.concatenate([X_s9, X_t], axis=1)
        X = X_t
        labels = load_labels(dataset)
        y = right(labels)
        return X, y

    run_exp_over_sb(sb_list, load_fn)


if __name__ == "__main__":
    main()


def run_exp_over_sb(sb_list, load_fn: Callable[[str, str], tuple]):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    table = []
    table2 = []
    for sb in sb_list:
        try:
            X_train, y_train = load_fn(sb, "train")
            X_test, y_test = load_fn(sb, "val")
            clf = LogisticRegression(penalty="l1", solver="liblinear")
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            train_score = f1_score(y_train, train_pred)
            val_pred = clf.predict(X_test)
            val_score = f1_score(y_test, val_pred)
            f19 = compute_f_k(y_test, val_pred, 19)
            row = [sb, val_score, f19]
            table.append(row)
            row2 = [sb, val_score]
            for t in clf.coef_[0]:
                row2.append("{:.3f}".format(t))

            row2.append("{:.3f}".format(clf.intercept_[0]))
            table2.append(row2)
        except FileNotFoundError as e:
            print(e)
    print_table(table)
    print_table(table2)


def compute_ig_over_sb(sb_list, load_fn: Callable[[str, str], tuple]):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    table = []
    for sb in sb_list:
        try:
            X_train, y_train = load_fn(sb, "train")
            n_feature = X_train.shape[1]
            row = [sb]
            for i in range(n_feature):
                ig = information_gain(X_train, y_train, i)[0]
                row.append(ig)
            table.append(row)
        except FileNotFoundError as e:
            print(e)
    print_table(table)
    df = pd.DataFrame(table, columns=["sb"] + ["IG_{}".format(i+1) for i in range(n_feature)])
    for i in range(n_feature):
        print(i, np.average(df["IG_{}".format(i+1)]))
